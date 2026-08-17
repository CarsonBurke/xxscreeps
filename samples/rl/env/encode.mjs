/**
 * Encode visible Screeps rooms as ViT-style 5×5 patch tensors + actor/target tables.
 * Shared schema: ../schema.json
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from 'xxscreeps/game/constants/index.js';
import { parseRoomName } from 'xxscreeps/game/room/name.js';
import { RoomPosition } from 'xxscreeps/game/position.js';
import { checkCreateConstructionSite } from 'xxscreeps/mods/construction/room.js';
import { structureFactories } from 'xxscreeps/mods/construction/symbols.js';

const schema = JSON.parse(
	fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '../schema.json'), 'utf8'),
);

export const SCHEMA = schema;
const {
	roomSize, patchSize, patchesPerSide, patchesPerRoom,
	maxRooms, maxActors, maxCreepActors, maxStructureActors, maxTargets, intentSlots,
	maxBodyParts,
	tileFeat, actorFeat, targetFeat, intentTypes, intentSpecs, amountBins, constructionTypes,
	actionOutcomes, bodyPartTypes, bodyPartCosts, maxRoomEnergy,
} = schema;

const INTENT = Object.fromEntries(intentTypes.map((n, i) => [ n, i ]));
export { INTENT };
if (!Array.isArray(schema.actorFeatures) || schema.actorFeatures.length !== actorFeat) {
	throw new Error(`schema actorFeatures must contain exactly actorFeat=${actorFeat} names`);
}
if (intentSlots !== 1) {
	throw new Error(`the current policy requires intentSlots=1, got ${intentSlots}`);
}
const CONSTRUCTION_TYPES = constructionTypes.map(type => C[type.toUpperCase()] || type);
const BODY_PART_CONST = bodyPartTypes.map(type => {
	const part = C[type.toUpperCase()];
	if (!part) throw new Error(`schema bodyPartTypes contains unknown part ${JSON.stringify(type)}`);
	return part;
});
const CONSTRUCTION_TILES = roomSize * roomSize;
const CONSTRUCTION_MASK_BYTES = Math.ceil(CONSTRUCTION_TILES / 8);
if (bodyPartCosts.length !== schema.bodyPartTypes.length || bodyPartCosts.some(cost => cost <= 0)) {
	throw new Error('schema bodyPartCosts must align one positive cost per bodyPartTypes entry');
}
const MIN_BODY_COST = Math.min(...bodyPartCosts);
const TARGET_INTENT_NAMES = intentTypes.filter(name =>
	intentSpecs[name]?.factors?.includes('target')
);

/**
 * Observation wire format (RL_OBS_FMT):
 *   json  — legacy decimal arrays (debug only, huge)
 *   b64   — per-field base64 float32/uint8
 *   pack  — single base64 blob of concatenated fields (text-pipe)
 *   bin   — length-prefixed raw frames on stdout (best; default in Python)
 */
const OBS_FMT = process.env.RL_OBS_FMT || 'bin';
const ANCHOR_ROOM = process.env.RL_ROOM || 'W7N3';

function f32ToWire(f32) {
	if (OBS_FMT === 'json') return Array.from(f32);
	// base64 of little-endian float32 bytes
	return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString('base64');
}

function u8ToWire(arr) {
	// arr is 0/1 number array or Uint8Array
	const u8 = arr instanceof Uint8Array ? arr : Uint8Array.from(arr, v => (v ? 1 : 0));
	if (OBS_FMT === 'json') return Array.from(u8);
	return Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString('base64');
}

/** Concatenate typed arrays into one Buffer (for pack encoding). */
function packBlob(parts) {
	const total = parts.reduce((n, p) => n + p.byteLength, 0);
	const out = Buffer.allocUnsafe(total);
	let o = 0;
	for (const p of parts) {
		const buf = Buffer.from(p.buffer, p.byteOffset, p.byteLength);
		buf.copy(out, o);
		o += p.byteLength;
	}
	return out.toString('base64');
}

function toU8(arr) {
	return arr instanceof Uint8Array ? arr : Uint8Array.from(arr, v => (v ? 1 : 0));
}

/**
 * Convert a tile coordinate to its schema patch-major byte offset once. This is
 * used by both the immutable terrain snapshot and direct dynamic painting.
 */
const packedTileOffsets = new Uint32Array(roomSize * roomSize);
for (let y = 0; y < roomSize; y++) {
	for (let x = 0; x < roomSize; x++) {
		const patch = Math.floor(y / patchSize) * patchesPerSide + Math.floor(x / patchSize);
		const cell = (y % patchSize) * patchSize + (x % patchSize);
		packedTileOffsets[y * roomSize + x] = (patch * patchSize * patchSize + cell) * tileFeat;
	}
}

/**
 * Terrain is immutable per room name in the sim. Cache it in final patch-major
 * order so an encode needs one native bulk copy instead of a tile clone followed
 * by 70k interpreted byte assignments per room.
 */
const terrainBaseCache = new Map();

function getTerrainBase(roomName, terrain) {
	let base = terrainBaseCache.get(roomName);
	if (base) return base;
	base = new Uint8Array(roomSize * roomSize * tileFeat);
	for (let y = 0; y < roomSize; y++) {
		for (let x = 0; x < roomSize; x++) {
			const o = packedTileOffsets[y * roomSize + x];
			const ter = terrain.get(x, y);
			if (ter === C.TERRAIN_MASK_WALL) base[o + 2] = 255;
			else if (ter === C.TERRAIN_MASK_SWAMP) base[o + 1] = 255;
			else base[o + 0] = 255;
			if (x === 0 || y === 0 || x === 49 || y === 49) base[o + 27] = 255;
		}
	}
	terrainBaseCache.set(roomName, base);
	return base;
}

const PATCH_BYTES_PER_ROOM = patchesPerRoom * patchSize * patchSize * tileFeat;

function createObservationBuffers() {
	return {
		roomMask: new Uint8Array(maxRooms),
		roomCoords: new Float32Array(maxRooms * 2),
		patches: new Uint8Array(maxRooms * PATCH_BYTES_PER_ROOM),
		actors: new Float32Array(maxActors * actorFeat),
		actorMask: new Uint8Array(maxActors),
		actorOutcome: new Uint8Array(maxActors),
		targets: new Float32Array(maxTargets * targetFeat),
		targetMask: new Uint8Array(maxTargets),
		intentMask: new Uint8Array(maxActors * intentSlots * intentTypes.length),
		dirMask: new Uint8Array(maxActors * intentSlots * 8),
		targetSelectMask: new Uint8Array(intentTypes.length * maxTargets),
		amountMask: new Uint8Array(
			maxActors * intentSlots * intentTypes.length * amountBins.length,
		),
		constructionMask: new Uint8Array(
			maxRooms * CONSTRUCTION_TYPES.length * CONSTRUCTION_MASK_BYTES,
		),
	};
}

function resetObservationBuffers(buffers) {
	// Patches are overwritten room-by-room from a complete terrain snapshot. Only
	// the synthetic one-room payload for zero visible rooms needs explicit clearing.
	buffers.roomMask.fill(0);
	buffers.roomCoords.fill(0);
	buffers.actors.fill(0);
	buffers.actorMask.fill(0);
	buffers.actorOutcome.fill(0);
	buffers.targets.fill(0);
	buffers.targetMask.fill(0);
	buffers.intentMask.fill(0);
	buffers.dirMask.fill(0);
	buffers.targetSelectMask.fill(0);
	buffers.amountMask.fill(0);
	buffers.constructionMask.fill(0);
	return buffers;
}

// Keep two snapshots: the just-returned binary observation remains intact while
// the next tick is encoded. The server frames it synchronously before a third
// acquisition, eliminating steady-state tensor allocation without aliasing
// adjacent observations.
const observationBufferPool = [ createObservationBuffers(), createObservationBuffers() ];
let observationBufferCursor = 0;

function acquireObservationBuffers() {
	const buffers = observationBufferPool[observationBufferCursor];
	observationBufferCursor = (observationBufferCursor + 1) % observationBufferPool.length;
	return resetObservationBuffers(buffers);
}

const STRUCT = {
	controller: 0, spawn: 1, extension: 2, road: 3, container: 4,
	storage: 5, tower: 6, constructedWall: 7, rampart: 8, link: 9, other: 10,
};

function clamp01(x) {
	if (!Number.isFinite(x)) return 0;
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

function q8(x) {
	return Math.round(clamp01(x) * 255);
}

function bodyFrac(creep, type) {
	if (!creep?.body?.length) return 0;
	let n = 0;
	for (const p of creep.body) if (p.type === type && p.hits > 0) n++;
	return n / creep.body.length;
}

function bodyCount(creep, type, activeOnly = true) {
	if (!creep?.body?.length) return 0;
	let n = 0;
	for (const p of creep.body) if (p.type === type && (!activeOnly || p.hits > 0)) n++;
	return n;
}

function structureChannel(struct) {
	if (!struct) return STRUCT.other;
	const t = struct.structureType;
	if (t in STRUCT) return STRUCT[t];
	return STRUCT.other;
}

function constructionSiteName(Game, roomName, structureType) {
	if (structureType !== C.STRUCTURE_SPAWN) return undefined;
	return `rlcs_${Game.time}_${roomName.replace(/[^a-zA-Z0-9]/g, '')}`;
}

// Simulator room wrappers can be recreated across tick loads, so object identity
// is not a stable cache key. The semantic signature below is the invalidation
// contract; room name identifies the slot across those wrappers.
const constructionMaskCache = new Map();

function constructionStateSignature(Game, room) {
	const objects = [
		...room.find(C.FIND_STRUCTURES),
		...room.find(C.FIND_CONSTRUCTION_SITES),
		...room.find(C.FIND_SOURCES),
		...room.find(C.FIND_MINERALS),
	].map(object => [
		object.id, object.structureType || object.constructor?.name,
		object.pos.x, object.pos.y, object.name || '',
	].join(':')).sort();
	return [
		room.controller?.level || 0,
		room.controller?.my ? 1 : 0,
		room['#user'] || '',
		Object.keys(Game.constructionSites || {}).length,
		...objects,
	].join('|');
}

/** Exact bit-packed legality from the same validator used by the engine intent. */
function fillConstructionMask(Game, roomNames, mask) {
	for (let roomIndex = 0; roomIndex < roomNames.length; roomIndex++) {
		const roomName = roomNames[roomIndex];
		const room = Game.rooms[roomName];
		// Lightweight encode fixtures are deliberately plain objects. Exact masks
		// only exist for real engine rooms with registered construction factories.
		if (!room?.controller?.my || typeof room['#lookFor'] !== 'function') continue;
		const signature = constructionStateSignature(Game, room);
		let cached = constructionMaskCache.get(roomName);
		if (cached?.signature === signature) {
			mask.set(
				cached.mask,
				roomIndex * CONSTRUCTION_TYPES.length * CONSTRUCTION_MASK_BYTES,
			);
			continue;
		}
		const roomMask = new Uint8Array(
			CONSTRUCTION_TYPES.length * CONSTRUCTION_MASK_BYTES,
		);
		for (let typeIndex = 0; typeIndex < CONSTRUCTION_TYPES.length; typeIndex++) {
			const structureType = CONSTRUCTION_TYPES[typeIndex];
			if (!structureFactories.has(structureType)) continue;
			const name = constructionSiteName(Game, roomName, structureType);
			const base = typeIndex * CONSTRUCTION_MASK_BYTES;
			for (let tile = 0; tile < CONSTRUCTION_TILES; tile++) {
				const x = tile % roomSize;
				const y = Math.floor(tile / roomSize);
				const pos = new RoomPosition(x, y, roomName);
				if (checkCreateConstructionSite(room, pos, structureType, name) === C.OK) {
					roomMask[base + (tile >> 3)] |= 1 << (tile & 7);
				}
			}
		}
		constructionMaskCache.set(roomName, { signature, mask: roomMask });
		mask.set(
			roomMask,
			roomIndex * CONSTRUCTION_TYPES.length * CONSTRUCTION_MASK_BYTES,
		);
	}
}

/**
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {string} userId
 */
export function encodeObservation(Game, userId, actorOutcomes = null) {
	const roomNames = Object.keys(Game.rooms).sort();
	return encodeObservationFromRooms(Game, userId, roomNames, actorOutcomes);
}

/**
 * Encode using an explicit room map (from player view or stitched peeks).
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {string} userId
 * @param {string[]} roomNames
 */
export function encodeObservationFromRooms(Game, userId, roomNames, actorOutcomes = null) {
	// Avoid StructureController.reservation here: the simulator's synthetic
	// user registry does not always contain the reserving user, while the engine
	// stores the authoritative expiry and allegiance directly on the room state.
	const controllerReservation = controller => (
		controller?.structureType === C.STRUCTURE_CONTROLLER
		&& Number(controller['#reservationEndTime'] || 0) > Number(Game.time || 0)
	);
	const allRoomNames = [ ...new Set(roomNames) ].filter(name => Game.rooms[name]);
	const visibleRoomCount = allRoomNames.length;
	let fullStructureActors = 0;
	for (const roomName of allRoomNames) {
		const room = Game.rooms[roomName];
		if (room.controller?.my) fullStructureActors += 1;
		fullStructureActors += room.find(C.FIND_MY_STRUCTURES)
			.filter(st => st.structureType === C.STRUCTURE_SPAWN || st.structureType === C.STRUCTURE_TOWER)
			.length;
	}
	const fullCreepActors = Object.values(Game.creeps)
		.filter(c => c?.my && allRoomNames.includes(c.room.name)).length;
	const fullActorCount = fullStructureActors + fullCreepActors;
	roomNames = allRoomNames
		.filter(name => Game.rooms[name])
		// Stable slots are part of the learned representation. Ownership changes
		// must not permute every room token after a successful claim.
		.sort((a, b) => (a === ANCHOR_ROOM ? -1 : b === ANCHOR_ROOM ? 1 : a.localeCompare(b)))
		.slice(0, maxRooms);
	// Tensor storage is a rotating snapshot pool. Every field is reset before use;
	// spatial rooms are overwritten from complete packed terrain bases below.
	const {
		roomMask, roomCoords, patches, actors, actorMask, actorOutcome,
		targets, targetMask,
		intentMask, dirMask, targetSelectMask, amountMask, constructionMask,
	} = acquireObservationBuffers();
	const nIntent = intentTypes.length;
	// Compact candidate table: [nIntent, maxTargets]. Actor-specific capability is
	// carried by intentMask; same-room compatibility is derived from entity room ids
	// in the model. This avoids O(actors × slots × intents × targets) observations.
	// Intent-specific amounts avoid aliases such as unaffordable spawn bodies and
	// construction types that are illegal at the current controller level.
	const anchorCoord = parseRoomName(ANCHOR_ROOM);
	for (let ri = 0; ri < roomNames.length; ri++) {
		const rc = parseRoomName(roomNames[ri]);
		roomCoords[ri * 2] = clamp01(0.5 + (rc.rx - anchorCoord.rx) / (2 * maxRooms));
		roomCoords[ri * 2 + 1] = clamp01(0.5 + (rc.ry - anchorCoord.ry) / (2 * maxRooms));
	}

	/** @type {{ kind: string, id: string, room: string, x: number, y: number, ref: any }[]} */
	const actorList = [];
	/** @type {{ kind: string, id: string, room: string, x: number, y: number, ref: any }[]} */
	let targetList = [];

	// --- collect targets with economy priority (sources/ctrl/spawns before walls)
	const ECONOMY_STRUCT = new Set([
		C.STRUCTURE_CONTROLLER, C.STRUCTURE_SPAWN, C.STRUCTURE_EXTENSION,
		C.STRUCTURE_STORAGE, C.STRUCTURE_CONTAINER, C.STRUCTURE_TOWER, C.STRUCTURE_LINK,
	]);
	for (const roomName of roomNames) {
		const room = Game.rooms[roomName];
		if (!room) continue;
		for (const s of room.find(C.FIND_SOURCES)) {
			targetList.push({ kind: 'source', id: s.id, room: roomName, x: s.pos.x, y: s.pos.y, ref: s, pri: 0 });
		}
		for (const st of room.find(C.FIND_STRUCTURES)) {
			const pri = st.structureType === C.STRUCTURE_CONTROLLER ? 1
				: ECONOMY_STRUCT.has(st.structureType) ? 2
					: 8; // roads/walls last
			targetList.push({
				kind: 'structure', id: st.id, room: roomName, x: st.pos.x, y: st.pos.y, ref: st, pri,
			});
		}
		for (const r of room.find(C.FIND_DROPPED_RESOURCES)) {
			targetList.push({ kind: 'resource', id: r.id, room: roomName, x: r.pos.x, y: r.pos.y, ref: r, pri: 3 });
		}
		for (const site of room.find(C.FIND_CONSTRUCTION_SITES)) {
			targetList.push({ kind: 'site', id: site.id, room: roomName, x: site.pos.x, y: site.pos.y, ref: site, pri: 4 });
		}
		for (const c of room.find(C.FIND_CREEPS)) {
			targetList.push({ kind: 'creep', id: c.id, room: roomName, x: c.pos.x, y: c.pos.y, ref: c, pri: 5 });
		}
		for (const m of room.find(C.FIND_MINERALS)) {
			const extractor = room.lookForAt(C.LOOK_STRUCTURES, m.pos.x, m.pos.y)
				.find(st => st.structureType === C.STRUCTURE_EXTRACTOR);
			const harvestable = m.mineralAmount > 0 && extractor?.my
				&& extractor.isActive?.() && extractor.cooldown === 0;
			targetList.push({
				kind: 'mineral', id: m.id, room: roomName, x: m.pos.x, y: m.pos.y,
				ref: m, pri: 6, harvestable,
			});
		}
	}
	const fullTargetCount = targetList.length;
	// Balance the fixed target budget across rooms and semantic categories. A
	// mature room's extensions must not evict every site, creep, or build position.
	const category = target => {
		if (target.kind === 'source' || target.kind === 'mineral') return 0;
		if (target.kind === 'structure' && [
			C.STRUCTURE_CONTROLLER, C.STRUCTURE_SPAWN, C.STRUCTURE_STORAGE,
			C.STRUCTURE_CONTAINER, C.STRUCTURE_TOWER, C.STRUCTURE_LINK,
		].includes(target.ref?.structureType)) return 1;
		if (target.kind === 'resource') return 2;
		if (target.kind === 'site') return 3;
		if (target.kind === 'creep') return 4;
		if (target.ref?.structureType === C.STRUCTURE_EXTENSION) return 5;
		return 7;
	};
	const queues = roomNames.map(roomName => {
		const grouped = Array.from({ length: 8 }, () => []);
		for (const target of targetList.filter(item => item.room === roomName)) {
			grouped[category(target)].push(target);
		}
		for (const group of grouped) group.sort((a, b) => (a.pri - b.pri) || a.id.localeCompare(b.id));
		return { grouped, cursor: 0 };
	});
	const packedTargets = [];
	while (packedTargets.length < maxTargets) {
		let added = false;
		for (const queue of queues) {
			for (let offset = 0; offset < queue.grouped.length; offset++) {
				const index = (queue.cursor + offset) % queue.grouped.length;
				if (!queue.grouped[index].length) continue;
				packedTargets.push(queue.grouped[index].shift());
				queue.cursor = (index + 1) % queue.grouped.length;
				added = true;
				break;
			}
			if (packedTargets.length >= maxTargets) break;
		}
		if (!added) break;
	}
	targetList = packedTargets;
	const targetsUsed = Math.min(maxTargets, targetList.length);
	for (let i = 0; i < targetsUsed; i++) {
		const t = targetList[i];
		targetMask[i] = 1;
		const o = i * targetFeat;
		const kind = {
			source: 0, mineral: 1, structure: 2, resource: 3, creep: 4, site: 5, position: 6,
		}[t.kind] ?? 6;
		targets[o + 0] = kind / 6;
		targets[o + 1] = t.x / 49;
		targets[o + 2] = t.y / 49;
		targets[o + 3] = roomNames.indexOf(t.room) / Math.max(1, maxRooms - 1);
		if (t.kind === 'source') {
			targets[o + 4] = clamp01(t.ref.energy / (t.ref.energyCapacity || 3000));
			targets[o + 8] = clamp01((t.ref.ticksToRegeneration || 0) / 300);
			targets[o + 9] = clamp01((t.ref.energyCapacity || 0) / 3000);
		} else if (t.kind === 'structure') {
			targets[o + 5] = structureChannel(t.ref) / 10;
			targets[o + 6] = clamp01(t.ref.hits / (t.ref.hitsMax || 1));
			targets[o + 7] = t.ref.my ? 1 : 0;
			if (t.ref.store) {
				targets[o + 4] = clamp01((t.ref.store[C.RESOURCE_ENERGY] || 0) / (t.ref.store.getCapacity?.(C.RESOURCE_ENERGY) || 1));
				targets[o + 8] = clamp01((t.ref.store[C.RESOURCE_ENERGY] || 0) / 10000);
				targets[o + 9] = clamp01((t.ref.store.getCapacity?.(C.RESOURCE_ENERGY) || 0) / 10000);
				targets[o + 13] = clamp01((t.ref.store[C.RESOURCE_ENERGY] || 0) / 1000000);
				targets[o + 14] = clamp01((t.ref.store.getCapacity?.(C.RESOURCE_ENERGY) || 0) / 1000000);
			}
			if (t.ref.structureType === C.STRUCTURE_CONTROLLER) {
				const reserved = controllerReservation(t.ref);
				targets[o + 10] = (t.ref.level || 0) / 8;
				targets[o + 11] = clamp01((t.ref.progress || 0) / (t.ref.progressTotal || 1));
				targets[o + 12] = reserved ? 1 : 0;
				targets[o + 15] = reserved && t.ref.room?.['#user'] === userId ? 1 : 0;
			}
		} else if (t.kind === 'creep') {
			targets[o + 4] = t.ref.my ? 1 : 0;
			targets[o + 5] = clamp01((t.ref.store?.[C.RESOURCE_ENERGY] || 0) / (t.ref.store?.getCapacity?.() || 1));
			targets[o + 6] = bodyFrac(t.ref, C.WORK);
			targets[o + 7] = bodyFrac(t.ref, C.ATTACK);
			targets[o + 8] = clamp01((t.ref.ticksToLive || 0) / 1500);
			targets[o + 9] = clamp01(bodyCount(t.ref, C.WORK) / 50);
			targets[o + 10] = clamp01(bodyCount(t.ref, C.CARRY) / 50);
			targets[o + 11] = clamp01(bodyCount(t.ref, C.MOVE) / 50);
			targets[o + 12] = clamp01(bodyCount(t.ref, C.CLAIM) / 50);
			targets[o + 13] = clamp01((t.ref.store?.[C.RESOURCE_ENERGY] || 0) / 2000);
			targets[o + 14] = clamp01((t.ref.store?.getCapacity?.() || 0) / 2000);
		} else if (t.kind === 'resource') {
			targets[o + 4] = clamp01((t.ref.amount || 0) / 1000);
			targets[o + 8] = clamp01((t.ref.amount || 0) / 10000);
		} else if (t.kind === 'site') {
			targets[o + 4] = clamp01(t.ref.progress / (t.ref.progressTotal || 1));
			targets[o + 5] = structureChannel({ structureType: t.ref.structureType }) / 10;
		}
	}

	// Candidate semantics are global and actor-independent. Actor capability and
	// room locality are applied separately by intentMask and the policy. Building
	// this table once avoids repeating the same target scan for every creep.
	const tSel = (intentIdx, ti) => intentIdx * maxTargets + ti;
	for (let ti = 0; ti < targetsUsed; ti++) {
		const t = targetList[ti];
		const st = t.kind === 'structure' ? t.ref?.structureType : null;
		const isSink = st === C.STRUCTURE_SPAWN || st === C.STRUCTURE_EXTENSION
			|| st === C.STRUCTURE_STORAGE || st === C.STRUCTURE_CONTAINER
			|| st === C.STRUCTURE_TOWER || st === C.STRUCTURE_LINK;
		const isWithdrawSource = st === C.STRUCTURE_STORAGE
			|| st === C.STRUCTURE_CONTAINER || st === C.STRUCTURE_LINK;
		const isCtrl = st === C.STRUCTURE_CONTROLLER;
		const my = t.ref?.my;
		const unowned = t.ref?.owner == null;
		const reservation = controllerReservation(t.ref);
		const myReservation = Boolean(
			reservation && t.ref?.room?.['#user'] === userId,
		);

		if ((t.kind === 'source' && (t.ref?.energy || 0) > 0)
			|| (t.kind === 'mineral' && t.harvestable)) {
			targetSelectMask[tSel(INTENT.harvest, ti)] = 1;
		}
		if (isSink && (my || unowned)) {
			const freeE = t.ref?.store?.getFreeCapacity?.(C.RESOURCE_ENERGY);
			if (freeE == null || freeE > 0) targetSelectMask[tSel(INTENT.transfer, ti)] = 1;
			const energy = t.ref?.store?.[C.RESOURCE_ENERGY] || 0;
			if (isWithdrawSource && energy > 0) {
				targetSelectMask[tSel(INTENT.withdraw, ti)] = 1;
			}
		}
		if (t.kind === 'creep' && my && !t.ref?.spawning
			&& (t.ref?.store?.getFreeCapacity?.() ?? 0) > 0) {
			targetSelectMask[tSel(INTENT.transfer, ti)] = 1;
		}
		if (t.kind === 'resource' && (t.ref?.amount || 0) > 0) {
			targetSelectMask[tSel(INTENT.pickup, ti)] = 1;
		}
		if (isCtrl && my) targetSelectMask[tSel(INTENT.upgradeController, ti)] = 1;
		if (isCtrl && !my && unowned && (!reservation || myReservation)) {
			targetSelectMask[tSel(INTENT.claimController, ti)] = 1;
			targetSelectMask[tSel(INTENT.reserveController, ti)] = 1;
		}
		if (isCtrl && !my && (t.ref?.owner || (reservation && !myReservation))) {
			targetSelectMask[tSel(INTENT.attackController, ti)] = 1;
		}
		if (t.kind === 'site') targetSelectMask[tSel(INTENT.build, ti)] = 1;
		if (t.kind === 'structure' && !isCtrl) {
			const hits = t.ref?.hits ?? 1;
			const hitsMax = t.ref?.hitsMax ?? 1;
			if ((my || unowned) && hits < hitsMax) {
				targetSelectMask[tSel(INTENT.repair, ti)] = 1;
			}
			if (!my) targetSelectMask[tSel(INTENT.dismantle, ti)] = 1;
		}
		if (t.kind === 'creep' && !my) {
			targetSelectMask[tSel(INTENT.attack, ti)] = 1;
			targetSelectMask[tSel(INTENT.rangedAttack, ti)] = 1;
		}
		if (t.kind === 'structure' && !my && !isCtrl) {
			targetSelectMask[tSel(INTENT.attack, ti)] = 1;
			targetSelectMask[tSel(INTENT.rangedAttack, ti)] = 1;
		}
		if (t.kind === 'creep' && my) {
			targetSelectMask[tSel(INTENT.heal, ti)] = 1;
			targetSelectMask[tSel(INTENT.rangedHeal, ti)] = 1;
		}
	}

	// --- actors: separately budget strategic structures and mobile creeps. ---
	const structureActors = [];
	for (const roomName of roomNames) {
		const room = Game.rooms[roomName];
		if (room.controller?.my) {
			const pos = room.controller.pos;
			structureActors.push({
				kind: 'room', id: `room:${roomName}`, room: roomName,
				x: pos.x, y: pos.y, ref: room,
			});
		}
		for (const st of room.find(C.FIND_MY_STRUCTURES)) {
			if (st.structureType === C.STRUCTURE_SPAWN || st.structureType === C.STRUCTURE_TOWER) {
				structureActors.push({
					kind: 'structure', id: st.id, room: roomName, x: st.pos.x, y: st.pos.y, ref: st,
				});
			}
		}
	}
	actorList.push(...structureActors.slice(0, maxStructureActors));
	let creepActors = 0;
	for (const name of Object.keys(Game.creeps).sort()) {
		const c = Game.creeps[name];
		if (!c?.my || !roomNames.includes(c.room.name)) continue;
		if (creepActors >= maxCreepActors) break;
		actorList.push({ kind: 'creep', id: name, room: c.room.name, x: c.pos.x, y: c.pos.y, ref: c });
		creepActors += 1;
	}
	const actorsUsed = Math.min(maxActors, actorList.length);
	fillConstructionMask(Game, roomNames, constructionMask);

	for (let ai = 0; ai < actorsUsed; ai++) {
		const a = actorList[ai];
		actorMask[ai] = 1;
		const previousOutcome = actorOutcomes?.get?.(a.id) ?? actorOutcomes?.[a.id] ?? 0;
		actorOutcome[ai] = previousOutcome >= 0 && previousOutcome < actionOutcomes.length
			? previousOutcome
			: actionOutcomes.indexOf('other');
		const o = ai * actorFeat;
		actors[o + 0] = a.kind === 'creep' ? 0 : 1;
		actors[o + 1] = a.x / 49;
		actors[o + 2] = a.y / 49;
		actors[o + 3] = roomNames.indexOf(a.room) / Math.max(1, maxRooms - 1);
		if (a.kind === 'creep') {
			const c = a.ref;
			const cap = c.store?.getCapacity?.() || 0;
			const energy = c.store?.[C.RESOURCE_ENERGY] || 0;
			const free = c.store?.getFreeCapacity?.() ?? Math.max(0, cap - energy);
			for (let part = 0; part < bodyPartTypes.length; part++) {
				actors[o + 4 + part] = bodyCount(c, BODY_PART_CONST[part], false) / maxBodyParts;
				actors[o + 12 + part] = bodyCount(c, BODY_PART_CONST[part]) / maxBodyParts;
			}
			actors[o + 20] = clamp01(energy / (cap || 1));
			actors[o + 21] = clamp01(free / (cap || 1));
			actors[o + 22] = clamp01(energy / maxRoomEnergy);
			actors[o + 23] = clamp01(cap / maxRoomEnergy);
			actors[o + 24] = clamp01(c.fatigue / 10);
			actors[o + 25] = clamp01(c.hits / c.hitsMax);
			actors[o + 26] = c.ticksToLive != null ? clamp01(c.ticksToLive / 1500) : 1;
			actors[o + 27] = c.spawning ? 1 : 0;
			actors[o + 28] = energy === 0 ? 1 : 0;
			actors[o + 29] = free === 0 && cap > 0 ? 1 : 0;
			actors[o + 30] = c.fatigue > 0 ? 1 : 0;
		} else if (a.kind === 'structure') {
			const st = a.ref;
			const cap = st.store?.getCapacity?.(C.RESOURCE_ENERGY) || 0;
			const energy = st.store?.[C.RESOURCE_ENERGY] || 0;
			const free = st.store?.getFreeCapacity?.(C.RESOURCE_ENERGY)
				?? Math.max(0, cap - energy);
			actors[o + 20] = clamp01(energy / (cap || 1));
			actors[o + 21] = clamp01(free / (cap || 1));
			actors[o + 22] = clamp01(energy / maxRoomEnergy);
			actors[o + 23] = clamp01(cap / maxRoomEnergy);
			actors[o + 25] = clamp01(st.hits / (st.hitsMax || 1));
			actors[o + 27] = st.spawning ? 1 : 0;
			actors[o + 28] = energy === 0 ? 1 : 0;
			actors[o + 29] = free === 0 && cap > 0 ? 1 : 0;
			actors[o + 31] = st.structureType === C.STRUCTURE_SPAWN ? 1 : 0;
			actors[o + 32] = st.structureType === C.STRUCTURE_TOWER ? 1 : 0;
			// Fixed scaling preserves the room's exact integer spawn budget without
			// saturating any standard room (maximum capacity 12,900).
			actors[o + 33] = clamp01((st.room?.energyAvailable || 0) / maxRoomEnergy);
			actors[o + 34] = clamp01((st.room?.energyCapacityAvailable || 0) / maxRoomEnergy);
		} else {
			// Strategic room actor. Position is the controller only as a spatial anchor;
			// construction itself uses the explicit room/type/tile action factors.
			actors[o + 25] = 1;
			actors[o + 35] = 1;
		}

		// Schema-v2 has one executable goal per entity. The singleton slot dimension is
		// retained on the wire so actions remain an explicit per-actor sequence axis.
		const slot0I = (ai * intentSlots) * nIntent;
		const slot0D = (ai * intentSlots) * 8;
		const baseI = slot0I;
		const baseD = slot0D;
		const aSel = (intentIdx, bin) => (
			((ai * intentSlots) * nIntent + intentIdx) * amountBins.length + bin
		);

		// always allow none
		intentMask[baseI + INTENT.none] = 1;
		if (a.kind === 'creep') {
			const c = a.ref;
			if (c.spawning) continue;
			const has = t => c.body.some(p => p.type === t && p.hits > 0);
			const energy = c.store?.[C.RESOURCE_ENERGY] || 0;
			const free = c.store?.getFreeCapacity?.() ?? 0;

			if (c.fatigue === 0 && has(C.MOVE)) {
				intentMask[baseI + INTENT.move] = 1;
				for (let d = 0; d < 8; d++) dirMask[baseD + d] = 1;
			}
			// Capability and resources open an intent; the global candidate table and
			// actor-local compatibility close it when no executable goal exists. This
			// preserves cross-room macros instead of requiring a target in the current room.
			if (has(C.WORK)) {
				intentMask[baseI + INTENT.harvest] = 1;
				intentMask[baseI + INTENT.dismantle] = 1;
			}
			if (has(C.WORK) && energy > 0) {
				intentMask[baseI + INTENT.upgradeController] = 1;
				intentMask[baseI + INTENT.build] = 1;
				intentMask[baseI + INTENT.repair] = 1;
			}
			if (has(C.CARRY)) {
				if (energy > 0) {
					intentMask[baseI + INTENT.transfer] = 1;
					intentMask[baseI + INTENT.drop] = 1;
					amountMask[aSel(INTENT.transfer, 0)] = 1;
					amountMask[aSel(INTENT.drop, 0)] = 1;
					for (let b = 1; b < amountBins.length; b++) {
						if (amountBins[b] <= energy) {
							amountMask[aSel(INTENT.transfer, b)] = 1;
							amountMask[aSel(INTENT.drop, b)] = 1;
						}
					}
				}
				if (free > 0) {
					intentMask[baseI + INTENT.withdraw] = 1;
					intentMask[baseI + INTENT.pickup] = 1;
					amountMask[aSel(INTENT.withdraw, 0)] = 1;
					for (let b = 1; b < amountBins.length; b++) {
						if (amountBins[b] <= free) amountMask[aSel(INTENT.withdraw, b)] = 1;
					}
				}
			}
			if (has(C.ATTACK)) intentMask[baseI + INTENT.attack] = 1;
			if (has(C.RANGED_ATTACK)) intentMask[baseI + INTENT.rangedAttack] = 1;
			if (has(C.HEAL)) {
				intentMask[baseI + INTENT.heal] = 1;
				intentMask[baseI + INTENT.rangedHeal] = 1;
			}
			if (has(C.CLAIM)) {
				intentMask[baseI + INTENT.claimController] = 1;
				intentMask[baseI + INTENT.reserveController] = 1;
				intentMask[baseI + INTENT.attackController] = 1;
			}

		} else if (a.kind === 'room') {
			const roomIndex = roomNames.indexOf(a.room);
			const base = roomIndex * CONSTRUCTION_TYPES.length * CONSTRUCTION_MASK_BYTES;
			let canConstruct = false;
			for (let index = 0; index < CONSTRUCTION_TYPES.length * CONSTRUCTION_MASK_BYTES; index++) {
				if (constructionMask[base + index]) { canConstruct = true; break; }
			}
			if (canConstruct) intentMask[baseI + INTENT.createConstructionSite] = 1;
		} else if (a.ref.structureType === C.STRUCTURE_SPAWN) {
			const energy = a.ref.room?.energyAvailable
				?? (a.ref.store?.[C.RESOURCE_ENERGY] || 0);
			// At least one valid body exists iff the room can afford the cheapest
			// schema part. Exact joint count×order affordability is conditioned in
			// the policy; the engine remains the race-safe execution boundary.
			if (!a.ref.spawning && energy >= MIN_BODY_COST) {
				intentMask[baseI + INTENT.spawnCreep] = 1;
			}
		} else if (a.ref.structureType === C.STRUCTURE_TOWER) {
			const tw = a.ref;
			const e = tw.store?.[C.RESOURCE_ENERGY] || 0;
			if (e > 0) {
				intentMask[baseI + INTENT.attack] = 1;
				intentMask[baseI + INTENT.repair] = 1;
				intentMask[baseI + INTENT.heal] = 1;
			}
		}

	}

	// Every advertised target-taking intent must have an executable argument.
	// This closes masked-but-invalid no-ops while preserving cross-room goals for
	// mobile creeps. Structures and construction anchors remain room-local.
	for (let ai = 0; ai < actorsUsed; ai++) {
		const actor = actorList[ai];
		const actorIsMobile = actor.kind === 'creep';
		for (const name of TARGET_INTENT_NAMES) {
			const intent = INTENT[name];
			const offset = (ai * intentSlots) * nIntent + intent;
			if (!intentMask[offset]) continue;
			const localOnly = !actorIsMobile;
			let hasCandidate = false;
			for (let ti = 0; ti < targetsUsed; ti++) {
				if (!targetSelectMask[intent * maxTargets + ti]) continue;
				if (localOnly && targetList[ti].room !== actor.room) continue;
				hasCandidate = true;
				break;
			}
			if (!hasCandidate) {
				for (let slot = 0; slot < intentSlots; slot++) {
					intentMask[((ai * intentSlots + slot) * nIntent) + intent] = 0;
				}
			}
		}
	}

	// --- tile / patch grid per room
	const roomsUsed = Math.min(maxRooms, roomNames.length);
	if (roomsUsed === 0) patches.fill(0, 0, PATCH_BYTES_PER_ROOM);
	for (let ri = 0; ri < roomsUsed; ri++) {
		roomMask[ri] = 1;
		const roomName = roomNames[ri];
		const room = Game.rooms[roomName];
		const roomBase = ri * PATCH_BYTES_PER_ROOM;
		patches.set(getTerrainBase(roomName, room.getTerrain()), roomBase);
		const paintOffset = (x, y) => (
			x < 0 || y < 0 || x >= roomSize || y >= roomSize
				? -1
				: roomBase + packedTileOffsets[y * roomSize + x]
		);

		for (const st of room.find(C.FIND_STRUCTURES)) {
			const o = paintOffset(st.pos.x, st.pos.y);
			if (o < 0) continue;
			patches[o + 3 + Math.min(structureChannel(st), 10)] = 255;
			patches[o + 14] = q8(st.hits / (st.hitsMax || 1));
			if (st.store) {
				patches[o + 15] = q8((st.store[C.RESOURCE_ENERGY] || 0) / (st.store.getCapacity?.(C.RESOURCE_ENERGY) || 1));
			}
			if (st.my) patches[o + 25] = 255;
			if (st.structureType === C.STRUCTURE_CONTROLLER) {
				patches[o + 16] = q8((st.progress || 0) / (st.progressTotal || 1));
				patches[o + 17] = q8((st.level || 0) / 8);
			}
		}
		for (const s of room.find(C.FIND_SOURCES)) {
			const o = paintOffset(s.pos.x, s.pos.y);
			if (o < 0) continue;
			patches[o + 18] = 255;
			patches[o + 19] = q8(s.energy / (s.energyCapacity || 3000));
		}
		for (const c of room.find(C.FIND_CREEPS)) {
			const o = paintOffset(c.pos.x, c.pos.y);
			if (o < 0) continue;
			if (c.my) {
				patches[o + 20] = 255;
				patches[o + 21] = q8(bodyFrac(c, C.WORK));
				patches[o + 22] = q8(bodyFrac(c, C.CARRY));
				patches[o + 23] = q8(bodyFrac(c, C.MOVE));
			} else {
				patches[o + 24] = 255;
			}
		}
		for (const site of room.find(C.FIND_CONSTRUCTION_SITES)) {
			const o = paintOffset(site.pos.x, site.pos.y);
			if (o < 0) continue;
			patches[o + 26] = q8(site.progress / (site.progressTotal || 1));
		}
	}

	// global scalars
	let rclMax = 0;
	let stored = 0;
	let controlProgress = 0;
	for (const roomName of roomNames) {
		const room = Game.rooms[roomName];
		if (room.controller?.my) {
			rclMax = Math.max(rclMax, room.controller.level);
			controlProgress += room.controller.progress || 0;
		}
		for (const st of room.find(C.FIND_MY_STRUCTURES)) {
			if (st.store) stored += st.store[C.RESOURCE_ENERGY] || 0;
		}
	}

	// Pack only active rooms for patches (usually 1) — main IPC win vs maxRooms=4 zeros.
	const rUse = Math.max(1, roomsUsed);
	const patchBytes = rUse * PATCH_BYTES_PER_ROOM;
	const patchPacked = patches.subarray(0, patchBytes);

	const shapes = {
		patches: [ rUse, patchesPerRoom, patchSize * patchSize * tileFeat ],
		roomCoords: [ maxRooms, 2 ],
		actors: [ maxActors, actorFeat ],
		actorOutcome: [ maxActors ],
		targets: [ maxTargets, targetFeat ],
		intentMask: [ maxActors, intentSlots, intentTypes.length ],
		dirMask: [ maxActors, intentSlots, 8 ],
		targetSelectMask: [ nIntent, maxTargets ],
		amountMask: [ maxActors, intentSlots, nIntent, amountBins.length ],
		constructionMask: [ maxRooms, CONSTRUCTION_TYPES.length, CONSTRUCTION_MASK_BYTES ],
	};
	const actorMeta = actorList.slice(0, actorsUsed).map((a, index) => ({
		kind: a.kind, id: a.id, objectId: a.ref?.id ?? null,
		room: a.room, x: a.x, y: a.y,
		outcome: actionOutcomes[actorOutcome[index]],
	}));
	const targetMeta = targetList.slice(0, targetsUsed).map(t => ({
		kind: t.kind, id: t.id, room: t.room, x: t.x, y: t.y,
		structureType: t.ref?.structureType ?? null,
		harvestable: Boolean(t.harvestable),
		my: Boolean(t.ref?.my),
		owned: Boolean(t.ref?.owner),
		reserved: controllerReservation(t.ref),
		myReservation: Boolean(controllerReservation(t.ref) && t.ref?.room?.['#user'] === userId),
	}));
	const globals = {
		rclMax,
		storedEnergy: stored,
		controlProgress,
		creeps: Object.keys(Game.creeps).length,
		gcl: Game.gcl?.level ?? 1,
		bucket: Game.cpu?.bucket ?? 10000,
		visibleRooms: visibleRoomCount,
		actorCount: fullActorCount,
		targetCount: fullTargetCount,
		roomOverflow: visibleRoomCount > maxRooms ? 1 : 0,
		actorOverflow: (fullActorCount > maxActors
			|| fullCreepActors > maxCreepActors
			|| fullStructureActors > maxStructureActors) ? 1 : 0,
		targetOverflow: fullTargetCount > maxTargets ? 1 : 0,
	};

	const rm = toU8(roomMask);
	const am = toU8(actorMask);
	const tm = toU8(targetMask);
	const im = toU8(intentMask);
	const dm = toU8(dirMask);
	const tsm = toU8(targetSelectMask);
	const amm = toU8(amountMask);
	// Shared field order for pack/bin: u8 patches | f32 actors, targets, room coords | u8 masks…
	const tensorParts = [
		patchPacked, actors, targets, roomCoords,
		rm, am, actorOutcome, tm, im, dm, tsm, amm, constructionMask,
	];

	// bin: TypedArrays for server to frame (never JSON-stringified)
	if (OBS_FMT === 'bin') {
		return {
			schemaVersion: schema.version,
			encoding: 'bin',
			time: Game.time,
			roomNames: roomNames.slice(0, maxRooms),
			roomsUsed: rUse,
			shapes,
			actorMeta,
			targetMeta,
			globals,
			_raw: {
				patches: patchPacked,
				actors,
				targets,
				roomCoords,
				roomMask: rm,
				actorMask: am,
				actorOutcome,
				targetMask: tm,
				intentMask: im,
				dirMask: dm,
				targetSelectMask: tsm,
				amountMask: amm,
				constructionMask,
			},
		};
	}

	// pack: one base64 blob — order must match Python pack/bin decoder
	if (OBS_FMT === 'pack') {
		return {
			schemaVersion: schema.version,
			encoding: 'pack',
			time: Game.time,
			roomNames: roomNames.slice(0, maxRooms),
			shapes,
			actorMeta,
			targetMeta,
			globals,
			blob: packBlob(tensorParts),
		};
	}

	return {
		schemaVersion: schema.version,
		encoding: OBS_FMT,
		time: Game.time,
		roomNames: roomNames.slice(0, maxRooms),
		roomMask: u8ToWire(roomMask),
		patches: u8ToWire(patchPacked),
		shapes,
		actors: f32ToWire(actors),
		actorMask: u8ToWire(actorMask),
		actorOutcome: u8ToWire(actorOutcome),
		actorMeta,
		targets: f32ToWire(targets),
		roomCoords: f32ToWire(roomCoords),
		targetMask: u8ToWire(targetMask),
		targetMeta,
		intentMask: u8ToWire(intentMask),
		dirMask: u8ToWire(dirMask),
		targetSelectMask: u8ToWire(targetSelectMask),
		amountMask: u8ToWire(amountMask),
		constructionMask: u8ToWire(constructionMask),
		globals,
	};
}
