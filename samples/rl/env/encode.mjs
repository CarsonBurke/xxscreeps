/**
 * Encode visible Screeps rooms as ViT-style 5×5 patch tensors + actor/target tables.
 * Shared schema: ../schema.json
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from 'xxscreeps/game/constants/index.js';
import { parseRoomName } from 'xxscreeps/game/room/name.js';

const schema = JSON.parse(
	fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '../schema.json'), 'utf8'),
);

export const SCHEMA = schema;
const {
	roomSize, patchSize, patchesPerSide, patchesPerRoom,
	maxRooms, maxActors, maxCreepActors, maxStructureActors, maxTargets, intentSlots,
	tileFeat, actorFeat, targetFeat, intentTypes, intentSpecs, amountBins, constructionTypes,
} = schema;

const INTENT = Object.fromEntries(intentTypes.map((n, i) => [ n, i ]));
export { INTENT };
if (intentSlots !== 1) {
	throw new Error(`schema-v2 requires intentSlots=1, got ${intentSlots}`);
}
const CONSTRUCTION_TYPES = constructionTypes.map(type => C[type.toUpperCase()] || type);
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
 * Terrain is immutable per room name in the sim. Cache plain/swamp/wall + edge
 * channels (0,1,2,27) so each tick only re-paints dynamic entity layers.
 * Keyed by roomName → Float32Array(roomSize² × tileFeat) with only static feats set.
 */
const terrainBaseCache = new Map();

function getTerrainBase(roomName, terrain) {
	let base = terrainBaseCache.get(roomName);
	if (base) return base;
	base = new Uint8Array(roomSize * roomSize * tileFeat);
	for (let y = 0; y < roomSize; y++) {
		for (let x = 0; x < roomSize; x++) {
			const o = (y * roomSize + x) * tileFeat;
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

function bodyCount(creep, type) {
	if (!creep?.body?.length) return 0;
	let n = 0;
	for (const p of creep.body) if (p.type === type && p.hits > 0) n++;
	return n;
}

function structureChannel(struct) {
	if (!struct) return STRUCT.other;
	const t = struct.structureType;
	if (t in STRUCT) return STRUCT[t];
	return STRUCT.other;
}

function freeBuildPositions(room) {
	const seen = new Set();
	const positions = [];
	const myStructures = room.find(C.FIND_MY_STRUCTURES);
	const mySites = room.find(C.FIND_MY_CONSTRUCTION_SITES);
	const structures = room.find(C.FIND_STRUCTURES);
	const sites = room.find(C.FIND_CONSTRUCTION_SITES);
	const sources = room.find(C.FIND_SOURCES);
	const minerals = room.find(C.FIND_MINERALS);
	const anchors = [ ...myStructures, ...mySites ];
	const blocked = new Set();
	for (const object of [ ...structures, ...sites, ...sources, ...minerals ]) {
		blocked.add(`${object.pos.x},${object.pos.y}`);
	}
	for (const anchor of anchors) {
		for (const { dx, dy } of schema.directions) {
			const x = anchor.pos.x + dx;
			const y = anchor.pos.y + dy;
			const key = `${x},${y}`;
			if (seen.has(key) || x < 1 || x > 48 || y < 1 || y > 48) continue;
			seen.add(key);
			if (room.getTerrain().get(x, y) === C.TERRAIN_MASK_WALL) continue;
			if (!blocked.has(key)) positions.push({ x, y });
		}
	}
	return positions;
}

/**
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {string} userId
 */
export function encodeObservation(Game, userId) {
	const roomNames = Object.keys(Game.rooms).sort();
	return encodeObservationFromRooms(Game, userId, roomNames);
}

/**
 * Encode using an explicit room map (from player view or stitched peeks).
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {string} userId
 * @param {string[]} roomNames
 */
export function encodeObservationFromRooms(Game, userId, roomNames) {
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
	// Typed masks from the start — avoid number[] → Uint8Array copies on every tick
	const roomMask = new Uint8Array(maxRooms);
	const roomCoords = new Float32Array(maxRooms * 2);
	// All spatial channels are normalized to [0,1]. Quantize them on the wire and
	// in rollout storage: 4x less IPC/RAM with <0.4% absolute feature error.
	const patches = new Uint8Array(maxRooms * patchesPerRoom * patchSize * patchSize * tileFeat);
	const actors = new Float32Array(maxActors * actorFeat);
	const actorMask = new Uint8Array(maxActors);
	const targets = new Float32Array(maxTargets * targetFeat);
	const targetMask = new Uint8Array(maxTargets);
	const nIntent = intentTypes.length;
	const intentMask = new Uint8Array(maxActors * intentSlots * nIntent);
	const dirMask = new Uint8Array(maxActors * intentSlots * 8);
	// Compact candidate table: [nIntent, maxTargets]. Actor-specific capability is
	// carried by intentMask; same-room compatibility is derived from entity room ids
	// in the model. This avoids O(actors × slots × intents × targets) observations.
	const targetSelectMask = new Uint8Array(nIntent * maxTargets);
	// Intent-specific amounts avoid aliases such as unaffordable spawn bodies and
	// construction types that are illegal at the current controller level.
	const amountMask = new Uint8Array(
		maxActors * intentSlots * nIntent * amountBins.length,
	);
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
		if (room.controller?.my) {
			for (const { x, y } of freeBuildPositions(room)) {
				targetList.push({
					kind: 'position', id: `build:${roomName}:${x}:${y}`,
					room: roomName, x, y, ref: null, pri: 4.5,
				});
			}
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
		if (target.kind === 'position') return 4;
		if (target.kind === 'creep') return 5;
		if (target.ref?.structureType === C.STRUCTURE_EXTENSION) return 6;
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
		if (t.kind === 'creep' && my && (t.ref?.store?.getFreeCapacity?.() ?? 0) > 0) {
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
		if (t.kind === 'position') {
			targetSelectMask[tSel(INTENT.createConstructionSite, ti)] = 1;
		}
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
	// Construction capacity is a room property, not an actor property. Snapshot it
	// once per observation so dozens of builders do not each repeat structure/site
	// scans for every supported type.
	const globalSites = Object.keys(Game.constructionSites || {}).length;
	const constructionAmountsByRoom = new Map();
	for (const roomName of roomNames) {
		const room = Game.rooms[roomName];
		const bins = new Uint8Array(amountBins.length);
		const rcl = room?.controller?.level || 0;
		if (rcl > 0 && globalSites < (C.MAX_CONSTRUCTION_SITES ?? 100)) {
			const existing = new Map();
			const planned = new Map();
			for (const structure of room.find(C.FIND_STRUCTURES)) {
				existing.set(structure.structureType, (existing.get(structure.structureType) || 0) + 1);
			}
			for (const site of room.find(C.FIND_CONSTRUCTION_SITES)) {
				planned.set(site.structureType, (planned.get(site.structureType) || 0) + 1);
			}
			for (let b = 0; b < Math.min(CONSTRUCTION_TYPES.length, amountBins.length); b++) {
				const structureType = CONSTRUCTION_TYPES[b];
				const allowed = C.CONTROLLER_STRUCTURES?.[structureType]?.[rcl] ?? 0;
				if ((existing.get(structureType) || 0) + (planned.get(structureType) || 0) < allowed) {
					bins[b] = 1;
				}
			}
		}
		constructionAmountsByRoom.set(roomName, bins);
	}

	for (let ai = 0; ai < actorsUsed; ai++) {
		const a = actorList[ai];
		actorMask[ai] = 1;
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
			actors[o + 4] = bodyFrac(c, C.WORK);
			actors[o + 5] = bodyFrac(c, C.CARRY);
			actors[o + 6] = bodyFrac(c, C.MOVE);
			actors[o + 7] = bodyFrac(c, C.ATTACK);
			actors[o + 8] = bodyFrac(c, C.RANGED_ATTACK);
			actors[o + 9] = bodyFrac(c, C.HEAL);
			actors[o + 10] = bodyFrac(c, C.CLAIM);
			actors[o + 11] = clamp01(energy / (cap || 1));
			actors[o + 12] = clamp01(c.fatigue / 10);
			actors[o + 13] = clamp01(c.hits / c.hitsMax);
			actors[o + 14] = c.ticksToLive != null ? clamp01(c.ticksToLive / 1500) : 1;
			// Phase / capacity (was dead 16–23) — E6
			actors[o + 16] = clamp01(free / (cap || 1));
			actors[o + 17] = energy === 0 ? 1 : 0;
			actors[o + 18] = free === 0 && cap > 0 ? 1 : 0;
			actors[o + 19] = c.fatigue > 0 ? 1 : 0;
			actors[o + 20] = clamp01(Math.min(energy, 2000) / 2000);
			actors[o + 21] = clamp01(Math.min(cap, 2000) / 2000);
			actors[o + 22] = clamp01(bodyCount(c, C.WORK) / 50);
			actors[o + 23] = clamp01(bodyCount(c, C.CARRY) / 50);
			actors[o + 24] = clamp01(bodyCount(c, C.MOVE) / 50);
			actors[o + 25] = clamp01(bodyCount(c, C.CLAIM) / 50);
			actors[o + 26] = clamp01(c.body.length / 50);
		} else {
			const st = a.ref;
			actors[o + 4] = st.structureType === C.STRUCTURE_SPAWN ? 1 : 0;
			actors[o + 5] = st.structureType === C.STRUCTURE_TOWER ? 1 : 0;
			actors[o + 11] = clamp01((st.store?.[C.RESOURCE_ENERGY] || 0) / (st.store?.getCapacity?.(C.RESOURCE_ENERGY) || 1));
			actors[o + 13] = clamp01(st.hits / (st.hitsMax || 1));
			actors[o + 15] = st.spawning ? 1 : 0;
			actors[o + 22] = clamp01((st.room?.energyAvailable || 0) / 10000);
			actors[o + 23] = clamp01((st.room?.energyCapacityAvailable || 0) / 10000);
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
			let canConstruct = false;
			const constructionBins = constructionAmountsByRoom.get(c.room.name);
			if (constructionBins && INTENT.createConstructionSite != null) {
				for (let b = 0; b < constructionBins.length; b++) {
					if (!constructionBins[b]) continue;
					amountMask[aSel(INTENT.createConstructionSite, b)] = 1;
					canConstruct = true;
				}
			}
			if (canConstruct) {
				intentMask[baseI + INTENT.createConstructionSite] = 1;
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

		} else if (a.ref.structureType === C.STRUCTURE_SPAWN) {
			const sp = a.ref;
			const e = sp.room?.energyAvailable ?? (sp.store?.[C.RESOURCE_ENERGY] || 0);
			// Expose exactly the body templates affordable from room-wide energy.
			let affordable = false;
			if (!sp.spawning) {
				const templates = schema.bodyTemplates || [];
				for (let b = 0; b < Math.min(amountBins.length, templates.length); b++) {
					if (e >= (templates[b].cost ?? Infinity)) {
						amountMask[aSel(INTENT.spawnCreep, b)] = 1;
						affordable = true;
					}
				}
			}
			if (affordable) {
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
			const localOnly = !actorIsMobile || name === 'createConstructionSite';
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
	for (let ri = 0; ri < roomsUsed; ri++) {
		roomMask[ri] = 1;
		const roomName = roomNames[ri];
		const room = Game.rooms[roomName];
		const terrain = room.getTerrain();
		// Clone cached terrain base (plain/swamp/wall/edge); paint dynamics on top
		const tileScratch = new Uint8Array(getTerrainBase(roomName, terrain));

		const paint = (x, y, fn) => {
			if (x < 0 || y < 0 || x >= roomSize || y >= roomSize) return;
			fn((y * roomSize + x) * tileFeat);
		};

		for (const st of room.find(C.FIND_STRUCTURES)) {
			paint(st.pos.x, st.pos.y, o => {
				tileScratch[o + 3 + Math.min(structureChannel(st), 10)] = 255;
				tileScratch[o + 14] = q8(st.hits / (st.hitsMax || 1));
				if (st.store) {
					tileScratch[o + 15] = q8((st.store[C.RESOURCE_ENERGY] || 0) / (st.store.getCapacity?.(C.RESOURCE_ENERGY) || 1));
				}
				if (st.my) tileScratch[o + 25] = 255;
				if (st.structureType === C.STRUCTURE_CONTROLLER) {
					tileScratch[o + 16] = q8((st.progress || 0) / (st.progressTotal || 1));
					tileScratch[o + 17] = q8((st.level || 0) / 8);
				}
			});
		}
		for (const s of room.find(C.FIND_SOURCES)) {
			paint(s.pos.x, s.pos.y, o => {
				tileScratch[o + 18] = 255;
				tileScratch[o + 19] = q8(s.energy / (s.energyCapacity || 3000));
			});
		}
		for (const c of room.find(C.FIND_CREEPS)) {
			paint(c.pos.x, c.pos.y, o => {
				if (c.my) {
					tileScratch[o + 20] = 255;
					tileScratch[o + 21] = q8(bodyFrac(c, C.WORK));
					tileScratch[o + 22] = q8(bodyFrac(c, C.CARRY));
					tileScratch[o + 23] = q8(bodyFrac(c, C.MOVE));
				} else {
					tileScratch[o + 24] = 255;
				}
			});
		}
		for (const site of room.find(C.FIND_CONSTRUCTION_SITES)) {
			paint(site.pos.x, site.pos.y, o => {
				tileScratch[o + 26] = q8(site.progress / (site.progressTotal || 1));
			});
		}

		// pack into 5×5 patches (row-major patch grid)
		for (let py = 0; py < patchesPerSide; py++) {
			for (let px = 0; px < patchesPerSide; px++) {
				const pIdx = py * patchesPerSide + px;
				const base = ((ri * patchesPerRoom + pIdx) * patchSize * patchSize) * tileFeat;
				let k = 0;
				for (let ly = 0; ly < patchSize; ly++) {
					for (let lx = 0; lx < patchSize; lx++) {
						const x = px * patchSize + lx;
						const y = py * patchSize + ly;
						const src = (y * roomSize + x) * tileFeat;
						for (let f = 0; f < tileFeat; f++) {
							patches[base + k * tileFeat + f] = tileScratch[src + f];
						}
						k++;
					}
				}
			}
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
	const patchFloats = rUse * patchesPerRoom * patchSize * patchSize * tileFeat;
	const patchPacked = patches.subarray(0, patchFloats);

	const shapes = {
		patches: [ rUse, patchesPerRoom, patchSize * patchSize * tileFeat ],
		roomCoords: [ maxRooms, 2 ],
		actors: [ maxActors, actorFeat ],
		targets: [ maxTargets, targetFeat ],
		intentMask: [ maxActors, intentSlots, intentTypes.length ],
		dirMask: [ maxActors, intentSlots, 8 ],
		targetSelectMask: [ nIntent, maxTargets ],
		amountMask: [ maxActors, intentSlots, nIntent, amountBins.length ],
	};
	const actorMeta = actorList.slice(0, actorsUsed).map(a => ({
		kind: a.kind, id: a.id, room: a.room, x: a.x, y: a.y,
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
		rm, am, tm, im, dm, tsm, amm,
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
				targetMask: tm,
				intentMask: im,
				dirMask: dm,
				targetSelectMask: tsm,
				amountMask: amm,
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
		actorMeta,
		targets: f32ToWire(targets),
		roomCoords: f32ToWire(roomCoords),
		targetMask: u8ToWire(targetMask),
		targetMeta,
		intentMask: u8ToWire(intentMask),
		dirMask: u8ToWire(dirMask),
		targetSelectMask: u8ToWire(targetSelectMask),
		amountMask: u8ToWire(amountMask),
		globals,
	};
}
