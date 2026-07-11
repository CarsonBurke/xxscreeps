/**
 * Encode visible Screeps rooms as ViT-style 5×5 patch tensors + actor/target tables.
 * Shared schema: ../schema.json
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from 'xxscreeps/game/constants/index.js';

const schema = JSON.parse(
	fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '../schema.json'), 'utf8'),
);

export const SCHEMA = schema;
const {
	roomSize, patchSize, patchesPerSide, patchesPerRoom,
	maxRooms, maxActors, maxTargets, intentSlots,
	tileFeat, actorFeat, targetFeat, intentTypes, amountBins,
} = schema;

const INTENT = Object.fromEntries(intentTypes.map((n, i) => [ n, i ]));
export { INTENT };

const STRUCT = {
	controller: 0, spawn: 1, extension: 2, road: 3, container: 4,
	storage: 5, tower: 6, constructedWall: 7, rampart: 8, link: 9, other: 10,
};

function clamp01(x) {
	if (!Number.isFinite(x)) return 0;
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

function bodyFrac(creep, type) {
	if (!creep?.body?.length) return 0;
	let n = 0;
	for (const p of creep.body) if (p.type === type && p.hits > 0) n++;
	return n / creep.body.length;
}

function structureChannel(struct) {
	if (!struct) return STRUCT.other;
	const t = struct.structureType;
	if (t in STRUCT) return STRUCT[t];
	return STRUCT.other;
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
	roomNames = [ ...roomNames ].sort();
	const roomMask = new Array(maxRooms).fill(0);
	const patches = new Float32Array(maxRooms * patchesPerRoom * patchSize * patchSize * tileFeat);
	const actors = new Float32Array(maxActors * actorFeat);
	const actorMask = new Array(maxActors).fill(0);
	const targets = new Float32Array(maxTargets * targetFeat);
	const targetMask = new Array(maxTargets).fill(0);
	const intentMask = new Array(maxActors * intentSlots * intentTypes.length).fill(0);
	const dirMask = new Array(maxActors * intentSlots * 8).fill(0);
	const targetSelectMask = new Array(maxActors * intentSlots * maxTargets).fill(0);
	const amountMask = new Array(maxActors * intentSlots * amountBins.length).fill(0);

	/** @type {{ kind: string, id: string, room: string, x: number, y: number, ref: any }[]} */
	const actorList = [];
	/** @type {{ kind: string, id: string, room: string, x: number, y: number, ref: any }[]} */
	const targetList = [];

	// --- collect targets first (sources, minerals, structures, resources, my+hostile creeps)
	for (const roomName of roomNames) {
		const room = Game.rooms[roomName];
		if (!room) continue;
		for (const s of room.find(C.FIND_SOURCES)) {
			targetList.push({ kind: 'source', id: s.id, room: roomName, x: s.pos.x, y: s.pos.y, ref: s });
		}
		for (const m of room.find(C.FIND_MINERALS)) {
			targetList.push({ kind: 'mineral', id: m.id, room: roomName, x: m.pos.x, y: m.pos.y, ref: m });
		}
		for (const st of room.find(C.FIND_STRUCTURES)) {
			targetList.push({
				kind: 'structure', id: st.id, room: roomName, x: st.pos.x, y: st.pos.y, ref: st,
			});
		}
		for (const r of room.find(C.FIND_DROPPED_RESOURCES)) {
			targetList.push({ kind: 'resource', id: r.id, room: roomName, x: r.pos.x, y: r.pos.y, ref: r });
		}
		for (const c of room.find(C.FIND_CREEPS)) {
			targetList.push({ kind: 'creep', id: c.id, room: roomName, x: c.pos.x, y: c.pos.y, ref: c });
		}
		for (const site of room.find(C.FIND_CONSTRUCTION_SITES)) {
			targetList.push({ kind: 'site', id: site.id, room: roomName, x: site.pos.x, y: site.pos.y, ref: site });
		}
	}
	const targetsUsed = Math.min(maxTargets, targetList.length);
	const targetIndex = new Map();
	for (let i = 0; i < targetsUsed; i++) {
		const t = targetList[i];
		targetIndex.set(t.id, i);
		targetMask[i] = 1;
		const o = i * targetFeat;
		const kind = { source: 0, mineral: 1, structure: 2, resource: 3, creep: 4, site: 5 }[t.kind] ?? 6;
		targets[o + 0] = kind / 6;
		targets[o + 1] = t.x / 49;
		targets[o + 2] = t.y / 49;
		targets[o + 3] = roomNames.indexOf(t.room) / Math.max(1, maxRooms - 1);
		if (t.kind === 'source') {
			targets[o + 4] = clamp01(t.ref.energy / (t.ref.energyCapacity || 3000));
		} else if (t.kind === 'structure' && t.ref.store) {
			targets[o + 4] = clamp01((t.ref.store[C.RESOURCE_ENERGY] || 0) / (t.ref.store.getCapacity?.(C.RESOURCE_ENERGY) || 1));
			targets[o + 5] = structureChannel(t.ref) / 10;
			targets[o + 6] = clamp01(t.ref.hits / (t.ref.hitsMax || 1));
			targets[o + 7] = t.ref.my ? 1 : 0;
		} else if (t.kind === 'creep') {
			targets[o + 4] = t.ref.my ? 1 : 0;
			targets[o + 5] = clamp01((t.ref.store?.[C.RESOURCE_ENERGY] || 0) / (t.ref.store?.getCapacity?.() || 1));
			targets[o + 6] = bodyFrac(t.ref, C.WORK);
			targets[o + 7] = bodyFrac(t.ref, C.ATTACK);
		} else if (t.kind === 'resource') {
			targets[o + 4] = clamp01((t.ref.amount || 0) / 1000);
		} else if (t.kind === 'site') {
			targets[o + 4] = clamp01(t.ref.progress / (t.ref.progressTotal || 1));
			targets[o + 5] = structureChannel({ structureType: t.ref.structureType }) / 10;
		}
	}

	// --- actors: my creeps + my spawns/towers
	for (const name of Object.keys(Game.creeps).sort()) {
		const c = Game.creeps[name];
		if (!c?.my) continue;
		actorList.push({ kind: 'creep', id: name, room: c.room.name, x: c.pos.x, y: c.pos.y, ref: c });
	}
	for (const roomName of roomNames) {
		const room = Game.rooms[roomName];
		for (const st of room.find(C.FIND_MY_STRUCTURES)) {
			if (st.structureType === C.STRUCTURE_SPAWN || st.structureType === C.STRUCTURE_TOWER) {
				actorList.push({
					kind: 'structure', id: st.id, room: roomName, x: st.pos.x, y: st.pos.y, ref: st,
				});
			}
		}
	}
	const actorsUsed = Math.min(maxActors, actorList.length);

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
			actors[o + 4] = bodyFrac(c, C.WORK);
			actors[o + 5] = bodyFrac(c, C.CARRY);
			actors[o + 6] = bodyFrac(c, C.MOVE);
			actors[o + 7] = bodyFrac(c, C.ATTACK);
			actors[o + 8] = bodyFrac(c, C.RANGED_ATTACK);
			actors[o + 9] = bodyFrac(c, C.HEAL);
			actors[o + 10] = bodyFrac(c, C.CLAIM);
			actors[o + 11] = clamp01((c.store?.[C.RESOURCE_ENERGY] || 0) / (c.store?.getCapacity?.() || 1));
			actors[o + 12] = clamp01(c.fatigue / 10);
			actors[o + 13] = clamp01(c.hits / c.hitsMax);
			actors[o + 14] = c.ticksToLive != null ? clamp01(c.ticksToLive / 1500) : 1;
		} else {
			const st = a.ref;
			actors[o + 4] = st.structureType === C.STRUCTURE_SPAWN ? 1 : 0;
			actors[o + 5] = st.structureType === C.STRUCTURE_TOWER ? 1 : 0;
			actors[o + 11] = clamp01((st.store?.[C.RESOURCE_ENERGY] || 0) / (st.store?.getCapacity?.(C.RESOURCE_ENERGY) || 1));
			actors[o + 13] = clamp01(st.hits / (st.hitsMax || 1));
			actors[o + 15] = st.spawning ? 1 : 0;
		}

		// intent / dir / target / amount masks per slot
		for (let slot = 0; slot < intentSlots; slot++) {
			const baseI = (ai * intentSlots + slot) * intentTypes.length;
			const baseD = (ai * intentSlots + slot) * 8;
			const baseT = (ai * intentSlots + slot) * maxTargets;
			const baseA = (ai * intentSlots + slot) * amountBins.length;

			// always allow none
			intentMask[baseI + INTENT.none] = 1;
			// amount bin 0 ("all"/n/a) always ok when amount used
			amountMask[baseA + 0] = 1;

			if (a.kind === 'creep') {
				const c = a.ref;
				const has = t => c.body.some(p => p.type === t && p.hits > 0);
				const energy = c.store?.[C.RESOURCE_ENERGY] || 0;
				const free = c.store?.getFreeCapacity?.() ?? 0;

				if (c.fatigue === 0 && has(C.MOVE)) {
					intentMask[baseI + INTENT.move] = 1;
					for (let d = 0; d < 8; d++) dirMask[baseD + d] = 1;
				}
				if (has(C.WORK)) {
					intentMask[baseI + INTENT.harvest] = 1;
					intentMask[baseI + INTENT.build] = 1;
					intentMask[baseI + INTENT.repair] = 1;
					intentMask[baseI + INTENT.upgradeController] = 1;
					intentMask[baseI + INTENT.dismantle] = 1;
				}
				if (has(C.CARRY)) {
					if (energy > 0) {
						intentMask[baseI + INTENT.transfer] = 1;
						intentMask[baseI + INTENT.drop] = 1;
						for (let b = 0; b < amountBins.length; b++) amountMask[baseA + b] = 1;
					}
					if (free > 0) {
						intentMask[baseI + INTENT.withdraw] = 1;
						intentMask[baseI + INTENT.pickup] = 1;
						for (let b = 0; b < amountBins.length; b++) amountMask[baseA + b] = 1;
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

				// nearby targets (range ≤ 3) for interactive intents
				for (let ti = 0; ti < targetsUsed; ti++) {
					const t = targetList[ti];
					if (t.room !== a.room) continue;
					const dist = Math.max(Math.abs(t.x - a.x), Math.abs(t.y - a.y));
					if (dist > 3 && t.kind !== 'source' && t.kind !== 'structure') continue;
					if (dist > 1 && (t.kind === 'source' || t.kind === 'resource')) continue;
					targetSelectMask[baseT + ti] = 1;
				}
			} else if (a.ref.structureType === C.STRUCTURE_SPAWN) {
				intentMask[baseI + INTENT.spawnCreep] = 1;
			} else if (a.ref.structureType === C.STRUCTURE_TOWER) {
				intentMask[baseI + INTENT.attack] = 1;
				intentMask[baseI + INTENT.repair] = 1;
				intentMask[baseI + INTENT.heal] = 1;
				for (let ti = 0; ti < targetsUsed; ti++) {
					const t = targetList[ti];
					if (t.room === a.room) targetSelectMask[baseT + ti] = 1;
				}
			}
		}
	}

	// --- tile / patch grid per room
	const roomsUsed = Math.min(maxRooms, roomNames.length);
	for (let ri = 0; ri < roomsUsed; ri++) {
		roomMask[ri] = 1;
		const room = Game.rooms[roomNames[ri]];
		const terrain = room.getTerrain();
		const tileScratch = new Float32Array(roomSize * roomSize * tileFeat);

		// base terrain
		for (let y = 0; y < roomSize; y++) {
			for (let x = 0; x < roomSize; x++) {
				const o = (y * roomSize + x) * tileFeat;
				const ter = terrain.get(x, y);
				if (ter === C.TERRAIN_MASK_WALL) tileScratch[o + 2] = 1;
				else if (ter === C.TERRAIN_MASK_SWAMP) tileScratch[o + 1] = 1;
				else tileScratch[o + 0] = 1;
				if (x === 0 || y === 0 || x === 49 || y === 49) tileScratch[o + 27] = 1;
			}
		}

		const paint = (x, y, fn) => {
			if (x < 0 || y < 0 || x >= roomSize || y >= roomSize) return;
			fn((y * roomSize + x) * tileFeat);
		};

		for (const st of room.find(C.FIND_STRUCTURES)) {
			paint(st.pos.x, st.pos.y, o => {
				tileScratch[o + 3 + Math.min(structureChannel(st), 10)] = 1;
				tileScratch[o + 14] = clamp01(st.hits / (st.hitsMax || 1));
				if (st.store) {
					tileScratch[o + 15] = clamp01((st.store[C.RESOURCE_ENERGY] || 0) / (st.store.getCapacity?.(C.RESOURCE_ENERGY) || 1));
				}
				if (st.my) tileScratch[o + 25] = 1;
				if (st.structureType === C.STRUCTURE_CONTROLLER) {
					tileScratch[o + 16] = clamp01((st.progress || 0) / (st.progressTotal || 1));
					tileScratch[o + 17] = (st.level || 0) / 8;
				}
			});
		}
		for (const s of room.find(C.FIND_SOURCES)) {
			paint(s.pos.x, s.pos.y, o => {
				tileScratch[o + 18] = 1;
				tileScratch[o + 19] = clamp01(s.energy / (s.energyCapacity || 3000));
			});
		}
		for (const c of room.find(C.FIND_CREEPS)) {
			paint(c.pos.x, c.pos.y, o => {
				if (c.my) {
					tileScratch[o + 20] = 1;
					tileScratch[o + 21] = bodyFrac(c, C.WORK);
					tileScratch[o + 22] = bodyFrac(c, C.CARRY);
					tileScratch[o + 23] = bodyFrac(c, C.MOVE);
				} else {
					tileScratch[o + 24] = 1;
				}
			});
		}
		for (const site of room.find(C.FIND_CONSTRUCTION_SITES)) {
			paint(site.pos.x, site.pos.y, o => {
				tileScratch[o + 26] = clamp01(site.progress / (site.progressTotal || 1));
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

	return {
		schemaVersion: schema.version,
		time: Game.time,
		roomNames: roomNames.slice(0, maxRooms),
		roomMask,
		patches: Array.from(patches),
		// shape hints for Python
		shapes: {
			patches: [ maxRooms, patchesPerRoom, patchSize * patchSize * tileFeat ],
			actors: [ maxActors, actorFeat ],
			targets: [ maxTargets, targetFeat ],
			intentMask: [ maxActors, intentSlots, intentTypes.length ],
			dirMask: [ maxActors, intentSlots, 8 ],
			targetSelectMask: [ maxActors, intentSlots, maxTargets ],
			amountMask: [ maxActors, intentSlots, amountBins.length ],
		},
		actors: Array.from(actors),
		actorMask,
		actorMeta: actorList.slice(0, actorsUsed).map(a => ({
			kind: a.kind, id: a.id, room: a.room, x: a.x, y: a.y,
		})),
		targets: Array.from(targets),
		targetMask,
		targetMeta: targetList.slice(0, targetsUsed).map(t => ({
			kind: t.kind, id: t.id, room: t.room, x: t.x, y: t.y,
		})),
		intentMask,
		dirMask,
		targetSelectMask,
		amountMask,
		globals: {
			rclMax,
			storedEnergy: stored,
			controlProgress,
			creeps: Object.keys(Game.creeps).length,
			gcl: Game.gcl?.level ?? 1,
			bucket: Game.cpu?.bucket ?? 10000,
		},
	};
}
