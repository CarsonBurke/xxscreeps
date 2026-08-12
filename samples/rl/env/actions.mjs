/** Apply one factorized goal/action per policy-controlled actor. */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from 'xxscreeps/game/constants/index.js';
import { RoomPosition } from 'xxscreeps/game/position.js';

const schema = JSON.parse(
	fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '../schema.json'), 'utf8'),
);

const {
	intentTypes, intentSpecs, amountBins, directions, bodyTemplates, constructionTypes, maxRooms,
} = schema;
const BODY_TEMPLATES = (bodyTemplates || [
	{ name: 'worker_basic', body: [ 'work', 'carry', 'move', 'move' ], cost: 250 },
]).map(t => ({
	...t,
	parts: t.body.map(p => {
		const key = String(p).toUpperCase();
		return C[key] || C.MOVE;
	}),
}));
const DIR_CONST = [
	C.TOP, C.TOP_RIGHT, C.RIGHT, C.BOTTOM_RIGHT,
	C.BOTTOM, C.BOTTOM_LEFT, C.LEFT, C.TOP_LEFT,
];
const CONSTRUCTION_TYPES = constructionTypes.map(type => C[type.toUpperCase()] || type);

/**
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {{
 *   actorMeta: { kind: string, id: string }[],
 *   targetMeta: { kind: string, id: string, room: string }[],
 * }} meta
 * @param {{
 *   types: number[][],
 *   dirs: number[][],
 *   targets: number[][],
 *   amounts: number[][],
 * }} actions  // [actor][slot]
 */
export function applyActions(Game, meta, actions) {
	const results = [];
	const { actorMeta, targetMeta } = meta;
	const nActors = Math.min(actorMeta.length, actions.types.length);
	pruneStepCache(Game.time);

	for (let ai = 0; ai < nActors; ai++) {
		const actor = actorMeta[ai];
		const slots = actions.types[ai]?.length ?? 0;
		for (let slot = 0; slot < slots; slot++) {
			const typeIdx = actions.types[ai][slot] | 0;
			const type = intentTypes[typeIdx] || 'none';
			if (type === 'none') continue;

			const dirIdx = actions.dirs[ai]?.[slot] | 0;
			const tgtIdx = actions.targets[ai]?.[slot] | 0;
			const amtIdx = actions.amounts[ai]?.[slot] | 0;
			const amountBin = amountBins[Math.min(amtIdx, amountBins.length - 1)] ?? 0;

			let code = -1;
			let executed = false;
			let resultTarget = null;
			try {
				if (actor.kind === 'creep') {
					const creep = Game.creeps[actor.id];
					if (!creep) {
						results.push({ actor: actor.id, type, code: -1, err: 'missing creep' });
						continue;
					}
					const metaT = targetMeta[tgtIdx];
					resultTarget = metaT;
					const target = resolveTarget(Game, metaT);
					if (!targetLegalForIntent(type, metaT, target)) {
						results.push({ actor: actor.id, type, code: C.ERR_INVALID_TARGET, err: 'kind filter' });
						continue;
					}
					const ready = readyForPrimitiveIntent(creep, type, target);
					code = runCreepIntent(creep, type, dirIdx, target, amountBin, amtIdx);
					executed = ready && code === C.OK;
				} else {
					const struct = Game.getObjectById?.(actor.id);
					// also search spawns by id from rooms
					const obj = struct || findStructure(Game, actor.id);
					if (!obj) {
						results.push({ actor: actor.id, type, code: -1, err: 'missing structure' });
						continue;
					}
					const metaT = targetMeta[tgtIdx];
					resultTarget = metaT;
					if (!targetLegalForStructure(obj, type, metaT)) {
						results.push({ actor: actor.id, type, code: C.ERR_INVALID_TARGET, err: 'kind filter' });
						continue;
					}
					const target = resolveTarget(Game, metaT);
					// amtIdx selects body template for spawnCreep (not energy amount)
					code = runStructureIntent(Game, obj, type, target, amtIdx);
					executed = code === C.OK;
				}
			} catch (err) {
				results.push({ actor: actor.id, type, code: -99, err: String(err?.message || err) });
				continue;
			}
			results.push({
				actor: actor.id,
				type,
				code,
				executed,
				targetKind: resultTarget?.kind,
				targetStructureType: resultTarget?.structureType,
			});
		}
	}
	return results;
}

function findStructure(Game, id) {
	for (const roomName of Object.keys(Game.rooms)) {
		for (const st of Game.rooms[roomName].find(C.FIND_STRUCTURES)) {
			if (st.id === id) return st;
		}
	}
	return null;
}

function positionalTarget(meta) {
	if (!Number.isFinite(meta?.x) || !Number.isFinite(meta?.y) || !meta?.room) return null;
	return {
		pos: new RoomPosition(meta.x, meta.y, meta.room),
		__unresolvedTarget: true,
	};
}

function resolveTarget(Game, meta) {
	if (!meta) return null;
	if (meta.kind === 'creep') {
		for (const name of Object.keys(Game.creeps)) {
			if (Game.creeps[name].id === meta.id) return Game.creeps[name];
		}
		// name as id fallback
		return Game.creeps[meta.id] || null;
	}
	if (Game.getObjectById) {
		const o = Game.getObjectById(meta.id);
		if (o) return o;
	}
	const room = Game.rooms[meta.room];
	if (!room) return positionalTarget(meta);
	const lists = [
		room.find(C.FIND_SOURCES),
		room.find(C.FIND_MINERALS),
		room.find(C.FIND_STRUCTURES),
		room.find(C.FIND_DROPPED_RESOURCES),
		room.find(C.FIND_CONSTRUCTION_SITES),
		room.find(C.FIND_CREEPS),
	];
	for (const list of lists) {
		for (const o of list) if (o.id === meta.id) return o;
	}
	// Known-room observations may name a goal before it is locally visible to
	// the acting creep. A positional proxy lets the macro navigate across exits;
	// once scouted, the next tick resolves the real engine object for the intent.
	return positionalTarget(meta);
}

/** Reject wrong target kinds so illegal bilinear picks become no-ops (not engine spam). */
function targetLegalForIntent(type, meta, target) {
	if (!intentSpecs[type]?.factors?.includes('target')) {
		return true;
	}
	if (!target) {
		return false;
	}
	const kind = meta?.kind;
	const st = target.structureType ?? meta?.structureType;
	switch (type) {
		case 'harvest':
			if (kind === 'source') return true;
			if (target.__unresolvedTarget) return kind === 'mineral' && Boolean(meta?.harvestable);
			if (kind !== 'mineral' || (target.mineralAmount ?? 0) <= 0) return false;
			return target.room?.lookForAt(C.LOOK_STRUCTURES, target.pos.x, target.pos.y)
				.some(st => st.structureType === C.STRUCTURE_EXTRACTOR
					&& st.my && st.isActive?.() && st.cooldown === 0) ?? false;
		case 'transfer':
			return kind === 'structure' || kind === 'creep';
		case 'withdraw':
			return kind === 'structure' && (
				st === C.STRUCTURE_STORAGE
				|| st === C.STRUCTURE_CONTAINER
				|| st === C.STRUCTURE_LINK
			);
		case 'pickup':
			return kind === 'resource';
		case 'upgradeController':
			return st === C.STRUCTURE_CONTROLLER;
		case 'claimController':
		case 'reserveController':
		case 'attackController':
			return st === C.STRUCTURE_CONTROLLER;
		case 'build':
			return kind === 'site';
		case 'repair':
		case 'dismantle':
			return kind === 'structure';
		case 'attack':
		case 'rangedAttack':
			return kind === 'creep' || kind === 'structure';
		case 'heal':
		case 'rangedHeal':
			return kind === 'creep';
		case 'createConstructionSite':
			return kind === 'position';
		default:
			return true;
	}
}

function targetLegalForStructure(struct, type, meta) {
	if (struct.structureType === C.STRUCTURE_SPAWN) return type === 'spawnCreep';
	if (struct.structureType !== C.STRUCTURE_TOWER) return false;
	if (type === 'attack' || type === 'heal') return meta?.kind === 'creep';
	if (type === 'repair') return meta?.kind === 'structure';
	return false;
}

function legalAmount(requested, available) {
	const cap = Math.max(0, Number(available) || 0);
	if (cap <= 0) return 0;
	if (!requested || requested <= 0) return cap;
	return Math.min(requested, cap);
}

const APPROACH_RANGE = Object.freeze({
	harvest: 1,
	transfer: 1,
	withdraw: 1,
	pickup: 1,
	upgradeController: 3,
	build: 3,
	repair: 3,
	attack: 1,
	rangedAttack: 3,
	heal: 1,
	rangedHeal: 3,
	claimController: 1,
	reserveController: 1,
	attackController: 1,
	dismantle: 1,
});

/** Distinguish issuing the requested primitive from a successful macro approach step. */
function readyForPrimitiveIntent(creep, type, target) {
	const range = APPROACH_RANGE[type];
	if (range == null) return true;
	const pos = target?.pos || target;
	return Boolean(pos && creep.pos.inRangeTo(pos, range));
}

function chebyshev(a, b) {
	return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
}

/** Remaining PathFinder steps: creepId|gx,gy,room|r → [{x,y}, …] */
const stepCache = new Map();
const reservedSteps = new Set();
let stepCacheTick = -1;

/**
 * Keep PathFinder remainder across consecutive ticks (same goal key).
 * Full clear only on time jump/rewind — old code wiped every tick so the
 * cache was write-only (red-team P1).
 */
function pruneStepCache(time) {
	if (stepCacheTick < 0) {
		stepCacheTick = time;
		return;
	}
	if (time === stepCacheTick) return;
	if (time !== stepCacheTick + 1) {
		stepCache.clear();
	}
	reservedSteps.clear();
	stepCacheTick = time;
}

function goalKey(creep, pos, range) {
	const id = creep.id || creep.name || '?';
	const room = pos.roomName || creep.pos.roomName;
	return `${id}|${pos.x},${pos.y},${room}|r${range}`;
}

function chebyshevStep(creep, pos) {
	const dx = Math.sign(pos.x - creep.pos.x);
	const dy = Math.sign(pos.y - creep.pos.y);
	const map = {
		'0,-1': C.TOP, '1,-1': C.TOP_RIGHT, '1,0': C.RIGHT, '1,1': C.BOTTOM_RIGHT,
		'0,1': C.BOTTOM, '-1,1': C.BOTTOM_LEFT, '-1,0': C.LEFT, '-1,-1': C.TOP_LEFT,
	};
	const dir = map[`${dx},${dy}`] ?? C.TOP;
	const next = { x: creep.pos.x + dx, y: creep.pos.y + dy, roomName: creep.pos.roomName };
	return moveWithReservation(creep, next, pos, dir);
}

/** Use the engine's exit-aware, cached inter-room router for strategic goals. */
function roomExitStep(creep, destination) {
	return creep.moveTo(destination, {
		reusePath: 10,
		serializeMemory: true,
		plainCost: 2,
		swampCost: 10,
		maxOps: 4000,
		maxRooms: Math.max(2, maxRooms),
	});
}

function stepKey(roomName, x, y) {
	return `${roomName}:${x},${y}`;
}

function walkable(room, x, y) {
	// Exit tiles are executable movement states, not walls around the learned world.
	if (x < 0 || x > 49 || y < 0 || y > 49) return false;
	if (room.getTerrain().get(x, y) === C.TERRAIN_MASK_WALL) return false;
	const structures = room.lookForAt?.(C.LOOK_STRUCTURES, x, y) || [];
	return structures.every(st =>
		st.structureType === C.STRUCTURE_ROAD
		|| st.structureType === C.STRUCTURE_CONTAINER
		|| (st.structureType === C.STRUCTURE_RAMPART && st.my)
	);
}

/** Reserve unique destinations while allowing native following and swaps. */
function moveWithReservation(creep, intended, goal, fallbackDir = null) {
	const room = creep.room;
	const candidates = [ intended ];
	for (let d = 0; d < 8; d++) {
		candidates.push({
			x: creep.pos.x + directions[d].dx,
			y: creep.pos.y + directions[d].dy,
			roomName: creep.pos.roomName,
		});
	}
	if (!goal?.roomName || goal.roomName === room.name) {
		candidates.sort((a, b) => chebyshev(a, goal) - chebyshev(b, goal));
	}
	for (const next of candidates) {
		const key = stepKey(next.roomName || room.name, next.x, next.y);
		if (reservedSteps.has(key)) continue;
		if ((next.roomName || room.name) === room.name && !walkable(room, next.x, next.y)) continue;
		reservedSteps.add(key);
		const nextPos = new RoomPosition(next.x, next.y, next.roomName || room.name);
		return creep.move(creep.pos.getDirectionTo(nextPos));
	}
	if (fallbackDir != null) return creep.move(fallbackDir);
	return C.ERR_BUSY;
}

/**
 * One step toward dest via PathFinder when available; else Chebyshev greedy dir.
 * Caches remaining path tiles so sticky approachOr does not re-A* every tick.
 */
function moveToward(creep, dest, range = 1) {
	if (!dest?.pos && dest?.x == null) return C.ERR_INVALID_TARGET;
	const pos = dest.pos || dest;
	if (creep.pos.inRangeTo(pos, range)) return C.OK;
	if (creep.fatigue > 0) return C.ERR_TIRED;
	if (pos.roomName && pos.roomName !== creep.room.name) {
		return roomExitStep(creep, pos);
	}

	const gkey = goalKey(creep, pos, range);
	let steps = stepCache.get(gkey);
	if (steps?.length) {
		// Drop tiles we've already occupied
		while (
			steps.length
			&& steps[0].x === creep.pos.x
			&& steps[0].y === creep.pos.y
		) {
			steps.shift();
		}
		if (steps.length) {
			const next = steps[0];
			const code = moveWithReservation(creep, next, pos);
			if (code === C.OK || code === C.ERR_TIRED) {
				stepCache.set(gkey, steps);
				return code;
			}
		}
		stepCache.delete(gkey);
	}

	// Use one dynamics contract for train and evaluation. Greedy navigation remains
	// an explicit benchmark mode, not the training default.
	const navMode = process.env.RL_NAV || 'pathfinder';
	if (navMode !== 'cheap') {
		try {
			if (typeof PathFinder !== 'undefined' && PathFinder.search) {
				// Same-room goals must never route through an exit and strand creeps
				// in the expansion room. Cross-room goals retain the configured bound.
				const searchMaxRooms = pos.roomName === creep.room.name ? 1 : maxRooms;
				const res = PathFinder.search(creep.pos, { pos, range }, {
					plainCost: 2, swampCost: 10, maxOps: 640, maxRooms: searchMaxRooms,
				});
				if (res.path?.length) {
					const rest = res.path.map(p => ({ x: p.x, y: p.y, roomName: p.roomName }));
					const first = rest[0];
					stepCache.set(gkey, rest.slice(1));
					if (stepCache.size > 64) {
						const oldest = stepCache.keys().next().value;
						stepCache.delete(oldest);
					}
					return moveWithReservation(creep, first, pos);
				}
			}
		} catch { /* fall through */ }
	}
	return chebyshevStep(creep, pos);
}

/**
 * Economy intents: if out of range, path toward target instead of pure fail.
 * Teaches target-as-goal under H+C reward (nav assist, not reward shaping).
 */
function approachOr(creep, target, range, act) {
	if (!target) return C.ERR_INVALID_TARGET;
	const pos = target.pos || target;
	if (creep.pos.inRangeTo(pos, range)) return act();
	return moveToward(creep, target, range);
}

function runCreepIntent(creep, type, dirIdx, target, amountBin, amountIndex = 0) {
	switch (type) {
		case 'move':
			// Dir-only only. amount is NOT in move's logπ (type-gated), so amountBin>0
			// was a latent hijack: random bins path to target-table[0]=source and break
			// deliver/upgrade approach. Path-to-goal lives in harvest/transfer via approachOr.
			return creep.move(DIR_CONST[dirIdx % 8]);
		case 'harvest': {
			// Always approachOr so RL_NAV=cheap|pathfinder applies (no silent moveTo bypass).
			const src = target || null;
			return approachOr(creep, src, 1, () => creep.harvest(src));
		}
		case 'transfer': {
			if (target?.id === creep.id) return C.ERR_INVALID_TARGET;
			const carried = creep.store?.[C.RESOURCE_ENERGY] || 0;
			const targetFree = target?.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) ?? carried;
			const amt = legalAmount(amountBin, Math.min(carried, targetFree));
			if (amt <= 0) return C.ERR_FULL;
			return approachOr(creep, target, 1, () => creep.transfer(target, C.RESOURCE_ENERGY, amt));
		}
		case 'withdraw': {
			return approachOr(creep, target, 1, () => {
				const available = target?.store?.[C.RESOURCE_ENERGY] || 0;
				const free = creep.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) ?? 0;
				const amt = legalAmount(amountBin, Math.min(available, free));
				if (amt <= 0) return C.ERR_NOT_ENOUGH_RESOURCES;
				return creep.withdraw(target, C.RESOURCE_ENERGY, amt);
			});
		}
		case 'pickup':
			return approachOr(creep, target, 1, () => creep.pickup(target));
		case 'drop': {
			const amt = legalAmount(amountBin, creep.store?.[C.RESOURCE_ENERGY] || 0);
			if (amt <= 0) return C.ERR_NOT_ENOUGH_RESOURCES;
			return creep.drop(C.RESOURCE_ENERGY, amt);
		}
		case 'upgradeController': {
			const ctrl = (target && target.structureType === C.STRUCTURE_CONTROLLER)
				? target
				: creep.room?.controller;
			if (!ctrl) return C.ERR_INVALID_TARGET;
			// approachOr respects RL_NAV (cheap=Chebyshev, pathfinder=PF). No moveTo bypass.
			return approachOr(creep, ctrl, 3, () => creep.upgradeController(ctrl));
		}
		case 'build':
			return approachOr(creep, target, 3, () => creep.build(target));
		case 'repair':
			return approachOr(creep, target, 3, () => creep.repair(target));
		case 'attack':
			return approachOr(creep, target, 1, () => creep.attack(target));
		case 'rangedAttack':
			return approachOr(creep, target, 3, () => creep.rangedAttack(target));
		case 'heal':
			return approachOr(creep, target, 1, () => creep.heal(target));
		case 'rangedHeal':
			return approachOr(creep, target, 3, () => creep.rangedHeal(target));
		case 'claimController':
			return approachOr(creep, target, 1, () => creep.claimController(target));
		case 'reserveController':
			return approachOr(creep, target, 1, () => creep.reserveController(target));
		case 'attackController':
			return approachOr(creep, target, 1, () => creep.attackController(target));
		case 'dismantle':
			return approachOr(creep, target, 1, () => creep.dismantle(target));
		case 'createConstructionSite': {
			// The target pointer already denotes an encoder-verified free build tile.
			// Keeping location in one categorical avoids invalid anchor×direction products.
			const room = creep.room;
			if (target?.pos?.roomName && target.pos.roomName !== room.name) {
				return C.ERR_INVALID_TARGET;
			}
			const x = target?.pos?.x;
			const y = target?.pos?.y;
			if (!Number.isFinite(x) || !Number.isFinite(y)) return C.ERR_INVALID_TARGET;
			if (x < 1 || x > 48 || y < 1 || y > 48) return C.ERR_INVALID_TARGET;
			const structureType = CONSTRUCTION_TYPES[
				Math.max(0, Math.min(CONSTRUCTION_TYPES.length - 1, amountIndex | 0))
			] || C.STRUCTURE_EXTENSION;
			return room.createConstructionSite(x, y, structureType);
		}
		default:
			return C.ERR_NO_BODYPART;
	}
}

function runStructureIntent(Game, struct, type, target, amountBin = 0) {
	if (struct.structureType === C.STRUCTURE_SPAWN && type === 'spawnCreep') {
		// amount head selects body template index (0 = basic WCM)
		const idx = Math.max(0, Math.min(BODY_TEMPLATES.length - 1, amountBin | 0));
		const tmpl = BODY_TEMPLATES[idx] || BODY_TEMPLATES[0];
		const energy = struct.room?.energyAvailable ?? (struct.store?.[C.RESOURCE_ENERGY] || 0);
		const body = tmpl.parts;
		const cost = tmpl.cost ?? 250;
		if (energy < cost) return C.ERR_NOT_ENOUGH_ENERGY;
		const suffix = String(struct.id || 'spawn').replace(/[^a-zA-Z0-9]/g, '').slice(-8);
		const name = `rl_${Game.time}_${suffix}`;
		return struct.spawnCreep(body, name);
	}
	if (struct.structureType === C.STRUCTURE_TOWER) {
		if (type === 'attack' && target) return struct.attack(target);
		if (type === 'repair' && target) return struct.repair(target);
		if (type === 'heal' && target) return struct.heal(target);
	}
	return C.ERR_INVALID_ARGS;
}
