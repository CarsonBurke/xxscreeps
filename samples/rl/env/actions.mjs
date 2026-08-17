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
	intentTypes, intentSpecs, amountBins, directions, constructionTypes, maxRooms,
	maxBodyParts, bodyPartTypes, actionOutcomes,
} = schema;
const DIR_CONST = [
	C.TOP, C.TOP_RIGHT, C.RIGHT, C.BOTTOM_RIGHT,
	C.BOTTOM, C.BOTTOM_LEFT, C.LEFT, C.TOP_LEFT,
];
const CONSTRUCTION_TYPES = constructionTypes.map(type => C[type.toUpperCase()] || type);
const BODY_PART_CONST = bodyPartTypes.map(type => {
	const part = C[type.toUpperCase()];
	if (!part) throw new Error(`schema bodyPartTypes contains unknown part ${JSON.stringify(type)}`);
	return part;
});
const OUTCOME = Object.fromEntries(actionOutcomes.map((name, index) => [ name, index ]));

/** Stable categorical feedback for the next observation of this actor. */
export function actionOutcome(code, type = '', actorKind = '', executed = true) {
	if (Number(code) === C.OK && actorKind === 'creep' && type !== 'move' && !executed) {
		return OUTCOME.approaching;
	}
	if (Number(code) === C.ERR_NOT_ENOUGH_ENERGY) {
		return type === 'spawnCreep' || actorKind === 'structure'
			? OUTCOME.not_enough_energy
			: OUTCOME.not_enough_resources;
	}
	switch (Number(code)) {
		case C.OK: return OUTCOME.ok;
		case C.ERR_BUSY: return OUTCOME.busy;
		case C.ERR_INVALID_TARGET: return OUTCOME.invalid_target;
		case C.ERR_INVALID_ARGS: return OUTCOME.invalid_args;
		case C.ERR_NO_BODYPART: return OUTCOME.no_bodypart;
		case C.ERR_FULL: return OUTCOME.full;
		case C.ERR_TIRED: return OUTCOME.tired;
		case C.ERR_NAME_EXISTS: return OUTCOME.name_exists;
		case C.ERR_NO_PATH: return OUTCOME.no_path;
		case C.ERR_NOT_OWNER: return OUTCOME.not_owner;
		case C.ERR_RCL_NOT_ENOUGH: return OUTCOME.rcl_not_enough;
		default: return OUTCOME.other;
	}
}

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
 *   bodyCounts: number[][][],
 *   bodyOrder: number[][][],
 *   constructionTypes: number[][],
 *   constructionTiles: number[][],
 * }} actions  // [actor][slot]
 */
export function applyActions(Game, meta, actions) {
	const results = [];
	const { actorMeta, targetMeta } = meta;
	const nActors = Math.min(actorMeta.length, actions.types.length);
	pruneStepCache(Game.time);
	const context = createExecutionContext(Game, actorMeta, targetMeta, actions, nActors);

	for (let ai = 0; ai < nActors; ai++) {
		const actor = actorMeta[ai];
		// The schema exposes one primary action per actor. Never let malformed or
		// stale callers smuggle multiple engine intents through a longer row.
		const slots = Math.min(1, actions.types[ai]?.length ?? 0);
		for (let slot = 0; slot < slots; slot++) {
			const typeIdx = actions.types[ai][slot] | 0;
			if (typeIdx < 0 || typeIdx >= intentTypes.length) {
				results.push({ actor: actor.id, type: 'invalid', code: C.ERR_INVALID_ARGS });
				continue;
			}
			const type = intentTypes[typeIdx];
			if (type === 'none') continue;

			const dirIdx = actions.dirs[ai]?.[slot] | 0;
			const tgtIdx = actions.targets[ai]?.[slot] | 0;
			const amtIdx = actions.amounts[ai]?.[slot] | 0;
			const bodyCounts = actions.bodyCounts?.[ai]?.[slot];
			const bodyOrder = actions.bodyOrder?.[ai]?.[slot];
			const constructionTypeValue = actions.constructionTypes?.[ai]?.[slot];
			const constructionType = constructionTypeValue == null
				? -1 : Number(constructionTypeValue);
			const constructionTileValue = actions.constructionTiles?.[ai]?.[slot];
			const constructionTile = constructionTileValue == null
				? -1 : Number(constructionTileValue);
			const factors = intentSpecs[type]?.factors || [];
			if (
				(factors.includes('direction') && (dirIdx < 0 || dirIdx >= DIR_CONST.length))
				|| (factors.includes('target') && (tgtIdx < 0 || tgtIdx >= targetMeta.length))
				|| (factors.includes('amount') && (amtIdx < 0 || amtIdx >= amountBins.length))
				|| (factors.includes('constructionType') && (
					!Number.isInteger(constructionType)
					|| constructionType < 0 || constructionType >= CONSTRUCTION_TYPES.length
				))
				|| (factors.includes('constructionTile') && (
					!Number.isInteger(constructionTile)
					|| constructionTile < 0 || constructionTile >= schema.roomSize * schema.roomSize
				))
			) {
				results.push({ actor: actor.id, type, code: C.ERR_INVALID_ARGS });
				continue;
			}
			const amountBin = amountBins[amtIdx] ?? 0;
			const body = factors.includes('bodyCounts')
				? decodeSpawnBody(bodyCounts, bodyOrder)
				: null;
			if (factors.includes('bodyCounts') && !body) {
				results.push({ actor: actor.id, type, code: C.ERR_INVALID_ARGS });
				continue;
			}

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
					const target = context.resolveTarget(tgtIdx);
					if (!targetLegalForIntent(type, metaT, target)) {
						results.push({ actor: actor.id, type, code: C.ERR_INVALID_TARGET, err: 'kind filter' });
						continue;
					}
					const ready = readyForPrimitiveIntent(creep, type, target);
					code = runCreepIntent(
						creep, type, dirIdx, target, amountBin, amtIdx, context,
					);
					executed = ready && code === C.OK;
				} else if (actor.kind === 'room') {
					const room = Game.rooms[actor.room];
					if (!room || actor.id !== `room:${actor.room}`) {
						results.push({ actor: actor.id, type, code: -1, err: 'missing room' });
						continue;
					}
					code = runRoomIntent(
						Game, room, type, constructionType, constructionTile,
					);
					executed = code === C.OK;
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
					const target = context.resolveTarget(tgtIdx);
					code = runStructureIntent(Game, obj, type, target, amtIdx, body);
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
				...(type === 'spawnCreep' && code === C.OK ? {
					spawnBodyLength: body.length,
					spawnBodyParts: body.map(part => BODY_PART_CONST.indexOf(part)),
				} : {}),
				...(type === 'createConstructionSite' ? {
					constructionType,
					constructionTile,
				} : {}),
				targetKind: resultTarget?.kind,
				targetId: resultTarget?.id,
				targetStructureType: resultTarget?.structureType,
			});
		}
	}
	return results;
}

function createExecutionContext(Game, actorMeta, targetMeta, actions, nActors) {
	const targetCache = new Map();
	const context = {
		reservedSteps: new Set(),
		stationarySteps: new Set(),
		energyStores: new Map(),
		resolveTarget(index) {
			if (!targetCache.has(index)) {
				targetCache.set(index, resolveTarget(Game, targetMeta[index]));
			}
			return targetCache.get(index);
		},
	};
	const movers = new Set();
	for (let ai = 0; ai < nActors; ai++) {
		const actor = actorMeta[ai];
		if (actor?.kind !== 'creep') continue;
		const creep = Game.creeps[actor.id];
		const canMove = creep?.body?.some?.(part => part.type === C.MOVE && part.hits > 0) ?? true;
		if (!creep || creep.fatigue > 0 || !canMove) continue;
		const typeIdx = actions.types[ai]?.[0] | 0;
		const type = intentTypes[typeIdx];
		if (type === 'move') {
			const directionIndex = actions.dirs[ai]?.[0] | 0;
			if (directionIndex < 0 || directionIndex >= DIR_CONST.length) continue;
			movers.add(creep.id);
			continue;
		}
		const range = APPROACH_RANGE[type];
		if (range == null) continue;
		const targetIndex = actions.targets[ai]?.[0] | 0;
		if (targetIndex < 0 || targetIndex >= targetMeta.length) continue;
		const metaT = targetMeta[targetIndex];
		const target = context.resolveTarget(targetIndex);
		if (!targetLegalForIntent(type, metaT, target)) continue;
		const pos = target?.pos || target;
		if (pos && !creep.pos.inRangeTo(pos, range)) movers.add(creep.id);
	}
	const visibleCreeps = new Map();
	for (const creep of Object.values(Game.creeps || {})) visibleCreeps.set(creep.id, creep);
	for (const room of Object.values(Game.rooms || {})) {
		for (const creep of room.find?.(C.FIND_CREEPS) || []) visibleCreeps.set(creep.id, creep);
	}
	for (const creep of visibleCreeps.values()) {
		if (!creep?.pos || movers.has(creep.id)) continue;
		context.stationarySteps.add(stepKey(creep.pos.roomName, creep.pos.x, creep.pos.y));
	}
	return context;
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
	if (Game.getObjectById) {
		const object = Game.getObjectById(meta.id);
		if (object) return object;
	}
	if (meta.kind === 'creep') {
		// name as id fallback
		const byName = Game.creeps[meta.id];
		if (byName) return byName;
		for (const creep of Object.values(Game.creeps)) {
			if (creep.id === meta.id) return creep;
		}
		return null;
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
let stepCacheTick = -1;

/**
 * Episode and segment starts must begin with a cold executor. Ids are
 * regenerated deterministically per seed, so a surviving route keyed by a
 * recycled creep id would steer a different creep, and a restored snapshot
 * whose tick happens to follow the previous tick would keep a foreign route.
 */
export function resetNavigationCaches() {
	stepCache.clear();
	stepCacheTick = -1;
}

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
	stepCacheTick = time;
}

function goalKey(creep, pos, range) {
	const id = creep.id || creep.name || '?';
	const room = pos.roomName || creep.pos.roomName;
	return `${id}|${pos.x},${pos.y},${room}|r${range}`;
}

function chebyshevStep(creep, pos, context) {
	const dx = Math.sign(pos.x - creep.pos.x);
	const dy = Math.sign(pos.y - creep.pos.y);
	const next = { x: creep.pos.x + dx, y: creep.pos.y + dy, roomName: creep.pos.roomName };
	return moveWithReservation(creep, next, pos, context);
}

/** Use the engine's exit-aware, cached inter-room router for strategic goals. */
function roomExitStep(creep, destination, context) {
	let moveToCode = C.ERR_NO_PATH;
	try {
		if (typeof creep.moveTo === 'function') {
			moveToCode = creep.moveTo(destination, {
				reusePath: 10,
				serializeMemory: true,
				plainCost: 2,
				swampCost: 10,
				maxOps: 4000,
				maxRooms: Math.max(2, maxRooms),
			});
			if (moveToCode !== C.ERR_NO_PATH) return moveToCode;
		}
	} catch { /* try the explicit bounded router below */ }
	try {
		if (typeof PathFinder !== 'undefined' && PathFinder.search) {
			const result = PathFinder.search(creep.pos, { pos: destination, range: 1 }, {
				plainCost: 2, swampCost: 10, maxOps: 4000, maxRooms: Math.max(2, maxRooms),
			});
			if (result.path?.length) {
				return moveWithReservation(creep, result.path[0], destination, context);
			}
		}
	} catch { /* preserve the native failure */ }
	return moveToCode;
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

function isExitTile(pos) {
	return pos && (pos.x === 0 || pos.x === 49 || pos.y === 0 || pos.y === 49);
}

/** Reserve unique destinations while allowing native following and swaps. */
function moveWithReservation(creep, intended, goal, context) {
	const room = creep.room;
	const sameRoomGoal = !goal?.roomName || goal.roomName === room.name;
	const candidates = [ intended ];
	for (let d = 0; d < 8; d++) {
		candidates.push({
			x: creep.pos.x + directions[d].dx,
			y: creep.pos.y + directions[d].dy,
			roomName: creep.pos.roomName,
		});
	}
	if (sameRoomGoal) {
		candidates.sort((a, b) => chebyshev(a, goal) - chebyshev(b, goal));
	}
	for (const next of candidates) {
		// Cached/engine paths may transiently include the actor's current tile.
		// Passing direction 0/undefined to creep.move is ERR_INVALID_TARGET;
		// skip it and choose an actual step instead.
		if ((next.roomName || room.name) === creep.pos.roomName
			&& next.x === creep.pos.x && next.y === creep.pos.y) continue;
		// Congestion fallback must not turn an ordinary local macro into an
		// accidental room transition. Explicit move actions and actual cross-room
		// goals retain native exit movement.
		if (sameRoomGoal && !isExitTile(goal) && isExitTile(next)) continue;
		const key = stepKey(next.roomName || room.name, next.x, next.y);
		if (context.reservedSteps.has(key) || context.stationarySteps.has(key)) continue;
		if ((next.roomName || room.name) === room.name && !walkable(room, next.x, next.y)) continue;
		context.reservedSteps.add(key);
		const nextPos = new RoomPosition(next.x, next.y, next.roomName || room.name);
		const direction = creep.pos.getDirectionTo(nextPos);
		if (!Number.isInteger(direction) || direction < C.TOP || direction > C.TOP_LEFT) {
			context.reservedSteps.delete(key);
			return C.ERR_BUSY;
		}
		return creep.move(direction);
	}
	// Never bypass reservations with an untracked fallback. ERR_BUSY is a team
	// contention signal, not an invalid primitive.
	return C.ERR_BUSY;
}

function moveInDirection(creep, dirIdx, context) {
	const direction = directions[dirIdx];
	if (!direction) return C.ERR_INVALID_ARGS;
	const next = {
		x: creep.pos.x + direction.dx,
		y: creep.pos.y + direction.dy,
		roomName: creep.pos.roomName,
	};
	// Direct moves across an exit are native Screeps intents. The neighbouring
	// room's wrapped coordinate is not available without parsing room names, so
	// leave this uncommon reservation to the engine instead of rejecting a legal
	// action as an out-of-bounds path.
	if (next.x < 0 || next.x > 49 || next.y < 0 || next.y > 49) {
		return creep.move(DIR_CONST[dirIdx]);
	}
	const key = stepKey(next.roomName, next.x, next.y);
	if (context.reservedSteps.has(key) || context.stationarySteps.has(key)) return C.ERR_BUSY;
	if (!walkable(creep.room, next.x, next.y)) return C.ERR_NO_PATH;
	context.reservedSteps.add(key);
	return creep.move(DIR_CONST[dirIdx]);
}

/**
 * One step toward dest via PathFinder when available; else Chebyshev greedy dir.
 * Caches remaining path tiles so sticky approachOr does not re-A* every tick.
 */
function moveToward(creep, dest, range = 1, context) {
	if (!dest?.pos && dest?.x == null) return C.ERR_INVALID_TARGET;
	const pos = dest.pos || dest;
	if (creep.pos.inRangeTo(pos, range)) return C.OK;
	if (creep.fatigue > 0) return C.ERR_TIRED;
	if (pos.roomName && pos.roomName !== creep.room.name) {
		return roomExitStep(creep, pos, context);
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
			const code = moveWithReservation(creep, next, pos, context);
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
					// Keep the first tile until the next tick proves the move landed.
					// Dropping it eagerly made a blocked creep skip ahead in its cache.
					stepCache.set(gkey, rest);
					if (stepCache.size > 64) {
						const oldest = stepCache.keys().next().value;
						stepCache.delete(oldest);
					}
					return moveWithReservation(creep, first, pos, context);
				}
			}
		} catch { /* fall through */ }
	}
	return chebyshevStep(creep, pos, context);
}

/**
 * Economy intents: if out of range, path toward target instead of pure fail.
 * Teaches target-as-goal under H+C reward (nav assist, not reward shaping).
 */
function approachOr(creep, target, range, act, context) {
	if (!target) return C.ERR_INVALID_TARGET;
	const pos = target.pos || target;
	if (creep.pos.inRangeTo(pos, range)) return act();
	return moveToward(creep, target, range, context);
}

function energyStore(context, target) {
	const key = target?.id;
	if (!key) return null;
	let projected = context.energyStores.get(key);
	if (!projected) {
		const energy = Math.max(0, Number(target.store?.[C.RESOURCE_ENERGY]) || 0);
		const free = Math.max(
			0, Number(target.store?.getFreeCapacity?.(C.RESOURCE_ENERGY)) || 0,
		);
		projected = { withdrawAvailable: energy, depositFree: free };
		context.energyStores.set(key, projected);
	}
	return projected;
}

function runCreepIntent(
	creep, type, dirIdx, target, amountBin, amountIndex = 0, context,
) {
	switch (type) {
		case 'move':
			// Dir-only only. amount is NOT in move's logπ (type-gated), so amountBin>0
			// was a latent hijack: random bins path to target-table[0]=source and break
			// deliver/upgrade approach. Path-to-goal lives in harvest/transfer via approachOr.
			return moveInDirection(creep, dirIdx, context);
		case 'harvest': {
			// Always approachOr so RL_NAV=cheap|pathfinder applies (no silent moveTo bypass).
			const src = target || null;
			return approachOr(creep, src, 1, () => creep.harvest(src), context);
		}
		case 'transfer': {
			if (target?.id === creep.id) return C.ERR_INVALID_TARGET;
			return approachOr(creep, target, 1, () => {
				const carried = creep.store?.[C.RESOURCE_ENERGY] || 0;
				const projected = energyStore(context, target);
				const rawFree = target?.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) ?? 0;
				const available = Math.min(carried, projected?.depositFree ?? rawFree);
				const amt = legalAmount(amountBin, available);
				if (amt <= 0) return rawFree > 0 ? C.ERR_BUSY : C.ERR_FULL;
				const code = creep.transfer(target, C.RESOURCE_ENERGY, amt);
				if (code === C.OK && projected) projected.depositFree -= amt;
				return code;
			}, context);
		}
		case 'withdraw': {
			return approachOr(creep, target, 1, () => {
				const rawAvailable = target?.store?.[C.RESOURCE_ENERGY] || 0;
				const projected = energyStore(context, target);
				const available = projected?.withdrawAvailable ?? rawAvailable;
				const free = creep.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) ?? 0;
				const amt = legalAmount(amountBin, Math.min(available, free));
				if (amt <= 0) {
					return rawAvailable > 0 ? C.ERR_BUSY : C.ERR_NOT_ENOUGH_RESOURCES;
				}
				const code = creep.withdraw(target, C.RESOURCE_ENERGY, amt);
				if (code === C.OK && projected) projected.withdrawAvailable -= amt;
				return code;
			}, context);
		}
		case 'pickup':
			return approachOr(creep, target, 1, () => creep.pickup(target), context);
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
			return approachOr(creep, ctrl, 3, () => creep.upgradeController(ctrl), context);
		}
		case 'build':
			return approachOr(creep, target, 3, () => creep.build(target), context);
		case 'repair':
			return approachOr(creep, target, 3, () => creep.repair(target), context);
		case 'attack':
			return approachOr(creep, target, 1, () => creep.attack(target), context);
		case 'rangedAttack':
			return approachOr(creep, target, 3, () => creep.rangedAttack(target), context);
		case 'heal':
			return approachOr(creep, target, 1, () => creep.heal(target), context);
		case 'rangedHeal':
			return approachOr(creep, target, 3, () => creep.rangedHeal(target), context);
		case 'claimController':
			return approachOr(creep, target, 1, () => creep.claimController(target), context);
		case 'reserveController':
			return approachOr(creep, target, 1, () => creep.reserveController(target), context);
		case 'attackController':
			return approachOr(creep, target, 1, () => creep.attackController(target), context);
		case 'dismantle':
			return approachOr(creep, target, 1, () => creep.dismantle(target), context);
		default:
			return C.ERR_NO_BODYPART;
	}
}

function constructionSiteName(Game, roomName, structureType) {
	if (structureType !== C.STRUCTURE_SPAWN) return undefined;
	return `rlcs_${Game.time}_${roomName.replace(/[^a-zA-Z0-9]/g, '')}`;
}

function runRoomIntent(Game, room, type, constructionTypeIndex, constructionTile) {
	if (type !== 'createConstructionSite') return C.ERR_INVALID_ARGS;
	const structureType = CONSTRUCTION_TYPES[constructionTypeIndex];
	if (!structureType) return C.ERR_INVALID_ARGS;
	const x = constructionTile % schema.roomSize;
	const y = Math.floor(constructionTile / schema.roomSize);
	return room.createConstructionSite(
		x, y, structureType, constructionSiteName(Game, room.name, structureType),
	);
}

function runStructureIntent(Game, struct, type, target, amountBin = 0, spawnBody = null) {
	if (struct.structureType === C.STRUCTURE_SPAWN && type === 'spawnCreep') {
		if (!Array.isArray(spawnBody) || spawnBody.length === 0) return C.ERR_INVALID_ARGS;
		const suffix = String(struct.id || 'spawn').replace(/[^a-zA-Z0-9]/g, '').slice(-8);
		const name = `rl_${Game.time}_${suffix}`;
		// Encoder/model affordability conditioning prevents predictable failures;
		// the engine remains the final authority for same-tick races and returns
		// categorical feedback on any rejected attempt. There is no hidden queue.
		return struct.spawnCreep(spawnBody, name);
	}
	if (struct.structureType === C.STRUCTURE_TOWER) {
		if (type === 'attack' && target) return struct.attack(target);
		if (type === 'repair' && target) return struct.repair(target);
		if (type === 'heal' && target) return struct.heal(target);
	}
	return C.ERR_INVALID_ARGS;
}

function decodeSpawnBody(counts, order) {
	const typeCount = BODY_PART_CONST.length;
	if (!counts || !order || counts.length !== typeCount || order.length !== typeCount) {
		return null;
	}
	const seen = new Uint8Array(typeCount);
	let length = 0;
	let nonzeroTypes = 0;
	for (let type = 0; type < typeCount; type++) {
		const count = Number(counts[type]);
		const orderedType = Number(order[type]);
		if (!Number.isInteger(count) || count < 0 || count > maxBodyParts) return null;
		if (!Number.isInteger(orderedType) || orderedType < 0 || orderedType >= typeCount) return null;
		if (seen[orderedType]) return null;
		seen[orderedType] = 1;
		length += count;
		if (count > 0) nonzeroTypes += 1;
	}
	if (length < 1 || length > maxBodyParts) return null;
	// There is only one representation of an executed grouped body: all nonzero
	// types occupy the active prefix, and zero-count types follow in ascending
	// order. This removes probability aliases from irrelevant suffix orderings.
	for (let index = 0; index < typeCount; index++) {
		const type = Number(order[index]);
		if ((index < nonzeroTypes) !== (Number(counts[type]) > 0)) return null;
		if (index > nonzeroTypes && type < Number(order[index - 1])) return null;
	}
	const body = [];
	for (const type of order) {
		for (let index = 0; index < counts[type]; index++) body.push(BODY_PART_CONST[type]);
	}
	return body;
}
