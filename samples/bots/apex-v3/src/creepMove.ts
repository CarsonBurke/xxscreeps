/**
 * Civil movement: cached ignoreCreeps paths + traffic registration.
 * Multi-room: path to exit tile, then registerDirection through the exit.
 */
import { getCachedPath, findPathIgnoreCreeps, type PathOpts } from './pathCache';
import { registerMove, registerDirection, markImmovable } from './trafficCore';

interface CreepPathState {
	steps: RoomPosition[];
	goalKey: string;
	index: number;
	cachedAt: number;
	stuck: number;
	lastX: number;
	lastY: number;
	lastRoom: string;
}

const creepState = new Map<string, CreepPathState>();

function goalKey(pos: RoomPosition, range: number): string {
	return `${pos.roomName}:${pos.x},${pos.y},r${range}`;
}

/** Direction to step off the map edge when already on an exit tile. */
function edgeExitDirection(pos: RoomPosition): DirectionConstant | null {
	if (pos.y === 0) return TOP;
	if (pos.y === 49) return BOTTOM;
	if (pos.x === 0) return LEFT;
	if (pos.x === 49) return RIGHT;
	return null;
}

/**
 * Register next path step — same-room tile or exit direction into next room.
 */
function registerStep(creep: Creep, step: RoomPosition): void {
	if (step.roomName === creep.room.name) {
		registerMove(creep, step.x, step.y);
		return;
	}
	// Next step is in another room: go to exit, or step through if already on edge.
	const exitConst = creep.room.findExitTo(step.roomName);
	if (exitConst === ERR_NO_PATH || typeof exitConst !== 'number') return;

	const edgeDir = edgeExitDirection(creep.pos);
	// On the correct edge facing the target room?
	if (edgeDir != null) {
		// Verify this edge leads to step.roomName
		const exits = creep.room.find(exitConst as FindConstant) as RoomPosition[];
		const onThisExit = exits.some(e => e.isEqualTo(creep.pos));
		if (onThisExit) {
			registerDirection(creep, edgeDir);
			return;
		}
	}

	const exitPos = creep.pos.findClosestByPath(exitConst as FindConstant, {
		ignoreCreeps: true,
	}) as RoomPosition | null;
	if (exitPos) {
		if (creep.pos.isEqualTo(exitPos)) {
			const d = edgeExitDirection(creep.pos);
			if (d) registerDirection(creep, d);
		} else {
			registerMove(creep, exitPos.x, exitPos.y);
		}
	}
}

export function goTo(
	creep: Creep,
	target: RoomPosition | { pos: RoomPosition },
	opts: PathOpts & { range?: number; immovableAtGoal?: boolean } = {},
): ScreepsReturnCode {
	const to = 'pos' in target && (target as { pos: RoomPosition }).pos
		? (target as { pos: RoomPosition }).pos
		: (target as RoomPosition);
	const range = opts.range != null ? opts.range : 1;

	if (creep.fatigue > 0) return ERR_TIRED;
	if (creep.spawning) return ERR_BUSY;

	// Same-room arrival
	if (creep.pos.roomName === to.roomName && creep.pos.inRangeTo(to, range)) {
		if (opts.immovableAtGoal) markImmovable(creep);
		return OK;
	}

	const gKey = goalKey(to, range);
	const ttl = opts.ttl != null ? opts.ttl : 40;
	let st = creepState.get(creep.name);

	const needNew =
		!st ||
		st.goalKey !== gKey ||
		st.lastRoom !== creep.room.name ||
		Game.time - st.cachedAt > ttl ||
		st.index >= st.steps.length;

	if (needNew) {
		const path = getCachedPath(creep.pos, to, { ...opts, range });
		st = {
			steps: path,
			goalKey: gKey,
			index: 0,
			cachedAt: Game.time,
			stuck: 0,
			lastX: creep.pos.x,
			lastY: creep.pos.y,
			lastRoom: creep.room.name,
		};
		creepState.set(creep.name, st);
	}

	if (st!.lastX === creep.pos.x && st!.lastY === creep.pos.y && st!.lastRoom === creep.room.name) {
		st!.stuck++;
		if (st!.stuck >= 3) {
			const fresh = findPathIgnoreCreeps(creep.pos, to, { ...opts, range });
			st = {
				steps: fresh,
				goalKey: gKey,
				index: 0,
				cachedAt: Game.time,
				stuck: 0,
				lastX: creep.pos.x,
				lastY: creep.pos.y,
				lastRoom: creep.room.name,
			};
			creepState.set(creep.name, st);
		}
	} else {
		st!.stuck = 0;
		st!.lastX = creep.pos.x;
		st!.lastY = creep.pos.y;
		st!.lastRoom = creep.room.name;
	}

	if (!st!.steps.length) {
		// Fallback: direct exit toward goal room
		if (to.roomName !== creep.room.name) {
			const exitConst = creep.room.findExitTo(to.roomName);
			if (typeof exitConst === 'number' && exitConst >= 0) {
				const edgeDir = edgeExitDirection(creep.pos);
				const exits = creep.room.find(exitConst as FindConstant) as RoomPosition[];
				if (edgeDir && exits.some(e => e.isEqualTo(creep.pos))) {
					registerDirection(creep, edgeDir);
					return OK;
				}
				const exitPos = creep.pos.findClosestByPath(exitConst as FindConstant, {
					ignoreCreeps: true,
				}) as RoomPosition | null;
				if (exitPos) {
					if (creep.pos.isEqualTo(exitPos)) {
						const d = edgeExitDirection(creep.pos);
						if (d) registerDirection(creep, d);
					} else registerMove(creep, exitPos.x, exitPos.y);
					return OK;
				}
			}
		}
		return ERR_NO_PATH;
	}

	// Skip steps we're already on
	while (st!.index < st!.steps.length) {
		const step = st!.steps[st!.index]!;
		if (step.roomName === creep.room.name && creep.pos.isEqualTo(step)) {
			st!.index++;
			continue;
		}
		// Adjacent in same room
		if (step.roomName === creep.room.name && creep.pos.isNearTo(step)) {
			registerMove(creep, step.x, step.y);
			return OK;
		}
		// Next step other room or not adjacent — use registerStep
		if (step.roomName !== creep.room.name || !creep.pos.isNearTo(step)) {
			// If not adjacent and same room, might need repath
			if (step.roomName === creep.room.name && !creep.pos.isNearTo(step)) {
				const fresh = findPathIgnoreCreeps(creep.pos, to, { ...opts, range });
				if (!fresh.length) return ERR_NO_PATH;
				st!.steps = fresh;
				st!.index = 0;
				st!.cachedAt = Game.time;
				registerStep(creep, fresh[0]!);
				return OK;
			}
			registerStep(creep, step);
			return OK;
		}
		st!.index++;
	}

	const again = findPathIgnoreCreeps(creep.pos, to, { ...opts, range });
	if (!again.length) return ERR_NO_PATH;
	st!.steps = again;
	st!.index = 0;
	registerStep(creep, again[0]!);
	return OK;
}

export function goDo(
	creep: Creep,
	target: RoomObject | RoomPosition,
	range: number,
	action: () => ScreepsReturnCode,
): ScreepsReturnCode {
	if (!target) return ERR_INVALID_TARGET;
	const pos = target instanceof RoomPosition ? target : (target as RoomObject).pos;
	if (creep.pos.roomName === pos.roomName && creep.pos.inRangeTo(pos, range)) {
		if (range <= 1) markImmovable(creep);
		return action();
	}
	return goTo(creep, pos, { range, preferRoads: true });
}

export function moveToRoom(creep: Creep, roomName: string): ScreepsReturnCode {
	if (creep.room.name === roomName) return OK;
	const route = Game.map.findRoute(creep.room.name, roomName);
	if (route === ERR_NO_PATH || !route.length) return ERR_NO_PATH;
	const exitConst = route[0]!.exit;

	const exits = creep.room.find(exitConst) as RoomPosition[];
	const onExit = exits.some(e => e.isEqualTo(creep.pos));
	if (onExit) {
		const d = edgeExitDirection(creep.pos);
		if (d) {
			registerDirection(creep, d);
			return OK;
		}
	}

	const exitPos = creep.pos.findClosestByPath(exitConst, { ignoreCreeps: true }) as RoomPosition | null;
	if (!exitPos) return ERR_NO_PATH;
	// Path to exit tile (range 0), then next tick we step through
	return goTo(creep, exitPos, { range: 0, preferRoads: true, maxRooms: 1 });
}

export function clearCreepPath(name: string): void {
	creepState.delete(name);
}

/** Drop dead creep path state occasionally */
export function gcCreepPaths(): void {
	for (const name of creepState.keys()) {
		if (!Game.creeps[name]) creepState.delete(name);
	}
}
