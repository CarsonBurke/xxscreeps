/**
 * End-tick traffic: creeps register move intents; we resolve swaps / yields / exits.
 */
import { movePriority } from './movePriorities';

export interface MoveIntent {
	creep: Creep;
	/** Same-room tile target (preferred) */
	x?: number;
	y?: number;
	/** Exit / free move by direction (multi-room) */
	direction?: DirectionConstant;
	roomName: string;
	priority: number;
}

const intentsByRoom = new Map<string, MoveIntent[]>();
const byCreep = new Map<string, MoveIntent>();
const immovable = new Set<string>();

export function resetTraffic(): void {
	intentsByRoom.clear();
	byCreep.clear();
	immovable.clear();
}

export function markImmovable(creep: Creep): void {
	immovable.add(creep.name);
}

function removePrev(creepName: string): void {
	const prev = byCreep.get(creepName);
	if (!prev) return;
	const list = intentsByRoom.get(prev.roomName);
	if (list) {
		const i = list.indexOf(prev);
		if (i >= 0) list.splice(i, 1);
	}
	byCreep.delete(creepName);
}

function addIntent(intent: MoveIntent): void {
	removePrev(intent.creep.name);
	byCreep.set(intent.creep.name, intent);
	let list = intentsByRoom.get(intent.roomName);
	if (!list) {
		list = [];
		intentsByRoom.set(intent.roomName, list);
	}
	list.push(intent);
}

/**
 * Register desired next tile in the creep's current room.
 */
export function registerMove(creep: Creep, x: number, y: number): void {
	if (creep.fatigue > 0 || creep.spawning) return;
	if (x < 0 || x > 49 || y < 0 || y > 49) return;
	// Same tile — no-op
	if (x === creep.pos.x && y === creep.pos.y) return;

	addIntent({
		creep,
		x,
		y,
		roomName: creep.room.name,
		priority: movePriority(creep),
	});
}

/**
 * Register a pure direction move (room exits / edge step).
 */
export function registerDirection(creep: Creep, direction: DirectionConstant): void {
	if (creep.fatigue > 0 || creep.spawning) return;
	if (!direction) return;
	addIntent({
		creep,
		direction,
		roomName: creep.room.name,
		priority: movePriority(creep),
	});
}

function posKey(x: number, y: number): string {
	return `${x},${y}`;
}

function creepAt(room: Room, x: number, y: number): Creep | undefined {
	return room.lookForAt(LOOK_CREEPS, x, y)[0];
}

export function runTraffic(): void {
	for (const [roomName, intents] of intentsByRoom) {
		const room = Game.rooms[roomName];
		if (!room || !intents.length) continue;
		resolveRoom(room, intents);
	}
	resetTraffic();
}

function resolveRoom(room: Room, intents: MoveIntent[]): void {
	const moved = new Set<string>();

	// Direction intents first (exits) — no tile conflict in-room
	for (const it of intents) {
		if (it.direction && !moved.has(it.creep.name)) {
			it.creep.move(it.direction);
			moved.add(it.creep.name);
		}
	}

	const tileIntents = intents.filter(it => it.x != null && it.y != null && !moved.has(it.creep.name));

	const wanted = new Map<string, MoveIntent[]>();
	for (const it of tileIntents) {
		const k = posKey(it.x!, it.y!);
		let arr = wanted.get(k);
		if (!arr) {
			arr = [];
			wanted.set(k, arr);
		}
		arr.push(it);
	}

	// Mutual swaps
	for (const it of tileIntents) {
		if (moved.has(it.creep.name)) continue;
		const occ = creepAt(room, it.x!, it.y!);
		if (!occ || occ.name === it.creep.name) continue;
		const otherIntent = byCreep.get(occ.name);
		if (!otherIntent || otherIntent.x == null || otherIntent.y == null) continue;
		if (otherIntent.x === it.creep.pos.x && otherIntent.y === it.creep.pos.y) {
			const d1 = it.creep.pos.getDirectionTo(it.x!, it.y!);
			const d2 = occ.pos.getDirectionTo(otherIntent.x, otherIntent.y);
			if (d1) it.creep.move(d1);
			if (d2) occ.move(d2);
			moved.add(it.creep.name);
			moved.add(occ.name);
		}
	}

	const rest = tileIntents
		.filter(it => !moved.has(it.creep.name))
		.sort((a, b) => b.priority - a.priority);

	for (const it of rest) {
		if (moved.has(it.creep.name)) continue;
		const k = posKey(it.x!, it.y!);
		const competitors = (wanted.get(k) || []).filter(c => !moved.has(c.creep.name));
		competitors.sort((a, b) => b.priority - a.priority);
		if (competitors[0] && competitors[0].creep.name !== it.creep.name) continue;

		const dir = it.creep.pos.getDirectionTo(it.x!, it.y!);
		if (!dir) continue;

		const occ = creepAt(room, it.x!, it.y!);
		if (!occ || occ.name === it.creep.name) {
			it.creep.move(dir);
			moved.add(it.creep.name);
			continue;
		}
		if (immovable.has(occ.name) || moved.has(occ.name)) continue;

		const occPrio = movePriority(occ);
		const occIntent = byCreep.get(occ.name);

		if (occIntent && (occIntent.direction || (occIntent.x !== it.x || occIntent.y !== it.y))) {
			it.creep.move(dir);
			moved.add(it.creep.name);
			continue;
		}

		if (occPrio < it.priority && !occIntent) {
			const pushDir = pushDirection(room, occ, it.creep.pos);
			if (pushDir) {
				occ.move(pushDir);
				it.creep.move(dir);
				moved.add(it.creep.name);
				moved.add(occ.name);
			}
		}
	}
}

function pushDirection(room: Room, creep: Creep, from: RoomPosition): DirectionConstant | null {
	const dirs: DirectionConstant[] = [
		TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT, LEFT, TOP_LEFT,
	];
	dirs.sort((a, b) => {
		const pa = posInDir(creep.pos, a);
		const pb = posInDir(creep.pos, b);
		return from.getRangeTo(pb.x, pb.y) - from.getRangeTo(pa.x, pa.y);
	});
	for (const d of dirs) {
		const p = posInDir(creep.pos, d);
		if (p.x < 0 || p.x > 49 || p.y < 0 || p.y > 49) continue;
		if (room.getTerrain().get(p.x, p.y) === TERRAIN_MASK_WALL) continue;
		const blocked = room.lookForAt(LOOK_STRUCTURES, p.x, p.y).some(s =>
			s.structureType !== STRUCTURE_ROAD &&
			s.structureType !== STRUCTURE_CONTAINER &&
			!(s.structureType === STRUCTURE_RAMPART && (s as StructureRampart).my),
		);
		if (blocked) continue;
		if (room.lookForAt(LOOK_CREEPS, p.x, p.y).length) continue;
		return d;
	}
	return null;
}

function posInDir(pos: RoomPosition, d: DirectionConstant): { x: number; y: number } {
	const dx = [0, 0, 1, 1, 1, 0, -1, -1, -1];
	const dy = [0, -1, -1, 0, 1, 1, 1, 0, -1];
	return { x: pos.x + dx[d]!, y: pos.y + dy[d]! };
}
