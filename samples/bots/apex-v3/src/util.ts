import config = require('./config');
import { asPos, type Coord } from './codec';
import { CreepMem } from './memoryKeys';
import { getRole as getRoleEnum, setRole as setRoleEnum, type Role } from './creepMem';
// Import creepMove only (not traffic barrel) to avoid util↔relay cycles.
import {
	goTo as trafficGoTo,
	goDo as trafficGoDo,
	moveToRoom as trafficMoveToRoom,
} from './creepMove';

const BODY_COST: Partial<Record<BodyPartConstant, number>> = {
	move: 50, work: 100, carry: 50, attack: 80,
	ranged_attack: 150, heal: 250, claim: 600, tough: 10,
};

export function bodyCost(body: BodyPartConstant[]): number {
	let sum = 0;
	for (let i = 0; i < body.length; i++) {
		sum += BODY_COST[body[i]!] || BODYPART_COST[body[i]!] || 0;
	}
	return sum;
}

export function energyOf(store: StoreDefinition | StoreDefinitionUnlimited | undefined | null): number {
	if (!store) return 0;
	if (typeof store.getUsedCapacity === 'function') {
		const v = store.getUsedCapacity(RESOURCE_ENERGY);
		if (v != null) return v;
	}
	return store[RESOURCE_ENERGY] || 0;
}

export function freeCapacity(
	store: StoreDefinition | StoreDefinitionUnlimited | undefined | null,
	resource: ResourceConstant = RESOURCE_ENERGY,
): number {
	if (!store) return 0;
	if (typeof store.getFreeCapacity === 'function') {
		let v = store.getFreeCapacity(resource);
		if (v == null || Number.isNaN(v)) v = store.getFreeCapacity();
		if (v != null && !Number.isNaN(v)) return v;
	}
	return 0;
}

export function creepMem(creep: Creep): CreepMemory {
	return creep.memory;
}

export function getRole(creep: Creep): Role {
	return getRoleEnum(creep);
}

export function setRole(creep: Creep, role: Role): void {
	setRoleEnum(creep, role);
}

export function updateWorking(creep: Creep, flag: number = CreepMem.working): boolean {
	const m = creep.memory;
	const energy = energyOf(creep.store);
	const free = freeCapacity(creep.store, RESOURCE_ENERGY);
	if (energy <= 0) m[flag] = false;
	else if (free <= 0) m[flag] = true;
	return !!m[flag];
}

export function ownedRooms(): Room[] {
	const rooms: Room[] = [];
	for (const name in Game.rooms) {
		const room = Game.rooms[name]!;
		if (room.controller && room.controller.my) rooms.push(room);
	}
	return rooms;
}

export function roomNameToXY(roomName: string): Coord | null {
	const m = /^([WE])(\d+)([NS])(\d+)$/.exec(roomName);
	if (!m) return null;
	let x = Number(m[2]);
	let y = Number(m[4]);
	if (m[1] === 'W') x = -x - 1;
	if (m[3] === 'N') y = -y - 1;
	return { x, y };
}

export function xyToRoomName(x: number, y: number): string {
	const ew = x < 0 ? 'W' + (-x - 1) : 'E' + x;
	const ns = y < 0 ? 'N' + (-y - 1) : 'S' + y;
	return ew + ns;
}

export function adjacentRoomNames(roomName: string): string[] {
	const xy = roomNameToXY(roomName);
	if (!xy) return [];
	return ([[0, -1], [1, 0], [0, 1], [-1, 0]] as const)
		.map(([dx, dy]) => xyToRoomName(xy.x + dx, xy.y + dy));
}

export function isHighway(roomName: string): boolean {
	const m = /^[WE](\d+)[NS](\d+)$/.exec(roomName);
	if (!m) return false;
	return Number(m[1]) % 10 === 0 || Number(m[2]) % 10 === 0;
}

export function isSourceKeeperRoom(roomName: string): boolean {
	const m = /^[WE](\d+)[NS](\d+)$/.exec(roomName);
	if (!m) return false;
	const x = Number(m[1]) % 10;
	const y = Number(m[2]) % 10;
	return x >= 4 && x <= 6 && y >= 4 && y <= 6;
}

/**
 * Can we field a melee SK killer at this spawn capacity?
 * SK ~5000 hits, ~400 dps: need ~19A19M (~2470e) to win solo before dying.
 */
export function canAffordSkKiller(spawnCapacity: number): boolean {
	const minE = (config as { skKillerMinEnergy?: number }).skKillerMinEnergy ?? 2500;
	return spawnCapacity >= minE;
}

/** Source Keeper NPC (not player/invader). */
export function isSourceKeeperCreep(creep: Creep): boolean {
	if (!creep || !creep.owner) return false;
	const u = creep.owner.username;
	if (u === 'Source Keeper' || u === 'SourceKeeper') return true;
	if (creep.name && String(creep.name).indexOf('Keeper') === 0) return true;
	return false;
}

export function militaryParts(creep: Creep): number {
	let n = 0;
	for (const p of creep.body) {
		if (p.hits <= 0) continue;
		if (p.type === ATTACK || p.type === RANGED_ATTACK || p.type === HEAL) n += 1;
		else if (p.type === WORK) n += 0.25;
	}
	return n;
}

export function hostileThreat(room: Room): number {
	let t = 0;
	for (const c of room.find(FIND_HOSTILE_CREEPS)) t += militaryParts(c);
	return t;
}

/**
 * Will this creep still be productive after we account for a replacement?
 * @param spawnEta ticks to spawn + travel a replacement (bodyParts*3 + pathLen).
 * Built-in 80-tick slack so we re-queue before death and avoid understaffed sources.
 */
export function stillAlive(creep: Creep, spawnEta = 0): boolean {
	if (creep.ticksToLive === undefined) return true; // still spawning
	return creep.ticksToLive > 80 + spawnEta;
}

/**
 * Horizon for counting a creep toward staffing: if TTL ≤ this, treat as already gone
 * so spawn re-queues the replacement while the old one still works.
 */
export function replaceHorizon(bodyParts: number, pathLen = 0): number {
	const spawnT = (typeof CREEP_SPAWN_TIME !== 'undefined' ? CREEP_SPAWN_TIME : 3);
	return Math.max(0, bodyParts) * spawnT + Math.max(0, pathLen) + 10;
}

export function spawnTimeForBody(body: BodyPartConstant[] | BodyPartDefinition[]): number {
	return body.length * CREEP_SPAWN_TIME;
}

export function estimatePathLength(fromPos: RoomPosition, toPos: RoomPosition): number {
	if (fromPos.roomName === toPos.roomName) {
		const path = fromPos.findPathTo(toPos, { ignoreCreeps: true, maxOps: 2000 });
		return path.length || fromPos.getRangeTo(toPos);
	}
	const roomDist = Game.map.getRoomLinearDistance(fromPos.roomName, toPos.roomName);
	return roomDist * 50 + 20;
}

// Movement: traffic stack (cached ignoreCreeps paths + end-tick resolve)
export function moveToRoom(creep: Creep, roomName: string): ScreepsReturnCode {
	return trafficMoveToRoom(creep, roomName);
}

export function goDo(
	creep: Creep,
	target: RoomObject | RoomPosition,
	range: number,
	action: () => ScreepsReturnCode,
): ScreepsReturnCode {
	return trafficGoDo(creep, target, range, action);
}

export function goTo(
	creep: Creep,
	target: RoomPosition | { pos: RoomPosition },
	opts?: { range?: number; preferRoads?: boolean },
): ScreepsReturnCode {
	return trafficGoTo(creep, target, opts);
}

export type Pickup =
	| { type: 'pickup'; target: Resource }
	| { type: 'withdraw'; target: AnyStoreStructure };

export function nearestEnergyPickup(
	pos: RoomPosition,
	range = 50,
	opts: { minAmount?: number; links?: boolean } = {},
): Pickup | null {
	const room = Game.rooms[pos.roomName];
	if (!room) return null;
	const minAmount = opts.minAmount != null ? opts.minAmount : 20;
	let best: Pickup | null = null;
	let bestScore = Infinity;

	const drops = pos.findInRange(FIND_DROPPED_RESOURCES, Math.min(range, 50), {
		filter: r => r.resourceType === RESOURCE_ENERGY && r.amount >= minAmount,
	});
	for (const d of drops) {
		const score = pos.getRangeTo(d) - d.amount / 1000;
		if (score < bestScore) {
			bestScore = score;
			best = { type: 'pickup', target: d };
		}
	}

	const structs = room.find(FIND_STRUCTURES, {
		filter: s => {
			const st = s as AnyStoreStructure;
			if (!('store' in st) || !st.store) return false;
			if (energyOf(st.store) < minAmount) return false;
			const t = st.structureType;
			if (t === STRUCTURE_CONTAINER || t === STRUCTURE_STORAGE || t === STRUCTURE_TERMINAL) return true;
			if (t === STRUCTURE_LINK && opts.links) return true;
			return false;
		},
	}) as AnyStoreStructure[];

	for (const s of structs) {
		const r = pos.getRangeTo(s);
		if (r > range) continue;
		const score = r - energyOf(s.store) / 5000;
		if (score < bestScore) {
			bestScore = score;
			best = { type: 'withdraw', target: s };
		}
	}
	return best;
}

export function doPickup(creep: Creep, pickup: Pickup): ScreepsReturnCode {
	if (pickup.type === 'pickup') {
		return goDo(creep, pickup.target, 1, () => creep.pickup(pickup.target));
	}
	return goDo(creep, pickup.target, 1, () => creep.withdraw(pickup.target, RESOURCE_ENERGY));
}

export function lowBucket(): boolean {
	return Game.cpu.bucket < config.lowBucketThreshold;
}

export function creepsFor(filter: (c: Creep) => boolean): Creep[] {
	const out: Creep[] = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name]!;
		if (filter(c)) out.push(c);
	}
	return out;
}

export function posFromStored(roomName: string, c: Coord): RoomPosition {
	return asPos(c, roomName);
}

export { BODY_COST };
