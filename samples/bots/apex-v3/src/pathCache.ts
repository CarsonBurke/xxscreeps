/**
 * Shared path cache — ignore creeps, terrain + structures only.
 * Routes keyed by origin/goal/options; reused until TTL or structure version bump.
 */
import { packXY } from './codec';

export interface PathOpts {
	range?: number;
	preferRoads?: boolean;
	maxRooms?: number;
	maxOps?: number;
	/** Cache TTL ticks (default 40) */
	ttl?: number;
}

interface CacheEntry {
	/** Same-room steps as packed x+y*50; multi-room as "x,y,room" strings */
	steps: string[];
	tick: number;
	hits: number;
}

const cache = new Map<string, CacheEntry>();
/** Bump when we place/destroy structures that block routes (optional). */
let structureEpoch = 0;

export function bumpStructureEpoch(): void {
	structureEpoch++;
	cache.clear();
}

function keyOf(from: RoomPosition, to: RoomPosition, opts: PathOpts): string {
	const pr = opts.preferRoads !== false ? 1 : 0;
	const r = opts.range != null ? opts.range : 1;
	return `${from.roomName}|${packXY(from.x, from.y)}>${to.roomName}|${packXY(to.x, to.y)}|r${r}|p${pr}|e${structureEpoch}`;
}

function stepKey(pos: RoomPosition): string {
	return `${pos.x},${pos.y},${pos.roomName}`;
}

function parseStep(s: string): RoomPosition {
	const [xs, ys, room] = s.split(',');
	return new RoomPosition(Number(xs), Number(ys), room!);
}

const OBSTACLE: Set<string> = new Set([
	STRUCTURE_SPAWN, STRUCTURE_EXTENSION, STRUCTURE_WALL, STRUCTURE_LINK,
	STRUCTURE_STORAGE, STRUCTURE_TOWER, STRUCTURE_OBSERVER, STRUCTURE_POWER_SPAWN,
	STRUCTURE_LAB, STRUCTURE_TERMINAL, STRUCTURE_NUKER, STRUCTURE_FACTORY,
	STRUCTURE_CONTROLLER,
	STRUCTURE_INVADER_CORE as string,
]);

function roomCallback(preferRoads: boolean): (roomName: string) => CostMatrix | boolean {
	return (roomName: string) => {
		const room = Game.rooms[roomName];
		const cm = new PathFinder.CostMatrix();
		if (!room) return cm;
		const structs = room.find(FIND_STRUCTURES);
		for (let i = 0; i < structs.length; i++) {
			const s = structs[i]!;
			if (s.structureType === STRUCTURE_ROAD) {
				if (preferRoads) cm.set(s.pos.x, s.pos.y, 1);
				continue;
			}
			if (s.structureType === STRUCTURE_CONTAINER) continue;
			if (s.structureType === STRUCTURE_RAMPART) {
				// Only own ramparts are walkable
				if ((s as StructureRampart).my) continue;
				cm.set(s.pos.x, s.pos.y, 255);
				continue;
			}
			if (OBSTACLE.has(s.structureType) || !s.hits) {
				cm.set(s.pos.x, s.pos.y, 255);
			}
		}
		// Construction sites that block
		const sites = room.find(FIND_MY_CONSTRUCTION_SITES);
		for (let i = 0; i < sites.length; i++) {
			const site = sites[i]!;
			if (OBSTACLE.has(site.structureType)) {
				cm.set(site.pos.x, site.pos.y, 255);
			}
		}
		return cm;
	};
}

/** PathFinder with ignoreCreeps (no creep costs in matrix). */
export function findPathIgnoreCreeps(
	from: RoomPosition,
	to: RoomPosition,
	opts: PathOpts = {},
): RoomPosition[] {
	const range = opts.range != null ? opts.range : 1;
	const preferRoads = opts.preferRoads !== false;
	const res = PathFinder.search(
		from,
		{ pos: to, range },
		{
			plainCost: preferRoads ? 2 : 1,
			swampCost: preferRoads ? 10 : 5,
			roomCallback: roomCallback(preferRoads),
			maxOps: opts.maxOps || 4000,
			maxRooms: opts.maxRooms || 16,
		},
	);
	if (res.incomplete && res.path.length === 0) return [];
	return res.path;
}

/**
 * Cached path from → to. Returns array of positions (next steps after `from`).
 */
export function getCachedPath(
	from: RoomPosition,
	to: RoomPosition,
	opts: PathOpts = {},
): RoomPosition[] {
	const ttl = opts.ttl != null ? opts.ttl : 40;
	const k = keyOf(from, to, opts);
	const hit = cache.get(k);
	if (hit && Game.time - hit.tick < ttl && hit.steps.length) {
		hit.hits++;
		return hit.steps.map(parseStep);
	}
	const path = findPathIgnoreCreeps(from, to, opts);
	const steps = path.map(stepKey);
	cache.set(k, { steps, tick: Game.time, hits: 1 });
	// Soft cap cache size
	if (cache.size > 200) {
		const first = cache.keys().next().value;
		if (first != null) cache.delete(first);
	}
	return path;
}

export function pathCacheStats(): { size: number; epoch: number } {
	return { size: cache.size, epoch: structureEpoch };
}

export function clearPathCache(): void {
	cache.clear();
}
