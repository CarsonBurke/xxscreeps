/**
 * Traffic manager: CostMatrix caching, road preference, parking seats,
 * and role-tuned moveTo wrappers for Apex v2.
 */
const config = require('config');
const { safeGet, directionTo } = require('util');

// Per-tick in-memory matrices (isolate-local).
const matrixCache = {};
const matrixTick = {};

const BLOCKING = new Set([
	STRUCTURE_SPAWN, STRUCTURE_EXTENSION, STRUCTURE_TOWER, STRUCTURE_STORAGE,
	STRUCTURE_TERMINAL, STRUCTURE_LINK, STRUCTURE_LAB, STRUCTURE_WALL,
	STRUCTURE_OBSERVER, STRUCTURE_POWER_SPAWN, STRUCTURE_NUKER, STRUCTURE_FACTORY,
	STRUCTURE_EXTRACTOR,
]);

/**
 * Build a CostMatrix with road preference + structure blockers only.
 * Terrain stays on PathFinder plainCost/swampCost — no 50×50 terrain fill.
 * Rebuilt every config.costMatrixRefresh ticks per room.
 */
function getCostMatrix(roomName) {
	const room = Game.rooms[roomName];
	if (!room) return new PathFinder.CostMatrix();

	const refresh = config.costMatrixRefresh || 25;
	if (matrixCache[roomName] && matrixTick[roomName] != null &&
		Game.time - matrixTick[roomName] < refresh) {
		return matrixCache[roomName];
	}

	const matrix = new PathFinder.CostMatrix();
	const road = config.roadCost || 1;
	// Packed [x,y,cost, ...] for fast apply into moveTo's working matrix.
	const ops = [];

	for (const s of room.find(FIND_STRUCTURES)) {
		const x = s.pos.x;
		const y = s.pos.y;
		let cost = -1;
		if (s.structureType === STRUCTURE_ROAD) {
			cost = road;
		} else if (s.structureType === STRUCTURE_CONTAINER) {
			// Mildly higher than road so paths prefer actual roads.
			if (matrix.get(x, y) === 0) cost = road + 1;
		} else if (s.structureType === STRUCTURE_RAMPART) {
			if (!s.my && !s.isPublic) cost = 255;
		} else if (BLOCKING.has(s.structureType)) {
			cost = 255;
		}
		if (cost >= 0) {
			matrix.set(x, y, cost);
			ops.push(x, y, cost);
		}
	}

	matrix._apexOps = ops;
	matrixCache[roomName] = matrix;
	matrixTick[roomName] = Game.time;
	return matrix;
}

/** Apply cached matrix onto moveTo's working costMatrix (only non-zero cells). */
function applyMatrix(cached, costMatrix) {
	const ops = cached && cached._apexOps;
	if (!ops) return;
	for (let i = 0; i < ops.length; i += 3) {
		costMatrix.set(ops[i], ops[i + 1], ops[i + 2]);
	}
}

function reuseFor(creep) {
	const role = creep.memory && creep.memory.role;
	const table = config.pathReuseByRole || {};
	return table[role] != null ? table[role] : (table.default || 15);
}

/**
 * Move creep toward target with CostMatrix + role reusePath.
 */
function moveCreep(creep, target, opts = {}) {
	if (!target) return ERR_INVALID_TARGET;
	const range = opts.range != null ? opts.range : 0;
	const pos = target.pos || target;
	if (creep.pos.inRangeTo(pos, range)) return OK;

	const reusePath = opts.reusePath != null ? opts.reusePath : reuseFor(creep);
	const far = creep.pos.getRangeTo(pos) > 3;
	const ignoreCreeps = opts.ignoreCreeps != null ? opts.ignoreCreeps : far;

	// Prefer native moveTo with costCallback for road preference.
	return creep.moveTo(pos, {
		reusePath,
		ignoreCreeps,
		maxRooms: opts.maxRooms != null ? opts.maxRooms : 16,
		range,
		plainCost: config.plainCost || 2,
		swampCost: config.swampCost || 10,
		costCallback(roomName, costMatrix) {
			applyMatrix(getCostMatrix(roomName), costMatrix);
			if (typeof opts.costCallback === 'function') {
				opts.costCallback(roomName, costMatrix);
			}
		},
		visualizePathStyle: Memory._apexVisuals && !require('util').criticalBucket()
			? { stroke: opts.stroke || '#88f', opacity: 0.25, lineStyle: 'dashed' }
			: undefined,
	});
}

/**
 * Go to target and perform action when in range.
 */
function goDo(creep, target, range, action, opts = {}) {
	if (!target) return ERR_INVALID_TARGET;
	if (creep.pos.inRangeTo(target, range)) {
		return action();
	}
	return moveCreep(creep, target, { ...opts, range });
}

/**
 * Open terrain tiles adjacent to a position (for parking seats).
 */
function openSpotsNear(pos, range = 1) {
	const room = Game.rooms[pos.roomName];
	if (!room) return [];
	const terrain = room.getTerrain();
	const spots = [];
	for (let dx = -range; dx <= range; dx++) {
		for (let dy = -range; dy <= range; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = pos.x + dx;
			const y = pos.y + dy;
			if (x < 1 || x > 48 || y < 1 || y > 48) continue;
			if (terrain.get(x, y) === TERRAIN_MASK_WALL) continue;
			const p = new RoomPosition(x, y, pos.roomName);
			const blocked = p.lookFor(LOOK_STRUCTURES).some(s =>
				s.structureType !== STRUCTURE_ROAD &&
				s.structureType !== STRUCTURE_RAMPART &&
				s.structureType !== STRUCTURE_CONTAINER);
			if (blocked) continue;
			spots.push(p);
		}
	}
	return spots;
}

/**
 * Miner seat: prefer container tile, else best open spot on source.
 * Reserves one adjacent spot for hauler parking when config.reserveHaulerSpot.
 */
function minerSeat(source) {
	if (!source) return null;
	const containers = source.pos.findInRange(FIND_STRUCTURES, 1, {
		filter: s => s.structureType === STRUCTURE_CONTAINER,
	});
	if (containers.length) return containers[0].pos;

	const spots = openSpotsNear(source.pos, 1);
	if (!spots.length) return source.pos;
	// Prefer spots not reserved as hauler parks (heuristic: more open neighbors).
	spots.sort((a, b) => openSpotsNear(a, 1).length - openSpotsNear(b, 1).length);
	// With reserveHaulerSpot, pick a seat that leaves an adjacent free tile.
	if (config.reserveHaulerSpot && spots.length > 1) {
		return spots[spots.length - 1];
	}
	return spots[0];
}

/**
 * Hauler parking position near source — not on miner seat.
 */
function haulerPark(source, minerPos) {
	if (!source) return null;
	const seat = minerPos || minerSeat(source);
	const spots = openSpotsNear(source.pos, config.haulerParkRange || 1);
	// Prefer container if free and not miner seat.
	const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
		filter: s => s.structureType === STRUCTURE_CONTAINER,
	})[0];
	if (container && (!seat || !container.pos.isEqualTo(seat))) {
		return container.pos;
	}
	const free = spots.filter(p => !seat || !p.isEqualTo(seat));
	if (!free.length) return spots[0] || null;
	// Prefer closest to seat for transfer range, but not on it.
	if (seat) free.sort((a, b) => a.getRangeTo(seat) - b.getRangeTo(seat));
	return free[0];
}

/**
 * Park creep at a position (idle without blocking key tiles).
 */
function parkAt(creep, pos) {
	if (!pos) return ERR_INVALID_TARGET;
	if (creep.pos.isEqualTo(pos)) return OK;
	return moveCreep(creep, pos, { range: 0, reusePath: 30, ignoreCreeps: false });
}

/**
 * Move toward another room using map route + cached costs.
 */
function moveToRoom(creep, roomName) {
	if (creep.room.name === roomName) return OK;
	const exit = creep.room.findExitTo(roomName);
	if (exit === ERR_NO_PATH || exit === ERR_INVALID_ARGS) {
		const { isHighway, isSourceKeeperRoom } = require('util');
		const route = Game.map.findRoute(creep.room.name, roomName, {
			routeCallback(room) {
				if (isHighway(room)) return 1.5;
				if (isSourceKeeperRoom(room)) return 4;
				const mem = Memory.intel && Memory.intel[room];
				if (mem && mem.owner && mem.owner !== 'mine' && mem.threat > 0) return 10;
				return 1;
			},
		});
		if (route === ERR_NO_PATH || !route.length) return ERR_NO_PATH;
		const step = route[0];
		const ex = creep.pos.findClosestByPath(step.exit);
		if (ex) return moveCreep(creep, ex, { maxRooms: 1, reusePath: 30 });
		return ERR_NO_PATH;
	}
	const pos = creep.pos.findClosestByPath(exit);
	if (!pos) return ERR_NO_PATH;
	return moveCreep(creep, pos, { maxRooms: 1, reusePath: 20 });
}

/**
 * Record path heat for road planning (tile visits by haulers/miners).
 */
function recordHeat(creep) {
	if (!creep || !creep.room) return;
	const role = creep.memory && creep.memory.role;
	if (!role) return;
	if (role !== 'hauler' && role !== 'remoteHauler' && role !== 'miner' &&
		role !== 'remoteMiner' && role !== 'bootstrap') return;

	const room = creep.room;
	room.memory.apex ||= {};
	room.memory.apex.heat ||= {};
	const key = `${creep.pos.x},${creep.pos.y}`;
	room.memory.apex.heat[key] = (room.memory.apex.heat[key] || 0) + 1;
}

/**
 * Decay path heat periodically.
 */
function decayHeat(room) {
	const interval = config.roadHeatDecayInterval || 200;
	room.memory.apex ||= {};
	if (room.memory.apex.heatAt && Game.time - room.memory.apex.heatAt < interval) return;
	room.memory.apex.heatAt = Game.time;
	const heat = room.memory.apex.heat;
	if (!heat) return;
	for (const k of Object.keys(heat)) {
		heat[k] = Math.floor(heat[k] * 0.5);
		if (heat[k] <= 0) delete heat[k];
	}
}

function harvestEnergy(creep) {
	const room = creep.room;
	const { energyOf, isEnergySourceStructure, byRange } = require('util');

	const drop = creep.pos.findClosestByPath(FIND_DROPPED_RESOURCES, {
		filter: r => r.resourceType === RESOURCE_ENERGY && r.amount > 20,
	});
	if (drop) {
		if (creep.pos.isNearTo(drop)) return creep.pickup(drop);
		return moveCreep(creep, drop);
	}

	const tomb = creep.pos.findClosestByPath(FIND_TOMBSTONES, {
		filter: t => energyOf(t.store) > 20,
	});
	if (tomb) {
		if (creep.pos.isNearTo(tomb)) return creep.withdraw(tomb, RESOURCE_ENERGY);
		return moveCreep(creep, tomb);
	}

	// Link-first: storage/controller-adjacent links are sinks; source links supply.
	if (config.linkFirst) {
		const sourceLinks = room.find(FIND_MY_STRUCTURES, {
			filter: s => s.structureType === STRUCTURE_LINK &&
				isEnergySourceStructure(s, room) && energyOf(s.store) > 50,
		});
		if (sourceLinks.length) {
			sourceLinks.sort(byRange(creep.pos));
			const link = sourceLinks[0];
			if (creep.pos.isNearTo(link)) return creep.withdraw(link, RESOURCE_ENERGY);
			return moveCreep(creep, link);
		}
	}

	const structs = room.find(FIND_STRUCTURES, {
		filter: s => isEnergySourceStructure(s, room) && energyOf(s.store) > 50,
	});
	if (structs.length) {
		structs.sort((a, b) => energyOf(b.store) - energyOf(a.store));
		const s = structs[0];
		if (creep.pos.isNearTo(s)) return creep.withdraw(s, RESOURCE_ENERGY);
		return moveCreep(creep, s);
	}

	const source = creep.pos.findClosestByPath(FIND_SOURCES_ACTIVE);
	if (source) {
		if (creep.pos.isNearTo(source)) return creep.harvest(source);
		return moveCreep(creep, source);
	}
	return ERR_NOT_FOUND;
}

function deliverEnergy(creep, opts = {}) {
	const room = creep.room;
	const { energyOf, isFillTarget, fillPriority } = require('util');

	// Link-first delivery: dump into storage link if present and hungry.
	if (config.linkFirst && room.storage) {
		const storageLink = room.storage.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK &&
				require('util').freeCapacity(s.store, RESOURCE_ENERGY) > 50,
		})[0];
		if (storageLink && creep.pos.getRangeTo(storageLink) <= creep.pos.getRangeTo(room.storage)) {
			// Prefer filling spawn network first.
		}
	}

	const targets = room.find(FIND_MY_STRUCTURES, {
		filter: s => isFillTarget(s, room),
	});
	targets.sort((a, b) => {
		const pa = fillPriority(a);
		const pb = fillPriority(b);
		if (pa !== pb) return pa - pb;
		return energyOf(a.store) - energyOf(b.store);
	});

	if (targets.length) {
		const t = creep.pos.findClosestByPath(targets.slice(0, 8)) || targets[0];
		return goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
	}

	if ((opts.includeController || opts.preferUpgrade) && room.controller && room.controller.my) {
		return goDo(creep, room.controller, 3, () => creep.upgradeController(room.controller));
	}

	const idle = room.storage || room.find(FIND_MY_SPAWNS)[0];
	if (idle && !creep.pos.inRangeTo(idle, 2)) moveCreep(creep, idle, { range: 2, reusePath: 30 });
	return ERR_NOT_FOUND;
}

module.exports = {
	getCostMatrix,
	moveCreep,
	goDo,
	moveToRoom,
	openSpotsNear,
	minerSeat,
	haulerPark,
	parkAt,
	recordHeat,
	decayHeat,
	harvestEnergy,
	deliverEnergy,
	reuseFor,
	safeGet,
	directionTo,
};
