/**
 * Construction planner — compact spawn-centric layout + infrastructure on demand.
 *
 * Layout philosophy (bunker-lite):
 *  - Extensions fill a checkerboard diamond around the first spawn (roads on other tiles).
 *  - Towers near center for coverage.
 *  - Storage / terminal / link cluster near spawn.
 *  - Containers next to sources and controller.
 *  - Roads along spawn→source and spawn→controller paths.
 */
const config = require('config');
const { energyOf } = require('util');

function maxOf(type, rcl) {
	const table = CONTROLLER_STRUCTURES[type];
	if (!table) return 0;
	return table[rcl] || 0;
}

function countStructures(room, type) {
	return room.find(FIND_STRUCTURES, { filter: s => s.structureType === type }).length +
		room.find(FIND_MY_CONSTRUCTION_SITES, { filter: s => s.structureType === type }).length;
}

function canBuild(room, type) {
	const rcl = room.controller ? room.controller.level : 0;
	return countStructures(room, type) < maxOf(type, rcl);
}

function trySite(room, x, y, type) {
	if (x < 1 || x > 48 || y < 1 || y > 48) return false;
	const pos = new RoomPosition(x, y, room.name);
	const terrain = room.getTerrain();
	if (terrain.get(x, y) === TERRAIN_MASK_WALL) return false;
	// Don't pile sites on existing structures (except roads under other things — skip).
	const look = pos.look();
	for (const o of look) {
		if (o.type === LOOK_STRUCTURES) {
			const st = o.structure.structureType;
			if (st === STRUCTURE_ROAD && type !== STRUCTURE_ROAD) continue;
			if (st === STRUCTURE_RAMPART) continue;
			return false;
		}
		if (o.type === LOOK_CONSTRUCTION_SITES) return false;
	}
	const r = room.createConstructionSite(x, y, type);
	return r === OK;
}

/** Spiral / ring offsets around center, ordered by Chebyshev distance. */
function ringOffsets(maxR) {
	const cells = [];
	for (let r = 1; r <= maxR; r++) {
		for (let dx = -r; dx <= r; dx++) {
			for (let dy = -r; dy <= r; dy++) {
				if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
				cells.push({ dx, dy, r });
			}
		}
	}
	return cells;
}

function planExtensions(room, spawn) {
	if (!canBuild(room, STRUCTURE_EXTENSION)) return;
	const rcl = room.controller.level;
	const need = maxOf(STRUCTURE_EXTENSION, rcl) - countStructures(room, STRUCTURE_EXTENSION);
	if (need <= 0) return;

	let placed = 0;
	for (const { dx, dy } of ringOffsets(6)) {
		if (placed >= need) break;
		// Checkerboard: extensions on tiles where (x+y) matches spawn parity + 1 for spacing with roads.
		const x = spawn.pos.x + dx;
		const y = spawn.pos.y + dy;
		if ((x + y) % 2 === (spawn.pos.x + spawn.pos.y) % 2) {
			// Road lattice tile — skip for extensions.
			if (trySite(room, x, y, STRUCTURE_ROAD)) {
				/* road ok */
			}
			continue;
		}
		if (trySite(room, x, y, STRUCTURE_EXTENSION)) placed++;
	}
}

function planTowers(room, spawn) {
	if (!canBuild(room, STRUCTURE_TOWER)) return;
	const need = maxOf(STRUCTURE_TOWER, room.controller.level) - countStructures(room, STRUCTURE_TOWER);
	let placed = 0;
	// Prefer positions ~3 tiles from spawn for coverage.
	const candidates = [
		[ 2, 0 ], [ -2, 0 ], [ 0, 2 ], [ 0, -2 ],
		[ 2, 2 ], [ -2, 2 ], [ 2, -2 ], [ -2, -2 ],
		[ 3, 1 ], [ -3, 1 ], [ 1, 3 ], [ 1, -3 ],
	];
	for (const [ dx, dy ] of candidates) {
		if (placed >= need) break;
		if (trySite(room, spawn.pos.x + dx, spawn.pos.y + dy, STRUCTURE_TOWER)) placed++;
	}
}

function planStorageCluster(room, spawn) {
	const rcl = room.controller.level;
	const spots = [
		[ 0, -2 ], [ 0, 2 ], [ -2, 0 ], [ 2, 0 ],
		[ -1, -2 ], [ 1, -2 ], [ -2, -1 ], [ -2, 1 ],
	];
	const order = [];
	if (rcl >= 4 && canBuild(room, STRUCTURE_STORAGE)) order.push(STRUCTURE_STORAGE);
	if (rcl >= 6 && canBuild(room, STRUCTURE_TERMINAL)) order.push(STRUCTURE_TERMINAL);
	if (rcl >= 5 && canBuild(room, STRUCTURE_LINK)) order.push(STRUCTURE_LINK);

	let si = 0;
	for (const type of order) {
		while (si < spots.length) {
			const [ dx, dy ] = spots[si++];
			if (trySite(room, spawn.pos.x + dx, spawn.pos.y + dy, type)) break;
		}
	}
}

function planExtractor(room) {
	if (!canBuild(room, STRUCTURE_EXTRACTOR)) return;
	const mineral = room.find(FIND_MINERALS)[0];
	if (!mineral) return;
	trySite(room, mineral.pos.x, mineral.pos.y, STRUCTURE_EXTRACTOR);
}

function nearestOpenNear(pos, range = 1) {
	const terrain = Game.map.getRoomTerrain(pos.roomName);
	const room = Game.rooms[pos.roomName];
	if (!room) return null;
	let best = null;
	let bestScore = Infinity;
	for (let dx = -range; dx <= range; dx++) {
		for (let dy = -range; dy <= range; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = pos.x + dx;
			const y = pos.y + dy;
			if (x < 1 || x > 48 || y < 1 || y > 48) continue;
			if (terrain.get(x, y) === TERRAIN_MASK_WALL) continue;
			const p = new RoomPosition(x, y, pos.roomName);
			const blocked = p.lookFor(LOOK_STRUCTURES).some(s =>
				s.structureType !== STRUCTURE_ROAD && s.structureType !== STRUCTURE_RAMPART &&
				s.structureType !== STRUCTURE_CONTAINER);
			if (blocked) continue;
			const score = Math.abs(dx) + Math.abs(dy);
			if (score < bestScore) {
				bestScore = score;
				best = p;
			}
		}
	}
	return best;
}

function planContainers(room) {
	if (room.controller.level < config.containerRcl) return;
	// Source containers.
	for (const source of room.find(FIND_SOURCES)) {
		const near = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		});
		const sites = source.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		});
		if (near.length || sites.length) continue;
		const spot = nearestOpenNear(source.pos, 1);
		if (spot) trySite(room, spot.x, spot.y, STRUCTURE_CONTAINER);
	}
	// Controller container.
	const ctrl = room.controller;
	if (ctrl) {
		const near = ctrl.pos.findInRange(FIND_STRUCTURES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_LINK,
		});
		const sites = ctrl.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_LINK,
		});
		if (!near.length && !sites.length) {
			const spot = nearestOpenNear(ctrl.pos, 2);
			if (spot) trySite(room, spot.x, spot.y, STRUCTURE_CONTAINER);
		}
	}
}

function planSourceLinks(room) {
	if (room.controller.level < config.linkRcl) return;
	if (!canBuild(room, STRUCTURE_LINK)) return;
	for (const source of room.find(FIND_SOURCES)) {
		const links = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK,
		});
		const sites = source.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK,
		});
		if (links.length || sites.length) continue;
		const spot = nearestOpenNear(source.pos, 2);
		if (spot && trySite(room, spot.x, spot.y, STRUCTURE_LINK)) {
			// one per tick budget
			return;
		}
	}
}

function planRoads(room, spawn) {
	// Roads are a mid-game luxury — early RCL needs energy on upgrades/creeps first.
	if (room.controller.level < 2) return;
	const sites = room.find(FIND_MY_CONSTRUCTION_SITES);
	if (sites.length > 8) return;
	// Cap concurrent road sites.
	const roadSites = sites.filter(s => s.structureType === STRUCTURE_ROAD);
	if (roadSites.length > 10) return;

	const goals = [
		...room.find(FIND_SOURCES).map(s => s.pos),
		room.controller && room.controller.pos,
		room.storage && room.storage.pos,
	].filter(Boolean);

	let budget = 5;
	for (const goal of goals) {
		if (budget <= 0) break;
		const path = spawn.pos.findPathTo(goal, {
			ignoreCreeps: true,
			maxOps: 4000,
			ignoreRoads: false,
		});
		for (const step of path) {
			if (budget <= 0) break;
			// Skip last step on the goal structure.
			if (step.x === goal.x && step.y === goal.y) continue;
			const pos = new RoomPosition(step.x, step.y, room.name);
			const hasRoad = pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_ROAD);
			const hasSite = pos.lookFor(LOOK_CONSTRUCTION_SITES).length > 0;
			if (!hasRoad && !hasSite) {
				if (trySite(room, step.x, step.y, STRUCTURE_ROAD)) budget--;
			}
		}
	}
}

function planRamparts(room, spawn) {
	// Light shell around spawn/storage at RCL 4+.
	if (room.controller.level < 4) return;
	if (energyOf(room.storage && room.storage.store) < 20_000 && room.controller.level < 6) return;

	const cores = [ spawn ];
	if (room.storage) cores.push(room.storage);
	if (room.terminal) cores.push(room.terminal);
	for (const tower of room.find(FIND_MY_STRUCTURES, { filter: s => s.structureType === STRUCTURE_TOWER })) {
		cores.push(tower);
	}

	let budget = 3;
	for (const core of cores) {
		for (let dx = -1; dx <= 1; dx++) {
			for (let dy = -1; dy <= 1; dy++) {
				if (budget <= 0) return;
				const x = core.pos.x + dx;
				const y = core.pos.y + dy;
				const pos = new RoomPosition(x, y, room.name);
				const has = pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_RAMPART);
				const site = pos.lookFor(LOOK_CONSTRUCTION_SITES).some(s => s.structureType === STRUCTURE_RAMPART);
				if (!has && !site && trySite(room, x, y, STRUCTURE_RAMPART)) budget--;
			}
		}
	}
}

/**
 * Run planner for a colony. Limits sites created per tick.
 */
function run(room) {
	if (!room.controller || !room.controller.my) return;
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (!spawn) return;

	// Global site budget — avoid pathing chaos.
	const totalSites = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	if (totalSites > 25) return;

	// Only plan every few ticks per room.
	const mem = room.memory;
	if (mem._planAt && Game.time - mem._planAt < 10) return;
	mem._planAt = Game.time;

	planExtensions(room, spawn);
	planTowers(room, spawn);
	planStorageCluster(room, spawn);
	planContainers(room);
	planSourceLinks(room);
	planExtractor(room);
	planRoads(room, spawn);
	planRamparts(room, spawn);
}

/**
 * Remote infrastructure: container next to each source when visible.
 */
function runRemote(room) {
	if (!room) return;
	for (const source of room.find(FIND_SOURCES)) {
		const near = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		});
		const sites = source.pos.findInRange(FIND_CONSTRUCTION_SITES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		});
		if (near.length || sites.length) continue;
		const spot = nearestOpenNear(source.pos, 1);
		if (spot) room.createConstructionSite(spot.x, spot.y, STRUCTURE_CONTAINER);
	}
}

module.exports = { run, runRemote, canBuild, maxOf, countStructures };
