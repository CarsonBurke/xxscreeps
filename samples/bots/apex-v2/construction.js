/**
 * Construction planner v2:
 *  - RCL-gated planner with site budget
 *  - Complete existing sites of a type before placing more of that type
 *  - Fewer road sites via path heat (high-traffic tiles only)
 *  - Extensions / towers / storage cluster / containers / links / ramparts
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

function countSites(room, type) {
	return room.find(FIND_MY_CONSTRUCTION_SITES, {
		filter: s => !type || s.structureType === type,
	}).length;
}

function canBuild(room, type) {
	const rcl = room.controller ? room.controller.level : 0;
	return countStructures(room, type) < maxOf(type, rcl);
}

/**
 * Site budget: refuse new sites if over global budget, or if same type still has unfinished sites
 * when finishSitesBeforeMore is set (except roads which use their own cap).
 */
function allowNewSite(room, type, passBudget) {
	if (passBudget && passBudget.left <= 0) return false;
	const totalBudget = config.siteBudget || 18;
	if (countSites(room) >= totalBudget) return false;

	if (config.finishSitesBeforeMore && type !== STRUCTURE_ROAD) {
		const existing = countSites(room, type);
		// Allow at most a few concurrent of same type; prefer finishing first.
		if (existing >= 3) return false;
	}
	if (type === STRUCTURE_ROAD) {
		if (countSites(room, STRUCTURE_ROAD) >= (config.maxRoadSites || 8)) return false;
	}
	return true;
}

function trySite(room, x, y, type, passBudget) {
	if (!allowNewSite(room, type, passBudget)) return false;
	if (x < 1 || x > 48 || y < 1 || y > 48) return false;
	const pos = new RoomPosition(x, y, room.name);
	const terrain = room.getTerrain();
	if (terrain.get(x, y) === TERRAIN_MASK_WALL) return false;
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
	if (r === OK) {
		if (passBudget) passBudget.left--;
		return true;
	}
	return false;
}

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

function planExtensions(room, spawn, budget) {
	if (!canBuild(room, STRUCTURE_EXTENSION)) return;
	// Finish existing extension sites first.
	if (config.finishSitesBeforeMore && countSites(room, STRUCTURE_EXTENSION) > 0) return;

	const rcl = room.controller.level;
	const need = maxOf(STRUCTURE_EXTENSION, rcl) - countStructures(room, STRUCTURE_EXTENSION);
	if (need <= 0) return;

	let placed = 0;
	for (const { dx, dy } of ringOffsets(6)) {
		if (placed >= need || budget.left <= 0) break;
		const x = spawn.pos.x + dx;
		const y = spawn.pos.y + dy;
		if ((x + y) % 2 === (spawn.pos.x + spawn.pos.y) % 2) {
			// Road lattice — only place road if heat warrants later; skip here.
			continue;
		}
		if (trySite(room, x, y, STRUCTURE_EXTENSION, budget)) placed++;
	}
}

function planTowers(room, spawn, budget) {
	if (!canBuild(room, STRUCTURE_TOWER)) return;
	if (config.finishSitesBeforeMore && countSites(room, STRUCTURE_TOWER) > 0) return;
	const need = maxOf(STRUCTURE_TOWER, room.controller.level) - countStructures(room, STRUCTURE_TOWER);
	let placed = 0;
	const candidates = [
		[ 2, 0 ], [ -2, 0 ], [ 0, 2 ], [ 0, -2 ],
		[ 2, 2 ], [ -2, 2 ], [ 2, -2 ], [ -2, -2 ],
		[ 3, 1 ], [ -3, 1 ], [ 1, 3 ], [ 1, -3 ],
	];
	for (const [ dx, dy ] of candidates) {
		if (placed >= need || budget.left <= 0) break;
		if (trySite(room, spawn.pos.x + dx, spawn.pos.y + dy, STRUCTURE_TOWER, budget)) placed++;
	}
}

function planStorageCluster(room, spawn, budget) {
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
		if (config.finishSitesBeforeMore && countSites(room, type) > 0) continue;
		while (si < spots.length && budget.left > 0) {
			const [ dx, dy ] = spots[si++];
			if (trySite(room, spawn.pos.x + dx, spawn.pos.y + dy, type, budget)) break;
		}
	}
}

function planExtractor(room, budget) {
	if (!canBuild(room, STRUCTURE_EXTRACTOR)) return;
	if (config.finishSitesBeforeMore && countSites(room, STRUCTURE_EXTRACTOR) > 0) return;
	const mineral = room.find(FIND_MINERALS)[0];
	if (!mineral) return;
	trySite(room, mineral.pos.x, mineral.pos.y, STRUCTURE_EXTRACTOR, budget);
}

function nearestOpenNear(pos, range = 1) {
	const room = Game.rooms[pos.roomName];
	if (!room) return null;
	const terrain = room.getTerrain();
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

function planContainers(room, budget) {
	if (room.controller.level < config.containerRcl) return;
	if (config.finishSitesBeforeMore && countSites(room, STRUCTURE_CONTAINER) > 0) {
		// Still allow one missing source if no sites near that source — skip globally for simplicity.
		return;
	}

	for (const source of room.find(FIND_SOURCES)) {
		if (budget.left <= 0) return;
		const near = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		});
		const sites = source.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		});
		if (near.length || sites.length) continue;
		const spot = nearestOpenNear(source.pos, 1);
		if (spot) trySite(room, spot.x, spot.y, STRUCTURE_CONTAINER, budget);
	}

	const ctrl = room.controller;
	if (ctrl && budget.left > 0) {
		const near = ctrl.pos.findInRange(FIND_STRUCTURES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_LINK,
		});
		const sites = ctrl.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_LINK,
		});
		if (!near.length && !sites.length) {
			const spot = nearestOpenNear(ctrl.pos, 2);
			if (spot) trySite(room, spot.x, spot.y, STRUCTURE_CONTAINER, budget);
		}
	}
}

function planSourceLinks(room, budget) {
	if (room.controller.level < config.linkRcl) return;
	if (!canBuild(room, STRUCTURE_LINK)) return;
	if (config.finishSitesBeforeMore && countSites(room, STRUCTURE_LINK) > 0) return;

	for (const source of room.find(FIND_SOURCES)) {
		if (budget.left <= 0) return;
		const links = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK,
		});
		const sites = source.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK,
		});
		if (links.length || sites.length) continue;
		const spot = nearestOpenNear(source.pos, 2);
		if (spot && trySite(room, spot.x, spot.y, STRUCTURE_LINK, budget)) return;
	}
}

/**
 * Roads only on high-traffic heat tiles or critical spawn→source/controller paths.
 * Much less spam than v1.
 */
function planRoads(room, spawn, budget) {
	if (countSites(room, STRUCTURE_ROAD) >= (config.maxRoadSites || 8)) return;

	const heat = (room.memory.apex && room.memory.apex.heat) || {};
	const threshold = config.roadHeatThreshold || 8;

	// 1) Heat-based roads (preferred).
	const hot = [];
	for (const key of Object.keys(heat)) {
		if (heat[key] >= threshold) {
			const [ x, y ] = key.split(',').map(Number);
			hot.push({ x, y, h: heat[key] });
		}
	}
	hot.sort((a, b) => b.h - a.h);

	let placed = 0;
	for (const tile of hot) {
		if (budget.left <= 0 || placed >= 4) break;
		const pos = new RoomPosition(tile.x, tile.y, room.name);
		const hasRoad = pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_ROAD);
		const hasSite = pos.lookFor(LOOK_CONSTRUCTION_SITES).length > 0;
		if (!hasRoad && !hasSite) {
			if (trySite(room, tile.x, tile.y, STRUCTURE_ROAD, budget)) placed++;
		}
	}

	// 2) Seed critical paths once if almost no roads exist yet.
	const roadCount = room.find(FIND_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_ROAD,
	}).length;
	if (roadCount < 5 && budget.left > 0) {
		const goals = [
			...room.find(FIND_SOURCES).map(s => s.pos),
			room.controller && room.controller.pos,
		].filter(Boolean);

		for (const goal of goals) {
			if (budget.left <= 0) break;
			const path = spawn.pos.findPathTo(goal, {
				ignoreCreeps: true,
				maxOps: 4000,
			});
			// Place every other tile to reduce spam; builders will still connect over time via heat.
			for (let i = 0; i < path.length; i += 2) {
				if (budget.left <= 0) break;
				const step = path[i];
				if (step.x === goal.x && step.y === goal.y) continue;
				const pos = new RoomPosition(step.x, step.y, room.name);
				const hasRoad = pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_ROAD);
				const hasSite = pos.lookFor(LOOK_CONSTRUCTION_SITES).length > 0;
				if (!hasRoad && !hasSite) {
					trySite(room, step.x, step.y, STRUCTURE_ROAD, budget);
				}
			}
		}
	}
}

function planRamparts(room, spawn, budget) {
	if (room.controller.level < 4) return;
	if (energyOf(room.storage && room.storage.store) < 20_000 && room.controller.level < 6) return;
	if (config.finishSitesBeforeMore && countSites(room, STRUCTURE_RAMPART) > 2) return;

	const cores = [ spawn ];
	if (room.storage) cores.push(room.storage);
	if (room.terminal) cores.push(room.terminal);
	for (const tower of room.find(FIND_MY_STRUCTURES, { filter: s => s.structureType === STRUCTURE_TOWER })) {
		cores.push(tower);
	}

	let placed = 0;
	for (const core of cores) {
		for (let dx = -1; dx <= 1; dx++) {
			for (let dy = -1; dy <= 1; dy++) {
				if (budget.left <= 0 || placed >= 3) return;
				const x = core.pos.x + dx;
				const y = core.pos.y + dy;
				const pos = new RoomPosition(x, y, room.name);
				const has = pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_RAMPART);
				const site = pos.lookFor(LOOK_CONSTRUCTION_SITES).some(s => s.structureType === STRUCTURE_RAMPART);
				if (!has && !site && trySite(room, x, y, STRUCTURE_RAMPART, budget)) placed++;
			}
		}
	}
}

/**
 * RCL-gated planner order with site budget.
 */
function run(room) {
	if (!room.controller || !room.controller.my) return;
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (!spawn) return;

	const totalSites = countSites(room);
	if (totalSites >= (config.siteBudget || 18)) return;

	const mem = room.memory;
	if (mem._planAt && Game.time - mem._planAt < 10) return;
	mem._planAt = Game.time;

	const rcl = room.controller.level;
	const budget = { left: config.sitesPerPass || 6 };

	// Priority order by RCL needs.
	if (rcl >= 1) planExtensions(room, spawn, budget);
	if (rcl >= 3) planTowers(room, spawn, budget);
	if (rcl >= 2) planContainers(room, budget);
	if (rcl >= 4) planStorageCluster(room, spawn, budget);
	if (rcl >= 5) planSourceLinks(room, budget);
	if (rcl >= 6) planExtractor(room, budget);
	if (rcl >= 2) planRoads(room, spawn, budget);
	if (rcl >= 4) planRamparts(room, spawn, budget);
}

/**
 * Remote infrastructure gated by FSM phase.
 */
function runRemote(room, phase) {
	if (!room) return;
	if (phase === 'scout' || phase === 'abandoned') return;

	// Containers from 'container' phase onward.
	if (phase === 'reserve') return; // wait until reserve advances

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

module.exports = { run, runRemote, canBuild, maxOf, countStructures, countSites };
