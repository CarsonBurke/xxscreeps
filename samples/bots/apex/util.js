/**
 * Pure helpers — no Game side effects except where noted.
 */

const BODY_COST = {
	move: 50, work: 100, carry: 50, attack: 80,
	ranged_attack: 150, heal: 250, claim: 600, tough: 10,
};

function bodyCost(body) {
	let sum = 0;
	for (const part of body) sum += BODY_COST[part] || 0;
	return sum;
}

function clamp(n, lo, hi) {
	return Math.max(lo, Math.min(hi, n));
}

function byRange(pos) {
	return (a, b) => pos.getRangeTo(a) - pos.getRangeTo(b);
}

function energyOf(storeLike) {
	if (!storeLike) return 0;
	if (typeof storeLike.getUsedCapacity === 'function') {
		const v = storeLike.getUsedCapacity(RESOURCE_ENERGY);
		if (v != null) return v;
	}
	return storeLike[RESOURCE_ENERGY] || storeLike.energy || 0;
}

function freeCapacity(storeLike, resource = RESOURCE_ENERGY) {
	if (!storeLike) return 0;
	if (typeof storeLike.getFreeCapacity === 'function') {
		// Prefer resource-specific free space; fall back to total free.
		let v = storeLike.getFreeCapacity(resource);
		if (v == null || Number.isNaN(v)) v = storeLike.getFreeCapacity();
		if (v != null && !Number.isNaN(v)) return v;
	}
	// Legacy plain stores
	if (resource === RESOURCE_ENERGY && storeLike.energyCapacity != null) {
		return Math.max(0, (storeLike.energyCapacity || 0) - (storeLike.energy || 0));
	}
	return 0;
}

function usedCapacity(storeLike, resource = RESOURCE_ENERGY) {
	if (!storeLike) return 0;
	if (typeof storeLike.getUsedCapacity === 'function') {
		const v = storeLike.getUsedCapacity(resource);
		if (v != null && !Number.isNaN(v)) return v;
	}
	return storeLike[resource] || 0;
}

/** Working-state helper: false when empty, true when full. Sticky otherwise. */
function updateWorking(creep, flag = 'working') {
	const energy = energyOf(creep.store);
	const free = freeCapacity(creep.store, RESOURCE_ENERGY);
	if (energy <= 0) creep.memory[flag] = false;
	else if (free <= 0) creep.memory[flag] = true;
	return !!creep.memory[flag];
}

/** True if structure can accept energy (spawn/extension/tower/storage/terminal/container/link). */
function wantsEnergy(structure) {
	if (!structure || !structure.store) return false;
	return freeCapacity(structure.store, RESOURCE_ENERGY) > 0;
}

const FILL_ORDER = {
	[STRUCTURE_SPAWN]: 1,
	[STRUCTURE_EXTENSION]: 2,
	[STRUCTURE_TOWER]: 3,
	[STRUCTURE_STORAGE]: 4,
	[STRUCTURE_TERMINAL]: 5,
	[STRUCTURE_CONTAINER]: 6,
	[STRUCTURE_LINK]: 7,
	[STRUCTURE_LAB]: 8,
	[STRUCTURE_NUKER]: 9,
	[STRUCTURE_POWER_SPAWN]: 10,
};

function fillPriority(structure) {
	return FILL_ORDER[structure.structureType] || 99;
}

/** Structures that should receive energy from haulers (not source containers). */
function isFillTarget(structure, room) {
	if (!wantsEnergy(structure)) return false;
	const t = structure.structureType;
	if (t === STRUCTURE_SPAWN || t === STRUCTURE_EXTENSION) return true;
	if (t === STRUCTURE_TOWER) {
		// Keep towers topped but don't starve spawns.
		return energyOf(structure.store) < structure.store.getCapacity(RESOURCE_ENERGY) * 0.9;
	}
	if (t === STRUCTURE_STORAGE || t === STRUCTURE_TERMINAL) return true;
	if (t === STRUCTURE_CONTAINER) {
		// Only controller-adjacent containers are sinks; source containers are sources.
		const ctrl = room.controller;
		if (ctrl && structure.pos.inRangeTo(ctrl, 3)) return true;
		return false;
	}
	if (t === STRUCTURE_LINK) {
		// Controller / storage links are sinks; source links are sources.
		const ctrl = room.controller;
		const storage = room.storage;
		if (ctrl && structure.pos.inRangeTo(ctrl, 2)) return true;
		if (storage && structure.pos.inRangeTo(storage, 2)) return true;
		return false;
	}
	return false;
}

function isEnergySourceStructure(structure, room) {
	if (!structure || !structure.store) return false;
	if (energyOf(structure.store) <= 0) return false;
	const t = structure.structureType;
	if (t === STRUCTURE_STORAGE || t === STRUCTURE_TERMINAL) return true;
	if (t === STRUCTURE_CONTAINER) {
		const ctrl = room.controller;
		// Source-side / remote containers.
		if (ctrl && structure.pos.inRangeTo(ctrl, 3)) return false;
		return true;
	}
	if (t === STRUCTURE_LINK) {
		const ctrl = room.controller;
		const storage = room.storage;
		if (ctrl && structure.pos.inRangeTo(ctrl, 2)) return false;
		if (storage && structure.pos.inRangeTo(storage, 2)) return false;
		return true;
	}
	return false;
}

/**
 * Direction from a to b as a Screeps direction constant (1–8), or null if same tile.
 */
function directionTo(from, to) {
	const dx = Math.sign(to.x - from.x);
	const dy = Math.sign(to.y - from.y);
	const map = {
		'0,-1': TOP,
		'1,-1': TOP_RIGHT,
		'1,0': RIGHT,
		'1,1': BOTTOM_RIGHT,
		'0,1': BOTTOM,
		'-1,1': BOTTOM_LEFT,
		'-1,0': LEFT,
		'-1,-1': TOP_LEFT,
	};
	return map[`${dx},${dy}`] || null;
}

function roomNameToXY(roomName) {
	const m = /^([WE])(\d+)([NS])(\d+)$/.exec(roomName);
	if (!m) return null;
	let x = Number(m[2]);
	let y = Number(m[4]);
	if (m[1] === 'W') x = -x - 1;
	if (m[3] === 'N') y = -y - 1;
	return { x, y };
}

function xyToRoomName(x, y) {
	const ew = x < 0 ? 'W' + (-x - 1) : 'E' + x;
	const ns = y < 0 ? 'N' + (-y - 1) : 'S' + y;
	return ew + ns;
}

function adjacentRoomNames(roomName) {
	const xy = roomNameToXY(roomName);
	if (!xy) return [];
	const deltas = [ [ 0, -1 ], [ 1, 0 ], [ 0, 1 ], [ -1, 0 ] ];
	return deltas.map(([ dx, dy ]) => xyToRoomName(xy.x + dx, xy.y + dy));
}

function isHighway(roomName) {
	const m = /^[WE](\d+)[NS](\d+)$/.exec(roomName);
	if (!m) return false;
	return Number(m[1]) % 10 === 0 || Number(m[2]) % 10 === 0;
}

function isSourceKeeperRoom(roomName) {
	const m = /^[WE](\d+)[NS](\d+)$/.exec(roomName);
	if (!m) return false;
	const x = Number(m[1]) % 10;
	const y = Number(m[2]) % 10;
	return x >= 4 && x <= 6 && y >= 4 && y <= 6;
}

function militaryParts(creep) {
	if (!creep || !creep.body) return 0;
	let n = 0;
	for (const p of creep.body) {
		if (
			p.hits > 0 &&
			(p.type === ATTACK || p.type === RANGED_ATTACK || p.type === HEAL || p.type === WORK)
		) {
			// WORK can dismantle; count as threat if hostile.
			if (p.type === WORK) n += 0.25;
			else n += 1;
		}
	}
	return n;
}

function hostileThreat(room) {
	if (!room) return 0;
	let threat = 0;
	for (const c of room.find(FIND_HOSTILE_CREEPS)) {
		threat += militaryParts(c);
	}
	return threat;
}

function ownedRooms() {
	const rooms = [];
	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		if (room.controller && room.controller.my) rooms.push(room);
	}
	return rooms;
}

function creepsByRole() {
	const map = {};
	for (const name in Game.creeps) {
		const creep = Game.creeps[name];
		const role = creep.memory && creep.memory.role;
		if (!role) continue;
		(map[role] ||= []).push(creep);
	}
	return map;
}

function creepsInRoom(roomName, role) {
	const out = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (role && (!c.memory || c.memory.role !== role)) continue;
		if (c.memory && c.memory.home === roomName) out.push(c);
		else if (!c.memory || !c.memory.home) {
			if (c.room.name === roomName) out.push(c);
		}
	}
	return out;
}

function creepsForTarget(targetRoom, role) {
	const out = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if (role && c.memory.role !== role) continue;
		if (c.memory.targetRoom === targetRoom || c.memory.remote === targetRoom) out.push(c);
	}
	return out;
}

/**
 * Go to target and perform `action()` when in range. Returns action result or move result.
 */
function goDo(creep, target, range, action) {
	if (!target) return ERR_INVALID_TARGET;
	if (creep.pos.inRangeTo(target, range)) {
		return action();
	}
	return creep.moveTo(target, {
		reusePath: 20,
		// ignoreCreeps when far reduces traffic jams for haulers/miners.
		ignoreCreeps: creep.pos.getRangeTo(target) > 3,
		maxRooms: 16,
		visualizePathStyle: Memory._apexVisuals
			? { stroke: '#88f', opacity: 0.3, lineStyle: 'dashed' }
			: undefined,
	});
}

function harvestEnergy(creep) {
	// Prefer dropped / tombstones / containers / storage / active sources.
	// Use range fallbacks when path search fails (common with many construction sites).
	const room = creep.room;
	const moveOpts = { reusePath: 10, maxOps: 4000, ignoreCreeps: true };

	const drop = creep.pos.findClosestByRange(FIND_DROPPED_RESOURCES, {
		filter: r => r.resourceType === RESOURCE_ENERGY && r.amount > 20,
	});
	if (drop) {
		if (creep.pos.isNearTo(drop)) return creep.pickup(drop);
		return creep.moveTo(drop, moveOpts);
	}

	const tomb = creep.pos.findClosestByRange(FIND_TOMBSTONES, {
		filter: t => energyOf(t.store) > 20,
	});
	if (tomb) {
		if (creep.pos.isNearTo(tomb)) return creep.withdraw(tomb, RESOURCE_ENERGY);
		return creep.moveTo(tomb, moveOpts);
	}

	const structs = room.find(FIND_STRUCTURES, {
		filter: s => isEnergySourceStructure(s, room) && energyOf(s.store) > 50,
	});
	if (structs.length) {
		structs.sort((a, b) => energyOf(b.store) - energyOf(a.store) || creep.pos.getRangeTo(a) - creep.pos.getRangeTo(b));
		const s = structs[0];
		if (creep.pos.isNearTo(s)) return creep.withdraw(s, RESOURCE_ENERGY);
		return creep.moveTo(s, moveOpts);
	}

	const source =
		creep.pos.findClosestByPath(FIND_SOURCES_ACTIVE, { maxOps: 4000 }) ||
		creep.pos.findClosestByRange(FIND_SOURCES_ACTIVE) ||
		creep.pos.findClosestByRange(FIND_SOURCES);
	if (source) {
		if (creep.pos.isNearTo(source)) return creep.harvest(source);
		return creep.moveTo(source, moveOpts);
	}
	return ERR_NOT_FOUND;
}

function deliverEnergy(creep, opts = {}) {
	const room = creep.room;
	const preferUpgrade = opts.preferUpgrade;
	const includeController = opts.includeController;

	const targets = room.find(FIND_MY_STRUCTURES, {
		filter: s => isFillTarget(s, room),
	});
	// Prioritize spawn network, then towers, then storage.
	targets.sort((a, b) => {
		const pa = fillPriority(a);
		const pb = fillPriority(b);
		if (pa !== pb) return pa - pb;
		// Prefer emptier among same type.
		return energyOf(a.store) - energyOf(b.store);
	});

	if (targets.length) {
		const t = creep.pos.findClosestByPath(targets.slice(0, 8)) || targets[0];
		return goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
	}

	if (includeController && room.controller && room.controller.my) {
		return goDo(creep, room.controller, 3, () => creep.upgradeController(room.controller));
	}

	if (preferUpgrade && room.controller && room.controller.my) {
		return goDo(creep, room.controller, 3, () => creep.upgradeController(room.controller));
	}

	// Idle near storage/spawn.
	const idle = room.storage || room.find(FIND_MY_SPAWNS)[0];
	if (idle && !creep.pos.inRangeTo(idle, 2)) creep.moveTo(idle, { reusePath: 30 });
	return ERR_NOT_FOUND;
}

function moveToRoom(creep, roomName) {
	if (creep.room.name === roomName) return OK;
	const exit = creep.room.findExitTo(roomName);
	if (exit === ERR_NO_PATH || exit === ERR_INVALID_ARGS) {
		// Multi-hop via map route.
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
		if (ex) return creep.moveTo(ex, { reusePath: 30, maxRooms: 1 });
		return ERR_NO_PATH;
	}
	const pos = creep.pos.findClosestByPath(exit);
	if (!pos) return ERR_NO_PATH;
	return creep.moveTo(pos, { reusePath: 20, maxRooms: 1 });
}

function log(...args) {
	if (Game.time % 10 === 0) console.log('Apex', Game.time, ...args);
}

module.exports = {
	BODY_COST,
	bodyCost,
	clamp,
	byRange,
	energyOf,
	freeCapacity,
	usedCapacity,
	wantsEnergy,
	fillPriority,
	isFillTarget,
	isEnergySourceStructure,
	directionTo,
	roomNameToXY,
	xyToRoomName,
	adjacentRoomNames,
	isHighway,
	isSourceKeeperRoom,
	militaryParts,
	hostileThreat,
	ownedRooms,
	creepsByRole,
	creepsInRoom,
	creepsForTarget,
	goDo,
	harvestEnergy,
	deliverEnergy,
	moveToRoom,
	updateWorking,
	log,
};
