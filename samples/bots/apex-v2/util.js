/**
 * Pure helpers — limited Game side effects. Apex v2.
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

/** Safe object lookup — always guards null/undefined ids. */
function safeGet(id) {
	if (!id) return null;
	const obj = Game.getObjectById(id);
	return obj || null;
}

function energyOf(storeLike) {
	if (!storeLike) return 0;
	if (typeof storeLike.getUsedCapacity === 'function') {
		return storeLike.getUsedCapacity(RESOURCE_ENERGY) || 0;
	}
	return storeLike[RESOURCE_ENERGY] || storeLike.energy || 0;
}

function freeCapacity(storeLike, resource = RESOURCE_ENERGY) {
	if (!storeLike) return 0;
	if (typeof storeLike.getFreeCapacity === 'function') {
		return storeLike.getFreeCapacity(resource) || 0;
	}
	return 0;
}

function usedCapacity(storeLike, resource = RESOURCE_ENERGY) {
	if (!storeLike) return 0;
	if (typeof storeLike.getUsedCapacity === 'function') {
		return storeLike.getUsedCapacity(resource) || 0;
	}
	return storeLike[resource] || 0;
}

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

function isFillTarget(structure, room) {
	if (!wantsEnergy(structure)) return false;
	const t = structure.structureType;
	if (t === STRUCTURE_SPAWN || t === STRUCTURE_EXTENSION) return true;
	if (t === STRUCTURE_TOWER) {
		return energyOf(structure.store) < structure.store.getCapacity(RESOURCE_ENERGY) * 0.9;
	}
	if (t === STRUCTURE_STORAGE || t === STRUCTURE_TERMINAL) return true;
	if (t === STRUCTURE_CONTAINER) {
		const ctrl = room.controller;
		if (ctrl && structure.pos.inRangeTo(ctrl, 3)) return true;
		return false;
	}
	if (t === STRUCTURE_LINK) {
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
 * Room-to-room path length cache (linear estimate refined when both rooms visible).
 * Memory.apex.pathCache[from|to] = { len, t }
 */
function cachedPathLength(fromRoom, toRoom) {
	if (!fromRoom || !toRoom) return 50;
	if (fromRoom === toRoom) return 0;
	Memory.apex ||= {};
	Memory.apex.pathCache ||= {};
	const key = fromRoom < toRoom ? `${fromRoom}|${toRoom}` : `${toRoom}|${fromRoom}`;
	const hit = Memory.apex.pathCache[key];
	if (hit && Game.time - hit.t < 1000) return hit.len;

	const linear = Game.map.getRoomLinearDistance(fromRoom, toRoom);
	let len = linear * 50 + 25;
	const route = Game.map.findRoute(fromRoom, toRoom, {
		routeCallback(room) {
			if (isHighway(room)) return 1.2;
			if (isSourceKeeperRoom(room)) return 5;
			return 1;
		},
	});
	if (route !== ERR_NO_PATH && route && route.length) {
		len = route.length * 50 + 15;
	}
	Memory.apex.pathCache[key] = { len, t: Game.time };
	return len;
}

/**
 * Estimate path length between two positions; uses room cache for multi-room.
 */
function estimatePathLength(fromPos, toPos) {
	if (!fromPos || !toPos) return 30;
	if (fromPos.roomName === toPos.roomName) {
		const cacheKey = `${fromPos.x},${fromPos.y}->${toPos.x},${toPos.y}`;
		Memory.apex ||= {};
		Memory.apex.localPathCache ||= {};
		const roomCache = Memory.apex.localPathCache[fromPos.roomName] ||= {};
		const hit = roomCache[cacheKey];
		if (hit && Game.time - hit.t < 500) return hit.len;
		const path = fromPos.findPathTo(toPos, { ignoreCreeps: true, maxOps: 2000 });
		const len = path.length || fromPos.getRangeTo(toPos);
		roomCache[cacheKey] = { len, t: Game.time };
		return len;
	}
	const roomLen = cachedPathLength(fromPos.roomName, toPos.roomName);
	// Approximate intra-room legs.
	return roomLen + 20;
}

function spawnTimeForBody(body) {
	// Each body part takes 3 ticks to spawn.
	return (body && body.length ? body.length : 3) * 3;
}

/**
 * Whether a living creep should still count toward demand
 * (false if TTL too low to finish another full spawn of itself).
 */
function stillAlive(creep, spawnEta = 0) {
	if (!creep) return false;
	if (creep.spawning) return true;
	if (creep.ticksToLive === undefined) return true;
	const config = require('config');
	const buffer = config.replaceTtlBuffer || 30;
	const minTtl = config.replaceMinTtl || 80;
	const threshold = Math.max(minTtl, spawnEta + buffer);
	return creep.ticksToLive > threshold;
}

function lowBucket() {
	const config = require('config');
	return Game.cpu.bucket < (config.lowBucketThreshold || 2000);
}

function criticalBucket() {
	const config = require('config');
	return Game.cpu.bucket < (config.criticalBucketThreshold || 500);
}

function log(...args) {
	if (Game.time % 10 === 0) console.log('Apex', Game.time, ...args);
}

module.exports = {
	BODY_COST,
	bodyCost,
	clamp,
	byRange,
	safeGet,
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
	cachedPathLength,
	estimatePathLength,
	spawnTimeForBody,
	stillAlive,
	lowBucket,
	criticalBucket,
	log,
};
