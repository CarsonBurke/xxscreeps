/**
 * Shared helpers for Apex v3.
 */
const BODY_COST = {
	move: 50, work: 100, carry: 50, attack: 80,
	ranged_attack: 150, heal: 250, claim: 600, tough: 10,
};

function bodyCost(body) {
	let sum = 0;
	for (let i = 0; i < body.length; i++) sum += BODY_COST[body[i]] || BODYPART_COST[body[i]] || 0;
	return sum;
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
		let v = storeLike.getFreeCapacity(resource);
		if (v == null || Number.isNaN(v)) v = storeLike.getFreeCapacity();
		if (v != null && !Number.isNaN(v)) return v;
	}
	return 0;
}

function updateWorking(creep, flag = 'working') {
	const energy = energyOf(creep.store);
	const free = freeCapacity(creep.store, RESOURCE_ENERGY);
	if (energy <= 0) creep.memory[flag] = false;
	else if (free <= 0) creep.memory[flag] = true;
	return !!creep.memory[flag];
}

function ownedRooms() {
	const rooms = [];
	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		if (room.controller && room.controller.my) rooms.push(room);
	}
	return rooms;
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
	return [ [ 0, -1 ], [ 1, 0 ], [ 0, 1 ], [ -1, 0 ] ]
		.map(([ dx, dy ]) => xyToRoomName(xy.x + dx, xy.y + dy));
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
		if (p.hits <= 0) continue;
		if (p.type === ATTACK || p.type === RANGED_ATTACK || p.type === HEAL) n += 1;
		else if (p.type === WORK) n += 0.25;
	}
	return n;
}

function hostileThreat(room) {
	if (!room) return 0;
	let t = 0;
	for (const c of room.find(FIND_HOSTILE_CREEPS)) t += militaryParts(c);
	return t;
}

function stillAlive(creep, spawnEta = 0) {
	if (!creep) return false;
	if (creep.ticksToLive === undefined) return true; // spawning
	return creep.ticksToLive > 80 + spawnEta;
}

function spawnTimeForBody(body) {
	const n = Array.isArray(body) ? body.length : (body && body.length) || 0;
	return n * (typeof CREEP_SPAWN_TIME === 'number' ? CREEP_SPAWN_TIME : 3);
}

function estimatePathLength(fromPos, toPos) {
	if (!fromPos || !toPos) return 30;
	if (fromPos.roomName === toPos.roomName) {
		const path = fromPos.findPathTo(toPos, { ignoreCreeps: true, maxOps: 2000 });
		return path.length || fromPos.getRangeTo(toPos);
	}
	const roomDist = Game.map.getRoomLinearDistance(fromPos.roomName, toPos.roomName);
	return roomDist * 50 + 20;
}

function moveToRoom(creep, roomName) {
	if (creep.room.name === roomName) return OK;
	const route = Game.map.findRoute(creep.room.name, roomName, {
		routeCallback(room) {
			if (isHighway(room)) return 1.5;
			if (isSourceKeeperRoom(room)) return 6;
			const mem = Memory.intel && Memory.intel[room];
			if (mem && mem.threat > 0) return 8;
			return 1;
		},
	});
	if (route === ERR_NO_PATH || !route.length) return ERR_NO_PATH;
	const exit = creep.pos.findClosestByPath(route[0].exit);
	if (!exit) return ERR_NO_PATH;
	return creep.moveTo(exit, { reusePath: 30, maxRooms: 1 });
}

function goDo(creep, target, range, action) {
	if (!target) return ERR_INVALID_TARGET;
	if (creep.pos.inRangeTo(target, range)) return action();
	return creep.moveTo(target, {
		reusePath: 15,
		ignoreCreeps: creep.pos.getRangeTo(target) > 3,
		maxOps: 4000,
	});
}

/** Energy piles / containers / storage near a position. */
function nearestEnergyPickup(pos, range = 50, opts = {}) {
	const room = Game.rooms[pos.roomName];
	if (!room) return null;
	const minAmount = opts.minAmount != null ? opts.minAmount : 20;
	let best = null;
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
			if (!s.store) return false;
			if (energyOf(s.store) < minAmount) return false;
			const t = s.structureType;
			if (t === STRUCTURE_CONTAINER || t === STRUCTURE_STORAGE || t === STRUCTURE_TERMINAL) return true;
			if (t === STRUCTURE_LINK && opts.links) return true;
			return false;
		},
	});
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

function doPickup(creep, pickup) {
	if (!pickup) return ERR_NOT_FOUND;
	if (pickup.type === 'pickup') {
		return goDo(creep, pickup.target, 1, () => creep.pickup(pickup.target));
	}
	return goDo(creep, pickup.target, 1, () => creep.withdraw(pickup.target, RESOURCE_ENERGY));
}

function lowBucket() {
	const config = require('config');
	return Game.cpu.bucket < (config.lowBucketThreshold || 2000);
}

function creepsFor(filter) {
	const out = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (filter(c)) out.push(c);
	}
	return out;
}

module.exports = {
	BODY_COST,
	bodyCost,
	energyOf,
	freeCapacity,
	updateWorking,
	ownedRooms,
	adjacentRoomNames,
	isHighway,
	isSourceKeeperRoom,
	militaryParts,
	hostileThreat,
	stillAlive,
	spawnTimeForBody,
	estimatePathLength,
	moveToRoom,
	goDo,
	nearestEnergyPickup,
	doPickup,
	lowBucket,
	creepsFor,
	roomNameToXY,
};
