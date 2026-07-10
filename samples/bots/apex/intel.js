/**
 * Room intelligence: scouting, threat tracking, remote candidate scoring.
 */
const config = require('config');
const {
	ownedRooms, adjacentRoomNames, isHighway, isSourceKeeperRoom,
	hostileThreat, militaryParts, log,
} = require('util');

function ensure() {
	Memory.intel ||= {};
	Memory.empire ||= { remotes: {}, attacks: [], claims: [] };
}

function recordRoom(room) {
	ensure();
	const name = room.name;
	const entry = Memory.intel[name] || {};
	entry.lastSeen = Game.time;
	entry.threat = hostileThreat(room);
	entry.sources = room.find(FIND_SOURCES).map(s => ({
		id: s.id,
		pos: { x: s.pos.x, y: s.pos.y, roomName: s.pos.roomName },
		energyCapacity: s.energyCapacity,
	}));
	const mineral = room.find(FIND_MINERALS)[0];
	entry.mineral = mineral
		? { id: mineral.id, type: mineral.mineralType, amount: mineral.mineralAmount }
		: null;

	if (room.controller) {
		entry.controller = {
			id: room.controller.id,
			level: room.controller.level,
			my: !!room.controller.my,
			owner: room.controller.owner && room.controller.owner.username,
			reservation: room.controller.reservation
				? {
					username: room.controller.reservation.username,
					ticks: room.controller.reservation.ticksToEnd,
				}
				: null,
			safeMode: room.controller.safeMode || 0,
			safeModeAvailable: room.controller.safeModeAvailable || 0,
		};
		if (room.controller.my) entry.owner = 'mine';
		else if (room.controller.owner) entry.owner = room.controller.owner.username;
		else if (room.controller.reservation) entry.owner = 'reserved:' + room.controller.reservation.username;
		else entry.owner = null;
	} else {
		entry.controller = null;
		entry.owner = isHighway(name) ? 'highway' : null;
	}

	// Source keeper rooms are poor early remotes.
	entry.sk = isSourceKeeperRoom(name);
	entry.highway = isHighway(name);
	entry.sourceCount = entry.sources.length;

	// Score as remote: more sources, no owner, not SK, closer preferred (scored by caller).
	let score = entry.sourceCount * 10;
	if (entry.owner && entry.owner !== 'mine') score = -100;
	if (entry.sk) score -= 20;
	if (entry.highway) score -= 50;
	if (entry.threat > 0) score -= entry.threat * 5;
	entry.remoteScore = score;

	Memory.intel[name] = entry;
	return entry;
}

function tick() {
	ensure();
	// Record every visible room.
	for (const name in Game.rooms) {
		recordRoom(Game.rooms[name]);
	}

	// Parse flag orders into empire intent.
	Memory.empire.attacks = [];
	Memory.empire.claims = [];
	Memory.empire.forcedRemotes = Memory.empire.forcedRemotes || {};
	Memory.empire.ignoreRooms = Memory.empire.ignoreRooms || {};

	for (const name in Game.flags) {
		const flag = Game.flags[name];
		const lower = name.toLowerCase();
		const room = flag.pos.roomName;
		if (lower.startsWith('attack')) {
			Memory.empire.attacks.push({ room, flag: name, pos: flag.pos });
		} else if (lower.startsWith('claim')) {
			Memory.empire.claims.push({ room, flag: name, pos: flag.pos });
		} else if (lower.startsWith('remote')) {
			Memory.empire.forcedRemotes[room] = flag.pos.roomName;
		} else if (lower.startsWith('ignore')) {
			Memory.empire.ignoreRooms[room] = Game.time;
		} else if (lower.startsWith('rally')) {
			Memory.empire.rally = { room, pos: flag.pos, flag: name };
		}
	}
}

function get(roomName) {
	ensure();
	return Memory.intel[roomName];
}

function isStale(roomName) {
	const e = get(roomName);
	return !e || Game.time - (e.lastSeen || 0) > config.intelStale;
}

/**
 * Pick remote mining rooms for a colony.
 * Returns array of room names, best first.
 */
function pickRemotes(colonyRoom) {
	ensure();
	const home = colonyRoom.name;
	const rcl = colonyRoom.controller ? colonyRoom.controller.level : 0;
	if (rcl < config.remoteMinRcl) return [];

	const candidates = new Set(adjacentRoomNames(home));
	// Depth-2 for high RCL.
	if (rcl >= 6) {
		for (const a of adjacentRoomNames(home)) {
			for (const b of adjacentRoomNames(a)) {
				if (b !== home) candidates.add(b);
			}
		}
	}

	// Forced remotes via flags.
	for (const r of Object.keys(Memory.empire.forcedRemotes || {})) {
		candidates.add(r);
	}

	const owned = new Set(ownedRooms().map(r => r.name));
	const scored = [];

	for (const name of candidates) {
		if (name === home) continue;
		if (owned.has(name)) continue;
		if (Memory.empire.ignoreRooms && Memory.empire.ignoreRooms[name]) continue;
		if (isHighway(name)) continue;

		const intel = get(name);
		let score = intel ? intel.remoteScore : 5; // unknown but adjacent: mild interest
		if (!intel) score = 5;

		// Prefer rooms we've seen with sources.
		if (intel && intel.sourceCount === 0 && intel.lastSeen) score -= 30;

		// Distance penalty via linear room distance.
		const dist = Game.map.getRoomLinearDistance(home, name);
		score -= dist * 3;

		// Threat memory.
		if (intel && intel.threat > 0 && Game.time - intel.lastSeen < config.remoteThreatCooldown) {
			score -= 40;
		}

		// Reservation by others.
		if (intel && intel.controller && intel.controller.reservation) {
			const res = intel.controller.reservation;
			if (res.username && !isMyReservation(res.username)) score -= 15;
		}

		if (score > 0) scored.push({ name, score });
	}

	scored.sort((a, b) => b.score - a.score);
	const max = config.maxRemotesPerColony;
	return scored.slice(0, max).map(s => s.name);
}

function isMyReservation(username) {
	// Our creeps reserve under our username; compare against any of our creeps' owner.
	for (const n in Game.creeps) {
		return Game.creeps[n].owner && Game.creeps[n].owner.username === username;
	}
	// Fallback: if we have no creeps, treat unknown as not ours.
	return false;
}

/**
 * Source mining slots: open terrain adjacent tiles.
 */
function sourceSpots(source) {
	const terrain = source.room.getTerrain();
	const spots = [];
	for (let dx = -1; dx <= 1; dx++) {
		for (let dy = -1; dy <= 1; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = source.pos.x + dx;
			const y = source.pos.y + dy;
			if (x < 0 || x > 49 || y < 0 || y > 49) continue;
			if (terrain.get(x, y) !== TERRAIN_MASK_WALL) {
				spots.push(new RoomPosition(x, y, source.room.name));
			}
		}
	}
	return spots;
}

function estimatePathLength(fromPos, toPos) {
	if (!fromPos || !toPos) return 30;
	if (fromPos.roomName === toPos.roomName) {
		const path = fromPos.findPathTo(toPos, { ignoreCreeps: true, maxOps: 2000 });
		return path.length || fromPos.getRangeTo(toPos);
	}
	// Rough multi-room estimate.
	const roomDist = Game.map.getRoomLinearDistance(fromPos.roomName, toPos.roomName);
	return roomDist * 50 + 20;
}

module.exports = {
	tick,
	recordRoom,
	get,
	isStale,
	pickRemotes,
	sourceSpots,
	estimatePathLength,
	log,
};
