/**
 * Room intelligence: scouting, threat tracking, remote candidate scoring,
 * expansion scoring (2 sources preferred, not SK/highway).
 */
const config = require('config');
const {
	ownedRooms, adjacentRoomNames, isHighway, isSourceKeeperRoom,
	hostileThreat, estimatePathLength, lowBucket,
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

	entry.sk = isSourceKeeperRoom(name);
	entry.highway = isHighway(name);
	entry.sourceCount = entry.sources.length;

	// Remote score: prefer 2 sources, avoid SK/highway/owned.
	let score = 0;
	const pref = config.preferredSourceCount || 2;
	if (entry.sourceCount >= pref) score += 25;
	else score += entry.sourceCount * 10;
	// Mild bonus for exactly 2 (standard remote).
	if (entry.sourceCount === 2) score += 5;
	if (entry.owner && entry.owner !== 'mine') score = -100;
	if (entry.sk) score -= 40;
	if (entry.highway) score -= 50;
	if (entry.threat > 0) score -= entry.threat * 5;
	entry.remoteScore = score;

	// Expansion score: claimable controllers, 2 sources preferred.
	let exp = entry.remoteScore;
	if (!room.controller || room.controller.my || room.controller.owner) exp = -100;
	if (entry.sourceCount === 2) exp += 15;
	if (entry.sourceCount > 2) exp += 5;
	if (entry.sk || entry.highway) exp = -100;
	entry.expandScore = exp;

	Memory.intel[name] = entry;
	return entry;
}

function tick() {
	ensure();
	for (const name in Game.rooms) {
		recordRoom(Game.rooms[name]);
	}

	Memory.empire.attacks = [];
	Memory.empire.claims = [];
	Memory.empire.forcedRemotes = Memory.empire.forcedRemotes || {};
	Memory.empire.ignoreRooms = Memory.empire.ignoreRooms || {};

	for (const name in Game.flags) {
		const flag = Game.flags[name];
		const lower = name.toLowerCase();
		const room = flag.pos.roomName;
		if (lower.startsWith('attack')) {
			Memory.empire.attacks.push({ room, flag: name, pos: { x: flag.pos.x, y: flag.pos.y, roomName: room } });
		} else if (lower.startsWith('claim')) {
			Memory.empire.claims.push({ room, flag: name, pos: { x: flag.pos.x, y: flag.pos.y, roomName: room } });
		} else if (lower.startsWith('remote')) {
			Memory.empire.forcedRemotes[room] = flag.pos.roomName;
		} else if (lower.startsWith('ignore')) {
			Memory.empire.ignoreRooms[room] = Game.time;
		} else if (lower.startsWith('rally')) {
			Memory.empire.rally = {
				room,
				pos: { x: flag.pos.x, y: flag.pos.y, roomName: room },
				flag: name,
			};
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

function isMyReservation(username) {
	for (const n in Game.creeps) {
		return Game.creeps[n].owner && Game.creeps[n].owner.username === username;
	}
	return false;
}

/**
 * Pick remote mining rooms for a colony. Returns room names, best first.
 */
function pickRemotes(colonyRoom) {
	ensure();
	const home = colonyRoom.name;
	const rcl = colonyRoom.controller ? colonyRoom.controller.level : 0;
	if (rcl < config.remoteMinRcl) return [];
	if (lowBucket()) {
		// Keep existing remotes only when CPU is tight.
		const m = colonyRoom.memory.apex;
		return (m && m.remotes) || [];
	}

	const candidates = new Set(adjacentRoomNames(home));
	if (rcl >= 6) {
		for (const a of adjacentRoomNames(home)) {
			for (const b of adjacentRoomNames(a)) {
				if (b !== home) candidates.add(b);
			}
		}
	}

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
		if (isSourceKeeperRoom(name)) continue;

		const intel = get(name);
		let score = intel ? intel.remoteScore : 5;
		if (!intel) score = 5;

		if (intel && intel.sourceCount === 0 && intel.lastSeen) score -= 30;
		// Prefer 2-source rooms strongly when known.
		if (intel && intel.sourceCount === 2) score += 10;
		if (intel && intel.sourceCount === 1) score += 0;
		if (intel && intel.sourceCount >= 3) score += 3;

		const dist = Game.map.getRoomLinearDistance(home, name);
		score -= dist * 3;

		if (intel && intel.threat > 0 && Game.time - intel.lastSeen < config.remoteThreatCooldown) {
			score -= 40;
		}

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

/**
 * Best expansion (claim) target near a colony.
 */
function pickExpansionTarget(colonyRoom) {
	ensure();
	const home = colonyRoom.name;
	const candidates = new Set(adjacentRoomNames(home));
	if (colonyRoom.controller && colonyRoom.controller.level >= 6) {
		for (const a of adjacentRoomNames(home)) {
			for (const b of adjacentRoomNames(a)) {
				if (b !== home) candidates.add(b);
			}
		}
	}
	const owned = new Set(ownedRooms().map(r => r.name));
	let best = null;
	let bestScore = 0;
	for (const name of candidates) {
		if (owned.has(name)) continue;
		if (Memory.empire.ignoreRooms && Memory.empire.ignoreRooms[name]) continue;
		if (isHighway(name) || isSourceKeeperRoom(name)) continue;
		const intel = get(name);
		if (!intel) continue;
		let score = intel.expandScore != null ? intel.expandScore : intel.remoteScore;
		if (intel.sourceCount === 2) score += 20;
		if (intel.owner) continue;
		if (intel.controller && intel.controller.owner) continue;
		score -= Game.map.getRoomLinearDistance(home, name) * 2;
		if (score > bestScore) {
			bestScore = score;
			best = name;
		}
	}
	return best;
}

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

module.exports = {
	tick,
	recordRoom,
	get,
	isStale,
	pickRemotes,
	pickExpansionTarget,
	sourceSpots,
	estimatePathLength,
};
