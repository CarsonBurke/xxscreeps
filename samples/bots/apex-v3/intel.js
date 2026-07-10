/**
 * Room intel + remote scoring + flag orders.
 */
const config = require('config');
const {
	ownedRooms, adjacentRoomNames, isHighway, isSourceKeeperRoom, hostileThreat,
} = require('util');

function ensure() {
	Memory.intel ||= {};
	Memory.empire ||= {
		remotes: {},
		attacks: [],
		claims: [],
		forcedRemotes: {},
		ignoreRooms: {},
		campaigns: {},
		economy: {},
		version: 3,
	};
}

function recordRoom(room) {
	ensure();
	const entry = Memory.intel[room.name] || {};
	entry.lastSeen = Game.time;
	entry.threat = hostileThreat(room);
	entry.sources = room.find(FIND_SOURCES).map(s => ({
		id: s.id,
		pos: { x: s.pos.x, y: s.pos.y, roomName: s.pos.roomName },
		energyCapacity: s.energyCapacity,
	}));
	entry.sourceCount = entry.sources.length;
	entry.sk = isSourceKeeperRoom(room.name);
	entry.highway = isHighway(room.name);
	if (room.controller) {
		entry.controller = {
			level: room.controller.level,
			my: !!room.controller.my,
			owner: room.controller.owner && room.controller.owner.username,
			reservation: room.controller.reservation
				? {
					username: room.controller.reservation.username,
					ticks: room.controller.reservation.ticksToEnd,
				}
				: null,
		};
		if (room.controller.my) entry.owner = 'mine';
		else if (room.controller.owner) entry.owner = room.controller.owner.username;
		else entry.owner = null;
	}
	let score = entry.sourceCount * 10;
	if (entry.owner && entry.owner !== 'mine') score = -100;
	if (entry.sk) score -= 20;
	if (entry.highway) score -= 50;
	if (entry.threat > 0) score -= entry.threat * 5;
	entry.remoteScore = score;
	Memory.intel[room.name] = entry;
	return entry;
}

function tick() {
	ensure();
	for (const name in Game.rooms) recordRoom(Game.rooms[name]);

	Memory.empire.attacks = [];
	Memory.empire.claims = [];
	Memory.empire.forcedRemotes = {};
	Memory.empire.ignoreRooms = {};
	for (const name in Game.flags) {
		const flag = Game.flags[name];
		const lower = name.toLowerCase();
		const room = flag.pos.roomName;
		if (lower.startsWith('attack')) Memory.empire.attacks.push({ room, flag: name, pos: flag.pos });
		else if (lower.startsWith('claim')) Memory.empire.claims.push({ room, flag: name, pos: flag.pos });
		else if (lower.startsWith('remote')) Memory.empire.forcedRemotes[room] = true;
		else if (lower.startsWith('ignore')) Memory.empire.ignoreRooms[room] = Game.time;
		else if (lower.startsWith('rally')) Memory.empire.rally = { room, pos: flag.pos, flag: name };
		else if (lower.startsWith('defend')) {
			Memory.empire.attacks.push({ room, flag: name, pos: flag.pos, defend: true });
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

function pickRemotes(colonyRoom) {
	ensure();
	const home = colonyRoom.name;
	const rcl = colonyRoom.controller ? colonyRoom.controller.level : 0;
	if (rcl < config.remoteMinRcl) return [];

	// Honor economy projection cap when present.
	let maxR = config.maxRemotesPerColony;
	if (Memory.empire.economy && Memory.empire.economy.maxRemotes != null) {
		maxR = Math.min(maxR, Memory.empire.economy.maxRemotes);
	}

	const candidates = new Set(adjacentRoomNames(home));
	if (rcl >= 6) {
		for (const a of adjacentRoomNames(home)) {
			for (const b of adjacentRoomNames(a)) {
				if (b !== home) candidates.add(b);
			}
		}
	}
	for (const r of Object.keys(Memory.empire.forcedRemotes || {})) candidates.add(r);

	const owned = new Set(ownedRooms().map(r => r.name));
	const scored = [];
	for (const name of candidates) {
		if (name === home || owned.has(name)) continue;
		if (Memory.empire.ignoreRooms[name]) continue;
		if (isHighway(name)) continue;
		const intel = get(name);
		let score = intel ? intel.remoteScore : 5;
		const dist = Game.map.getRoomLinearDistance(home, name);
		score -= dist * 3;
		if (intel && intel.threat > 0 && Game.time - intel.lastSeen < config.remoteThreatCooldown) {
			score -= 40;
		}
		// War module may mark room as contested / abandoned
		const camp = Memory.empire.campaigns && Object.values(Memory.empire.campaigns).find(
			c => c.room === name && c.status === 'cooldown',
		);
		if (camp) score -= 30;
		if (score > 0) scored.push({ name, score });
	}
	scored.sort((a, b) => b.score - a.score);
	return scored.slice(0, maxR).map(s => s.name);
}

module.exports = {
	tick,
	recordRoom,
	get,
	isStale,
	pickRemotes,
};
