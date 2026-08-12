/**
 * Room intel + intent sync.
 * Positions as Coord only. Room orders as RoomIntent enums (no flag-name strings).
 */
import config = require('./config');
import { asCoord, type Coord } from './codec';
import {
	ownedRooms, adjacentRoomNames, isHighway, isSourceKeeperRoom, hostileThreat,
	canAffordSkKiller,
} from './util';
import { RoomIntent, syncIntentsFromFlags, getRoomIntent } from './intents';

export interface SourceIntel {
	id: string;
	pos: Coord & { roomName: string };
	energyCapacity: number;
}

export interface RoomIntel {
	lastSeen: number;
	threat: number;
	sources: SourceIntel[];
	sourceCount: number;
	sk: boolean;
	highway: boolean;
	owner: string | null;
	/** Mirror of RoomIntent for consumers that read intel only */
	intent: RoomIntent;
	controller?: {
		level: number;
		my: boolean;
		owner?: string;
		reservation?: { username: string; ticks: number } | null;
	};
	remoteScore: number;
}

function ensureEmpire(): void {
	Memory.empire ||= {
		attacks: [],
		claims: [],
		defends: [],
		forcedRemotes: {},
		ignoreRooms: {},
		campaigns: {},
		economy: {},
		version: 3,
	};
	Memory.intel ||= {};
}

export function recordRoom(room: Room): RoomIntel {
	ensureEmpire();
	const map = Memory.intel as Record<string, RoomIntel>;
	const entry: RoomIntel = map[room.name] || {
		lastSeen: 0,
		threat: 0,
		sources: [],
		sourceCount: 0,
		sk: false,
		highway: false,
		owner: null,
		intent: RoomIntent.none,
		remoteScore: 0,
	};
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
	entry.intent = getRoomIntent(room.name);
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
	if (entry.intent === RoomIntent.ignore) score = -1000;
	if (entry.intent === RoomIntent.remote) score += 50;
	entry.remoteScore = score;
	map[room.name] = entry;
	return entry;
}

export function tick(): void {
	ensureEmpire();

	// Color → RoomIntent enums (no string name parsing)
	const intents = syncIntentsFromFlags();

	// Mirror room-name lists for spawn/empire (RoomIntent is source of truth)
	const empire = Memory.empire as {
		attacks: { room: string }[];
		claims: { room: string }[];
		defends: { room: string }[];
		forcedRemotes: Record<string, boolean>;
		ignoreRooms: Record<string, number>;
		rally?: { room: string; pos: Coord };
	};
	empire.attacks = intents.attacks.map(room => ({ room }));
	empire.defends = intents.defends.map(room => ({ room }));
	empire.claims = intents.claims.map(room => ({ room }));
	empire.forcedRemotes = {};
	for (const room of intents.remotes) empire.forcedRemotes[room] = true;
	empire.ignoreRooms = {};
	for (const room of intents.ignores) empire.ignoreRooms[room] = Game.time;
	empire.rally = intents.rally ?? undefined;

	for (const name in Game.rooms) recordRoom(Game.rooms[name]!);
}

export function get(roomName: string): RoomIntel | undefined {
	ensureEmpire();
	return (Memory.intel as Record<string, RoomIntel>)[roomName];
}

export function isStale(roomName: string): boolean {
	const e = get(roomName);
	return !e || Game.time - (e.lastSeen || 0) > config.intelStale;
}

/**
 * Every viable remote room, best-first. No artificial count cap.
 * Real filters only: not ours, not highway, not ignore, not fresh threat, not SK without combat.
 */
export function pickRemotes(colonyRoom: Room): string[] {
	ensureEmpire();
	const emp = Memory.empire!;
	const home = colonyRoom.name;
	const rcl = colonyRoom.controller ? colonyRoom.controller.level : 0;
	// Need a spawn to project packages — no spawn ⇒ no remotes (uptime constraint)
	if (!colonyRoom.find(FIND_MY_SPAWNS).length) return [];

	// SK only if we can field a melee killer at current spawn energy capacity
	let spawnCap = 0;
	for (const s of colonyRoom.find(FIND_MY_STRUCTURES)) {
		if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
			spawnCap += s.store.getCapacity(RESOURCE_ENERGY) || 0;
		}
	}
	const skOk = canAffordSkKiller(spawnCap);

	const candidates = new Set(adjacentRoomNames(home));
	// Always consider 2-hop: distance is scored, spawn duty filters later — not RCL
	for (const a of adjacentRoomNames(home)) {
		for (const b of adjacentRoomNames(a)) {
			if (b !== home) candidates.add(b);
		}
	}
	// Forced remotes: RoomIntent.remote
	for (const name in Memory.rooms || {}) {
		if (getRoomIntent(name) === RoomIntent.remote) candidates.add(name);
	}
	for (const r of Object.keys(emp.forcedRemotes || {})) candidates.add(r);

	const owned = new Set(ownedRooms().map(r => r.name));
	const scored: { name: string; score: number }[] = [];

	for (const name of candidates) {
		if (name === home || owned.has(name)) continue;
		if (getRoomIntent(name) === RoomIntent.ignore) continue;
		// Highways have no sources worth mining
		if (isHighway(name)) continue;
		const sk = isSourceKeeperRoom(name);
		// SK: skip until we can kill keepers — then harvest (higher e/t)
		if (sk && !skOk) continue;
		const inf = get(name);
		let score = inf ? inf.remoteScore : 12;
		if (inf && inf.sourceCount > 0) score += inf.sourceCount * 8;
		else if (!inf) score += 4;
		const dist = Game.map.getRoomLinearDistance(home, name);
		score -= dist * 3; // farther = worse logistics (real cost)
		if (sk) {
			// SK sources ~13.3 e/t each — worth more once combat is paid
			score += 25;
		}
		if (inf && inf.threat > 0 && Game.time - inf.lastSeen < config.remoteThreatCooldown) {
			// Player/invader threat: abandon (not SK keepers)
			if (!sk) continue;
		}
		if (inf && inf.owner && inf.owner !== 'mine') score -= 80;
		if (getRoomIntent(name) === RoomIntent.remote) score += 80;
		if (score > 0) scored.push({ name, score });
	}
	scored.sort((a, b) => b.score - a.score);
	// No slice cap — take every positive-score room
	return scored.map(s => s.name);
}
