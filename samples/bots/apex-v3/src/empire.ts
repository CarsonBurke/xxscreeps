// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * Empire heuristics — dynamic scoring for expansion, remotes, and support assignment.
 *
 * Philosophy:
 *  - Expand the moment free GCL exists (not gated on high RCL).
 *  - Prefer launching from the most *established* colonies.
 *  - Multiple homes can co-support one expansion (claimers / pioneers / military).
 *  - Remotes scale with estimated spare income and spawn bandwidth.
 *  - Combat support is requested when remotes/claims are contested (war.js optional).
 *
 * Writes Memory.empire.plan each tick for spawn / war / construction to read.
 */
const config = require('./config');
const intel = require('./intel');
const {
	ownedRooms, adjacentRoomNames, isHighway, isSourceKeeperRoom,
	energyOf, hostileThreat, estimatePathLength,
} = require('./util');

const PLAN_INTERVAL = 10;

function ensure() {
	Memory.empire ||= {
		attacks: [],
		claims: [],
		forcedRemotes: {},
		ignoreRooms: {},
		campaigns: {},
		economy: {},
		plan: {},
		version: 3,
	};
	return Memory.empire;
}

/**
 * How "established" is a colony? Higher = better base for expansion / remotes / war.
 * Heuristic blend — not a simulation, but ranks rooms usefully.
 */
function colonyStrength(room) {
	if (!room.controller || !room.controller.my) return 0;
	const rcl = room.controller.level || 1;
	const spawns = room.find(FIND_MY_SPAWNS);
	if (!spawns.length) return 0; // unseeded claim — needs support, cannot lead

	let energyCap = 0;
	let energyNow = 0;
	for (const s of room.find(FIND_MY_STRUCTURES)) {
		if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
			energyCap += s.store.getCapacity(RESOURCE_ENERGY) || 0;
			energyNow += energyOf(s.store);
		}
	}
	const storageE = room.storage ? energyOf(room.storage.store) : 0;
	const towers = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_TOWER,
	}).length;
	const threat = hostileThreat(room);
	const { getHome } = require('./creepMem');
	const creeps = Object.keys(Game.creeps).filter(n => {
		const c = Game.creeps[n];
		return getHome(c) === room.name;
	}).length;

	// Weighted score
	let score = 0;
	score += rcl * 100;
	score += Math.min(energyCap, 2000) / 10; // spawn bandwidth proxy
	score += Math.min(storageE, 200_000) / 2000;
	score += towers * 40;
	score += Math.min(creeps, 30) * 3;
	score += spawns.length * 50;
	score -= threat * 80;
	// Downgrade pressure — still usable but weaker
	const td = room.controller.ticksToDowngrade || Infinity;
	if (td < 5000) score -= 50;
	return score;
}

function freeGcl() {
	// Track owned room names across visibility gaps.
	const known = new Set(Memory.empire.ownedRooms || []);
	for (const r of ownedRooms()) known.add(r.name);
	// Drop rooms we can see that are no longer ours.
	for (const name of [ ...known ]) {
		const room = Game.rooms[name];
		if (room && room.controller && !room.controller.my) known.delete(name);
	}
	Memory.empire.ownedRooms = [ ...known ];

	const gcl = Game.gcl ? Game.gcl.level : 1;
	let myCount = 0;
	for (const name of known) {
		const room = Game.rooms[name];
		if (room && room.controller && room.controller.my) {
			myCount++;
			continue;
		}
		if (!room) {
			const inf = intel.get(name);
			// Trust memory until intel proves otherwise
			if (!inf || inf.owner === 'mine' || inf.owner == null) myCount++;
		}
	}
	return Math.max(0, gcl - myCount);
}

/**
 * Score a potential claim target from the perspective of an empire.
 */
function scoreClaimTarget(roomName, fromRooms) {
	if (isHighway(roomName) || isSourceKeeperRoom(roomName)) return -Infinity;
	if (Memory.empire.ignoreRooms && Memory.empire.ignoreRooms[roomName]) return -Infinity;

	const inf = intel.get(roomName);
	let score = 0;

	// Prefer 2 sources
	const sources = inf ? inf.sourceCount : 1;
	score += sources * 50;
	if (sources >= 2) score += 40;

	// Unknown rooms get mild interest so scouts/claims still try
	if (!inf) score += 15;
	if (inf && inf.owner && inf.owner !== 'mine') {
		// Occupied — only if war wants it
		score -= 200;
	}
	if (inf && inf.threat > 0) score -= inf.threat * 15;

	// Distance to nearest strong colony (closer = better logistics)
	let bestDist = Infinity;
	let bestSupport = 0;
	for (const room of fromRooms) {
		const d = Game.map.getRoomLinearDistance(room.name, roomName);
		const str = colonyStrength(room);
		if (d < bestDist || (d === bestDist && str > bestSupport)) {
			bestDist = d;
			bestSupport = str;
		}
		// Prefer adjacency
		if (d === 1) score += 30;
		if (d === 2) score += 10;
	}
	score -= bestDist * 25;
	score += Math.min(bestSupport, 500) / 20;

	// Forced claim flags
	const flagClaim = (Memory.empire.claims || []).some(c => c.room === roomName);
	if (flagClaim) score += 500;

	return score;
}

/**
 * Pick expansion targets when free GCL > 0.
 */
function pickExpansionTargets(colonies, free) {
	if (free <= 0) return [];

	const candidates = new Set();
	// Adjacent + depth-2 from every colony
	for (const room of colonies) {
		for (const a of adjacentRoomNames(room.name)) {
			candidates.add(a);
			for (const b of adjacentRoomNames(a)) candidates.add(b);
		}
	}
	for (const c of Memory.empire.claims || []) candidates.add(c.room);

	const owned = new Set(colonies.map(r => r.name));
	for (const name of Memory.empire.ownedRooms || []) owned.add(name);

	const scored = [];
	for (const name of candidates) {
		if (owned.has(name)) continue;
		const s = scoreClaimTarget(name, colonies);
		if (s > -100) scored.push({ room: name, score: s });
	}
	scored.sort((a, b) => b.score - a.score);
	return scored.slice(0, free);
}

/**
 * Which colonies should support an expansion target?
 * Returns homes sorted by support strength * proximity.
 */
function assignSupport(targetRoom, colonies, need) {
	const ranked = colonies.map(room => {
		const dist = Game.map.getRoomLinearDistance(room.name, targetRoom);
		const str = colonyStrength(room);
		// Closer and stronger wins; distant weak rooms sit out
		const supportScore = str / (1 + dist * 1.5) - dist * 20;
		return { room: room.name, dist, str, supportScore };
	}).filter(r => r.str > 0 && r.supportScore > 0);

	ranked.sort((a, b) => b.supportScore - a.supportScore);
	return ranked.slice(0, need);
}

/**
 * Remote capacity heuristic per colony — without full economy.js.
 * e/t spare ≈ sources*10 * efficiency - upkeep; remotes cost ~15-30 e/t each rough.
 */
/**
 * How many remote *rooms* to attach. No magic 6 / RCL ladder.
 * Return a large number so pickRemotes is the real filter (all positive-score rooms).
 * Spawn duty + EV in spawn.ts decide how many packages we can sustain.
 */
function remoteBudget(room, _strength) {
	if (!room.find(FIND_MY_SPAWNS).length) return 0; // no spawn uptime
	// Unbounded for practical purposes — pickRemotes already ranks/filters
	return 99;
}

/**
 * Military budget signal for war.js (energy we can "risk").
 */
function militaryBudget(room, strength) {
	const storageE = room.storage ? energyOf(room.storage.store) : 0;
	const rcl = room.controller.level;
	// Soft budget scales with strength; RCL1 can still spare a tiny defense force
	return Math.floor(strength * 20 + storageE * 0.05 + rcl * 500);
}

function tick() {
	const empire = ensure();
	if (empire.plan && empire.plan.tick && Game.time - empire.plan.tick < PLAN_INTERVAL) {
		return empire.plan;
	}

	const colonies = ownedRooms().filter(r => r.find(FIND_MY_SPAWNS).length > 0);
	const unseeded = ownedRooms().filter(r => r.find(FIND_MY_SPAWNS).length === 0);
	const free = freeGcl();

	// Strength table
	const strengths = {};
	for (const room of colonies) {
		strengths[room.name] = colonyStrength(room);
	}

	// Expansion: as soon as free GCL, pick targets
	const expansionTargets = pickExpansionTargets(colonies, free);
	const expansions = expansionTargets.map(t => {
		// Need claimer from best home + pioneers from top 1-2 homes
		const support = assignSupport(t.room, colonies, 3);
		const primary = support[0] || null;
		return {
			room: t.room,
			score: t.score,
			primaryHome: primary && primary.room,
			supportHomes: support.map(s => s.room),
			// Early RCL colonies can still claim if they're the only option
			needClaimer: true,
			needPioneers: 2,
			needEscort: t.score < 50 || (intel.get(t.room) && intel.get(t.room).threat > 0),
		};
	});

	// Also finish unseeded claims we already own
	for (const room of unseeded) {
		const support = assignSupport(room.name, colonies, 2);
		expansions.push({
			room: room.name,
			score: 1000, // finish what we started
			primaryHome: support[0] && support[0].room,
			supportHomes: support.map(s => s.room),
			needClaimer: false,
			needPioneers: 2,
			needEscort: hostileThreat(room) > 0,
			unseeded: true,
		});
	}

	// Per-colony plan
	const byColony = {};
	const { RoomIntent, setRoomIntent, getRoomIntent } = require('./intents');
	for (const room of colonies) {
		const str = strengths[room.name];
		const maxRemotes = remoteBudget(room, str);
		// pickRemotes already returns all viable rooms (no artificial cap)
		const remotes = maxRemotes > 0 ? intel.pickRemotes(room) : [];
		// Reassert RoomIntent.remote so flag sync wipe does not drop auto remotes
		for (const r of remotes) {
			const cur = getRoomIntent(r);
			if (
				cur === RoomIntent.attack ||
				cur === RoomIntent.claim ||
				cur === RoomIntent.ignore ||
				cur === RoomIntent.defend
			) continue;
			setRoomIntent(r, RoomIntent.remote);
		}
		// Expansions this room should help
		const help = expansions.filter(e =>
			e.primaryHome === room.name || (e.supportHomes || []).includes(room.name));

		byColony[room.name] = {
			strength: str,
			maxRemotes,
			remotes,
			militaryBudget: militaryBudget(room, str),
			// Spawn priority tilt: if we're the primary expansion home, claimers are urgent
			expandPrimary: expansions.some(e => e.primaryHome === room.name),
			helpExpansions: help.map(e => e.room),
			// Income focus vs expand focus
			mode: help.length && free > 0 ? 'expand' : (str < 200 ? 'bootstrap' : 'grow'),
		};
	}

	// Global intent list for spawn merge (Role / CreepMem numeric keys)
	const { Role, CreepMem, memInit } = require('./creepMem');
	const spawnIntents = [];
	for (const exp of expansions) {
		if (exp.needClaimer && exp.primaryHome) {
			spawnIntents.push({
				home: exp.primaryHome,
				role: Role.claimer,
				priority: 15, // high — GCL is the scarce resource
				memory: memInit({
					[CreepMem.role]: Role.claimer,
					[CreepMem.targetRoom]: exp.room,
				}),
				reason: 'free-gcl-expand',
			});
		}
		for (const home of exp.supportHomes || []) {
			spawnIntents.push({
				home,
				role: Role.builder,
				priority: 18,
				memory: memInit({
					[CreepMem.role]: Role.builder,
					[CreepMem.pioneer]: true,
					[CreepMem.targetRoom]: exp.room,
				}),
				context: {},
				reason: 'pioneer-support',
			});
		}
		if (exp.needEscort && exp.primaryHome) {
			spawnIntents.push({
				home: exp.primaryHome,
				role: Role.attacker,
				priority: 22,
				memory: memInit({
					[CreepMem.role]: Role.attacker,
					[CreepMem.targetRoom]: exp.room,
					[CreepMem.squad]: `exp_${exp.room}`,
				}),
				reason: 'expand-escort',
			});
		}
	}

	const plan = {
		tick: Game.time,
		freeGcl: free,
		strengths,
		expansions,
		byColony,
		spawnIntents,
		// Ranked colonies for "who leads the empire"
		leadColony: colonies.slice().sort((a, b) => strengths[b.name] - strengths[a.name])[0]
			? colonies.slice().sort((a, b) => strengths[b.name] - strengths[a.name])[0].name
			: null,
	};

	empire.plan = plan;
	return plan;
}

function planFor(roomName) {
	const plan = Memory.empire && Memory.empire.plan;
	if (!plan || !plan.byColony) return null;
	return plan.byColony[roomName] || null;
}

function spawnIntentsFor(roomName) {
	const plan = Memory.empire && Memory.empire.plan;
	if (!plan || !plan.spawnIntents) return [];
	return plan.spawnIntents.filter(i => i.home === roomName);
}

module.exports = {
	tick,
	planFor,
	spawnIntentsFor,
	colonyStrength,
	freeGcl,
	scoreClaimTarget,
};

export {};
