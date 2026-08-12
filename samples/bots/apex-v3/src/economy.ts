// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * Apex v3 — economy projections API + Memory.empire.economy writer.
 *
 * Pure-ish math lives in projections.js; this module:
 *   - exposes the public estimate* / projectColony API
 *   - walks live rooms when available
 *   - writes Memory.empire.economy each tick for spawn/combat consumers
 *
 * v3 roles (projections only — no AI here):
 *   static harvesters (drop/container), haulers, fillers, upgraders, builders
 *
 * Usage:
 *   const economy = require('./economy');
 *   economy.tick(colonies); // colonies: Room[] or { room, remotesIntel }[]
 *
 * Memory.empire.economy = {
 *   tick, incomeEt, maxRemotes, remotes: [...], spawnBusyPct, militaryBudgetEt,
 *   colonies: { [roomName]: projectColony result summary }
 * }
 */

'use strict';

// Flat sandbox: require('./projections'). Host Node may treat files as ESM — normalize.
const P = (function loadProjections() {
	const mod = require('./projections');
	return mod.CONST ? mod : (mod.default || mod);
})();
const { Role } = require('./creepMem');
const { getRemotes } = require('./roomMem');

const CONST = P.CONST || {};
const estimateSource = P.estimateSource;
const estimateHarvester = P.estimateHarvester;
const estimateHauler = P.estimateHauler;
const estimateReserver = P.estimateReserver;
const estimateRemotePackage = P.estimateRemotePackage;
const affordRemotes = P.affordRemotes;
const estimateStaffing = P.estimateStaffing;
const estimateLocalSources = P.estimateLocalSources;
const estimateRoomPathLen = P.estimateRoomPathLen;
const bodyCost = P.bodyCost;
const spawnTime = P.spawnTime;

// Re-export spawn priorities / constants for spawn.js & combat.js
const SPAWN_PRIORITY = CONST.SPAWN_PRIORITY || [];

// ---------------------------------------------------------------------------
// Live room helpers (safe when Game/room missing)
// ---------------------------------------------------------------------------

function extensionCount(room) {
	if (!room || !room.find) return 0;
	try {
		return room.find(FIND_MY_STRUCTURES, {
			filter: s => s.structureType === STRUCTURE_EXTENSION,
		}).length;
	} catch (_e) {
		return 0;
	}
}

function spawnCount(room) {
	if (!room || !room.find) return 1;
	try {
		const spawns = room.find(FIND_MY_SPAWNS);
		return Math.max(1, spawns.length);
	} catch (_e) {
		return 1;
	}
}

function spawnCapacity(room) {
	if (!room || !room.find) return 300;
	let cap = 0;
	try {
		const structs = room.find(FIND_MY_STRUCTURES);
		for (let i = 0; i < structs.length; i++) {
			const s = structs[i];
			if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
				if (s.store && typeof s.store.getCapacity === 'function') {
					cap += s.store.getCapacity(RESOURCE_ENERGY) || 0;
				} else if (s.energyCapacity != null) {
					cap += s.energyCapacity;
				}
			}
		}
	} catch (_e) {
		return 300;
	}
	return cap || 300;
}

function constructionSiteCount(room) {
	if (!room || !room.find) return 0;
	try {
		return room.find(FIND_CONSTRUCTION_SITES).length;
	} catch (_e) {
		return 0;
	}
}

function storageEnergy(room) {
	if (!room) return 0;
	try {
		if (room.storage && room.storage.store) {
			if (typeof room.storage.store.getUsedCapacity === 'function') {
				return room.storage.store.getUsedCapacity(RESOURCE_ENERGY) || 0;
			}
			return room.storage.store[RESOURCE_ENERGY] || 0;
		}
	} catch (_e) { /* ignore */ }
	return 0;
}

/**
 * Path length from room anchor (storage or first spawn) to a source.
 */
function pathLenToSource(room, source) {
	if (!room || !source || !source.pos) return 15;
	let from = null;
	if (room.storage) from = room.storage.pos;
	else {
		try {
			const spawns = room.find(FIND_MY_SPAWNS);
			if (spawns[0]) from = spawns[0].pos;
		} catch (_e) { /* ignore */ }
	}
	if (!from) return 15;

	// Prefer PathFinder / findPath when same room
	try {
		if (from.roomName === source.pos.roomName && typeof from.findPathTo === 'function') {
			const path = from.findPathTo(source.pos, { ignoreCreeps: true, maxOps: 2000 });
			if (path && path.length) return path.length;
		}
		if (typeof from.getRangeTo === 'function') {
			return Math.max(1, from.getRangeTo(source.pos));
		}
	} catch (_e) { /* ignore */ }
	return 15;
}

/**
 * Build local source descriptors from a live room.
 */
function localSourcesFromRoom(room) {
	if (!room || !room.find) {
		return [ { pathLen: 15 }, { pathLen: 20 } ]; // assume dual-source offline
	}
	try {
		const sources = room.find(FIND_SOURCES);
		return sources.map(s => ({
			id: s.id,
			pathLen: pathLenToSource(room, s),
			energyCapacity: s.energyCapacity,
			sk: false,
		}));
	} catch (_e) {
		return [ { pathLen: 15 } ];
	}
}

/**
 * Normalize remotesIntel into package inputs.
 *
 * remotesIntel shapes accepted:
 *   - string[] room names
 *   - { roomName, sources: [{id, pathLen}] }[]
 *   - { [roomName]: { sources: [...] } }
 */
function normalizeRemotesIntel(homeRoomName, remotesIntel) {
	if (!remotesIntel) return [];

	const out = [];

	if (Array.isArray(remotesIntel)) {
		for (const item of remotesIntel) {
			if (typeof item === 'string') {
				out.push({
					remoteRoomName: item,
					sources: sourcesForRemote(homeRoomName, item, null),
				});
			} else if (item && typeof item === 'object') {
				const name = item.roomName || item.remoteRoomName || item.name;
				if (!name) continue;
				out.push({
					remoteRoomName: name,
					sources: sourcesForRemote(homeRoomName, name, item.sources),
				});
			}
		}
		return out;
	}

	if (typeof remotesIntel === 'object') {
		for (const name of Object.keys(remotesIntel)) {
			const entry = remotesIntel[name] || {};
			out.push({
				remoteRoomName: name,
				sources: sourcesForRemote(homeRoomName, name, entry.sources),
			});
		}
	}
	return out;
}

function sourcesForRemote(homeRoomName, remoteRoomName, sources) {
	if (Array.isArray(sources) && sources.length) {
		return sources.map(s => ({
			id: s.id || null,
			pathLen: s.pathLen != null
				? s.pathLen
				: estimateRoomPathLen(homeRoomName, remoteRoomName) + 25,
			sk: !!s.sk,
		}));
	}

	// Live visibility
	if (typeof Game !== 'undefined' && Game.rooms && Game.rooms[remoteRoomName]) {
		const room = Game.rooms[remoteRoomName];
		try {
			const found = room.find(FIND_SOURCES);
			const base = estimateRoomPathLen(homeRoomName, remoteRoomName);
			return found.map(s => ({
				id: s.id,
				pathLen: base + 25,
				energyCapacity: s.energyCapacity,
				sk: false,
			}));
		} catch (_e) { /* fall through */ }
	}

	// Intel memory fallback
	if (typeof Memory !== 'undefined' && Memory.intel && Memory.intel[remoteRoomName]) {
		const info = Memory.intel[remoteRoomName];
		const n = info.sourceCount || info.sources || 1;
		const count = typeof n === 'number' ? n : (Array.isArray(n) ? n.length : 1);
		const base = estimateRoomPathLen(homeRoomName, remoteRoomName);
		const list = [];
		const srcArr = Array.isArray(info.sources) ? info.sources : null;
		for (let i = 0; i < Math.max(1, count); i++) {
			const s = srcArr && srcArr[i];
			list.push({
				id: s && s.id || null,
				pathLen: (s && s.pathLen != null) ? s.pathLen : base + 25,
				sk: !!(s && s.sk),
			});
		}
		return list;
	}

	// Unknown: assume 1 source at room-path + 25
	const base = estimateRoomPathLen(homeRoomName, remoteRoomName);
	return [ { id: null, pathLen: base + 25, sk: false } ];
}

function freeCpuRough() {
	if (typeof Game === 'undefined' || !Game.cpu) return null;
	try {
		const limit = Game.cpu.limit || 20;
		const used = Game.cpu.getUsed ? Game.cpu.getUsed() : 0;
		// Remaining budget this tick is not the same as free for economy — use bucket headroom proxy
		const bucket = Game.cpu.bucket != null ? Game.cpu.bucket : 10000;
		if (bucket < 1000) return 0;
		// Rough spare ms/tick for remotes
		return Math.max(0, limit - used - 4);
	} catch (_e) {
		return null;
	}
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Project one colony's economy.
 *
 * @param {Room|object} room  live Room or { name, controller: { level }, ... }
 * @param {object|array} remotesIntel
 * @returns {{
 *   roomName, rcl, incomeEt, upkeepEt, netEt, maxRemotes,
 *   packages, local, spawnBusyPct, militaryBudgetEt, recommendations, afford
 * }}
 */
function projectColony(room, remotesIntel) {
	const roomName = (room && room.name) || (room && room.roomName) || 'unknown';
	const rcl = (room && room.controller && room.controller.level) || (room && room.rcl) || 1;
	const spawns = spawnCount(room);
	const ext = extensionCount(room);
	const sites = constructionSiteCount(room);
	const stor = storageEnergy(room);
	const cap = spawnCapacity(room);

	// Local sources
	const localDesc = localSourcesFromRoom(room);
	const local = estimateLocalSources(localDesc, { roads: true });

	// Remote packages
	const remoteSpecs = normalizeRemotesIntel(roomName, remotesIntel);
	const packages = [];
	for (const spec of remoteSpecs) {
		const pkg = estimateRemotePackage(
			roomName,
			spec.remoteRoomName,
			spec.sources,
			{ roads: rcl >= 4 },
		);
		packages.push(pkg);
	}

	// Sort candidates by net e/t descending for affordability pass
	const ranked = packages.slice().sort((a, b) => (b.netEt || 0) - (a.netEt || 0));

	const afford = affordRemotes({
		colonyIncomeEt: local.netEt,
		spawnCount: spawns,
		rcl,
		freeCpu: freeCpuRough(),
		localSpawnBusyFrac: local.spawnBusyFrac,
		candidatePackages: ranked,
		maxRemotes: CONST.MAX_REMOTES_SOFT || 99,
	});

	const afforded = afford.afforded || [];
	let remoteIncomeEt = 0;
	let remoteUpkeepEt = 0;
	let remoteSpawnBusy = 0;
	for (const pkg of afforded) {
		remoteIncomeEt += pkg.deliveredEt || 0;
		remoteUpkeepEt += pkg.upkeepEt || 0;
		remoteSpawnBusy += pkg.spawnBusyFrac || 0;
	}

	const incomeEt = local.incomeEt + remoteIncomeEt;
	const upkeepEt = local.upkeepEt + remoteUpkeepEt;
	const spawnBusyFrac = local.spawnBusyFrac + remoteSpawnBusy;
	const spawnBusyPct = (spawnBusyFrac / Math.max(1, spawns)) * 100;

	const staffing = estimateStaffing({
		rcl,
		extensionCount: ext,
		incomeEt,
		upkeepEt,
		remoteCommitEt: remoteUpkeepEt,
		constructionSites: sites,
		storageEnergy: stor,
		spawnCapacity: cap,
	});

	// Merge recommendations: economy roles first (omit zero counts)
	const recommendations = [];

	if (local.harvesterCount > 0) {
		recommendations.push({
			role: Role.harvester,
			count: local.harvesterCount,
			reason: `static miner per local source (${local.sourceCount})`,
		});
	}
	if (local.haulerCount > 0) {
		recommendations.push({
			role: Role.hauler,
			count: local.haulerCount,
			reason: `local pipeline haulers for ${local.sourceCount} sources`,
		});
	}

	let remoteHarvesters = 0;
	let remoteHaulers = 0;
	let reservers = 0;
	for (const pkg of afforded) {
		remoteHarvesters += pkg.harvesterCount || 0;
		remoteHaulers += pkg.haulerCount || 0;
		reservers += 1;
	}
	if (remoteHarvesters > 0) {
		recommendations.push({
			role: Role.remoteHarvester,
			count: remoteHarvesters,
			reason: `1 per source across ${afforded.length} remotes`,
		});
	}
	if (remoteHaulers > 0) {
		recommendations.push({
			role: Role.remoteHauler,
			count: remoteHaulers,
			reason: 'pathLen × e/t pipeline demand',
		});
	}
	if (reservers > 0) {
		recommendations.push({
			role: Role.reserver,
			count: reservers,
			reason: '1 CLAIM upkeep per remote room',
		});
	}

	for (const r of staffing.recommendations) {
		if (r && r.count > 0) recommendations.push(r);
	}

	return {
		roomName,
		rcl,
		spawnCount: spawns,
		incomeEt,
		upkeepEt,
		netEt: incomeEt - upkeepEt,
		localIncomeEt: local.incomeEt,
		remoteIncomeEt,
		remoteUpkeepEt,
		maxRemotes: afford.maxRemotes,
		affordReason: afford.reason,
		affordDetail: afford.detail,
		packages: afforded,
		allPackages: packages,
		local,
		spawnBusyPct,
		spawnBusyFrac,
		militaryBudgetEt: staffing.militaryBudgetEt,
		surplusEt: staffing.surplusEt,
		staffing,
		recommendations,
		afford,
	};
}

/**
 * Empire tick: project all colonies and store Memory.empire.economy.
 *
 * @param {Array<Room|{room, remotesIntel}>} colonies
 * @param {object} [opts]
 * @param {object|array} [opts.remotesIntel]  default remotes for all if colony lacks own
 */
function tick(colonies, opts = {}) {
	const list = normalizeColoniesArg(colonies);
	const tickNow = (typeof Game !== 'undefined' && Game.time != null) ? Game.time : 0;

	let incomeEt = 0;
	let maxRemotes = 0;
	let spawnBusyPct = 0;
	let militaryBudgetEt = 0;
	const remotes = [];
	const colonyMap = {};

	for (const entry of list) {
		const room = entry.room;
		const remotesIntel = entry.remotesIntel != null ? entry.remotesIntel : opts.remotesIntel;
		const proj = projectColony(room, remotesIntel);

		incomeEt += proj.incomeEt;
		maxRemotes += proj.maxRemotes;
		militaryBudgetEt += proj.militaryBudgetEt || 0;
		// Weighted / max spawn busy across colonies (report empire max)
		if (proj.spawnBusyPct > spawnBusyPct) spawnBusyPct = proj.spawnBusyPct;

		for (const pkg of proj.packages) {
			remotes.push({
				home: proj.roomName,
				remote: pkg.remoteRoomName,
				deliveredEt: pkg.deliveredEt,
				upkeepEt: pkg.upkeepEt,
				netEt: pkg.netEt,
				spawnBusyPct: pkg.spawnBusyPct,
				sources: pkg.sourceCount,
				haulers: pkg.haulerCount,
			});
		}

		colonyMap[proj.roomName] = {
			incomeEt: proj.incomeEt,
			upkeepEt: proj.upkeepEt,
			netEt: proj.netEt,
			maxRemotes: proj.maxRemotes,
			spawnBusyPct: proj.spawnBusyPct,
			militaryBudgetEt: proj.militaryBudgetEt,
			recommendations: proj.recommendations,
			affordReason: proj.affordReason,
		};
	}

	const snapshot = {
		tick: tickNow,
		incomeEt,
		maxRemotes,
		remotes,
		spawnBusyPct,
		militaryBudgetEt,
		// Primary map (empire.js / spawn also accept byColony alias)
		colonies: colonyMap,
		byColony: colonyMap,
		// Flatten recommendations for spawn.js optional merge
		recommendations: Object.keys(colonyMap).flatMap(home => {
			const recs = colonyMap[home].recommendations || [];
			return recs.map(r => ({ home, ...r }));
		}),
	};

	// Persist for spawn / combat
	if (typeof Memory !== 'undefined') {
		Memory.empire = Memory.empire || {};
		Memory.empire.economy = snapshot;
	}

	return snapshot;
}

function normalizeColoniesArg(colonies) {
	if (!colonies) {
		// Auto-discover owned rooms when running in-game
		if (typeof Game !== 'undefined' && Game.rooms) {
			const rooms = [];
			for (const name in Game.rooms) {
				const room = Game.rooms[name];
				if (room.controller && room.controller.my) {
					rooms.push({ room, remotesIntel: remotesFromMemory(room) });
				}
			}
			return rooms;
		}
		return [];
	}

	if (!Array.isArray(colonies)) {
		// Single room
		if (colonies.room) return [ colonies ];
		return [ { room: colonies, remotesIntel: remotesFromMemory(colonies) } ];
	}

	return colonies.map(c => {
		if (c && c.room) return { room: c.room, remotesIntel: c.remotesIntel };
		return { room: c, remotesIntel: c && c.remotesIntel != null ? c.remotesIntel : remotesFromMemory(c) };
	});
}

/**
 * Pull remote list from common Apex memory shapes.
 */
function remotesFromMemory(room) {
	if (!room || typeof Memory === 'undefined') return [];
	const name = room.name;
	try {
		// RoomApexMem.remotes via room bag `a`
		const fromBag = getRemotes(name);
		if (fromBag.length) return fromBag;
		// empire plan cache
		if (Memory.empire && Memory.empire.plan && Memory.empire.plan.byColony && Memory.empire.plan.byColony[name]) {
			const c = Memory.empire.plan.byColony[name];
			if (c.remotes) return c.remotes;
		}
	} catch (_e) { /* ignore */ }
	return [];
}

/**
 * Read last projection (spawn/combat convenience).
 */
function last() {
	if (typeof Memory === 'undefined' || !Memory.empire) return null;
	return Memory.empire.economy || null;
}

/**
 * Military budget for a room from last snapshot, or 0.
 */
function militaryBudgetFor(roomName) {
	const snap = last();
	if (!snap) return 0;
	if (roomName && snap.colonies && snap.colonies[roomName]) {
		return snap.colonies[roomName].militaryBudgetEt || 0;
	}
	return snap.militaryBudgetEt || 0;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
	// Core projections (API contract)
	estimateSource,
	estimateHauler,
	estimateRemotePackage,
	projectColony,

	// Extended projections
	estimateHarvester,
	estimateReserver,
	estimateStaffing,
	estimateLocalSources,
	affordRemotes,
	estimateRoomPathLen,

	// Tick / memory
	tick,
	last,
	militaryBudgetFor,

	// Constants for spawn/combat
	SPAWN_PRIORITY,
	CONST,
	bodyCost,
	spawnTime,

	// projections module (escape hatch)
	projections: P,
};

export {};
