/**
 * Accurate max harvest e/t given source positions and spawn-time budget.
 *
 * Physics:
 *   source e/t = energyCapacity / ENERGY_REGEN_TIME  (normally 10)
 *   full mine  = ceil(e/t / HARVEST_POWER) WORK      (normally 5)
 *
 * Logistics package for one source (projections):
 *   static harvester body + hauler(s) with carry ≈ 2 * pathLen * e/t / 50
 *   packageDuty = Σ (bodyParts * CREEP_SPAWN_TIME / CREEP_LIFE_TIME)
 *
 * Spawn constraint:
 *   Σ packageDuty ≤ nSpawns  (each spawn sustains duty 1.0 continuously)
 *
 * Ceiling = greedy select packages by (e/t ÷ duty) until spawn budget exhausted.
 * This is the bar we compare realized harvestRate against — not a fixed 40.
 */
// @ts-nocheck
/* eslint-disable */
const P = (function loadP() {
	const mod = require('./projections');
	return mod.CONST ? mod : (mod.default || mod);
})();
const { estimatePathLength, adjacentRoomNames, isHighway, isSourceKeeperRoom } = require('./util');
const intel = require('./intel');

const HARVEST_POWER = (P.CONST && P.CONST.HARVEST_POWER) || 2;
const WORK_FULL = (P.CONST && P.CONST.STATIC_HARVESTER_WORK) || 5;
const LIFE = (P.CONST && P.CONST.CREEP_LIFE_TIME) || 1500;
const SPAWN_T = (P.CONST && P.CONST.CREEP_SPAWN_TIME) || 3;
const CARRY = (P.CONST && P.CONST.CARRY_CAPACITY) || 50;
const WASTE = (P.CONST && P.CONST.TRAVEL_WASTE_FACTOR) || 0.05;

function sourceEt(source) {
	const cap = source.energyCapacity || 3000;
	const regen = (typeof ENERGY_REGEN_TIME !== 'undefined' && ENERGY_REGEN_TIME) || 300;
	return cap / regen;
}

function dutyOfParts(n) {
	return (n * SPAWN_T) / LIFE;
}

/**
 * Build a logistics package for one source.
 * @param {object} opts
 * @param {string} opts.id
 * @param {string} opts.roomName
 * @param {number} opts.pathLen one-way tiles spawn/storage ↔ source
 * @param {number} opts.ePerTick
 * @param {boolean} [opts.remote]
 * @param {boolean} [opts.roads]
 * @param {number} [opts.workParts] capacity-limited WORK (default full)
 * @param {number} [opts.reserverDutyShare] remote room reserver duty split across sources
 */
function packageForSource(opts) {
	const ePerTick = opts.ePerTick;
	const pathLen = Math.max(1, opts.pathLen || 1);
	const roads = opts.roads !== false && !opts.remote;
	const needWork = Math.ceil(ePerTick / HARVEST_POWER);
	const workParts = opts.workParts != null ? opts.workParts : needWork;
	const harvestEt = Math.min(ePerTick, workParts * HARVEST_POWER);

	const harv = P.estimateHarvester({ pathLen, roads, workParts });
	const haul = P.estimateHauler(pathLen, harvestEt, { roads: roads || !opts.remote ? roads : false });
	// Remotes: off-road unless we know roads
	const haulFinal = opts.remote && opts.roads !== true
		? P.estimateHauler(pathLen, harvestEt, { roads: false })
		: haul;

	const minerDuty = harv.spawnBusyFrac;
	const haulDuty = haulFinal.spawnBusyFrac;
	const reserverDuty = opts.reserverDutyShare || 0;
	const packageDuty = minerDuty + haulDuty + reserverDuty;

	const deliveredEt = harvestEt * (opts.remote ? (1 - WASTE) : 1);
	const upkeepEt = harv.upkeepEt + haulFinal.upkeepEt + (opts.reserverUpkeepShare || 0);

	return {
		id: opts.id,
		roomName: opts.roomName,
		remote: !!opts.remote,
		pathLen,
		ePerTick,
		workParts,
		needWork,
		harvestEt,
		deliveredEt,
		harv,
		haul: haulFinal,
		minerDuty,
		haulDuty,
		reserverDuty,
		packageDuty,
		upkeepEt,
		/** e/t delivered per unit spawn duty — ranking key */
		efficiency: packageDuty > 0 ? deliveredEt / packageDuty : 0,
		haulerCount: haulFinal.haulerCount,
		carryPartsNeed: haulFinal.carryPartsNeed,
	};
}

/**
 * Enumerate local + candidate remote sources for ceiling estimate.
 * Remotes: intel-known adjacent (or 2-hop if RCL high) non-owned rooms with sources.
 */
function enumerateCandidates(homeRoom, opts = {}) {
	const spawns = homeRoom.find(FIND_MY_SPAWNS);
	if (!spawns.length) return [];
	const spawnPos = spawns[0].pos;
	const home = homeRoom.name;
	const rcl = homeRoom.controller ? homeRoom.controller.level : 1;
	const out = [];

	// --- Local ---
	for (const source of homeRoom.find(FIND_SOURCES)) {
		const pathLen = Math.max(1, estimatePathLength(spawnPos, source.pos));
		out.push(packageForSource({
			id: source.id,
			roomName: home,
			pathLen,
			ePerTick: sourceEt(source),
			remote: false,
			roads: true,
			workParts: opts.localWorkParts,
		}));
	}

	// --- Remote candidates (for ceiling — not only currently remoted) ---
	if (opts.includeRemotes === false) return out;

	const rooms = new Set(adjacentRoomNames(home));
	// Always include 2-hop: distance scored in packages, spawn duty filters — not RCL
	if (opts.deepRemotes !== false) {
		for (const a of adjacentRoomNames(home)) {
			for (const b of adjacentRoomNames(a)) {
				if (b !== home) rooms.add(b);
			}
		}
	}

	const remoteSourcesByRoom = {}; // room -> packages (for reserver split)

	for (const roomName of rooms) {
		if (roomName === home) continue;
		if (isHighway(roomName)) continue;
		// SK optional: include with lower efficiency via higher work
		const sk = isSourceKeeperRoom(roomName);
		if (sk && !opts.includeSk) continue;

		const owned = Game.rooms[roomName];
		if (owned && owned.controller && owned.controller.my) continue;

		const inf = intel.get && intel.get(roomName);
		let sourceList = [];
		if (owned) {
			sourceList = owned.find(FIND_SOURCES).map(s => ({
				id: s.id,
				pos: s.pos,
				ePerTick: sourceEt(s),
			}));
		} else if (inf && inf.sources && inf.sources.length) {
			sourceList = inf.sources.map(s => ({
				id: s.id,
				pos: s.pos ? new RoomPosition(s.pos.x, s.pos.y, s.pos.roomName || roomName) : null,
				ePerTick: (s.energyCapacity || 3000) / 300,
			}));
		} else {
			// Unknown: assume 1 source, path ≈ room dist * 50 + 25 (conservative)
			const d = Game.map.getRoomLinearDistance(home, roomName);
			sourceList = [{
				id: `${roomName}:unk0`,
				pos: null,
				ePerTick: 10,
				pathLenHint: d * 50 + 40,
			}];
		}

		if (!sourceList.length) continue;

		const reserver = P.estimateReserver
			? P.estimateReserver({ claimParts: 1, pathLen: Game.map.getRoomLinearDistance(home, roomName) * 50 })
			: { spawnBusyFrac: (4 * SPAWN_T) / 600, upkeepEt: 650 / 600 };
		const n = sourceList.length;
		const reserverDutyShare = (reserver.spawnBusyFrac || 0) / n;
		const reserverUpkeepShare = (reserver.upkeepEt || 0) / n;

		for (const s of sourceList) {
			let pathLen = s.pathLenHint;
			if (s.pos) {
				pathLen = Math.max(1, estimatePathLength(spawnPos, s.pos));
			} else if (pathLen == null) {
				pathLen = Game.map.getRoomLinearDistance(home, roomName) * 50 + 40;
			}
			const pkg = packageForSource({
				id: s.id,
				roomName,
				pathLen,
				ePerTick: s.ePerTick || 10,
				remote: true,
				roads: false,
				// SK ~13.33 e/t → ceil(13.33/2)=7 WORK; normal 10 e/t → 5
				workParts: sk ? 7 : WORK_FULL,
				reserverDutyShare,
				reserverUpkeepShare,
			});
			out.push(pkg);
			if (!remoteSourcesByRoom[roomName]) remoteSourcesByRoom[roomName] = [];
			remoteSourcesByRoom[roomName].push(pkg);
		}
	}

	return out;
}

/**
 * Greedy pack packages under spawn duty budget.
 * With leftover duty, fund a fractional package (miner+haul scaled together).
 */
function packUnderSpawnBudget(packages, nSpawns) {
	const spawnBudget = Math.max(0, nSpawns);
	const ranked = packages.slice().sort((a, b) => b.efficiency - a.efficiency);

	let duty = 0;
	let harvestEt = 0;
	let deliveredEt = 0;
	let upkeepEt = 0;
	const chosen = [];

	for (const p of ranked) {
		if (p.packageDuty <= 0) continue;
		if (duty + p.packageDuty <= spawnBudget + 1e-9) {
			duty += p.packageDuty;
			harvestEt += p.harvestEt;
			deliveredEt += p.deliveredEt;
			upkeepEt += p.upkeepEt;
			chosen.push({
				id: p.id,
				roomName: p.roomName,
				remote: p.remote,
				pathLen: p.pathLen,
				ePerTick: p.ePerTick,
				deliveredEt: p.deliveredEt,
				packageDuty: p.packageDuty,
				efficiency: p.efficiency,
				fraction: 1,
			});
			continue;
		}
		// Partial package with remaining duty; else try next cheaper package
		const rem = spawnBudget - duty;
		if (rem > 1e-6 && rem / p.packageDuty > 0.15) {
			const frac = rem / p.packageDuty;
			duty += rem;
			harvestEt += p.harvestEt * frac;
			deliveredEt += p.deliveredEt * frac;
			upkeepEt += p.upkeepEt * frac;
			chosen.push({
				id: p.id,
				roomName: p.roomName,
				remote: p.remote,
				pathLen: p.pathLen,
				ePerTick: p.ePerTick,
				deliveredEt: p.deliveredEt * frac,
				packageDuty: rem,
				efficiency: p.efficiency,
				fraction: frac,
			});
			break; // budget filled by partial
		}
		// remaining duty too small for this package — try lower-duty candidates
		if (rem <= 1e-6) break;
		continue;
	}

	const physicsEt = packages.reduce((a, p) => a + p.ePerTick, 0);
	const allDuty = packages.reduce((a, p) => a + p.packageDuty, 0);

	return {
		/** Sum of e/t of every candidate source (no spawn limit) */
		maxEtPhysics: physicsEt,
		/** Harvest e/t if every candidate fully staffed (same as physics for full WORK) */
		maxEtIfInfiniteSpawn: packages.reduce((a, p) => a + p.harvestEt, 0),
		/** Delivered e/t sustainable under spawn budget (THE ceiling) */
		maxEtSpawnBound: deliveredEt,
		/** Raw harvest under spawn budget (before remote waste) */
		maxHarvestEtSpawnBound: harvestEt,
		upkeepEt,
		netEt: deliveredEt - upkeepEt,
		dutyUsed: duty,
		spawnBudget,
		spareDuty: Math.max(0, spawnBudget - duty),
		/** Duty if we tried to run every candidate */
		dutyIfAll: allDuty,
		/** How many spawns needed to run all candidates continuously */
		spawnsToTakeAll: allDuty,
		chosen,
		ranked: ranked.map(p => ({
			id: p.id,
			room: p.roomName,
			remote: p.remote,
			pathLen: p.pathLen,
			ePerTick: p.ePerTick,
			deliveredEt: p.deliveredEt,
			duty: p.packageDuty,
			eff: p.efficiency,
			haulers: p.haulerCount,
		})),
	};
}

/**
 * Full ceiling estimate for a home room.
 * @param {Room} homeRoom
 * @param {object} [opts]
 * @param {number} [opts.nSpawns] override spawn count
 * @param {boolean} [opts.includeRemotes=true]
 * @param {boolean} [opts.includeSk=false]
 * @param {number} [opts.localWorkParts] cap WORK by energy capacity if set
 */
function estimateHarvestCeiling(homeRoom, opts = {}) {
	const nSpawns = opts.nSpawns != null
		? opts.nSpawns
		: homeRoom.find(FIND_MY_SPAWNS).length;
	if (!nSpawns) {
		return {
			maxEtPhysics: 0,
			maxEtSpawnBound: 0,
			maxHarvestEtSpawnBound: 0,
			spawnBudget: 0,
			dutyUsed: 0,
			spareDuty: 0,
			spawnsToTakeAll: 0,
			chosen: [],
			ranked: [],
			localEt: 0,
			remoteEt: 0,
			packages: [],
		};
	}

	const packages = enumerateCandidates(homeRoom, {
		includeRemotes: opts.includeRemotes !== false,
		includeSk: !!opts.includeSk,
		deepRemotes: !!opts.deepRemotes,
		localWorkParts: opts.localWorkParts,
	});

	const packed = packUnderSpawnBudget(packages, nSpawns);
	let localEt = 0;
	let remoteEt = 0;
	for (const c of packed.chosen) {
		if (c.remote) remoteEt += c.deliveredEt;
		else localEt += c.deliveredEt;
	}

	return {
		...packed,
		localEt,
		remoteEt,
		packageCount: packages.length,
		packages,
		nSpawns,
		/** Human summary */
		summary: {
			physics: +packed.maxEtPhysics.toFixed(2),
			spawnBound: +packed.maxEtSpawnBound.toFixed(2),
			local: +localEt.toFixed(2),
			remote: +remoteEt.toFixed(2),
			duty: `${packed.dutyUsed.toFixed(2)}/${packed.spawnBudget}`,
			spawnsForAll: +packed.spawnsToTakeAll.toFixed(2),
		},
	};
}

/**
 * Write ceiling + gap vs realized into Memory.empire.harvestCeiling[room]
 * and return the estimate.
 */
function updateHarvestCeiling(homeRoom, realizedEt, opts) {
	const est = estimateHarvestCeiling(homeRoom, opts);
	if (!Memory.empire) Memory.empire = {};
	Memory.empire.harvestCeiling = Memory.empire.harvestCeiling || {};
	Memory.empire.harvestCeiling[homeRoom.name] = {
		t: Game.time,
		maxEtPhysics: +est.maxEtPhysics.toFixed(2),
		maxEtSpawnBound: +est.maxEtSpawnBound.toFixed(2),
		maxHarvestEtSpawnBound: +est.maxHarvestEtSpawnBound.toFixed(2),
		localEt: +est.localEt.toFixed(2),
		remoteEt: +est.remoteEt.toFixed(2),
		realizedEt: realizedEt != null ? +Number(realizedEt).toFixed(2) : null,
		gapEt: realizedEt != null
			? +(est.maxEtSpawnBound - realizedEt).toFixed(2)
			: null,
		efficiency: realizedEt != null && est.maxEtSpawnBound > 0
			? +(realizedEt / est.maxEtSpawnBound).toFixed(3)
			: null,
		dutyUsed: +est.dutyUsed.toFixed(3),
		spawnBudget: est.spawnBudget,
		spareDuty: +est.spareDuty.toFixed(3),
		spawnsToTakeAll: +est.spawnsToTakeAll.toFixed(2),
		chosen: est.chosen,
		// top efficiency ranking (debug)
		top: (est.ranked || []).slice(0, 8),
	};
	return est;
}

module.exports = {
	estimateHarvestCeiling,
	updateHarvestCeiling,
	enumerateCandidates,
	packageForSource,
	packUnderSpawnBudget,
	HARVEST_POWER,
	WORK_FULL,
};

export {};
