/**
 * Apex v3 — pure economy projection math.
 *
 * No Memory writes, no Game side-effects beyond optional map distance reads.
 * Formulas documented in ECONOMY.md and inline.
 *
 * Screeps constants used with fallbacks so unit tests / offline require() work:
 *   ENERGY_REGEN_TIME=300, SOURCE_ENERGY_CAPACITY=3000, CREEP_SPAWN_TIME=3,
 *   CREEP_LIFE_TIME=1500, CREEP_CLAIM_LIFE_TIME=600, MAX_CREEP_SIZE=50,
 *   CARRY_CAPACITY=50, HARVEST_POWER=2, BODYPART_COST, ERR_NO_PATH.
 */

'use strict';

// ---------------------------------------------------------------------------
// Constants (mirror Screeps; overridable via exports.CONST for tests)
// ---------------------------------------------------------------------------

const BODYPART_COST_FALLBACK = {
	move: 50,
	work: 100,
	carry: 50,
	attack: 80,
	ranged_attack: 150,
	heal: 250,
	claim: 600,
	tough: 10,
};

const CONST = {
	/** Normal owned/neutral source: 3000 energy / 300 tick regen = 10 e/t. */
	SOURCE_ENERGY_CAPACITY: 3000,
	ENERGY_REGEN_TIME: 300,
	SOURCE_ET: 10,

	/**
	 * Source Keeper sources are larger (typically 4000) → ~13.33 e/t,
	 * but require combat upkeep and higher travel risk. Not defaulted into
	 * remote packages unless caller marks sources as SK.
	 */
	SK_SOURCE_ENERGY_CAPACITY: 4000,
	SK_SOURCE_ET: 4000 / 300, // ≈ 13.333

	HARVEST_POWER: 2, // energy/tick per WORK while harvesting
	/** WORK parts needed to fully mine a normal 10 e/t source. */
	STATIC_HARVESTER_WORK: 5,
	CARRY_CAPACITY: 50,
	CREEP_SPAWN_TIME: 3,
	CREEP_LIFE_TIME: 1500,
	CREEP_CLAIM_LIFE_TIME: 600,
	MAX_CREEP_SIZE: 50,

	/** Path length fallback: linear rooms * this + ROOM_PATH_BIAS. */
	ROOM_PATH_TILES: 50,
	ROOM_PATH_BIAS: 20,

	/** Fraction of pipeline energy lost to travel / drop waste (rough). */
	TRAVEL_WASTE_FACTOR: 0.05,

	/** Max CARRY parts we put on one hauler body (leave room for MOVE). */
	MAX_HAULER_CARRY: 32,
	MIN_HAULER_CARRY: 2,

	/** Fraction of energy surplus reserved for military after economy needs. */
	MILITARY_BUDGET_FACTOR: 0.35,

	/** RCL gate before remotes are considered affordable. */
	REMOTE_MIN_RCL: 3,
	/** Soft cap remotes per colony (spawn / CPU sanity). */
	MAX_REMOTES_SOFT: 6,

	/** Spawn priority order for consumers (spawn.js, combat). Lower index = higher prio. */
	SPAWN_PRIORITY: [
		'bootstrap',
		'defender',
		'harvester',
		'filler',
		'hauler',
		'remoteHarvester',
		'remoteHauler',
		'reserver',
		'upgrader',
		'builder',
		'repairer',
		'scout',
		'claimer',
		'attacker',
		'ranged',
		'healer',
		'dismantler',
	],
};

function bodyCostTable() {
	if (typeof BODYPART_COST !== 'undefined' && BODYPART_COST) return BODYPART_COST;
	return BODYPART_COST_FALLBACK;
}

function partCost(part) {
	const table = bodyCostTable();
	return table[part] || BODYPART_COST_FALLBACK[part] || 0;
}

function bodyCost(body) {
	let sum = 0;
	for (let i = 0; i < body.length; i++) sum += partCost(body[i]);
	return sum;
}

function spawnTime(bodyOrParts) {
	const n = typeof bodyOrParts === 'number'
		? bodyOrParts
		: (bodyOrParts && bodyOrParts.length) || 0;
	const t = (typeof CREEP_SPAWN_TIME !== 'undefined' ? CREEP_SPAWN_TIME : CONST.CREEP_SPAWN_TIME);
	return n * t;
}

function clamp(n, lo, hi) {
	return Math.max(lo, Math.min(hi, n));
}

function ceilDiv(a, b) {
	return Math.ceil(a / b);
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/**
 * Estimate path length (tiles) between two room names.
 * Prefer Game.map.findRoute when available; else linear * 50 + 20.
 */
function estimateRoomPathLen(homeRoomName, remoteRoomName) {
	if (!homeRoomName || !remoteRoomName) return CONST.ROOM_PATH_TILES + CONST.ROOM_PATH_BIAS;
	if (homeRoomName === remoteRoomName) return 0;

	if (typeof Game !== 'undefined' && Game.map) {
		try {
			if (typeof Game.map.findRoute === 'function') {
				const route = Game.map.findRoute(homeRoomName, remoteRoomName);
				const errNo = typeof ERR_NO_PATH !== 'undefined' ? ERR_NO_PATH : -2;
				if (route !== errNo && route && typeof route.length === 'number') {
					return route.length * CONST.ROOM_PATH_TILES + CONST.ROOM_PATH_BIAS;
				}
			}
			if (typeof Game.map.getRoomLinearDistance === 'function') {
				const linear = Game.map.getRoomLinearDistance(homeRoomName, remoteRoomName);
				return linear * CONST.ROOM_PATH_TILES + CONST.ROOM_PATH_BIAS;
			}
		} catch (_e) {
			// fall through
		}
	}
	// Offline / no map: crude room-name parse for linear distance
	const a = parseRoomXY(homeRoomName);
	const b = parseRoomXY(remoteRoomName);
	if (a && b) {
		const d = Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y)); // chebyshev-ish rooms
		// use manhattan-ish for travel estimate
		const man = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
		return Math.max(d, man) * CONST.ROOM_PATH_TILES + CONST.ROOM_PATH_BIAS;
	}
	return CONST.ROOM_PATH_TILES + CONST.ROOM_PATH_BIAS;
}

function parseRoomXY(roomName) {
	const m = /^([WE])(\d+)([NS])(\d+)$/.exec(roomName);
	if (!m) return null;
	let x = Number(m[2]);
	let y = Number(m[4]);
	if (m[1] === 'W') x = -x - 1;
	if (m[3] === 'N') y = -y - 1;
	return { x, y };
}

// ---------------------------------------------------------------------------
// 1. Source throughput
// ---------------------------------------------------------------------------

/**
 * Estimate steady-state energy/tick from a source.
 *
 * @param {object|number|null} sourceOrOpts
 *   - number: treated as pathLen only (throughput still 10 e/t)
 *   - Source-like: { energyCapacity?, ticksToRegeneration?, id?, sk? }
 *   - opts: { sk?: boolean, pathLen?: number, energyCapacity?: number }
 * @returns {{
 *   ePerTick: number,
 *   energyCapacity: number,
 *   regenTicks: number,
 *   pathLen: number|null,
 *   sk: boolean,
 *   note: string,
 * }}
 */
function estimateSource(sourceOrOpts) {
	let pathLen = null;
	let sk = false;
	let energyCapacity = CONST.SOURCE_ENERGY_CAPACITY;
	let regenTicks = CONST.ENERGY_REGEN_TIME;
	let id = null;

	if (typeof sourceOrOpts === 'number') {
		pathLen = sourceOrOpts;
	} else if (sourceOrOpts && typeof sourceOrOpts === 'object') {
		if (sourceOrOpts.pathLen != null) pathLen = sourceOrOpts.pathLen;
		if (sourceOrOpts.sk) sk = true;
		if (sourceOrOpts.id) id = sourceOrOpts.id;
		if (sourceOrOpts.energyCapacity != null) energyCapacity = sourceOrOpts.energyCapacity;
		// Live Source object
		if (sourceOrOpts.energyCapacity != null && sourceOrOpts.pos) {
			energyCapacity = sourceOrOpts.energyCapacity;
		}
		if (sourceOrOpts.ticksToRegeneration != null) {
			// capacity known from regen cycle only when full; keep defaults
		}
	}

	if (sk) {
		energyCapacity = energyCapacity === CONST.SOURCE_ENERGY_CAPACITY
			? CONST.SK_SOURCE_ENERGY_CAPACITY
			: energyCapacity;
	}

	const ePerTick = energyCapacity / regenTicks;
	const note = sk
		? 'SK source: higher throughput (~13.3 e/t) but requires combat upkeep and is riskier'
		: 'Normal owned/neutral source: 3000/300 = 10 e/t';

	return {
		ePerTick,
		energyCapacity,
		regenTicks,
		pathLen,
		sk,
		id,
		note,
	};
}

// ---------------------------------------------------------------------------
// 2. Static harvester body
// ---------------------------------------------------------------------------

/**
 * Static harvester: 5 WORK (mines 10 e/t) + 1 CARRY (bootstrap container/drop)
 * + enough MOVE to reach the seat once (then parks).
 *
 * Formula:
 *   workParts = 5 (ceil(SOURCE_ET / HARVEST_POWER))
 *   carryParts = 1
 *   moveParts  = max(1, ceil(nonMove / 2)) on roads-ish, or full for long remote walk
 *   cost       = Σ BODYPART_COST
 *   spawnTime  = parts * CREEP_SPAWN_TIME
 *   upkeepEt   = cost / CREEP_LIFE_TIME
 *
 * @param {object} [opts]
 * @param {number} [opts.pathLen=5]  tiles to seat (affects MOVE count for arrival)
 * @param {boolean} [opts.roads=true]
 * @param {boolean} [opts.sk=false]  SK → still 5 WORK base; caller may boost
 * @param {number} [opts.workParts]  override WORK count
 */
function estimateHarvester(opts = {}) {
	const pathLen = opts.pathLen != null ? opts.pathLen : 5;
	const roads = opts.roads !== false;
	const workParts = opts.workParts != null
		? opts.workParts
		: CONST.STATIC_HARVESTER_WORK;
	const carryParts = 1;

	// Static miner walks to seat once; over-MOVE wastes energy.
	// Short path / local: 1 MOVE is enough on plain if empty (fatigue from 6 non-move
	// needs 6 MOVE off-road, but once seated MOVE is idle — still need arrival).
	// Practical Apex body: 5W1C + MOVE scaled for one-way walk on roads.
	const nonMove = workParts + carryParts;
	let moveParts;
	if (pathLen <= 2 && roads) {
		moveParts = 1;
	} else if (roads) {
		moveParts = Math.max(1, Math.ceil(nonMove / 2));
	} else {
		// Off-road remote: full MOVE so fatigue doesn't strand the miner.
		moveParts = nonMove;
	}
	// Cap to MAX_CREEP_SIZE
	const maxSize = typeof MAX_CREEP_SIZE !== 'undefined' ? MAX_CREEP_SIZE : CONST.MAX_CREEP_SIZE;
	while (workParts + carryParts + moveParts > maxSize && moveParts > 1) moveParts--;

	const body = [];
	for (let i = 0; i < workParts; i++) body.push('work');
	for (let i = 0; i < carryParts; i++) body.push('carry');
	for (let i = 0; i < moveParts; i++) body.push('move');

	const cost = bodyCost(body);
	const st = spawnTime(body);
	const life = CONST.CREEP_LIFE_TIME;
	const harvestEt = workParts * CONST.HARVEST_POWER;

	return {
		body,
		parts: body.length,
		workParts,
		carryParts,
		moveParts,
		cost,
		spawnTime: st,
		lifetime: life,
		/** Energy/tick amortized body cost. */
		upkeepEt: cost / life,
		/** Harvest power e/t (capped by source). */
		harvestEt,
		/** Spawn duty cycle for one continuous harvester (fraction of one spawn). */
		spawnBusyFrac: st / life,
	};
}

// ---------------------------------------------------------------------------
// 3. Hauler demand for a route
// ---------------------------------------------------------------------------

/**
 * Hauler demand for steady pipeline home ↔ source.
 *
 * Formula:
 *   roundTripTiles ≈ 2 * pathLen
 *   energyInFlight  = roundTripTiles * ePerTick     // carry buffer to not starve
 *   carryPartsNeed  = ceil(energyInFlight / 50)
 *   body pattern on roads: [CARRY, CARRY, MOVE] → 2C per unit, 1M
 *   if carryPartsNeed > max per creep → multiple haulers
 *
 * @param {number} pathLen  tiles one-way (spawn/storage ↔ source)
 * @param {number} [ePerTick=10]
 * @param {object} [opts]
 * @param {boolean} [opts.roads=true]
 * @param {number} [opts.maxCarry]  max CARRY parts per creep
 * @returns hauler projection
 */
function estimateHauler(pathLen, ePerTick, opts = {}) {
	const rate = ePerTick != null ? ePerTick : CONST.SOURCE_ET;
	const len = Math.max(1, pathLen || 1);
	const roads = opts.roads !== false;
	const maxCarry = opts.maxCarry != null ? opts.maxCarry : CONST.MAX_HAULER_CARRY;
	const minCarry = opts.minCarry != null ? opts.minCarry : CONST.MIN_HAULER_CARRY;
	const maxSize = typeof MAX_CREEP_SIZE !== 'undefined' ? MAX_CREEP_SIZE : CONST.MAX_CREEP_SIZE;

	// energy needed in the pipeline for continuous delivery
	const roundTrip = 2 * len;
	const energyInFlight = roundTrip * rate;
	const carryPartsNeed = Math.max(minCarry, Math.ceil(energyInFlight / CONST.CARRY_CAPACITY));

	// Max CARRY parts on one creep given MOVE ratio and MAX_CREEP_SIZE
	// roads: 2C1M → parts = carry + ceil(carry/2); carry + ceil(carry/2) <= 50
	// off-road: 1C1M → carry + carry <= 50 → carry <= 25
	let maxCarryPerCreep;
	if (roads) {
		// c + ceil(c/2) <= maxSize → roughly c <= floor(maxSize * 2/3)
		maxCarryPerCreep = Math.min(maxCarry, Math.floor((maxSize * 2) / 3));
	} else {
		maxCarryPerCreep = Math.min(maxCarry, Math.floor(maxSize / 2));
	}

	const haulerCount = Math.max(1, ceilDiv(carryPartsNeed, maxCarryPerCreep));
	const carryPerHauler = Math.max(minCarry, Math.ceil(carryPartsNeed / haulerCount));

	const bodies = [];
	let totalCost = 0;
	let totalSpawnTime = 0;
	let totalParts = 0;

	for (let h = 0; h < haulerCount; h++) {
		const carry = carryPerHauler;
		const body = buildHaulerBody(carry, roads, maxSize);
		const cost = bodyCost(body);
		const st = spawnTime(body);
		bodies.push({ body, carryParts: carry, parts: body.length, cost, spawnTime: st });
		totalCost += cost;
		totalSpawnTime += st;
		totalParts += body.length;
	}

	const life = CONST.CREEP_LIFE_TIME;
	const capacity = carryPartsNeed * CONST.CARRY_CAPACITY;
	// Delivery rate when pipeline full: capacity / roundTrip
	const deliveryEt = capacity / roundTrip;

	return {
		pathLen: len,
		ePerTick: rate,
		roundTrip,
		energyInFlight,
		carryPartsNeed,
		haulerCount,
		carryPerHauler,
		bodies,
		/** Representative single body (first). */
		body: bodies[0].body,
		cost: totalCost,
		costEach: bodies[0].cost,
		spawnTime: totalSpawnTime,
		spawnTimeEach: bodies[0].spawnTime,
		parts: totalParts,
		lifetime: life,
		upkeepEt: totalCost / life,
		spawnBusyFrac: totalSpawnTime / life,
		deliveryEt: Math.min(rate, deliveryEt),
		maxCarryPerCreep,
	};
}

function buildHaulerBody(carryParts, roads, maxSize) {
	const body = [];
	let carry = carryParts;
	let move;
	if (roads) {
		// Prefer 2C:1M packing
		move = Math.max(1, Math.ceil(carry / 2));
	} else {
		move = carry;
	}
	while (carry + move > maxSize && carry > 1) {
		carry--;
		move = roads ? Math.max(1, Math.ceil(carry / 2)) : carry;
	}
	for (let i = 0; i < carry; i++) body.push('carry');
	for (let i = 0; i < move; i++) body.push('move');
	return body;
}

// ---------------------------------------------------------------------------
// 4. Reserver upkeep
// ---------------------------------------------------------------------------

/**
 * Reserver amortization.
 *
 * CLAIM creeps live CREEP_CLAIM_LIFE_TIME (600) ticks.
 * 1 CLAIM + 1 MOVE = 650 energy; each tick of reserveController adds
 * +1 reservation tick per CLAIM part (max reservation 5000).
 *
 * Continuous upkeep with 1 CLAIM: one reserver every 600 ticks.
 * upkeepEt = cost / 600
 *
 * With 2 CLAIM: can top up faster / less travel waste; cost 1300 / 600.
 *
 * @param {object} [opts]
 * @param {number} [opts.claimParts=1]
 * @param {number} [opts.pathLen]  optional travel; extra MOVE if long
 */
function estimateReserver(opts = {}) {
	const claimParts = opts.claimParts != null ? opts.claimParts : 1;
	const pathLen = opts.pathLen || 50;
	const maxSize = typeof MAX_CREEP_SIZE !== 'undefined' ? MAX_CREEP_SIZE : CONST.MAX_CREEP_SIZE;

	// MOVE: at least 1 per CLAIM for roads; full for long off-road
	const moveParts = Math.min(
		maxSize - claimParts,
		Math.max(claimParts, pathLen > 100 ? claimParts : Math.ceil(claimParts)),
	);

	const body = [];
	for (let i = 0; i < claimParts; i++) body.push('claim');
	for (let i = 0; i < moveParts; i++) body.push('move');

	const cost = bodyCost(body);
	const life = CONST.CREEP_CLAIM_LIFE_TIME;
	const st = spawnTime(body);

	return {
		body,
		parts: body.length,
		claimParts,
		moveParts,
		cost,
		spawnTime: st,
		lifetime: life,
		upkeepEt: cost / life,
		spawnBusyFrac: st / life,
		/** Reservation ticks generated over full life (idle travel ignored). */
		reservationTicks: claimParts * life,
	};
}

// ---------------------------------------------------------------------------
// 5. Remote room package
// ---------------------------------------------------------------------------

/**
 * Full remote package cost for one remote room.
 *
 * @param {string} homeRoomName
 * @param {string} remoteRoomName
 * @param {Array<{id?: string, pathLen?: number, sk?: boolean}>} sources
 * @param {object} [opts]
 * @returns package projection
 */
function estimateRemotePackage(homeRoomName, remoteRoomName, sources, opts = {}) {
	const srcList = Array.isArray(sources) && sources.length
		? sources
		: [ { pathLen: null } ]; // assume 1 source if unknown

	const roomPath = estimateRoomPathLen(homeRoomName, remoteRoomName);
	const roads = opts.roads !== false;
	const waste = opts.travelWaste != null ? opts.travelWaste : CONST.TRAVEL_WASTE_FACTOR;

	const reserver = estimateReserver({
		claimParts: opts.claimParts || 1,
		pathLen: roomPath,
	});

	const sourcePackages = [];
	let harvestEt = 0;
	let haulUpkeepEt = 0;
	let harvestUpkeepEt = 0;
	let totalSpawnTime = reserver.spawnTime;
	let totalCost = reserver.cost;
	let haulerCount = 0;
	let harvesterCount = 0;

	for (const s of srcList) {
		const srcEst = estimateSource(s);
		const pathLen = s.pathLen != null
			? s.pathLen
			: roomPath + 25; // room border + in-room average

		const harv = estimateHarvester({
			pathLen,
			roads: false, // remotes often lack full road nets initially
			sk: srcEst.sk,
			workParts: srcEst.sk ? 6 : CONST.STATIC_HARVESTER_WORK,
		});
		// Re-estimate with roads if opts say so
		const harvFinal = roads
			? estimateHarvester({ pathLen, roads: true, sk: srcEst.sk, workParts: harv.workParts })
			: harv;

		const haul = estimateHauler(pathLen, srcEst.ePerTick, { roads });

		const delivered = srcEst.ePerTick * (1 - waste);
		// Net after creep upkeep for this source
		const netEt = delivered - harvFinal.upkeepEt - haul.upkeepEt;

		sourcePackages.push({
			id: s.id || srcEst.id || null,
			pathLen,
			ePerTick: srcEst.ePerTick,
			deliveredEt: delivered,
			harvester: harvFinal,
			hauler: haul,
			netEt,
			sk: srcEst.sk,
		});

		harvestEt += srcEst.ePerTick;
		haulUpkeepEt += haul.upkeepEt;
		harvestUpkeepEt += harvFinal.upkeepEt;
		totalSpawnTime += harvFinal.spawnTime + haul.spawnTime;
		totalCost += harvFinal.cost + haul.cost;
		haulerCount += haul.haulerCount;
		harvesterCount += 1;
	}

	const deliveredEt = harvestEt * (1 - waste);
	const upkeepEt = reserver.upkeepEt + harvestUpkeepEt + haulUpkeepEt;
	const netEt = deliveredEt - upkeepEt;

	// Spawn busy % of ONE spawn to keep the package replaced continuously.
	// creeps replaced every lifetime; claim life is shorter — use weighted.
	// Approx: sum(spawnTime_i / life_i) * 100
	let spawnBusyFrac = reserver.spawnBusyFrac;
	for (const sp of sourcePackages) {
		spawnBusyFrac += sp.harvester.spawnBusyFrac + sp.hauler.spawnBusyFrac;
	}
	const spawnBusyPct = spawnBusyFrac * 100;

	return {
		homeRoomName,
		remoteRoomName,
		roomPath,
		sources: sourcePackages,
		sourceCount: sourcePackages.length,
		harvesterCount,
		haulerCount,
		reserver,
		/** Gross source e/t before waste. */
		harvestEt,
		/** e/t delivered at home after travel waste. */
		deliveredEt,
		/** Amortized creep body cost e/t. */
		upkeepEt,
		/** Net e/t after upkeep. */
		netEt,
		totalCost,
		totalSpawnTime,
		/** Percent of one spawn busy to sustain this package. */
		spawnBusyPct,
		spawnBusyFrac,
		travelWaste: waste,
	};
}

// ---------------------------------------------------------------------------
// 6. How many remotes can we afford?
// ---------------------------------------------------------------------------

/**
 * @param {object} input
 * @param {number} input.colonyIncomeEt  local sources net e/t (after local upkeep)
 * @param {number} input.spawnCount
 * @param {number} input.rcl
 * @param {number} [input.freeCpu]  optional spare CPU budget (ms/tick rough)
 * @param {number} [input.localSpawnBusyFrac]  fraction already used by local economy
 * @param {object[]} [input.candidatePackages]  from estimateRemotePackage, sorted best-first
 * @param {number} [input.maxRemotes]
 */
function affordRemotes(input) {
	const rcl = input.rcl || 1;
	const spawnCount = Math.max(1, input.spawnCount || 1);
	const incomeEt = input.colonyIncomeEt || 0;
	const freeCpu = input.freeCpu;
	const localBusy = input.localSpawnBusyFrac || 0;
	const maxSoft = input.maxRemotes != null ? input.maxRemotes : CONST.MAX_REMOTES_SOFT;
	const packages = input.candidatePackages || [];

	if (rcl < CONST.REMOTE_MIN_RCL) {
		return {
			maxRemotes: 0,
			reason: 'rcl',
			detail: `RCL ${rcl} < ${CONST.REMOTE_MIN_RCL}; remotes gated`,
			afforded: [],
			spawnHeadroom: spawnCount - localBusy,
			energyHeadroom: incomeEt,
		};
	}

	// Spawn headroom: each spawn provides 1.0 busy-frac capacity
	let spawnHeadroom = spawnCount - localBusy;
	let energyHeadroom = incomeEt; // can temporarily fund remote bodies from local income
	// Once remotes produce, net is positive; still need spawn + bootstrap energy

	const afforded = [];
	let reason = 'candidates';
	let blockedBy = null;

	// CPU rough: each remote package ~0.5–2 ms; if freeCpu known, ~1.0 ms each
	const cpuPerRemote = 1.0;
	let cpuHeadroom = freeCpu != null ? freeCpu : Infinity;

	for (let i = 0; i < packages.length; i++) {
		if (afforded.length >= maxSoft) {
			reason = 'cap';
			blockedBy = 'maxRemotes soft cap';
			break;
		}
		const pkg = packages[i];
		const needSpawn = pkg.spawnBusyFrac || (pkg.spawnBusyPct || 0) / 100;
		const needEnergyBootstrap = pkg.upkeepEt || 0; // need spare income to float upkeep until net positive
		// Prefer packages with positive net; still require spawn room
		if (needSpawn > spawnHeadroom + 1e-9) {
			reason = 'spawn';
			blockedBy = 'spawn bottleneck';
			break;
		}
		// Energy: need local surplus to cover upkeep until remote delivers;
		// if package is net positive, require only partial float (50% of upkeep)
		const floatNeed = pkg.netEt >= 0 ? needEnergyBootstrap * 0.5 : needEnergyBootstrap;
		if (floatNeed > energyHeadroom + 1e-9) {
			reason = 'energy';
			blockedBy = 'energy bottleneck';
			break;
		}
		if (cpuPerRemote > cpuHeadroom + 1e-9) {
			reason = 'cpu';
			blockedBy = 'cpu bottleneck';
			break;
		}
		// Hauler count sanity: very long routes can explode creep count
		if ((pkg.haulerCount || 0) + (pkg.harvesterCount || 0) > 12) {
			reason = 'hauler';
			blockedBy = 'hauler/creep count too high for path';
			// skip this candidate, try next
			continue;
		}

		afforded.push(pkg);
		spawnHeadroom -= needSpawn;
		// After package is up, energy headroom gains netEt
		energyHeadroom = energyHeadroom - floatNeed + Math.max(0, pkg.netEt || 0);
		if (freeCpu != null) cpuHeadroom -= cpuPerRemote;
	}

	if (!packages.length && reason === 'candidates') {
		reason = 'no_candidates';
		blockedBy = 'no remote package candidates provided';
	} else if (afforded.length && !blockedBy) {
		reason = 'ok';
		blockedBy = null;
	}

	return {
		maxRemotes: afforded.length,
		reason: blockedBy ? reason : (afforded.length ? 'ok' : reason),
		detail: blockedBy || (afforded.length ? `afford ${afforded.length} remotes` : 'none afforded'),
		afforded,
		spawnHeadroom,
		energyHeadroom,
		cpuHeadroom: freeCpu != null ? cpuHeadroom : null,
	};
}

// ---------------------------------------------------------------------------
// 7. Filler / upgrade / build staffing
// ---------------------------------------------------------------------------

/**
 * Simple staffing model for base logistics & GCL.
 *
 * fillers: 1 when extensions exist; body scales with extension count
 * upgraders: surplus after spawn-fill commitment + remote commitment
 * builders: sites / construction pressure
 *
 * @param {object} ctx
 * @param {number} ctx.rcl
 * @param {number} ctx.extensionCount
 * @param {number} ctx.incomeEt       gross colony + remote delivered
 * @param {number} ctx.upkeepEt       all economy creep upkeep
 * @param {number} ctx.remoteCommitEt remote package upkeep (subset)
 * @param {number} [ctx.constructionSites=0]
 * @param {number} [ctx.storageEnergy=0]
 * @param {number} [ctx.spawnCapacity=300]
 */
function estimateStaffing(ctx) {
	const rcl = ctx.rcl || 1;
	const ext = ctx.extensionCount || 0;
	const incomeEt = ctx.incomeEt || 0;
	const upkeepEt = ctx.upkeepEt || 0;
	const remoteCommitEt = ctx.remoteCommitEt || 0;
	const sites = ctx.constructionSites || 0;
	const storageEnergy = ctx.storageEnergy || 0;
	const spawnCap = ctx.spawnCapacity || 300;

	const recommendations = [];

	// --- Filler ---
	// 1 filler once extensions exist (RCL2+). Carry scales with fill volume.
	// Spawn+extensions capacity must be refilled each spawn cycle; rough fill rate
	// needed ≈ spawnCap / 50 ticks average drain under load → scale carry.
	let fillerCount = 0;
	let fillerCarry = 4;
	if (ext > 0 || rcl >= 2) {
		fillerCount = 1;
		// More extensions → bigger filler (2C per ~5 extensions, min 4, max 16)
		fillerCarry = clamp(4 + Math.floor(ext / 5) * 2, 4, 16);
		if (ext >= 30) fillerCount = 2; // RCL 6+ comfort
		recommendations.push({
			role: 'filler',
			count: fillerCount,
			reason: `extensions=${ext}; carry≈${fillerCarry} to keep spawns topped`,
			carryParts: fillerCarry,
		});
	}

	// Filler upkeep rough (roads body)
	const fillerHaul = fillerCount
		? estimateHauler(10, 5, { roads: true, maxCarry: fillerCarry, minCarry: fillerCarry })
		: null;
	// Override count: estimateHauler may split; for filler we force body size
	let fillerUpkeep = 0;
	if (fillerCount) {
		const body = buildHaulerBody(fillerCarry, true, CONST.MAX_CREEP_SIZE);
		// Add a bit of WORK? No — pure filler is carry/move. Cost * count / life
		fillerUpkeep = (bodyCost(body) * fillerCount) / CONST.CREEP_LIFE_TIME;
	}

	// Surplus after economy upkeep (including remotes already in upkeepEt)
	const surplusEt = incomeEt - upkeepEt - fillerUpkeep;

	// --- Upgraders ---
	// RCL8 downgrade prevention needs ~1 WORK continuous near end; before that push GCL.
	// Each upgrader WORK ≈ 1 e/t upgrade when fed; body ~ 2W1C1M unit on roads.
	let upgraderCount = 0;
	let upgraderWork = 0;
	if (rcl < 8) {
		// Commit ~60% of positive surplus to upgrade, min 1 if any surplus > 2
		const upgradeBudget = Math.max(0, surplusEt * 0.6);
		if (upgradeBudget >= 2 || storageEnergy > 10_000) {
			upgraderWork = Math.max(2, Math.floor(upgradeBudget));
			// ~5 WORK per upgrader mid-game
			const workEach = rcl >= 6 ? 10 : rcl >= 4 ? 5 : 2;
			upgraderCount = Math.max(1, Math.min(6, Math.ceil(upgraderWork / workEach)));
			recommendations.push({
				role: 'upgrader',
				count: upgraderCount,
				reason: `surplus≈${surplusEt.toFixed(1)} e/t; budget ${upgradeBudget.toFixed(1)} e/t upgrade`,
				workParts: workEach,
			});
		} else if (rcl >= 1) {
			// Always keep a minimal upgrader so controller never decays hard
			upgraderCount = 1;
			recommendations.push({
				role: 'upgrader',
				count: 1,
				reason: 'minimal controller progress / safety',
				workParts: 2,
			});
		}
	} else {
		// RCL8: 15 WORK cap on upgrade; 1–2 upgraders
		upgraderCount = storageEnergy > 50_000 ? 2 : 1;
		recommendations.push({
			role: 'upgrader',
			count: upgraderCount,
			reason: 'RCL8 maintenance upgrade (15 WORK cap)',
			workParts: 15,
		});
	}

	// --- Builders ---
	let builderCount = 0;
	if (sites > 0) {
		builderCount = sites > 10 ? 3 : sites > 3 ? 2 : 1;
		// Don't starve upgrade: if surplus thin, cap builders
		if (surplusEt < 5) builderCount = Math.min(builderCount, 1);
		recommendations.push({
			role: 'builder',
			count: builderCount,
			reason: `constructionSites=${sites}`,
		});
	} else if (storageEnergy > 80_000 && rcl >= 3) {
		// Walls / idle build pressure
		builderCount = 1;
		recommendations.push({
			role: 'builder',
			count: 1,
			reason: 'storage high — wall grind / repair pressure',
		});
	}

	// Military budget: surplus after staffing, * factor
	const staffedSurplus = Math.max(0, surplusEt - upgraderCount * 2 - builderCount * 2);
	const militaryBudgetEt = staffedSurplus * CONST.MILITARY_BUDGET_FACTOR;

	return {
		fillerCount,
		fillerCarry,
		fillerUpkeepEt: fillerUpkeep,
		upgraderCount,
		upgraderWork,
		builderCount,
		surplusEt,
		militaryBudgetEt,
		remoteCommitEt,
		spawnCapacity: spawnCap,
		recommendations,
	};
}

// ---------------------------------------------------------------------------
// Local colony income (owned room sources)
// ---------------------------------------------------------------------------

/**
 * Project local (owned room) mining package.
 * @param {Array<{id?, pathLen?, sk?}>|number} sources  list or source count
 * @param {object} [opts]
 */
function estimateLocalSources(sources, opts = {}) {
	let list;
	if (typeof sources === 'number') {
		list = [];
		for (let i = 0; i < sources; i++) list.push({ pathLen: opts.defaultPathLen || 15 });
	} else {
		list = sources || [];
	}

	const roads = opts.roads !== false;
	const parts = [];
	let incomeEt = 0;
	let upkeepEt = 0;
	let spawnBusyFrac = 0;
	let haulerCount = 0;

	for (const s of list) {
		const src = estimateSource(s);
		const pathLen = s.pathLen != null ? s.pathLen : (opts.defaultPathLen || 15);
		const harv = estimateHarvester({ pathLen, roads, sk: src.sk });
		const haul = estimateHauler(pathLen, src.ePerTick, { roads });
		const delivered = src.ePerTick; // local: negligible waste
		incomeEt += delivered;
		upkeepEt += harv.upkeepEt + haul.upkeepEt;
		spawnBusyFrac += harv.spawnBusyFrac + haul.spawnBusyFrac;
		haulerCount += haul.haulerCount;
		parts.push({
			id: s.id || null,
			pathLen,
			ePerTick: src.ePerTick,
			harvester: harv,
			hauler: haul,
			netEt: delivered - harv.upkeepEt - haul.upkeepEt,
		});
	}

	return {
		sources: parts,
		sourceCount: parts.length,
		incomeEt,
		upkeepEt,
		netEt: incomeEt - upkeepEt,
		spawnBusyFrac,
		haulerCount,
		harvesterCount: parts.length,
	};
}

module.exports = {
	CONST,
	BODYPART_COST_FALLBACK,
	bodyCost,
	spawnTime,
	clamp,
	estimateRoomPathLen,
	parseRoomXY,
	estimateSource,
	estimateHarvester,
	estimateHauler,
	estimateReserver,
	estimateRemotePackage,
	affordRemotes,
	estimateStaffing,
	estimateLocalSources,
	buildHaulerBody,
};
