// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * Apex spawn — continuous highest-EV economic spawning.
 *
 * No fixed "want N upgraders". Every idle spawn tick picks the creep that
 * unlocks the most economy per spawn-time (and energy) spent.
 *
 * Constraints
 *   • Source e/t        = energyCapacity / ENERGY_REGEN_TIME  (usually 10)
 *   • Full mine         = ceil(e/t / HARVEST_POWER) WORK     (usually 5)
 *   • Haul buffer       = 2 * pathLen * harvestEt  → CARRY parts
 *   • Spawn duty        = bodyParts * 3 / 1500 per continuous creep
 *   • One spawn sustains duty cycle ≤ 1.0
 *
 * Max harvest estimate (Memory.empire.spawnBudget[room]):
 *   maxEtUnconstrained  — mine every source fully (ignore spawn)
 *   maxEtSpawnBound     — greedy packages under current spawn count + capacity
 *   realizedEt          — min(e/t, live WORK*2) summed
 *   surplusEt           — realized after pipeline upkeep → feeds upgrade WORK
 */
const config = require('./config');
const body = require('./body');
const intel = require('./intel');
const empire = require('./empire');
const harvestBudget = require('./harvestBudget');
const P = (function loadP() {
	const mod = require('./projections');
	return mod.CONST ? mod : (mod.default || mod);
})();
const {
	energyOf, hostileThreat, stillAlive, replaceHorizon, estimatePathLength,
	isSourceKeeperRoom, canAffordSkKiller, isSourceKeeperCreep, militaryParts,
} = require('./util');
const {
	getRole, getHome, getSourceId, getTargetRoom, getRemote, getPioneer,
	memInit, roleName, parseRole, Role, CreepMem,
} = require('./creepMem');

const { bodyCost: costOf } = body;

let war = null;
try { war = require('./war'); } catch { /* optional */ }

const WORK_PER_SOURCE = (P.CONST && P.CONST.STATIC_HARVESTER_WORK) || 5;
const HARVEST_POWER = (P.CONST && P.CONST.HARVEST_POWER) || 2;
const CREEP_LIFE = (P.CONST && P.CONST.CREEP_LIFE_TIME) || 1500;
const SPAWN_PART_TIME = (P.CONST && P.CONST.CREEP_SPAWN_TIME) || 3;
const CARRY_CAP = (P.CONST && P.CONST.CARRY_CAPACITY) || 50;

// ---------------------------------------------------------------------------
// Live inventory — TTL-aware (dying creeps don't count as staffed)
// ---------------------------------------------------------------------------

/**
 * Horizon so a replacement can spawn + walk before the old creep dies.
 * stillAlive adds another 80 ticks of slack.
 */
function minerHorizon(workParts, pathLen) {
	// Off-road remote body ≈ work + carry + work MOVE ≈ 2*work+1 parts
	const parts = Math.max(3, (workParts || 2) * 2 + 1);
	return replaceHorizon(parts, pathLen || 0);
}

function haulerHorizon(carryParts, pathLen) {
	// roads 2C1M units or off-road 1C1M — rough part count
	const parts = Math.max(2, Math.ceil((carryParts || 4) * 1.5));
	return replaceHorizon(parts, pathLen || 0);
}

function homeCreeps(home) {
	const out = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		const h = getHome(c);
		if (h && h !== home) continue;
		// Include everyone still breathing for role counts; pipeline uses horizons
		if (c.ticksToLive !== undefined && c.ticksToLive <= 1) continue;
		out.push(c);
	}
	return out;
}

function workOnSource(sourceId, horizon = 0) {
	let w = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		const role = getRole(c);
		if ((role !== Role.harvester && role !== Role.remoteHarvester) ||
			getSourceId(c) !== sourceId || !stillAlive(c, horizon)) continue;
		w += c.getActiveBodyparts(WORK);
	}
	return w;
}

function haulersForSource(sourceId, horizon = 0) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		const role = getRole(c);
		if ((role === Role.hauler || role === Role.remoteHauler) &&
			getSourceId(c) === sourceId && stillAlive(c, horizon)) n++;
	}
	return n;
}

function harvestersOnSource(sourceId, horizon = 0) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		const role = getRole(c);
		if ((role === Role.harvester || role === Role.remoteHarvester) &&
			getSourceId(c) === sourceId && stillAlive(c, horizon)) n++;
	}
	return n;
}

/** Open non-wall tiles adjacent to a source (max concurrent harvesters). */
function harvestSeats(source) {
	if (!source || !source.pos || !source.room) return 3; // unknown: allow multi
	const terrain = source.room.getTerrain();
	let n = 0;
	for (let dx = -1; dx <= 1; dx++) {
		for (let dy = -1; dy <= 1; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = source.pos.x + dx;
			const y = source.pos.y + dy;
			if (x < 0 || x > 49 || y < 0 || y > 49) continue;
			if (terrain.get(x, y) !== TERRAIN_MASK_WALL) n++;
		}
	}
	return Math.max(1, n);
}

/**
 * How many miner creeps can still help this source?
 * Stack small bodies until WORK reaches full mine (usually 5), limited by open seats.
 */
function maxMinersForSource(needWorkFull, workCap, seats) {
	const w = Math.max(1, workCap || 1);
	const byWork = Math.ceil(needWorkFull / w);
	const bySeats = Math.max(1, seats || 3);
	return Math.max(1, Math.min(byWork, bySeats));
}

function totalParts(home, part, roles) {
	let n = 0;
	for (const c of homeCreeps(home)) {
		if (roles && roles.indexOf(getRole(c)) < 0) continue;
		n += c.getActiveBodyparts(part);
	}
	return n;
}

function roleCount(home, role) {
	let n = 0;
	for (const c of homeCreeps(home)) {
		if (getRole(c) === role) n++;
	}
	return n;
}

function creepsForTarget(targetRoom, role) {
	const out = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (role != null && getRole(c) !== role) continue;
		if (getTargetRoom(c) === targetRoom || getRemote(c) === targetRoom) out.push(c);
	}
	return out;
}

function spawnEnergy(room) {
	let capacity = 0;
	let available = 0;
	for (const s of room.find(FIND_MY_STRUCTURES)) {
		if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
			capacity += s.store.getCapacity(RESOURCE_ENERGY);
			available += energyOf(s.store);
		}
	}
	return { capacity, available };
}

/** Max WORK on one static miner at current energy capacity (body must fit). */
function maxWorkAtCapacity(capacity) {
	// Min body W+C+M = 200; each extra W = 100 (+ MOVE eventually).
	// Practical: 1W@200, 2W@300, 3W@400, 5W@550+
	if (capacity >= 550) return WORK_PER_SOURCE;
	if (capacity >= 400) return 3;
	if (capacity >= 300) return 2;
	if (capacity >= 200) return 1;
	return 0;
}

function dutyOfBody(parts) {
	return (parts * SPAWN_PART_TIME) / CREEP_LIFE;
}

function liveDuty(home) {
	let d = 0;
	for (const c of homeCreeps(home)) d += dutyOfBody(c.body.length);
	return d;
}

// ---------------------------------------------------------------------------
// Accurate max-harvest estimate under spawn + capacity
// ---------------------------------------------------------------------------

/**
 * For each local source: full-mine target, multi-harvester stacking, TTL staffing.
 * Creeps about to die (TTL ≤ replace horizon) do not count — spawn re-queues early.
 */
function analyzeSources(room) {
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	const spawnPos = spawn.pos;
	const { capacity } = spawnEnergy(room);
	const workCap = maxWorkAtCapacity(capacity) || 1;
	const sources = [];

	for (const source of room.find(FIND_SOURCES)) {
		const pathLen = Math.max(1, estimatePathLength(spawnPos, source.pos));
		const ePerTick = (source.energyCapacity || 3000) /
			((typeof ENERGY_REGEN_TIME !== 'undefined' && ENERGY_REGEN_TIME) || 300);
		const needWorkFull = Math.ceil(ePerTick / HARVEST_POWER); // 5
		const workParts = Math.min(needWorkFull, workCap);
		const fullHarvestEt = Math.min(ePerTick, needWorkFull * HARVEST_POWER);

		const harv = P.estimateHarvester({ pathLen, roads: true, workParts });
		const haulFull = P.estimateHauler(pathLen, fullHarvestEt, { roads: true });
		const needHaulers = Math.max(1, haulFull.haulerCount || 1);
		const seats = harvestSeats(source);
		const maxMiners = maxMinersForSource(needWorkFull, workCap, seats);

		// TTL: only count WORK/haulers that outlive spawn+walk of a replacement
		const mHorizon = minerHorizon(workParts, pathLen);
		const hHorizon = haulerHorizon(haulFull.carryPerHauler || 4, pathLen);
		const haveWork = workOnSource(source.id, mHorizon);
		const haveHarvesters = harvestersOnSource(source.id, mHorizon);
		const haveHaulers = haulersForSource(source.id, hHorizon);
		// Live realized (what is mining *right now*, including dying) for metrics
		const liveWork = workOnSource(source.id, 0);
		const realizedEt = Math.min(ePerTick, liveWork * HARVEST_POWER);

		const carryPer = haulFull.carryPerHauler ||
			Math.max(2, Math.ceil((2 * pathLen * fullHarvestEt) / CARRY_CAP));
		const haulCapEt = haveHaulers > 0
			? (haveHaulers * carryPer * CARRY_CAP) / (2 * pathLen)
			: 0;
		const deliveredEt = Math.min(realizedEt, haulCapEt || realizedEt);

		sources.push({
			id: source.id,
			pathLen,
			ePerTick,
			needWorkFull,
			workPartsTarget: workParts,
			harvestEtIfStaffed: fullHarvestEt,
			harv,
			haul: haulFull,
			packageDuty: harv.spawnBusyFrac * Math.ceil(needWorkFull / Math.max(1, workParts)) +
				haulFull.spawnBusyFrac,
			packageUpkeepEt: harv.upkeepEt + haulFull.upkeepEt,
			haveWork,
			haveHarvesters,
			haveHaulers,
			needHaulers,
			seats,
			maxMiners,
			mHorizon,
			hHorizon,
			realizedEt,
			haulCapEt,
			deliveredEt,
			// Stack multi-harvester WORK until full mine; TTL already peeled dying
			missingWork: Math.max(0, needWorkFull - haveWork),
			missingHaulers: Math.max(0, needHaulers - haveHaulers),
		});
	}
	return sources;
}

/**
 * Greedy: assign full source packages in efficiency order under spawn duty budget.
 * Returns max sustainable delivered e/t given spawn count + capacity-limited bodies.
 */
function estimateMaxHarvest(sources, nSpawns) {
	const spawnBudget = Math.max(1, nSpawns);
	// Unconstrained: full mine every source (ideal 5W, ignore capacity)
	let maxEtUnconstrained = 0;
	for (const s of sources) maxEtUnconstrained += s.ePerTick;

	// Full mine when WORK is stacked (multi-harvester); not single-body capacity
	let maxEtCapacityBound = 0;
	for (const s of sources) maxEtCapacityBound += s.ePerTick;

	// Spawn-bound: pick packages with best e/t per duty
	const ranked = sources.slice().sort((a, b) => {
		const ea = a.packageDuty > 0 ? a.harvestEtIfStaffed / a.packageDuty : 0;
		const eb = b.packageDuty > 0 ? b.harvestEtIfStaffed / b.packageDuty : 0;
		return eb - ea;
	});
	let duty = 0;
	let maxEtSpawnBound = 0;
	let upkeep = 0;
	const chosen = [];
	for (const s of ranked) {
		if (duty + s.packageDuty > spawnBudget + 1e-9) {
			// Partial: can we afford miner only?
			if (duty + s.harv.spawnBusyFrac <= spawnBudget + 1e-9) {
				duty += s.harv.spawnBusyFrac;
				// without haul, energy drops — count ~50% as recoverable short-term
				maxEtSpawnBound += s.harvestEtIfStaffed * 0.5;
				upkeep += s.harv.upkeepEt;
				chosen.push({ id: s.id, partial: 'miner-only' });
			}
			continue;
		}
		duty += s.packageDuty;
		maxEtSpawnBound += s.harvestEtIfStaffed;
		upkeep += s.packageUpkeepEt;
		chosen.push({ id: s.id, partial: false });
	}

	return {
		maxEtUnconstrained,
		maxEtCapacityBound,
		maxEtSpawnBound,
		packageDutyTotal: ranked.reduce((a, s) => a + s.packageDuty, 0),
		dutyUsed: duty,
		spawnBudget,
		upkeepEt: upkeep,
		// headroom for non-pipeline after full local package
		spareDuty: Math.max(0, spawnBudget - duty),
		chosen,
	};
}

// ---------------------------------------------------------------------------
// EV-scored spawn candidates — always fill idle spawn time
// ---------------------------------------------------------------------------

/**
 * Score = economic e/t unlocked per unit spawn duty (higher = better).
 * Bootstrap / builders get situational boosts when they unlock capacity.
 */
function collectCandidates(room, sources, maxH, capacity, available) {
	const home = room.name;
	const rcl = room.controller.level;
	const cands = [];
	const workCap = maxWorkAtCapacity(capacity);

	const realizedEt = sources.reduce((a, s) => a + s.realizedEt, 0);
	const deliveredEt = sources.reduce((a, s) => a + s.deliveredEt, 0);
	const pipelineUpkeep = sources.reduce((a, s) => {
		// Approximate live upkeep from existing pipeline creeps via duty * avg
		return a + (s.haveHarvesters > 0 ? s.harv.upkeepEt * Math.min(1, s.haveWork / Math.max(1, s.workPartsTarget)) : 0)
			+ (s.haveHaulers > 0 ? s.haul.upkeepEt * (s.haveHaulers / Math.max(1, s.needHaulers)) : 0);
	}, 0);

	// ----- Capacity / extension rush (binding: unlock 5W @550) -----
	const extMax = CONTROLLER_STRUCTURES && CONTROLLER_STRUCTURES[STRUCTURE_EXTENSION]
		? (CONTROLLER_STRUCTURES[STRUCTURE_EXTENSION][rcl] || 0) : 0;
	const extensions = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_EXTENSION,
	}).length;
	const extSites = room.find(FIND_MY_CONSTRUCTION_SITES, {
		filter: s => s.structureType === STRUCTURE_EXTENSION,
	}).length;
	const sites = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	const gapToFullMine = Math.max(0, maxH.maxEtUnconstrained - maxH.maxEtCapacityBound);
	/** Until 5 extensions built, this is the #1 economy unlock */
	const extRush = capacity < 550 && extMax > extensions;

	// ----- Per-source mining / haul (LOCAL foundation — beat remotes while under-mined) -----
	for (const s of sources) {
		const maxMiners = s.maxMiners || maxMinersForSource(s.needWorkFull, workCap, s.seats);
		const localBoost = 50; // local full mine before remote expansion

		// Haul first if WORK is mining but energy is stranding (cuts ~40% residual)
		if (s.missingHaulers > 0 && (s.realizedEt > 0.1 || s.haveWork > 0)) {
			const stranded = Math.max(0, s.realizedEt - s.haulCapEt);
			const etGain = stranded > 0.1 ? stranded : Math.max(1, s.haveWork * HARVEST_POWER * 0.5);
			const duty = (s.haul.spawnBusyFrac / Math.max(1, s.needHaulers)) || 0.02;
			const carry = Math.max(2, s.haul.carryPerHauler || s.haul.carryPartsNeed || 4);
			cands.push({
				tag: `haul:${s.id.slice(-4)}`,
				role: Role.hauler,
				score: (etGain + 2) / Math.max(1e-6, duty) + localBoost + 20,
				etGain,
				duty,
				memory: { [CreepMem.sourceId]: s.id },
				context: { carryParts: carry, pathLen: s.pathLen },
				minEnergy: 100,
			});
		}

		// Stack WORK to full mine. Never leave a source at 0 WORK (death spiral).
		if (s.missingWork > 0 && s.haveHarvesters < maxMiners && workCap >= 1) {
			if (s.haveWork > 0 && s.missingHaulers > 0) {
				// haul-first only when already mining
			} else {
				const addW = Math.max(1, Math.min(workCap, s.missingWork, s.workPartsTarget || workCap));
				const before = Math.min(s.ePerTick, s.haveWork * HARVEST_POWER);
				const after = Math.min(s.ePerTick, (s.haveWork + addW) * HARVEST_POWER);
				const etGain = Math.max(0.5, after - before);
				const duty = dutyOfBody(addW + 2);
				// Empty source = hard priority over fillers/builders
				const emptyBoost = s.haveWork === 0 ? 500 : localBoost;
				cands.push({
					tag: `mine:${s.id.slice(-4)}:w${s.haveWork}+${addW}`,
					role: Role.harvester,
					score: etGain / Math.max(1e-6, duty) + emptyBoost,
					etGain,
					duty,
					memory: { [CreepMem.sourceId]: s.id },
					context: { workParts: addW, pathLen: s.pathLen },
					minEnergy: 200,
				});
			}
		}
	}

	// Surplus before builders
	const upWork = totalParts(home, WORK, [Role.upgrader, Role.bootstrap]);
	let liveUpkeepEt = 0;
	for (const c of homeCreeps(home)) {
		liveUpkeepEt += costOf(c.body.map(p => p.type)) / CREEP_LIFE;
	}
	const surplusEt = Math.max(0, deliveredEt - liveUpkeepEt - 1);

	// Extensions: hard priority until 550 (only real unlock for 5W bodies)
	const builders = roleCount(home, Role.builder);
	if (extRush) {
		const etGain = Math.max(8, gapToFullMine + (550 - capacity) / 40);
		const dil = 1 + builders * 0.25;
		cands.push({
			tag: 'build:ext',
			role: Role.builder,
			score: (600 + etGain * 20) / dil, // beats remote spam
			etGain,
			duty: 0.02,
			memory: {},
			context: {},
			minEnergy: 200,
		});
		// Bootstrap as mobile builders during ext rush if few builders
		if (builders < 2) {
			cands.push({
				tag: 'bootstrap:ext',
				role: Role.bootstrap,
				score: 400 / (1 + roleCount(home, Role.bootstrap) * 0.3),
				etGain: 5,
				duty: 0.02,
				memory: {},
				context: {},
				minEnergy: 200,
			});
		}
	} else if (sites > 0 && realizedEt > 1) {
		const dil = 1 + builders * 0.4;
		cands.push({
			tag: 'build:sites',
			role: Role.builder,
			score: (2 + sites * 0.5 + (extSites > 0 ? 15 : 0)) / 0.03 / dil,
			etGain: 1.5,
			duty: 0.03,
			memory: {},
			context: {},
			minEnergy: 200,
		});
	}

	// Filler only when mining is online — never replace the harvester pipeline
	const localWork = sources.reduce((a, s) => a + s.haveWork, 0);
	if (localWork > 0 && (extensions > 0 || capacity > 300)) {
		const fillers = roleCount(home, Role.filler);
		const spawnHungry = available < capacity * 0.4;
		if ((spawnHungry && fillers < 2) || fillers < 1) {
			const dil = 1 + fillers * 0.6;
			cands.push({
				tag: 'filler',
				role: Role.filler,
				score: ((spawnHungry ? 8 : 3) / 0.025) / dil,
				etGain: spawnHungry ? 3 : 1,
				duty: 0.025,
				memory: {},
				context: {},
				minEnergy: 100,
			});
		}
	}

	// Upgrade: spend surplus — was left at 0 while remotes won EV
	const wantUpWork = Math.max(Math.floor(surplusEt), realizedEt >= 4 ? 1 : 0);
	const targetUpWork = wantUpWork;
	if (upWork < targetUpWork && available >= 200) {
		const missing = targetUpWork - upWork;
		// Strong when surplus sits unused (banner surp>0 upW=0)
		const score = 90 + missing * 50 + surplusEt * 25 + (extRush ? 0 : 40);
		cands.push({
			tag: 'upgrade',
			role: Role.upgrader,
			score,
			etGain: missing,
			duty: 0.03,
			memory: {},
			context: { rcl },
			minEnergy: 200,
			meta: { surplusEt, targetUpWork, upWork },
		});
	}

	// ----- Bootstrap: multipurpose when specialized pipeline is thin -----
	const boots = roleCount(home, Role.bootstrap);
	const specialists = roleCount(home, Role.harvester) + roleCount(home, Role.hauler);
	if (specialists < sources.length && boots < sources.length * 2 + 2) {
		cands.push({
			tag: 'bootstrap',
			role: Role.bootstrap,
			score: 5 / 0.02 / (1 + boots * 0.3),
			etGain: 3,
			duty: 0.02,
			memory: {},
			context: {},
			minEnergy: 200,
		});
	}

	// ----- Defense: enough to match threat parts (no hard “max 3”) -----
	const threat = hostileThreat(room);
	if (threat >= 1) {
		const want = Math.max(1, Math.ceil(threat / 2));
		const have = roleCount(home, Role.defender);
		if (have < want) {
			cands.push({
				tag: 'defend',
				role: Role.defender,
				score: 100 + (want - have) * 20,
				etGain: 0,
				duty: 0.04,
				memory: {},
				context: {},
				minEnergy: 200,
			});
		}
	}

	// ----- Soft: always have *some* positive-EV option if spawn idle -----
	// If nothing else, a bootstrap/upgrader/hauler still beats idling spawn time.
	if (!cands.length && available >= 200) {
		cands.push({
			tag: 'idle-upgrade',
			role: Role.upgrader,
			score: 0.5,
			etGain: 0.5,
			duty: 0.03,
			memory: {},
			context: { rcl },
			minEnergy: 200,
		});
	}

	cands.sort((a, b) => b.score - a.score);
	return {
		cands,
		realizedEt,
		deliveredEt,
		surplusEt,
		targetUpWork,
		upWork,
		liveUpkeepEt,
	};
}

/** WORK currently in a given room (remote miners that arrived), TTL-aware. */
function workInRoomOnSource(sourceId, roomName, horizon = 0) {
	let w = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		const role = getRole(c);
		if ((role !== Role.harvester && role !== Role.remoteHarvester) ||
			getSourceId(c) !== sourceId || !stillAlive(c, horizon)) continue;
		if (c.room.name !== roomName) continue;
		w += c.getActiveBodyparts(WORK);
	}
	return w;
}

/**
 * Live remote harvest e/t: WORK×HARVEST_POWER of remoteHarvesters in-room.
 * Uses horizon 0 (instant rate); staffing uses replace horizons separately.
 */
function remoteHarvestEtLive(home) {
	let et = 0;
	for (const c of homeCreeps(home)) {
		if (getRole(c) !== Role.remoteHarvester) continue;
		if (c.ticksToLive !== undefined && c.ticksToLive <= 1) continue;
		const remote = getRemote(c) || getTargetRoom(c);
		if (!remote || c.room.name !== remote) continue;
		et += c.getActiveBodyparts(WORK) * HARVEST_POWER;
	}
	return et;
}

/**
 * Abandon remote on player/invader threat — NOT Source Keepers.
 * Keepers are the content of SK remotes; treating them as abandon blocked all SK forever.
 */
function isRemoteAbandoned(remoteName) {
	const room = Game.rooms[remoteName];
	if (room) {
		const hostiles = room.find(FIND_HOSTILE_CREEPS).filter(c =>
			!isSourceKeeperCreep(c) && militaryParts(c) > 0);
		return hostiles.length > 0;
	}
	// No vision: use intel threat only for non-SK rooms
	if (isSourceKeeperRoom(remoteName)) return false;
	const inf = intel.get(remoteName);
	if (!inf || !(inf.threat > 0)) return false;
	const age = Game.time - (inf.lastSeen || 0);
	return age < (config.remoteThreatCooldown || 500);
}

/**
 * Remote packages are economic foundation — same EV band as local mining.
 * Always field them when we have remotes + spawn time; only skip rooms abandoned
 * by recent threat. Multi-harvester + TTL pre-replace keep sources full.
 */
/**
 * @param {object} [opts]
 * @param {boolean} [opts.localMineComplete] local sources have full WORK
 * @param {boolean} [opts.extRush] capacity < 550 — throttle remote packages
 */
function collectRemoteCandidates(room, remotes, capacity, _available, opts = {}) {
	const cands = [];
	const home = room.name;
	if (!remotes || !remotes.length) return cands;
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (!spawn) return cands;

	const localMineComplete = !!opts.localMineComplete;
	const extRush = !!opts.extRush;
	// During extension rush, only scout until local mine is full — don't burn energy on remote bodies
	const packagesAllowed = localMineComplete || !extRush || capacity >= 550;

	const workCap = maxWorkAtCapacity(capacity) || 1;
	const travelWaste = (P.CONST && P.CONST.TRAVEL_WASTE_FACTOR) || 0.05;
	const targetRemoteEt = config.targetRemoteEt || 40;
	const remoteLive = remoteHarvestEtLive(home);
	const underTarget = remoteLive < targetRemoteEt;
	// Remotes compete after local foundation; milder boost during ext rush
	const pushBoost = !packagesAllowed
		? 0
		: underTarget
			? (extRush ? 8 : 25) + Math.max(0, targetRemoteEt - remoteLive) * (extRush ? 1 : 4)
			: 5;

	// Scouts always (cheap vision) — even during ext rush
	for (const remoteName of remotes) {
		if (isRemoteAbandoned(remoteName)) continue;
		const inf = intel.get(remoteName);
		const needsScout = !inf ||
			!inf.sources || !inf.sources.length ||
			Game.time - (inf.lastSeen || 0) > (config.intelStale || 1000);
		if (!needsScout) continue;
		const aliveScouts = creepsForTarget(remoteName, Role.scout)
			.filter(c => stillAlive(c, replaceHorizon(1, 50)));
		if (aliveScouts.length > 0) continue;
		cands.push({
			tag: `scout:${remoteName}`,
			role: Role.scout,
			score: 60 + pushBoost, // below build:ext / local mine
			etGain: 3,
			duty: dutyOfBody(1),
			memory: {
				[CreepMem.targetRoom]: remoteName,
				[CreepMem.remote]: remoteName,
			},
			context: {},
			minEnergy: 50,
		});
	}

	if (!packagesAllowed) return cands;

	for (const remoteName of remotes) {
		// Abandoned from attack: pull packages until cooldown clears
		if (isRemoteAbandoned(remoteName)) continue;
		const sk = isSourceKeeperRoom(remoteName);
		// SK only if we can field melee killers — otherwise skip entirely
		if (sk && !canAffordSkKiller(capacity)) continue;

		const live = Game.rooms[remoteName];
		const inf = intel.get(remoteName);
		const skEt = 4000 / 300; // SK sources are larger
		let sourceList = [];
		if (live) {
			sourceList = live.find(FIND_SOURCES).map(s => ({
				id: s.id,
				pos: s.pos,
				ePerTick: sk
					? (s.energyCapacity || 4000) / 300
					: (s.energyCapacity || 3000) / 300,
			}));
		} else if (inf && inf.sources && inf.sources.length) {
			sourceList = inf.sources.map(s => ({
				id: s.id,
				pos: s.pos
					? new RoomPosition(s.pos.x, s.pos.y, s.pos.roomName || remoteName)
					: null,
				ePerTick: sk
					? (s.energyCapacity || 4000) / 300
					: (s.energyCapacity || 3000) / 300,
			}));
		} else {
			// Blind: wait for scout IDs for multi-source; assume 1 package until vision
			// (avoids collapsing all blind slots onto one WORK total)
			sourceList = [{
				id: null,
				pos: null,
				ePerTick: sk ? skEt : 10,
				blind: true,
			}];
		}

		// SK combat first: melee killers must be *in room* before miners enter
		if (sk) {
			const pathSk = Game.map.getRoomLinearDistance(home, remoteName) * 50 + 40;
			const killerParts = Math.min(50, 2 * (config.skKillerMinAttack || 19));
			const killerHorizon = replaceHorizon(killerParts, pathSk);
			const killers = creepsForTarget(remoteName, Role.attacker)
				.filter(c => stillAlive(c, killerHorizon) && !c.spawning);
			const killersInRoom = killers.filter(c => c.room.name === remoteName);
			const wantKillers = Math.min(2, Math.max(1, sourceList.length));
			if (killers.length < wantKillers) {
				// Score as unlocking full SK package e/t (otherwise normal remotes starve SK)
				const unlockEt = sourceList.length * skEt * (1 - travelWaste);
				const dutyK = dutyOfBody(killerParts);
				cands.push({
					tag: `skkill:${remoteName}`,
					role: Role.attacker,
					score: unlockEt / Math.max(1e-6, dutyK) + pushBoost + 50,
					etGain: unlockEt,
					duty: dutyK,
					memory: {
						[CreepMem.targetRoom]: remoteName,
						[CreepMem.remote]: remoteName,
						[CreepMem.squad]: `sk_${remoteName}`,
					},
					context: { skKiller: true },
					minEnergy: config.skKillerMinEnergy || 2500,
				});
			}
			// Miners only once a non-spawning killer is in the SK room
			if (killersInRoom.length < 1) continue;
		}

		// Reserver: neutral rooms only — SK has no claimable controller economy
		if (!sk && capacity >= 650 && (live || (inf && inf.sources))) {
			const reservers = creepsForTarget(remoteName, Role.reserver).filter(c => stillAlive(c, 50));
			const reservation = live && live.controller && live.controller.reservation;
			let myUser = null;
			for (const rn in Game.rooms) {
				const c = Game.rooms[rn].controller;
				if (c && c.my && c.owner) { myUser = c.owner.username; break; }
			}
			const ticksLeft = reservation && reservation.ticksToEnd != null ? reservation.ticksToEnd : 0;
			const isMine = !!(reservation && myUser && reservation.username === myUser);
			const needReserve = !isMine || ticksLeft < (config.reserveRefresh || 2000);
			if (needReserve && reservers.length < 1) {
				cands.push({
					tag: `reserve:${remoteName}`,
					role: Role.reserver,
					score: 12 + (underTarget ? 5 : 0),
					etGain: 1,
					duty: dutyOfBody(2),
					memory: {
						[CreepMem.targetRoom]: remoteName,
						[CreepMem.remote]: remoteName,
					},
					context: {},
					minEnergy: 650,
				});
			}
		}

		for (const src of sourceList) {
			const pathLen = src.pos
				? Math.max(1, estimatePathLength(spawn.pos, src.pos))
				: Game.map.getRoomLinearDistance(home, remoteName) * 50 + 40;
			const ePerTick = src.ePerTick || 10;
			// SK sources ~13.3 e/t → more WORK; normal 10 e/t → 5 WORK
			const needWorkFull = Math.ceil(ePerTick / HARVEST_POWER);
			const bodyW = Math.max(1, Math.min(Math.max(WORK_PER_SOURCE, needWorkFull), workCap));
			const fullHarvestEt = Math.min(ePerTick, needWorkFull * HARVEST_POWER);
			const deliveredEt = fullHarvestEt * (1 - travelWaste);
			const haul = P.estimateHauler(pathLen, fullHarvestEt, { roads: false });
			const needHaulers = Math.max(1, haul.haulerCount || 1);
			const mHorizon = minerHorizon(bodyW, pathLen);
			const hHorizon = haulerHorizon(haul.carryPerHauler || 6, pathLen);

			const haveW = src.id
				? workOnSource(src.id, mHorizon)
				: creepsForTarget(remoteName, Role.remoteHarvester)
					.filter(c => stillAlive(c, mHorizon))
					.reduce((a, c) => a + c.getActiveBodyparts(WORK), 0);
			const inRoomW = src.id ? workInRoomOnSource(src.id, remoteName, 0) : 0;
			const haveH = src.id
				? haulersForSource(src.id, hHorizon)
				: creepsForTarget(remoteName, Role.remoteHauler)
					.filter(c => stillAlive(c, hHorizon)).length;
			const missingWork = Math.max(0, needWorkFull - haveW);
			const missingHaulers = Math.max(0, needHaulers - haveH);
			const nRemoteMiners = src.id
				? harvestersOnSource(src.id, mHorizon)
				: creepsForTarget(remoteName, Role.remoteHarvester)
					.filter(c => stillAlive(c, mHorizon)).length;
			const liveSrc = src.id && live ? live.find(FIND_SOURCES).find(s => s.id === src.id) : null;
			const seats = liveSrc ? harvestSeats(liveSrc) : 3;
			const maxMiners = maxMinersForSource(needWorkFull, bodyW, seats);

			// Haul before more miners when drops/WORK already exist (cut residual)
			const minerAssigned = nRemoteMiners > 0 || haveW > 0;
			const dropsReady = live && live.find(FIND_DROPPED_RESOURCES, {
				filter: r => r.resourceType === RESOURCE_ENERGY && r.amount > 50,
			}).length > 0;
			if (missingHaulers > 0 && (minerAssigned || dropsReady || inRoomW > 0)) {
				const fracOnline = Math.max(0.35, Math.min(1, (inRoomW || haveW) / Math.max(1, needWorkFull)));
				const stranded = Math.max(0.5, deliveredEt * fracOnline);
				const duty = (haul.spawnBusyFrac / needHaulers) || 0.04;
				const carry = Math.max(4, haul.carryPerHauler || haul.carryPartsNeed || 6);
				cands.push({
					tag: `rhaul:${remoteName.slice(-4)}:${String(src.id || 'x').slice(-4)}`,
					role: Role.remoteHauler,
					score: stranded / Math.max(1e-6, duty) + pushBoost + 15,
					etGain: stranded,
					duty,
					memory: {
						...(src.id ? { [CreepMem.sourceId]: src.id } : {}),
						[CreepMem.targetRoom]: remoteName,
						[CreepMem.remote]: remoteName,
					},
					context: { carryParts: carry, pathLen, roads: false },
					minEnergy: 100,
				});
			}

			// Miners only if haul isn't the binding gap (or first miner on empty source)
			if (missingWork > 0 && nRemoteMiners < maxMiners) {
				if (haveW > 0 && missingHaulers > 0) {
					// wait for haul
				} else {
					const addW = Math.max(1, Math.min(bodyW, missingWork));
					const before = Math.min(ePerTick, haveW * HARVEST_POWER);
					const after = Math.min(ePerTick, (haveW + addW) * HARVEST_POWER);
					const etGain = Math.max(1, (after - before) * (1 - travelWaste));
					const duty = dutyOfBody(addW * 2 + 1);
					cands.push({
						tag: src.blind
							? `rmine:${remoteName.slice(-4)}:blind`
							: `rmine:${remoteName.slice(-4)}:${String(src.id).slice(-4)}:w${haveW}+${addW}`,
						role: Role.remoteHarvester,
						score: etGain / Math.max(1e-6, duty) + pushBoost,
						etGain,
						duty,
						memory: {
							...(src.id ? { [CreepMem.sourceId]: src.id } : {}),
							[CreepMem.targetRoom]: remoteName,
							[CreepMem.remote]: remoteName,
						},
						context: { workParts: addW, pathLen, roads: false },
						minEnergy: 200,
					});
				}
			}
		}
	}

	return cands;
}

// ---------------------------------------------------------------------------
// Plan + run
// ---------------------------------------------------------------------------

function planColony(room, remotes) {
	const home = room.name;
	const spawns = room.find(FIND_MY_SPAWNS);
	if (!spawns.length) return [];

	const { capacity, available } = spawnEnergy(room);
	const sources = analyzeSources(room);
	const maxH = estimateMaxHarvest(sources, spawns.length);
	const liveD = liveDuty(home);
	const scored = collectCandidates(room, sources, maxH, capacity, available);

	// Accurate ceiling: local + remote candidates, pathLen, package duty, greedy under nSpawns
	const workCap = maxWorkAtCapacity(capacity) || WORK_PER_SOURCE;
	const ceiling = harvestBudget.updateHarvestCeiling(room, scored.realizedEt, {
		nSpawns: spawns.length,
		includeRemotes: true,
		includeSk: canAffordSkKiller(capacity),
		localWorkParts: workCap >= WORK_PER_SOURCE ? undefined : workCap,
	});

	// Persist — realized vs spawn-bound ceiling is the real bar
	if (Memory.empire) {
		Memory.empire.spawnBudget = Memory.empire.spawnBudget || {};
		Memory.empire.spawnBudget[home] = {
			t: Game.time,
			// Legacy local-only fields
			maxEtUnconstrained: +maxH.maxEtUnconstrained.toFixed(2),
			maxEtCapacityBound: +maxH.maxEtCapacityBound.toFixed(2),
			maxEtSpawnBoundLocal: +maxH.maxEtSpawnBound.toFixed(2),
			// Full ceiling (local + remotes under spawn time)
			maxEtPhysics: ceiling.maxEtPhysics,
			maxEtSpawnBound: ceiling.maxEtSpawnBound,
			maxEtLocalBound: ceiling.localEt,
			maxEtRemoteBound: ceiling.remoteEt,
			spawnsToTakeAll: ceiling.spawnsToTakeAll,
			realizedEt: +scored.realizedEt.toFixed(2),
			deliveredEt: +scored.deliveredEt.toFixed(2),
			surplusEt: +scored.surplusEt.toFixed(2),
			// Like-for-like: realized harvest vs local capacity-bound (not full remote ceiling)
			harvestEfficiency: maxH.maxEtCapacityBound > 0
				? +(scored.realizedEt / maxH.maxEtCapacityBound).toFixed(3)
				: 0,
			/** Gap to full spawn-bound ceiling (local+remote packages) */
			gapEt: +(ceiling.maxEtSpawnBound - scored.realizedEt).toFixed(2),
			/** Gap to local full mine only */
			gapLocalEt: +(maxH.maxEtCapacityBound - scored.realizedEt).toFixed(2),
			spawnBudget: ceiling.spawnBudget,
			packageDuty: +ceiling.dutyUsed.toFixed(3),
			liveDuty: +liveD.toFixed(3),
			spawnUtil: +(liveD / Math.max(1, ceiling.spawnBudget)).toFixed(3),
			spareDuty: +ceiling.spareDuty.toFixed(3),
			targetUpWork: scored.targetUpWork,
			upWork: scored.upWork,
			capacity,
			remoteLiveEt: +remoteHarvestEtLive(home).toFixed(2),
			targetRemoteEt: config.targetRemoteEt || 40,
			remotes: (remotes || []).slice(),
			top: scored.cands.slice(0, 5).map(c => `${c.tag}@${c.score.toFixed(1)}`),
			ceilingChosen: (ceiling.chosen || []).slice(0, 12),
			sources: sources.map(s => ({
				pathLen: s.pathLen,
				ePerTick: s.ePerTick,
				haveWork: s.haveWork,
				needWork: s.needWorkFull, // full mine WORK (usually 5), not body size
				bodyW: s.workPartsTarget,
				haveHaulers: s.haveHaulers,
				needHaulers: s.needHaulers,
				realizedEt: +s.realizedEt.toFixed(2),
			})),
		};
	}

	// Emergency: no workers that harvest (fillers-only is still dead)
	const miners = homeCreeps(home).filter(c => {
		const r = getRole(c);
		return r === Role.harvester || r === Role.bootstrap || r === Role.remoteHarvester;
	}).length;
	if (homeCreeps(home).length === 0 || miners === 0) {
		return [{
			role: Role.bootstrap,
			priority: 0,
			memory: memInit({ [CreepMem.role]: Role.bootstrap, [CreepMem.home]: home }),
			context: {},
			energyMode: 'available',
			score: 1e9,
		}];
	}

	const reqs = scored.cands.map((c, i) => ({
		role: c.role,
		// lower number = sooner; invert score into priority band
		priority: 1000 - Math.min(999, Math.floor(c.score)),
		memory: memInit({
			[CreepMem.role]: c.role,
			[CreepMem.home]: home,
			...c.memory,
		}),
		context: c.context || {},
		energyMode: 'available', // always spend — idle spawn is wasted duty forever
		score: c.score,
		tag: c.tag,
		minEnergy: c.minEnergy || 100,
	}));

	// Empire expansion only if local mining is decent and we have spawn headroom
	const localCap = maxH.maxEtCapacityBound || maxH.maxEtSpawnBound || 1;
	const eff = scored.realizedEt / localCap;
	const spareDuty = (Memory.empire.spawnBudget && Memory.empire.spawnBudget[home]
		&& Memory.empire.spawnBudget[home].spareDuty) || maxH.spareDuty;
	if (eff > 0.6 && spareDuty > 0.05) {
		for (const intent of empire.spawnIntentsFor(home)) {
			const iRole = typeof intent.role === 'number' ? intent.role : parseRole(intent.role);
			if (iRole == null) continue;
			const iMem = intent.memory || {};
			const target = iMem[CreepMem.targetRoom] ?? iMem.targetRoom;
			if (iRole === Role.claimer && creepsForTarget(target, Role.claimer).length >= 1) continue;
			reqs.push({
				role: iRole,
				priority: 800,
				memory: memInit({
					[CreepMem.role]: iRole,
					[CreepMem.home]: home,
					[CreepMem.targetRoom]: target,
					[CreepMem.pioneer]: !!(iMem[CreepMem.pioneer] || iMem.pioneer),
					[CreepMem.squad]: iMem[CreepMem.squad] || iMem.squad,
				}),
				context: intent.context || {},
				energyMode: 'available',
				score: 1,
				tag: 'empire',
			});
		}
	}

	if (war && war.spawnRequests && spareDuty > 0.1 && eff > 0.5) {
		for (const req of war.spawnRequests(room)) {
			const rRole = typeof req.role === 'number' ? req.role : parseRole(req.role);
			if (rRole == null) continue;
			const mem = req.memory || {};
			if (mem[CreepMem.home] == null) mem[CreepMem.home] = home;
			if (mem[CreepMem.role] == null) mem[CreepMem.role] = rRole;
			reqs.push({
				role: rRole,
				priority: req.priority,
				memory: mem,
				context: req.context || {},
				energyMode: 'available',
				score: 0.5,
				tag: 'war',
			});
		}
	}

	// Remotes: scouts always; full packages after local mine is full (or past ext rush).
	const localMineComplete = sources.length === 0 ||
		sources.every(s => s.missingWork <= 0);
	const extRush = capacity < 550 &&
		(CONTROLLER_STRUCTURES && CONTROLLER_STRUCTURES[STRUCTURE_EXTENSION]
			? (CONTROLLER_STRUCTURES[STRUCTURE_EXTENSION][room.controller.level] || 0) : 0) >
		room.find(FIND_MY_STRUCTURES, { filter: s => s.structureType === STRUCTURE_EXTENSION }).length;
	const remoteCands = collectRemoteCandidates(room, remotes || [], capacity, available, {
		localMineComplete,
		extRush,
	});
	for (const c of remoteCands) {
		reqs.push({
			role: c.role,
			priority: 1000 - Math.min(999, Math.floor(c.score)),
			memory: memInit({
				[CreepMem.role]: c.role,
				[CreepMem.home]: home,
				...c.memory,
			}),
			context: c.context || {},
			energyMode: 'available',
			score: c.score,
			tag: c.tag,
			minEnergy: c.minEnergy || 100,
		});
	}

	reqs.sort((a, b) => (b.score || 0) - (a.score || 0) || a.priority - b.priority);
	return reqs;
}

function nameFor(role) {
	return `${roleName(role)}_${Game.time.toString(36)}_${Math.floor(Math.random() * 36).toString(36)}`;
}

function run(room, remotes) {
	const idleSpawns = room.find(FIND_MY_SPAWNS).filter(s => !s.spawning);
	if (!idleSpawns.length) return;

	const reqs = planColony(room, remotes);
	if (!reqs.length) return;

	let { capacity, available } = spawnEnergy(room);

	for (const spawn of idleSpawns) {
		let spawned = false;
		for (const req of reqs) {
			const minE = req.minEnergy || 100;
			if (available < minE) continue;

			// Spend whatever we have — grow into full bodies as capacity allows
			const energy = Math.min(available, capacity);
			const parts = body.build(req.role, energy, {
				...req.context,
				rcl: room.controller.level,
			});
			if (!parts.length) continue;
			if ((req.role === Role.harvester || req.role === Role.remoteHarvester ||
				req.role === Role.upgrader || req.role === Role.builder) &&
				!parts.includes(WORK)) continue;
			if (req.role === Role.attacker && req.context && req.context.skKiller &&
				!parts.includes(ATTACK)) continue;

			const cost = costOf(parts);
			if (cost > available) continue;

			const name = nameFor(req.role);
			const result = spawn.spawnCreep(parts, name, { memory: req.memory });
			if (result === OK) {
				const sb = Memory.empire && Memory.empire.spawnBudget && Memory.empire.spawnBudget[room.name];
				const eff = sb ? ` eff=${((sb.harvestEfficiency || 0) * 100).toFixed(0)}%` : '';
				const mx = sb ? ` ceil=${sb.maxEtSpawnBound}/${sb.maxEtPhysics} gap=${sb.gapEt}` : '';
				const rel = sb ? ` got=${sb.realizedEt}` : '';
				console.log(
					`Apex v4 spawn ${room.name}: ${roleName(req.role)} [${parts}] cost=${cost}` +
					` tag=${req.tag || '?'} score=${(req.score || 0).toFixed(1)}${eff}${mx}${rel}`,
				);
				available -= cost;
				spawned = true;
				break; // one creep per spawn per tick
			}
		}
		if (!spawned && available >= 200 && Game.time % 50 === 0) {
			console.log(`Apex v4 spawn ${room.name}: IDLE energy=${available}/${capacity} (no candidate fit)`);
		}
	}
}

module.exports = {
	run,
	planColony,
	spawnEnergy,
	analyzeSources,
	estimateMaxHarvest,
	WORK_PER_SOURCE,
	harvestBudget,
};

export {};
