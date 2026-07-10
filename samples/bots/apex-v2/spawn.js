/**
 * Apex v2 spawn system:
 *  - Single priority queue per colony with energy-wait hysteresis
 *  - Per-source demand: hauler count from path length + throughput
 *  - Name registry / no duplicate source claims
 *  - Dead-creep replacement from ticksToLive + spawn ETA
 *  - Bootstrap that does not over-spawn workers
 */
const config = require('config');
const body = require('body');
const intel = require('intel');
const combat = require('combat');
const stats = require('stats');
const {
	energyOf, hostileThreat, creepsInRoom, creepsForTarget,
	stillAlive, spawnTimeForBody, estimatePathLength, safeGet, lowBucket,
} = require('util');

const { bodyCost: costOf } = body;

/** Registry of claimed source ids → creep name (refreshed each plan). */
function sourceClaims() {
	const claims = {};
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		const role = c.memory.role;
		if ((role === 'miner' || role === 'remoteMiner') && c.memory.sourceId) {
			if (stillAlive(c, spawnTimeForBody(c.body))) {
				// First claimer wins; second is a duplicate to reassign.
				if (!claims[c.memory.sourceId]) claims[c.memory.sourceId] = name;
			}
		}
	}
	return claims;
}

function roleCount(home, role, extraFilter, spawnEta = 0) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory || c.memory.role !== role) continue;
		if (c.memory.home && c.memory.home !== home) continue;
		if (!c.memory.home && c.room.name !== home && role !== 'scout') continue;
		if (extraFilter && !extraFilter(c)) continue;
		if (!stillAlive(c, spawnEta)) continue;
		n++;
	}
	return n;
}

function minersOnSource(sourceId, spawnEta = 0) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if ((c.memory.role === 'miner' || c.memory.role === 'remoteMiner') &&
			c.memory.sourceId === sourceId &&
			stillAlive(c, spawnEta)) {
			n++;
		}
	}
	return n;
}

function haulersForSource(sourceId, spawnEta = 0) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if ((c.memory.role === 'hauler' || c.memory.role === 'remoteHauler') &&
			c.memory.sourceId === sourceId &&
			stillAlive(c, spawnEta)) {
			n++;
		}
	}
	return n;
}

/**
 * Hauler demand from path length + throughput.
 * Round-trip ≈ 2 * pathLen; capacity needed ≈ rate * roundTrip.
 * Each hauler carry parts hold 50 energy.
 */
function haulerDemand(pathLen, opts = {}) {
	const rate = config.sourceEnergyPerTick || 10;
	const roundTrip = Math.max(10, 2 * (pathLen || 20));
	const energyOnPipe = rate * roundTrip;
	// On roads haulers move 1:1 with 2C1M; effective delivery rate scales with carry.
	let carryParts = Math.ceil(energyOnPipe / 50);
	carryParts = Math.max(config.minHaulerCarryParts, Math.min(config.maxHaulerCarryParts, carryParts));
	// Prefer sizing one fat hauler; add a second if path is long or container is full.
	let count = 1;
	if (pathLen > 35 || energyOnPipe > config.maxHaulerCarryParts * 50) count = 2;
	if (pathLen > 60) count = 3;
	if (opts.containerFull) count = Math.max(count, 2);
	if (opts.containerVeryFull) count = Math.max(count, Math.min(3, count + 1));
	return { count, carryParts };
}

function containerFullness(source) {
	if (!source) return 0;
	const c = source.pos.findInRange(FIND_STRUCTURES, 1, {
		filter: s => s.structureType === STRUCTURE_CONTAINER,
	})[0];
	if (!c || !c.store) return 0;
	const cap = c.store.getCapacity(RESOURCE_ENERGY) || 2000;
	return energyOf(c.store) / cap;
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

function isOurs(username) {
	for (const n in Game.creeps) {
		const o = Game.creeps[n].owner;
		if (o) return o.username === username;
	}
	return false;
}

/** Unique creep names via registry in Memory. */
function nameFor(role) {
	Memory.apex ||= {};
	Memory.apex.nameSeq = (Memory.apex.nameSeq || 0) + 1;
	const seq = Memory.apex.nameSeq;
	const name = `${role}_${Game.time.toString(36)}_${seq.toString(36)}`;
	// Ensure no collision with living creeps.
	if (Game.creeps[name]) return `${name}_x`;
	return name;
}

/**
 * Build prioritized spawn requests for one colony.
 */
function planColony(room, remotes) {
	const home = room.name;
	const rcl = room.controller.level;
	const spawns = room.find(FIND_MY_SPAWNS);
	if (!spawns.length) return [];

	const { capacity, available } = spawnEnergy(room);
	const claims = sourceClaims();
	const reqs = [];
	const push = (role, priority, memory, context = {}, energyMode = 'capacity') => {
		reqs.push({
			role,
			priority,
			memory: { role, home, ...memory },
			context,
			energyMode,
		});
	};

	// Spawn ETA for a typical miner (for replacement threshold).
	const sampleMiner = body.build('miner', capacity, { workParts: config.sourceWorkParts });
	const minerEta = spawnTimeForBody(sampleMiner);

	const creepsHere = creepsInRoom(home);
	const totalCreeps = creepsHere.length;
	const threat = hostileThreat(room);
	const bucketLow = lowBucket();

	// ---- Emergency bootstrap ----
	const hasMiner = roleCount(home, 'miner', null, minerEta) + roleCount(home, 'bootstrap');
	const hasHauler = roleCount(home, 'hauler', null, minerEta) + roleCount(home, 'bootstrap');
	if (totalCreeps === 0 || (hasMiner === 0 && hasHauler === 0)) {
		push('bootstrap', 0, {}, {}, 'available');
		return reqs;
	}
	if (available < config.emergencyEnergy && hasHauler === 0) {
		push('bootstrap', 0, {}, {}, 'available');
	}

	// ---- Defense ----
	if (threat >= config.defendThreatParts) {
		const defenders = roleCount(home, 'defender');
		const want = Math.min(3, Math.ceil(threat / 3));
		if (defenders < want) {
			push('defender', 5, {}, {}, 'available');
		}
	}

	// ---- Bootstrap phase: limited multi-role workers ----
	const specialize = rcl >= config.bootstrapRcl || capacity >= (config.specializeCapacity || 550);
	if (!specialize && capacity < (config.specializeCapacity || 550)) {
		const boots = roleCount(home, 'bootstrap');
		const sources = room.find(FIND_SOURCES).length;
		// Cap: 2 per source, hard max from config — do not over-spawn.
		const wantBoots = Math.min(
			config.maxBootstrapWorkers || 6,
			Math.max(2, sources * 2),
		);
		if (boots < wantBoots) {
			push('bootstrap', 10, {}, {}, 'available');
		}
		// Early: also allow one miner if we can afford 550+ for specialization transition.
		if (capacity >= 400 && boots >= sources) {
			for (const source of room.find(FIND_SOURCES)) {
				if (minersOnSource(source.id, minerEta) < 1 && !claims[source.id]) {
					push('miner', 12, { sourceId: source.id }, { workParts: Math.min(3, config.sourceWorkParts) });
				}
			}
		}
		return reqs;
	}

	// ---- Local miners + haulers (per-source demand) ----
	const spawnPos = spawns[0].pos;
	for (const source of room.find(FIND_SOURCES)) {
		const mEta = minerEta;
		if (minersOnSource(source.id, mEta) < 1) {
			// Avoid duplicate claims if another miner already registered.
			if (!claims[source.id]) {
				push('miner', 10, { sourceId: source.id }, { workParts: config.sourceWorkParts });
			}
		}

		const pathLen = estimatePathLength(spawnPos, source.pos);
		const fullness = containerFullness(source);
		const demand = haulerDemand(pathLen, {
			containerFull: fullness > 0.6,
			containerVeryFull: fullness > 0.85,
		});
		const hEta = spawnTimeForBody(body.build('hauler', capacity, { carryParts: demand.carryParts }));
		if (haulersForSource(source.id, hEta) < demand.count) {
			push('hauler', 15, { sourceId: source.id }, { carryParts: demand.carryParts });
		}
	}

	// ---- Remotes (respect FSM phase + bucket) ----
	if (!bucketLow) {
		for (const remoteName of remotes) {
			const remoteMem = (room.memory.apex && room.memory.apex.remoteState &&
				room.memory.apex.remoteState[remoteName]) || {};
			const phase = remoteMem.phase || 'scout';
			const remoteIntel = intel.get(remoteName);
			const sources = (remoteIntel && remoteIntel.sources) || [];

			// Abandoned remotes: skip.
			if (phase === 'abandoned') continue;

			// Reserver once past scout.
			if (phase !== 'scout') {
				const ctrl = remoteIntel && remoteIntel.controller;
				const ticks = ctrl && ctrl.reservation ? ctrl.reservation.ticks : 0;
				const reservers = creepsForTarget(remoteName, 'reserver').filter(c => stillAlive(c, 50));
				const needReserve = !ctrl || !ctrl.reservation || ticks < config.reserveRefresh ||
					(ctrl.reservation && !isOurs(ctrl.reservation.username));
				if (needReserve && reservers.length < 1 && rcl >= 3) {
					push('reserver', 20, { targetRoom: remoteName }, {});
				}
			}

			// Mine/haul only in mine or haul phases.
			if (phase === 'mine' || phase === 'haul' || phase === 'container') {
				for (const src of sources) {
					if (phase === 'container') {
						// Wait for container phase completion before full mining crew;
						// allow one remote miner to help build via drop mining.
						if (minersOnSource(src.id, minerEta) < 1 && !claims[src.id]) {
							push('remoteMiner', 22, {
								sourceId: src.id,
								targetRoom: remoteName,
								remote: remoteName,
							}, {});
						}
						continue;
					}

					if (minersOnSource(src.id, minerEta) < 1 && !claims[src.id]) {
						push('remoteMiner', 22, {
							sourceId: src.id,
							targetRoom: remoteName,
							remote: remoteName,
						}, {});
					}

					if (phase === 'haul' || phase === 'mine') {
						const srcPos = new RoomPosition(src.pos.x, src.pos.y, src.pos.roomName || remoteName);
						const pathLen = estimatePathLength(spawnPos, srcPos);
						const remoteRoom = Game.rooms[remoteName];
						let fullness = 0;
						if (remoteRoom) {
							const liveSrc = safeGet(src.id);
							if (liveSrc) fullness = containerFullness(liveSrc);
						}
						const demand = haulerDemand(pathLen, {
							containerFull: fullness > 0.5,
							containerVeryFull: fullness > 0.8,
						});
						// Long remotes always want haul phase haulers.
						if (phase === 'mine' && demand.count > 1) {
							// During mine-only, still seed one hauler.
							demand.count = 1;
						}
						const hEta = spawnTimeForBody(
							body.build('remoteHauler', capacity, { pathLen, carryParts: demand.carryParts }),
						);
						if (haulersForSource(src.id, hEta) < demand.count) {
							push('remoteHauler', 25, {
								sourceId: src.id,
								targetRoom: remoteName,
								remote: remoteName,
							}, { pathLen, carryParts: demand.carryParts });
						}
					}
				}
			}
		}
	}

	// ---- Upgraders ----
	const storageE = room.storage ? energyOf(room.storage.store) : 0;
	const downGrade = room.controller.ticksToDowngrade || Infinity;
	let wantUpgraders = 1;
	if (downGrade < config.upgradeSafeTicks) wantUpgraders = Math.max(wantUpgraders, 2);
	if (storageE > config.upgradePushStorage) {
		wantUpgraders = Math.min(config.maxUpgraders, 2 + Math.floor(storageE / 50_000));
	}
	if (rcl === 8) wantUpgraders = Math.min(wantUpgraders, 2);
	const upEta = spawnTimeForBody(body.build('upgrader', capacity, { rcl }));
	if (roleCount(home, 'upgrader', null, upEta) < wantUpgraders) {
		push('upgrader', 30, {}, { rcl });
	}

	// ---- Builders ----
	const sites = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	let remoteSites = 0;
	for (const rn of remotes) {
		if (Game.rooms[rn]) {
			remoteSites += Game.rooms[rn].find(FIND_CONSTRUCTION_SITES).length;
		}
	}
	const wantBuilders = Math.min(
		config.maxBuilders,
		sites + remoteSites > 0 ? Math.ceil((sites + remoteSites) / 5) : 0,
	);
	if (roleCount(home, 'builder') < wantBuilders) {
		push('builder', 35, {}, {});
	}

	// ---- Repairer ----
	if (rcl >= 3 && roleCount(home, 'repairer') < 1) {
		push('repairer', 38, {}, {});
	}

	// ---- Mineral ----
	if (rcl >= 6) {
		const mineral = room.find(FIND_MINERALS)[0];
		const extractor = room.find(FIND_MY_STRUCTURES, {
			filter: s => s.structureType === STRUCTURE_EXTRACTOR,
		})[0];
		if (mineral && extractor && mineral.mineralAmount > 0 && roleCount(home, 'mineralMiner') < 1) {
			push('mineralMiner', 45, {}, {});
		}
	}

	// ---- Scouts (skip when bucket low) ----
	if (!bucketLow && Game.time % config.scoutInterval < 10 && roleCount(home, 'scout') < 1) {
		const { adjacentRoomNames } = require('util');
		const stale = intel.pickRemotes(room).filter(r => intel.isStale(r));
		const adj = adjacentRoomNames(home).filter(r => intel.isStale(r));
		const queue = [ ...new Set([ ...stale, ...adj ]) ].slice(0, 5);
		if (queue.length) {
			push('scout', 50, { targetRoom: queue[0], queue: queue.slice(1) }, {});
		}
	}

	// ---- Expansion claimers + pioneers ----
	const claimsFlags = (Memory.empire && Memory.empire.claims) || [];
	const ownedRoomsList = [];
	for (const rname in Game.rooms) {
		const rr = Game.rooms[rname];
		if (rr.controller && rr.controller.my) ownedRoomsList.push(rr);
	}
	const ownedCount = ownedRoomsList.length;
	const gclFree = (Game.gcl ? Game.gcl.level : 1) - ownedCount;

	for (const rr of ownedRoomsList) {
		if (rr.find(FIND_MY_SPAWNS).length > 0) continue;
		let extra = 0;
		for (const n in Game.creeps) {
			const c = Game.creeps[n];
			if (c.memory && c.memory.pioneer && c.memory.targetRoom === rr.name && stillAlive(c, 50)) {
				extra++;
			}
		}
		if (extra < 2) {
			push('builder', 8, {
				pioneer: true,
				targetRoom: rr.name,
				role: 'builder',
			}, {}, 'capacity');
		}
	}

	if (!bucketLow) {
		for (const claim of claimsFlags) {
			if (creepsForTarget(claim.room, 'claimer').filter(c => stillAlive(c, 50)).length >= 1) continue;
			if (gclFree < 1) continue;
			if (rcl < config.expandMinRcl && storageE < config.expandMinStorage) continue;
			push('claimer', 55, { targetRoom: claim.room }, {});
		}

		if (
			gclFree >= config.expandMinFreeGcl &&
			rcl >= config.expandMinRcl &&
			storageE >= config.expandMinStorage &&
			claimsFlags.length === 0
		) {
			const target = intel.pickExpansionTarget(room);
			if (target && creepsForTarget(target, 'claimer').length === 0) {
				push('claimer', 60, { targetRoom: target }, {});
			}
		}
	}

	// ---- Attack squads ----
	for (const req of combat.attackSpawnRequests(room)) {
		reqs.push({
			role: req.role,
			priority: req.priority,
			memory: { ...req.memory, home },
			context: req.context || {},
			energyMode: 'capacity',
		});
	}

	return reqs;
}

/**
 * Global priority sort + hysteresis: do not spawn tiny bodies when full body is almost affordable.
 */
function run(room, remotes) {
	const spawns = room.find(FIND_MY_SPAWNS).filter(s => !s.spawning);
	if (!spawns.length) return;

	const reqs = planColony(room, remotes);
	if (!reqs.length) return;

	const prioIndex = {};
	config.spawnPriority.forEach((r, i) => { prioIndex[r] = i; });
	reqs.sort((a, b) => {
		if (a.priority !== b.priority) return a.priority - b.priority;
		return (prioIndex[a.role] ?? 99) - (prioIndex[b.role] ?? 99);
	});

	let { capacity, available } = spawnEnergy(room);
	const hysteresis = config.spawnEnergyHysteresis || 0.92;

	for (const spawn of spawns) {
		for (const req of reqs) {
			// Ideal body at full capacity.
			const ideal = body.build(req.role, capacity, {
				...req.context,
				rcl: room.controller.level,
			});
			const idealCost = ideal.length ? costOf(ideal) : 0;

			let energyBudget;
			if (req.energyMode === 'available') {
				energyBudget = available;
			} else {
				// Hysteresis: wait until we can nearly afford the ideal body.
				if (idealCost > 0 && available < idealCost * hysteresis) {
					// Exception: if available is high relative to capacity, spawn scaled-down.
					if (available < capacity * 0.5 || available < idealCost * 0.55) {
						continue;
					}
				}
				energyBudget = Math.min(available, capacity);
			}

			const parts = body.build(req.role, energyBudget, {
				...req.context,
				rcl: room.controller.level,
			});
			if (!parts.length) continue;
			const cost = costOf(parts);
			if (cost > available) continue;

			// For non-emergency: reject bodies that are drastically undersized vs ideal
			// when we're only a little short of ideal (avoids spam of tiny haulers).
			if (
				req.energyMode !== 'available' &&
				idealCost > 0 &&
				cost < idealCost * 0.6 &&
				available >= idealCost * hysteresis
			) {
				// Can almost afford ideal — wait rather than spawn tiny.
				continue;
			}

			const name = nameFor(req.role);
			const result = spawn.spawnCreep(parts, name, {
				memory: req.memory,
				directions: [ TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT, LEFT, TOP_LEFT ],
			});
			if (result === OK) {
				console.log(`Apex v2 spawn ${room.name}: ${req.role} [${parts}] cost=${cost}`);
				stats.noteSpawnCost(cost);
				available -= cost;
				// One spawn action per free spawn structure.
				break;
			}
		}
	}
}

module.exports = {
	run,
	planColony,
	spawnEnergy,
	roleCount,
	haulerDemand,
	sourceClaims,
	nameFor,
};
