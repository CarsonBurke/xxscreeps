/**
 * Intelligent spawn queue.
 *
 * Each colony builds a prioritized list of spawn requests based on:
 *  - emergency bootstrap
 *  - defense
 *  - static miners / haulers (local + remote)
 *  - reservation
 *  - upgrade / build / repair
 *  - minerals, scouts, claimers, attack squads
 *
 * Bodies scale to available energy capacity (or currently available energy for emergencies).
 */
const config = require('config');
const body = require('body');
const intel = require('intel');
const combat = require('combat');
const {
	energyOf, hostileThreat, creepsInRoom, creepsForTarget,
} = require('util');

const { bodyCost: costOf } = body;

function roleCount(home, role, extraFilter) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory || c.memory.role !== role) continue;
		if (c.memory.home && c.memory.home !== home) continue;
		if (!c.memory.home && c.room.name !== home && role !== 'scout') continue;
		if (extraFilter && !extraFilter(c)) continue;
		// Ignore nearly-dead creeps so we pre-spawn replacements.
		if (c.ticksToLive !== undefined && c.ticksToLive < 80) continue;
		n++;
	}
	return n;
}

function minersOnSource(sourceId) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if ((c.memory.role === 'miner' || c.memory.role === 'remoteMiner') &&
			c.memory.sourceId === sourceId &&
			(c.ticksToLive === undefined || c.ticksToLive > 80)) {
			n++;
		}
	}
	return n;
}

function haulersForSource(sourceId) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if ((c.memory.role === 'hauler' || c.memory.role === 'remoteHauler') &&
			c.memory.sourceId === sourceId &&
			(c.ticksToLive === undefined || c.ticksToLive > 80)) {
			n++;
		}
	}
	return n;
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

function haulerPartsFor(pathLen) {
	return Math.max(
		config.minHaulerCarryParts,
		Math.min(config.maxHaulerCarryParts, Math.ceil((pathLen * config.haulerCarryPerPathTile) / 50) * 2),
	);
}

/**
 * Build spawn requests for one colony room.
 */
function planColony(room, remotes) {
	const home = room.name;
	const rcl = room.controller.level;
	const spawns = room.find(FIND_MY_SPAWNS);
	if (!spawns.length) return [];

	const { capacity, available } = spawnEnergy(room);
	const reqs = [];
	const push = (role, priority, memory, context = {}, energyMode = 'capacity') => {
		reqs.push({
			role,
			priority,
			memory: { role, home, ...memory },
			context,
			energyMode, // 'capacity' | 'available'
		});
	};

	const creepsHere = creepsInRoom(home);
	const totalCreeps = creepsHere.length;
	const threat = hostileThreat(room);

	// ---- Emergency bootstrap ----
	const hasMiner = roleCount(home, 'miner') + roleCount(home, 'bootstrap');
	const hasHauler = roleCount(home, 'hauler') + roleCount(home, 'bootstrap');
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

	// Early RCL: bootstrap workers instead of full specialization.
	if (rcl < config.bootstrapRcl && capacity < 550) {
		const boots = roleCount(home, 'bootstrap');
		const sources = room.find(FIND_SOURCES).length;
		const wantBoots = sources * 3;
		if (boots < wantBoots) {
			push('bootstrap', 10, {}, {}, 'available');
		}
		// Still allow a single builder-equivalent via bootstrap.
		return reqs;
	}

	// ---- Local miners + haulers ----
	const spawnPos = spawns[0].pos;
	for (const source of room.find(FIND_SOURCES)) {
		if (minersOnSource(source.id) < 1) {
			push('miner', 10, { sourceId: source.id }, { workParts: config.sourceWorkParts });
		}
		const pathLen = intel.estimatePathLength(spawnPos, source.pos);
		const carryParts = haulerPartsFor(pathLen);
		// One hauler per source; two if long path.
		const wantHaulers = pathLen > 30 ? 2 : 1;
		if (haulersForSource(source.id) < wantHaulers) {
			push('hauler', 15, { sourceId: source.id }, { carryParts });
		}
	}

	// ---- Remotes ----
	for (const remoteName of remotes) {
		const remoteIntel = intel.get(remoteName);
		const sources = (remoteIntel && remoteIntel.sources) || [];

		// Reserver
		const ctrl = remoteIntel && remoteIntel.controller;
		const ticks = ctrl && ctrl.reservation ? ctrl.reservation.ticks : 0;
		const reservers = creepsForTarget(remoteName, 'reserver').length;
		const needReserve = !ctrl || !ctrl.reservation || ticks < config.reserveRefresh ||
			(ctrl.reservation && !isOurs(ctrl.reservation.username));
		if (needReserve && reservers < 1 && rcl >= 3) {
			push('reserver', 20, { targetRoom: remoteName }, {});
		}

		for (const src of sources) {
			if (minersOnSource(src.id) < 1) {
				push('remoteMiner', 22, {
					sourceId: src.id,
					targetRoom: remoteName,
					remote: remoteName,
				}, {});
			}
			const srcPos = new RoomPosition(src.pos.x, src.pos.y, src.pos.roomName || remoteName);
			const pathLen = intel.estimatePathLength(spawnPos, srcPos);
			const carryParts = haulerPartsFor(pathLen);
			const wantH = pathLen > 40 ? 2 : 1;
			if (haulersForSource(src.id) < wantH) {
				push('remoteHauler', 25, {
					sourceId: src.id,
					targetRoom: remoteName,
					remote: remoteName,
				}, { pathLen, carryParts });
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
	if (rcl === 8) wantUpgraders = Math.min(wantUpgraders, 2); // one fat upgrader often enough
	const upgraders = roleCount(home, 'upgrader');
	if (upgraders < wantUpgraders) {
		push('upgrader', 30, {}, { rcl });
	}

	// ---- Builders ----
	const sites = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	// Count remote sites for our remotes if visible.
	let remoteSites = 0;
	for (const rn of remotes) {
		if (Game.rooms[rn]) {
			remoteSites += Game.rooms[rn].find(FIND_CONSTRUCTION_SITES).length;
		}
	}
	const wantBuilders = Math.min(config.maxBuilders, sites + remoteSites > 0 ? Math.ceil((sites + remoteSites) / 5) : 0);
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

	// ---- Scouts ----
	if (Game.time % config.scoutInterval < 10 && roleCount(home, 'scout') < 1) {
		const stale = intel.pickRemotes(room).filter(r => intel.isStale(r));
		const adj = require('util').adjacentRoomNames(home).filter(r => intel.isStale(r));
		const queue = [ ...new Set([ ...stale, ...adj ]) ].slice(0, 5);
		if (queue.length) {
			push('scout', 50, { targetRoom: queue[0], queue: queue.slice(1) }, {});
		}
	}

	// ---- Expansion claimers + pioneers for unseeded colonies ----
	const claims = (Memory.empire && Memory.empire.claims) || [];
	const ownedRoomsList = [];
	for (const rname in Game.rooms) {
		const rr = Game.rooms[rname];
		if (rr.controller && rr.controller.my) ownedRoomsList.push(rr);
	}
	const ownedCount = ownedRoomsList.length;
	const gclFree = (Game.gcl ? Game.gcl.level : 1) - ownedCount;

	// Claimed rooms without a spawn need pioneer builders.
	for (const rr of ownedRoomsList) {
		if (rr.find(FIND_MY_SPAWNS).length > 0) continue;
		const pioneers = roleCount(home, 'builder', c => c.memory.pioneer && c.memory.targetRoom === rr.name);
		// Also count bootstrap pioneers.
		let extra = 0;
		for (const n in Game.creeps) {
			const c = Game.creeps[n];
			if (c.memory && c.memory.pioneer && c.memory.targetRoom === rr.name) extra++;
		}
		if (pioneers + extra < 2) {
			push('builder', 8, {
				pioneer: true,
				targetRoom: rr.name,
				role: 'builder',
			}, {}, 'capacity');
		}
	}

	for (const claim of claims) {
		if (creepsForTarget(claim.room, 'claimer').length >= 1) continue;
		if (gclFree < 1) continue;
		if (rcl < config.expandMinRcl && storageE < config.expandMinStorage) continue;
		push('claimer', 55, { targetRoom: claim.room }, {});
	}

	// Auto-claim if GCL free and storage fat and high RCL.
	if (
		gclFree >= config.expandMinFreeGcl &&
		rcl >= config.expandMinRcl &&
		storageE >= config.expandMinStorage &&
		claims.length === 0
	) {
		for (const remoteName of remotes) {
			const ri = intel.get(remoteName);
			if (ri && ri.controller && !ri.controller.owner && creepsForTarget(remoteName, 'claimer').length === 0) {
				push('claimer', 60, { targetRoom: remoteName }, {});
				break;
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

function isOurs(username) {
	for (const n in Game.creeps) {
		const o = Game.creeps[n].owner;
		if (o) return o.username === username;
	}
	return false;
}

function nameFor(role) {
	return `${role}_${Game.time.toString(36)}_${Math.floor(Math.random() * 36).toString(36)}`;
}

/**
 * Try to spawn the highest-priority request that fits energy.
 */
function run(room, remotes) {
	const spawns = room.find(FIND_MY_SPAWNS).filter(s => !s.spawning);
	if (!spawns.length) return;

	const reqs = planColony(room, remotes);
	if (!reqs.length) return;

	// Sort by priority (lower = more important), then stable.
	reqs.sort((a, b) => a.priority - b.priority);

	const { capacity, available } = spawnEnergy(room);
	const prioIndex = {};
	config.spawnPriority.forEach((r, i) => { prioIndex[r] = i; });
	reqs.sort((a, b) => {
		if (a.priority !== b.priority) return a.priority - b.priority;
		return (prioIndex[a.role] ?? 99) - (prioIndex[b.role] ?? 99);
	});

	for (const spawn of spawns) {
		let spawned = false;
		for (const req of reqs) {
			const budget = req.energyMode === 'available' ? available : capacity;
			// Wait for full capacity for non-emergency so we spawn optimal bodies.
			if (req.energyMode !== 'available' && available < Math.min(capacity, budget) * 0.9 && available < budget) {
				// Allow if we've been waiting — use whatever we have above half.
				if (available < budget * 0.5) continue;
			}
			const energy = req.energyMode === 'available' ? available : Math.min(available, capacity);
			const parts = body.build(req.role, energy, {
				...req.context,
				rcl: room.controller.level,
			});
			if (!parts.length) continue;
			if (costOf(parts) > available) continue;

			const name = nameFor(req.role);
			const result = spawn.spawnCreep(parts, name, {
				memory: req.memory,
				directions: [ TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT, LEFT, TOP_LEFT ],
			});
			if (result === OK) {
				console.log(`Apex spawn ${room.name}: ${req.role} [${parts}] cost=${costOf(parts)}`);
				spawned = true;
				// Refresh available for second spawn in same room.
				break;
			}
		}
		if (!spawned) {
			// Idle: nothing to spawn this tick.
		}
	}
}

module.exports = {
	run,
	planColony,
	spawnEnergy,
	roleCount,
};
