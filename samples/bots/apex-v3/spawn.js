/**
 * Apex v3 spawn queue — delegation roles + optional war/economy recommendations.
 */
const config = require('config');
const body = require('body');
const intel = require('intel');
const empire = require('empire');
const {
	energyOf, hostileThreat, stillAlive, estimatePathLength,
} = require('util');

const { bodyCost: costOf } = body;

let war = null;
let economy = null;
try { war = require('war'); } catch { /* optional until integrated */ }
try { economy = require('economy'); } catch { /* optional */ }

function roleCount(home, role, extraFilter) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory || c.memory.role !== role) continue;
		if (c.memory.home && c.memory.home !== home) continue;
		if (extraFilter && !extraFilter(c)) continue;
		if (!stillAlive(c, 0)) continue;
		n++;
	}
	return n;
}

function harvestersOnSource(sourceId) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if ((c.memory.role === 'harvester' || c.memory.role === 'remoteHarvester') &&
			c.memory.sourceId === sourceId && stillAlive(c, 0)) n++;
	}
	return n;
}

function haulersForSource(sourceId) {
	let n = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if ((c.memory.role === 'hauler' || c.memory.role === 'remoteHauler') &&
			c.memory.sourceId === sourceId && stillAlive(c, 0)) n++;
	}
	return n;
}

function creepsForTarget(targetRoom, role) {
	const out = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if (role && c.memory.role !== role) continue;
		if (c.memory.targetRoom === targetRoom || c.memory.remote === targetRoom) out.push(c);
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

function haulerCarryParts(pathLen) {
	// energy needed ≈ 2 * pathLen * e/t ; each CARRY holds 50
	const ePerTick = config.sourceEnergyPerTick || 10;
	const energyCarry = 2 * pathLen * ePerTick;
	const parts = Math.ceil(energyCarry / 50);
	return Math.max(config.minHaulerCarryParts, Math.min(config.maxHaulerCarryParts, parts));
}

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
			energyMode,
		});
	};

	const threat = hostileThreat(room);
	const totalCreeps = Object.keys(Game.creeps).filter(n => {
		const c = Game.creeps[n];
		return c.memory && c.memory.home === home;
	}).length;

	// Emergency bootstrap
	const hasHarvester = roleCount(home, 'harvester') + roleCount(home, 'bootstrap');
	const hasMover = roleCount(home, 'hauler') + roleCount(home, 'filler') + roleCount(home, 'bootstrap');
	if (totalCreeps === 0 || (hasHarvester === 0 && hasMover === 0)) {
		push('bootstrap', 0, {}, {}, 'available');
		return reqs;
	}
	if (available < config.emergencyEnergy && hasMover === 0) {
		push('bootstrap', 0, {}, {}, 'available');
	}

	// Defense
	if (threat >= (config.defendThreatParts || 1)) {
		const want = Math.min(3, Math.ceil(threat / 3));
		if (roleCount(home, 'defender') < want) push('defender', 5, {}, {}, 'available');
	}

	// Early multi-skill
	if (rcl < config.bootstrapRcl && capacity < config.specializeCapacity) {
		const boots = roleCount(home, 'bootstrap');
		const want = Math.min(config.maxBootstrapWorkers, room.find(FIND_SOURCES).length * 2);
		if (boots < want) push('bootstrap', 10, {}, {}, 'available');
		return reqs;
	}

	const spawnPos = spawns[0].pos;

	// Local harvesters + haulers
	for (const source of room.find(FIND_SOURCES)) {
		if (harvestersOnSource(source.id) < 1) {
			push('harvester', 10, { sourceId: source.id }, { workParts: config.sourceWorkParts });
		}
		const pathLen = estimatePathLength(spawnPos, source.pos);
		const carryParts = haulerCarryParts(pathLen);
		const wantH = pathLen > 25 ? 2 : 1;
		if (haulersForSource(source.id) < wantH) {
			push('hauler', 15, { sourceId: source.id }, { carryParts, pathLen });
		}
	}

	// Fillers — once we have extensions or storage path
	const extensions = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_EXTENSION,
	}).length;
	const wantFillers = extensions > 0 || capacity >= 550 ? Math.min(config.maxFillers || 2, 1 + Math.floor(extensions / 20)) : 0;
	if (roleCount(home, 'filler') < wantFillers) {
		push('filler', 12, {}, {});
	}

	// Remotes
	for (const remoteName of remotes) {
		const remoteIntel = intel.get(remoteName);
		const sources = (remoteIntel && remoteIntel.sources) || [];
		const ctrl = remoteIntel && remoteIntel.controller;
		const ticks = ctrl && ctrl.reservation ? ctrl.reservation.ticks : 0;
		const needReserve = !ctrl || !ctrl.reservation || ticks < config.reserveRefresh;
		if (needReserve && creepsForTarget(remoteName, 'reserver').length < 1 && rcl >= 3) {
			push('reserver', 20, { targetRoom: remoteName }, {});
		}
		for (const src of sources) {
			if (harvestersOnSource(src.id) < 1) {
				push('remoteHarvester', 22, {
					sourceId: src.id,
					targetRoom: remoteName,
					remote: remoteName,
				}, { workParts: config.sourceWorkParts });
			}
			const srcPos = new RoomPosition(src.pos.x, src.pos.y, src.pos.roomName || remoteName);
			const pathLen = estimatePathLength(spawnPos, srcPos);
			const carryParts = haulerCarryParts(pathLen);
			const wantRH = pathLen > 40 ? 2 : 1;
			if (haulersForSource(src.id) < wantRH) {
				push('remoteHauler', 25, {
					sourceId: src.id,
					targetRoom: remoteName,
					remote: remoteName,
				}, { carryParts, pathLen });
			}
		}
	}

	// Upgraders
	const storageE = room.storage ? energyOf(room.storage.store) : 0;
	const downGrade = room.controller.ticksToDowngrade || Infinity;
	let wantUp = 1;
	if (downGrade < config.upgradeSafeTicks) wantUp = 2;
	if (storageE > config.upgradePushStorage) {
		wantUp = Math.min(config.maxUpgraders, 2 + Math.floor(storageE / 50_000));
	}
	if (roleCount(home, 'upgrader') < wantUp) push('upgrader', 30, {}, { rcl });

	// Builders
	const sites = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	let remoteSites = 0;
	for (const rn of remotes) {
		if (Game.rooms[rn]) remoteSites += Game.rooms[rn].find(FIND_CONSTRUCTION_SITES).length;
	}
	const wantB = Math.min(config.maxBuilders, (sites + remoteSites) > 0 ? Math.ceil((sites + remoteSites) / 5) : 0);
	if (roleCount(home, 'builder') < wantB) push('builder', 35, {}, {});

	// Empire expansion intents (GCL-driven claimers/pioneers/escorts)
	for (const intent of empire.spawnIntentsFor(home)) {
		if (intent.role === 'claimer') {
			if (creepsForTarget(intent.memory.targetRoom, 'claimer').length >= 1) continue;
		}
		if (intent.role === 'builder' && intent.memory && intent.memory.pioneer) {
			let pioneers = 0;
			for (const n in Game.creeps) {
				const c = Game.creeps[n];
				if (c.memory && c.memory.pioneer && c.memory.targetRoom === intent.memory.targetRoom) {
					pioneers++;
				}
			}
			if (pioneers >= 2) continue;
		}
		if (intent.role === 'attacker') {
			if (creepsForTarget(intent.memory.targetRoom, 'attacker').length >= 1) continue;
		}
		push(intent.role, intent.priority, intent.memory || {}, intent.context || {}, 'capacity');
	}

	// Economy module recommendations (optional staffing from projections)
	if (Memory.empire && Memory.empire.economy) {
		const eco = Memory.empire.economy;
		const recs = eco.recommendations
			|| (eco.colonies && eco.colonies[home] && eco.colonies[home].recommendations)
			|| (eco.byColony && eco.byColony[home] && eco.byColony[home].recommendations)
			|| [];
		for (const rec of recs) {
			if (rec.home && rec.home !== home) continue;
			if (!rec.role || rec.count == null) continue;
			const have = roleCount(home, rec.role);
			if (have < rec.count) {
				push(rec.role, rec.priority || 40, rec.memory || {}, rec.context || {});
			}
		}
	}

	// War spawn requests
	if (war && war.spawnRequests) {
		for (const req of war.spawnRequests(room)) {
			reqs.push({
				role: req.role,
				priority: req.priority,
				memory: { ...req.memory, home },
				context: req.context || {},
				energyMode: req.energyMode || 'capacity',
			});
		}
	}

	// Manual claim flags still work even if empire missed them
	const claims = (Memory.empire && Memory.empire.claims) || [];
	for (const claim of claims) {
		if (creepsForTarget(claim.room, 'claimer').length >= 1) continue;
		if (empire.freeGcl() < 1) continue;
		// Prefer strongest home: only primary support spawns flag claimers
		const cp = empire.planFor(home);
		const plan = Memory.empire.plan;
		const exp = plan && plan.expansions && plan.expansions.find(e => e.room === claim.room);
		if (exp && exp.primaryHome && exp.primaryHome !== home) continue;
		push('claimer', 16, { targetRoom: claim.room }, {});
	}

	return reqs;
}

function nameFor(role) {
	return `${role}_${Game.time.toString(36)}_${Math.floor(Math.random() * 36).toString(36)}`;
}

function run(room, remotes) {
	const spawns = room.find(FIND_MY_SPAWNS).filter(s => !s.spawning);
	if (!spawns.length) return;

	const reqs = planColony(room, remotes);
	if (!reqs.length) return;
	reqs.sort((a, b) => a.priority - b.priority);

	let { capacity, available } = spawnEnergy(room);
	const hyst = config.spawnEnergyHysteresis || 0.9;
	const fulfilled = new Set();

	for (const spawn of spawns) {
		for (let i = 0; i < reqs.length; i++) {
			if (fulfilled.has(i)) continue;
			const req = reqs[i];
			const budget = req.energyMode === 'available' ? available : capacity;
			if (req.energyMode !== 'available' && available < budget * hyst) continue;

			const energy = req.energyMode === 'available' ? available : Math.min(available, capacity);
			const parts = body.build(req.role, energy, {
				...req.context,
				rcl: room.controller.level,
			});
			if (!parts.length) continue;
			const cost = costOf(parts);
			if (cost > available) continue;

			const name = nameFor(req.role);
			const result = spawn.spawnCreep(parts, name, { memory: req.memory });
			if (result === OK) {
				console.log(`Apex v3 spawn ${room.name}: ${req.role} [${parts}] cost=${cost}`);
				available -= cost;
				fulfilled.add(i);
				break;
			}
		}
	}
}

module.exports = { run, planColony, spawnEnergy, haulerCarryParts };
