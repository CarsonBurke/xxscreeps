/**
 * Civilian + economic role runners (Apex v2).
 * Uses traffic for movement/parking; stats hooks; safeGet; drop-mine only until container.
 */
const config = require('config');
const {
	energyOf, freeCapacity, safeGet,
} = require('util');
const traffic = require('traffic');
const combat = require('combat');
const logistics = require('logistics');
const stats = require('stats');

const { goDo, moveToRoom, moveCreep, harvestEnergy, deliverEnergy, minerSeat, haulerPark, parkAt, recordHeat } = traffic;

// ---------- Bootstrap worker (early game) ----------
function bootstrap(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home && energyOf(creep.store) === 0) {
		moveToRoom(creep, home);
		return;
	}

	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	if (!creep.memory.working) {
		harvestEnergy(creep);
		recordHeat(creep);
		return;
	}

	const room = creep.room;
	const needFill = room.find(FIND_MY_STRUCTURES, {
		filter: s =>
			(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
			freeCapacity(s.store, RESOURCE_ENERGY) > 0,
	});
	if (needFill.length) {
		const t = creep.pos.findClosestByPath(needFill);
		if (t) goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
		return;
	}

	const site = creep.pos.findClosestByPath(FIND_MY_CONSTRUCTION_SITES);
	if (site) {
		goDo(creep, site, 3, () => {
			const r = creep.build(site);
			stats.noteBuild(creep, r);
			return r;
		});
		return;
	}

	if (room.controller && room.controller.my) {
		goDo(creep, room.controller, 3, () => {
			const r = creep.upgradeController(room.controller);
			stats.noteUpgrade(creep, r);
			return r;
		});
	}
}

// ---------- Static miner ----------
function miner(creep) {
	let source = safeGet(creep.memory.sourceId);
	if (!source) {
		const sources = creep.room.find(FIND_SOURCES);
		const taken = new Set();
		for (const n in Game.creeps) {
			const m = Game.creeps[n].memory;
			if (m && (m.role === 'miner' || m.role === 'remoteMiner') && m.sourceId && n !== creep.name) {
				taken.add(m.sourceId);
			}
		}
		const free = sources.find(s => !taken.has(s.id)) || sources[0];
		if (free) {
			creep.memory.sourceId = free.id;
			source = free;
		} else {
			return;
		}
	}

	if (creep.room.name !== source.pos.roomName) {
		moveToRoom(creep, source.pos.roomName);
		return;
	}

	// Prefer standing on container seat; park so haulers are not blocked.
	if (!creep.memory.seat) {
		const seatPos = minerSeat(source);
		if (seatPos) {
			const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
				filter: s => s.structureType === STRUCTURE_CONTAINER && s.pos.isEqualTo(seatPos),
			})[0];
			if (container) creep.memory.seat = container.id;
			else {
				creep.memory.seatX = seatPos.x;
				creep.memory.seatY = seatPos.y;
			}
		}
	}

	const seatStruct = safeGet(creep.memory.seat);
	if (seatStruct && !creep.pos.isEqualTo(seatStruct.pos)) {
		moveCreep(creep, seatStruct, { reusePath: 40, ignoreCreeps: false });
		recordHeat(creep);
		return;
	}
	if (!seatStruct && creep.memory.seatX != null) {
		const sp = new RoomPosition(creep.memory.seatX, creep.memory.seatY, source.pos.roomName);
		if (!creep.pos.isEqualTo(sp)) {
			moveCreep(creep, sp, { reusePath: 40 });
			return;
		}
	}

	if (!creep.pos.isNearTo(source)) {
		moveCreep(creep, source, { reusePath: 40, ignoreCreeps: true });
		return;
	}

	const harvestResult = creep.harvest(source);
	stats.noteHarvest(creep, harvestResult);
	recordHeat(creep);

	// Drip into link/container; drop only until container/link exists.
	if (energyOf(creep.store) > 0) {
		const link = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK && freeCapacity(s.store, RESOURCE_ENERGY) > 0,
		})[0];
		if (link && creep.pos.isNearTo(link)) {
			creep.transfer(link, RESOURCE_ENERGY);
			return;
		}
		const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER && freeCapacity(s.store) > 0,
		})[0];
		if (container && creep.pos.isNearTo(container)) {
			creep.transfer(container, RESOURCE_ENERGY);
			return;
		}
		// Drop mining only until container (or link) exists.
		const anyContainer = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		})[0];
		const anyLink = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK,
		})[0];
		if (config.dropMineUntilContainer && !anyContainer && !anyLink) {
			creep.drop(RESOURCE_ENERGY);
		}
		// If container is full, drop is wasteful — leave energy in carry (blocks harvest slightly)
		// or drop only if full and no free capacity.
		else if (anyContainer && freeCapacity(anyContainer.store) === 0 && energyOf(creep.store) > 0) {
			// Container full: drop adjacent so haulers can clear (better than blocking WORK).
			creep.drop(RESOURCE_ENERGY);
		}
	}
}

// ---------- Hauler ----------
function hauler(creep) {
	const home = creep.memory.home;
	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	if (!creep.memory.working) {
		if (home && creep.room.name !== home) {
			moveToRoom(creep, home);
			return;
		}

		// Dynamic reassignment: if assigned source is empty and another is fuller, switch.
		let sourceId = creep.memory.sourceId;
		let source = safeGet(sourceId);
		if (config.haulerFullnessBias) {
			const neediest = logistics.neediestSource(creep.room);
			if (neediest) {
				const curFull = source ? containerEnergy(source) : 0;
				const newFull = containerEnergy(neediest);
				if (!source || (newFull > curFull + 200 && newFull > 100)) {
					creep.memory.sourceId = neediest.id;
					source = neediest;
					sourceId = neediest.id;
				}
			}
		}

		if (source) {
			// Prefer pickup/withdraw without standing on miner seat.
			const pile = source.pos.findInRange(FIND_DROPPED_RESOURCES, 1, {
				filter: r => r.resourceType === RESOURCE_ENERGY,
			})[0];
			if (pile) {
				goDo(creep, pile, 1, () => creep.pickup(pile));
				recordHeat(creep);
				return;
			}
			const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
				filter: s => s.structureType === STRUCTURE_CONTAINER && energyOf(s.store) > 0,
			})[0];
			if (container) {
				goDo(creep, container, 1, () => creep.withdraw(container, RESOURCE_ENERGY));
				recordHeat(creep);
				return;
			}
			// Link-first: withdraw from source link if hauler is near.
			if (config.linkFirst) {
				const link = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
					filter: s => s.structureType === STRUCTURE_LINK && energyOf(s.store) > 50,
				})[0];
				if (link) {
					goDo(creep, link, 1, () => creep.withdraw(link, RESOURCE_ENERGY));
					return;
				}
			}
			// Park near source without blocking miner while waiting for energy.
			const park = haulerPark(source);
			if (park && !creep.pos.isEqualTo(park) && !creep.pos.isNearTo(source)) {
				parkAt(creep, park);
				return;
			}
		}
		harvestEnergy(creep);
		recordHeat(creep);
		return;
	}

	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	deliverEnergy(creep, { preferUpgrade: true });
	recordHeat(creep);
}

function containerEnergy(source) {
	if (!source) return 0;
	const c = source.pos.findInRange(FIND_STRUCTURES, 1, {
		filter: s => s.structureType === STRUCTURE_CONTAINER,
	})[0];
	if (c) return energyOf(c.store);
	const d = source.pos.findInRange(FIND_DROPPED_RESOURCES, 1, {
		filter: r => r.resourceType === RESOURCE_ENERGY,
	})[0];
	return d ? d.amount : 0;
}

// ---------- Upgrader ----------
function upgrader(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	const room = creep.room;
	const ctrl = room.controller;
	if (!ctrl || !ctrl.my) return;

	if (!creep.memory.working) {
		// Link-first at controller.
		const link = ctrl.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK && energyOf(s.store) > 0,
		})[0];
		if (link) {
			goDo(creep, link, 1, () => creep.withdraw(link, RESOURCE_ENERGY));
			return;
		}
		const container = ctrl.pos.findInRange(FIND_STRUCTURES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER && energyOf(s.store) > 0,
		})[0];
		if (container) {
			goDo(creep, container, 1, () => creep.withdraw(container, RESOURCE_ENERGY));
			return;
		}
		if (room.storage && energyOf(room.storage.store) > 1000) {
			goDo(creep, room.storage, 1, () => creep.withdraw(room.storage, RESOURCE_ENERGY));
			return;
		}
		harvestEnergy(creep);
		return;
	}

	goDo(creep, ctrl, 3, () => {
		const r = creep.upgradeController(ctrl);
		stats.noteUpgrade(creep, r);
		return r;
	});
}

// ---------- Builder ----------
function builder(creep) {
	const home = creep.memory.home;
	if (creep.memory.pioneer) {
		return pioneer(creep);
	}

	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	if (!creep.memory.working) {
		if (home && creep.room.name !== home) moveToRoom(creep, home);
		else harvestEnergy(creep);
		return;
	}

	// Prefer finishing existing sites (same type concentration handled by planner).
	let site = null;
	for (const rname in Game.rooms) {
		const r = Game.rooms[rname];
		if (!r.controller || !r.controller.my) continue;
		if (r.find(FIND_MY_SPAWNS).length) continue;
		const spawnSite = r.find(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_SPAWN,
		})[0];
		if (spawnSite) {
			site = spawnSite;
			break;
		}
	}
	if (!site) site = creep.pos.findClosestByPath(FIND_MY_CONSTRUCTION_SITES);
	if (!site) {
		// Prefer most-progressed site empire-wide.
		let best = null;
		let bestProg = -1;
		for (const rname in Game.rooms) {
			for (const s of Game.rooms[rname].find(FIND_MY_CONSTRUCTION_SITES)) {
				const prog = s.progress / Math.max(1, s.progressTotal);
				if (prog > bestProg) {
					bestProg = prog;
					best = s;
				}
			}
		}
		site = best;
	}
	if (site) {
		if (creep.room.name !== site.pos.roomName) {
			moveToRoom(creep, site.pos.roomName);
			return;
		}
		goDo(creep, site, 3, () => {
			const r = creep.build(site);
			stats.noteBuild(creep, r);
			return r;
		});
		return;
	}

	upgrader(creep);
}

function pioneer(creep) {
	const target = creep.memory.targetRoom;
	if (!target) {
		creep.memory.pioneer = false;
		return builder(creep);
	}
	if (creep.room.name !== target) {
		moveToRoom(creep, target);
		return;
	}
	const room = creep.room;
	if (room.find(FIND_MY_SPAWNS).length > 0) {
		creep.memory.home = room.name;
		creep.memory.role = 'bootstrap';
		creep.memory.pioneer = false;
		return;
	}

	const ctrl = room.controller;
	let spawnSite = room.find(FIND_MY_CONSTRUCTION_SITES, {
		filter: s => s.structureType === STRUCTURE_SPAWN,
	})[0];
	if (!spawnSite && ctrl) {
		const terrain = room.getTerrain();
		for (let r = 2; r <= 4 && !spawnSite; r++) {
			for (let dx = -r; dx <= r; dx++) {
				for (let dy = -r; dy <= r; dy++) {
					if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
					const x = ctrl.pos.x + dx;
					const y = ctrl.pos.y + dy;
					if (x < 2 || x > 47 || y < 2 || y > 47) continue;
					if (terrain.get(x, y) === TERRAIN_MASK_WALL) continue;
					if (room.createConstructionSite(x, y, STRUCTURE_SPAWN) === OK) {
						spawnSite = true;
						break;
					}
				}
				if (spawnSite) break;
			}
		}
		spawnSite = room.find(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_SPAWN,
		})[0];
	}

	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	if (!creep.memory.working) {
		const source = creep.pos.findClosestByPath(FIND_SOURCES_ACTIVE) ||
			creep.pos.findClosestByPath(FIND_SOURCES);
		if (source) goDo(creep, source, 1, () => {
			const r = creep.harvest(source);
			stats.noteHarvest(creep, r);
			return r;
		});
		return;
	}
	if (spawnSite && spawnSite !== true) {
		goDo(creep, spawnSite, 3, () => {
			const r = creep.build(spawnSite);
			stats.noteBuild(creep, r);
			return r;
		});
		return;
	}
	if (ctrl) goDo(creep, ctrl, 3, () => {
		const r = creep.upgradeController(ctrl);
		stats.noteUpgrade(creep, r);
		return r;
	});
}

// ---------- Repairer ----------
function repairer(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	if (!creep.memory.working) {
		harvestEnergy(creep);
		return;
	}

	const room = creep.room;
	const rcl = room.controller ? room.controller.level : 1;
	const wallTarget = (config.wallHitsTarget[rcl] || config.wallHitsTarget[8] || 10_000);

	const targets = room.find(FIND_STRUCTURES, {
		filter: s => {
			if (s.structureType === STRUCTURE_WALL || s.structureType === STRUCTURE_RAMPART) {
				const storageE = room.storage ? energyOf(room.storage.store) : 0;
				if (storageE < config.wallGrindMinStorage && rcl < 6) return false;
				return s.hits < wallTarget && s.hits < s.hitsMax;
			}
			return s.hits < s.hitsMax * config.repairThreshold;
		},
	});
	targets.sort((a, b) => {
		const crit = t => (t.structureType === STRUCTURE_CONTAINER || t.structureType === STRUCTURE_ROAD) ? 0 : 1;
		if (crit(a) !== crit(b)) return crit(a) - crit(b);
		return a.hits - b.hits;
	});

	const target = targets[0];
	if (target) {
		goDo(creep, target, 3, () => creep.repair(target));
		return;
	}
	builder(creep);
}

// ---------- Scout ----------
function scout(creep) {
	const target = creep.memory.targetRoom;
	if (!target) {
		creep.suicide();
		return;
	}
	if (creep.room.name === target) {
		creep.memory.arrived = creep.memory.arrived || Game.time;
		if (Game.time - creep.memory.arrived > 5) {
			const queue = creep.memory.queue;
			if (queue && queue.length) {
				creep.memory.targetRoom = queue.shift();
				creep.memory.arrived = null;
			} else {
				creep.suicide();
			}
		} else {
			moveCreep(creep, new RoomPosition(25, 25, creep.room.name), { reusePath: 20 });
		}
		return;
	}
	moveToRoom(creep, target);
}

// ---------- Reserver ----------
function reserver(creep) {
	const target = creep.memory.targetRoom;
	if (!target) return;
	if (creep.room.name !== target) {
		moveToRoom(creep, target);
		return;
	}
	const ctrl = creep.room.controller;
	if (!ctrl) return;
	goDo(creep, ctrl, 1, () => creep.reserveController(ctrl));
	if (config.sign && (!ctrl.sign || ctrl.sign.text !== config.sign)) {
		if (creep.pos.isNearTo(ctrl)) creep.signController(ctrl, config.sign);
	}
}

// ---------- Claimer ----------
function claimer(creep) {
	const target = creep.memory.targetRoom;
	if (!target) return;
	if (creep.room.name !== target) {
		moveToRoom(creep, target);
		return;
	}
	const ctrl = creep.room.controller;
	if (!ctrl) return;
	if (ctrl.my) {
		creep.suicide();
		return;
	}
	if (ctrl.owner && !ctrl.my) {
		goDo(creep, ctrl, 1, () => creep.attackController(ctrl));
		return;
	}
	if (ctrl.reservation && ctrl.reservation.username) {
		const mine = Object.values(Game.creeps)[0];
		const myName = mine && mine.owner && mine.owner.username;
		if (myName && ctrl.reservation.username !== myName) {
			goDo(creep, ctrl, 1, () => creep.attackController(ctrl));
			return;
		}
	}
	goDo(creep, ctrl, 1, () => creep.claimController(ctrl));
}

// ---------- Remote miner / hauler ----------
function remoteMiner(creep) {
	miner(creep);
}

function remoteHauler(creep) {
	const home = creep.memory.home;
	const remote = creep.memory.remote || creep.memory.targetRoom;

	if (creep.memory.working && energyOf(creep.store) === 0) creep.memory.working = false;
	if (!creep.memory.working && freeCapacity(creep.store) === 0) creep.memory.working = true;

	if (!creep.memory.working) {
		if (remote && creep.room.name !== remote) {
			moveToRoom(creep, remote);
			return;
		}
		const source = safeGet(creep.memory.sourceId);
		if (source) {
			const drop = source.pos.findInRange(FIND_DROPPED_RESOURCES, 2, {
				filter: r => r.resourceType === RESOURCE_ENERGY,
			})[0];
			if (drop) {
				goDo(creep, drop, 1, () => creep.pickup(drop));
				recordHeat(creep);
				return;
			}
			const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
				filter: s => s.structureType === STRUCTURE_CONTAINER && energyOf(s.store) > 0,
			})[0];
			if (container) {
				goDo(creep, container, 1, () => creep.withdraw(container, RESOURCE_ENERGY));
				recordHeat(creep);
				return;
			}
			const park = haulerPark(source);
			if (park && !creep.pos.isNearTo(source)) {
				parkAt(creep, park);
				return;
			}
		}
		harvestEnergy(creep);
		return;
	}

	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	deliverEnergy(creep);
	recordHeat(creep);
}

// ---------- Mineral miner ----------
function mineralMiner(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const room = creep.room;
	const mineral = room.find(FIND_MINERALS)[0];
	const extractor = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_EXTRACTOR,
	})[0];
	if (!mineral || !extractor) return;

	if (freeCapacity(creep.store) === 0) {
		const dest = room.terminal || room.storage;
		if (dest) {
			const type = Object.keys(creep.store).find(k => k !== 'energy' && creep.store[k] > 0) ||
				mineral.mineralType;
			goDo(creep, dest, 1, () => creep.transfer(dest, type));
		}
		return;
	}

	if (extractor.cooldown > 0) {
		if (!creep.pos.isNearTo(mineral)) moveCreep(creep, mineral, { reusePath: 30 });
		return;
	}
	goDo(creep, mineral, 1, () => creep.harvest(mineral));
}

const runners = {
	bootstrap,
	miner,
	hauler,
	upgrader,
	builder,
	repairer,
	scout,
	reserver,
	claimer,
	remoteMiner,
	remoteHauler,
	mineralMiner,
	defender: combat.runDefender,
	attacker: combat.runAttacker,
	ranged: combat.runRanged,
	healer: combat.runHealer,
	dismantler: combat.runDismantler,
};

function run(creep) {
	const role = creep.memory && creep.memory.role;
	if (!role) {
		creep.memory.role = 'bootstrap';
	}
	const fn = runners[creep.memory.role];
	if (!fn) return;
	try {
		fn(creep);
	} catch (err) {
		console.log(`Apex v2 role error ${creep.name} ${creep.memory.role}: ${err}`);
	}
}

module.exports = {
	run,
	runners,
	bootstrap,
	miner,
	hauler,
	upgrader,
	builder,
	repairer,
	scout,
	reserver,
	claimer,
	remoteMiner,
	remoteHauler,
	mineralMiner,
};
