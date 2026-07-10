/**
 * Civilian + economic role runners.
 */
const config = require('config');
const {
	goDo, harvestEnergy, deliverEnergy, moveToRoom, energyOf,
	freeCapacity, updateWorking,
} = require('util');
const combat = require('combat');

// ---------- Bootstrap worker (early game) ----------
function bootstrap(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home && energyOf(creep.store) === 0) {
		moveToRoom(creep, home);
		return;
	}

	const working = updateWorking(creep);

	if (!working) {
		harvestEnergy(creep);
		return;
	}

	const room = creep.room;
	// Priority: fill spawn/extensions → build → upgrade.
	const needFill = room.find(FIND_MY_STRUCTURES, {
		filter: s =>
			(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
			freeCapacity(s.store, RESOURCE_ENERGY) > 0,
	});
	if (needFill.length) {
		const t = creep.pos.findClosestByPath(needFill);
		goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
		return;
	}

	const site = creep.pos.findClosestByPath(FIND_MY_CONSTRUCTION_SITES);
	if (site) {
		goDo(creep, site, 3, () => creep.build(site));
		return;
	}

	if (room.controller && room.controller.my) {
		goDo(creep, room.controller, 3, () => creep.upgradeController(room.controller));
	}
}

// ---------- Static miner ----------
function miner(creep) {
	const source = Game.getObjectById(creep.memory.sourceId);
	if (!source) {
		// Rebind in current room.
		const sources = creep.room.find(FIND_SOURCES);
		const taken = new Set();
		for (const n in Game.creeps) {
			const m = Game.creeps[n].memory;
			if (m && m.role === 'miner' && m.sourceId) taken.add(m.sourceId);
		}
		const free = sources.find(s => !taken.has(s.id)) || sources[0];
		if (free) creep.memory.sourceId = free.id;
		return;
	}

	if (creep.room.name !== source.pos.roomName) {
		moveToRoom(creep, source.pos.roomName);
		return;
	}

	// Prefer standing on container.
	if (!creep.memory.seat) {
		const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		})[0];
		if (container) creep.memory.seat = container.id;
	}
	const seat = creep.memory.seat && Game.getObjectById(creep.memory.seat);
	if (seat && !creep.pos.isEqualTo(seat.pos)) {
		creep.moveTo(seat, { reusePath: 30 });
		return;
	}
	if (!creep.pos.isNearTo(source)) {
		creep.moveTo(source, { reusePath: 30, ignoreCreeps: true });
		return;
	}

	creep.harvest(source);

	// Drip into link/container if CARRY used.
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
		// Drop for haulers if no container yet.
		if (!container && !link) creep.drop(RESOURCE_ENERGY);
	}
}

// ---------- Hauler ----------
function hauler(creep) {
	const home = creep.memory.home;
	const working = updateWorking(creep);

	if (!working) {
		// Pickup assigned source container or any energy in home room.
		if (home && creep.room.name !== home) {
			moveToRoom(creep, home);
			return;
		}
		const sourceId = creep.memory.sourceId;
		if (sourceId) {
			const source = Game.getObjectById(sourceId);
			if (source) {
				const pile = source.pos.findInRange(FIND_DROPPED_RESOURCES, 1, {
					filter: r => r.resourceType === RESOURCE_ENERGY,
				})[0];
				if (pile) {
					goDo(creep, pile, 1, () => creep.pickup(pile));
					return;
				}
				const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
					filter: s => s.structureType === STRUCTURE_CONTAINER && energyOf(s.store) > 0,
				})[0];
				if (container) {
					goDo(creep, container, 1, () => creep.withdraw(container, RESOURCE_ENERGY));
					return;
				}
			}
		}
		harvestEnergy(creep);
		return;
	}

	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	deliverEnergy(creep, { preferUpgrade: true });
}

// ---------- Upgrader ----------
function upgrader(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const working = updateWorking(creep);

	const room = creep.room;
	const ctrl = room.controller;
	if (!ctrl || !ctrl.my) return;

	if (!working) {
		// Prefer controller link / container / storage.
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

	goDo(creep, ctrl, 3, () => creep.upgradeController(ctrl));
}

// ---------- Builder ----------
function builder(creep) {
	const home = creep.memory.home;
	// Pioneer mode: establish a brand-new colony (no spawn yet).
	if (creep.memory.pioneer) {
		return pioneer(creep);
	}

	const working = updateWorking(creep);

	if (!working) {
		if (home && creep.room.name !== home) moveToRoom(creep, home);
		else harvestEnergy(creep);
		return;
	}

	// Prefer spawn sites in unseeded colonies, then home sites, then any visible site.
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
		for (const rname in Game.rooms) {
			const sites = Game.rooms[rname].find(FIND_MY_CONSTRUCTION_SITES);
			if (sites.length) {
				site = sites[0];
				break;
			}
		}
	}
	if (site) {
		if (creep.room.name !== site.pos.roomName) {
			moveToRoom(creep, site.pos.roomName);
			return;
		}
		goDo(creep, site, 3, () => creep.build(site));
		return;
	}

	// No sites: act as upgrader.
	upgrader(creep);
}

/**
 * Bootstrap a claimed room: place/build spawn near controller, harvest locally.
 */
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
		// Colony established — become a normal bootstrap worker for the new home.
		creep.memory.home = room.name;
		creep.memory.role = 'bootstrap';
		creep.memory.pioneer = false;
		return;
	}

	const ctrl = room.controller;
	// Place spawn site if missing.
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
						spawnSite = true; // will re-find next tick
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

	const working = updateWorking(creep);

	if (!working) {
		const source = creep.pos.findClosestByRange(FIND_SOURCES_ACTIVE) || creep.pos.findClosestByRange(FIND_SOURCES);
		if (source) goDo(creep, source, 1, () => creep.harvest(source));
		return;
	}
	if (spawnSite) {
		goDo(creep, spawnSite, 3, () => creep.build(spawnSite));
		return;
	}
	if (ctrl) goDo(creep, ctrl, 3, () => creep.upgradeController(ctrl));
}

// ---------- Repairer ----------
function repairer(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const working = updateWorking(creep);

	if (!working) {
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
		// Critical first (roads/containers), then lowest hits ratio.
		const crit = t => (t.structureType === STRUCTURE_CONTAINER || t.structureType === STRUCTURE_ROAD) ? 0 : 1;
		if (crit(a) !== crit(b)) return crit(a) - crit(b);
		return a.hits - b.hits;
	});

	const target = targets[0];
	if (target) {
		goDo(creep, target, 3, () => creep.repair(target));
		return;
	}
	// Idle help build.
	builder(creep);
}

// ---------- Scout ----------
function scout(creep) {
	const target = creep.memory.targetRoom;
	if (!target) {
		// Wander adjacent unexplored.
		creep.suicide();
		return;
	}
	if (creep.room.name === target) {
		// Circle once then mark done.
		creep.memory.arrived = creep.memory.arrived || Game.time;
		if (Game.time - creep.memory.arrived > 5) {
			// Pick next stale room from memory queue or suicide.
			const queue = creep.memory.queue;
			if (queue && queue.length) {
				creep.memory.targetRoom = queue.shift();
				creep.memory.arrived = null;
			} else {
				creep.suicide();
			}
		} else {
			creep.moveTo(25, 25, { reusePath: 20 });
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
	// Attack reservation/ownership if needed, else claim.
	if (ctrl.owner && !ctrl.my) {
		goDo(creep, ctrl, 1, () => creep.attackController(ctrl));
		return;
	}
	if (ctrl.reservation && ctrl.reservation.username) {
		// If reserved by other, attack; if ours, claim still works when reservation ends — attack to clear.
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
	// Same as miner; memory.sourceId in remote room.
	miner(creep);
}

function remoteHauler(creep) {
	const home = creep.memory.home;
	const remote = creep.memory.remote || creep.memory.targetRoom;

	const working = updateWorking(creep);

	if (!working) {
		if (remote && creep.room.name !== remote) {
			moveToRoom(creep, remote);
			return;
		}
		const source = Game.getObjectById(creep.memory.sourceId);
		if (source) {
			const drop = source.pos.findInRange(FIND_DROPPED_RESOURCES, 2, {
				filter: r => r.resourceType === RESOURCE_ENERGY,
			})[0];
			if (drop) {
				goDo(creep, drop, 1, () => creep.pickup(drop));
				return;
			}
			const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
				filter: s => s.structureType === STRUCTURE_CONTAINER && energyOf(s.store) > 0,
			})[0];
			if (container) {
				goDo(creep, container, 1, () => creep.withdraw(container, RESOURCE_ENERGY));
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
			const type = Object.keys(creep.store).find(k => k !== 'energy' && creep.store[k] > 0) || mineral.mineralType;
			goDo(creep, dest, 1, () => creep.transfer(dest, type));
		}
		return;
	}

	if (extractor.cooldown > 0) {
		// Wait on spot.
		if (!creep.pos.isNearTo(mineral)) creep.moveTo(mineral, { reusePath: 30 });
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
		// Unassigned: bootstrap behavior.
		creep.memory.role = 'bootstrap';
	}
	const fn = runners[creep.memory.role];
	if (!fn) return;
	try {
		fn(creep);
	} catch (err) {
		console.log(`Apex role error ${creep.name} ${creep.memory.role}: ${err}`);
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
