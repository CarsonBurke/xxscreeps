/**
 * Apex v3 — strict role delegation.
 *
 *  harvester  — sits on source; drops energy or transfers into adjacent container
 *  hauler     — moves energy from sources → storage / spawn-side piles
 *  filler     — parks by spawn/extensions; fills them from nearby energy
 *  upgrader   — parks at controller; eats nearby container/drops only
 *  builder    — builds/repairs using container/drops (never harvests)
 *  bootstrap  — early-game multi-skill until capacity allows specialization
 */
const config = require('config');
const {
	energyOf, freeCapacity, updateWorking, goDo, moveToRoom,
	nearestEnergyPickup, doPickup, militaryParts,
} = require('util');

// Combat runners from combat.js / war.js
let combat = null;
try {
	combat = require('combat');
} catch {
	try {
		const war = require('war');
		combat = war.runners || war.combat || null;
	} catch {
		combat = null;
	}
}

// ---------- Bootstrap (emergency / early) ----------
function bootstrap(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home && energyOf(creep.store) === 0) {
		moveToRoom(creep, home);
		return;
	}
	const working = updateWorking(creep);
	if (!working) {
		const source = creep.pos.findClosestByRange(FIND_SOURCES_ACTIVE)
			|| creep.pos.findClosestByRange(FIND_SOURCES);
		if (source) goDo(creep, source, 1, () => creep.harvest(source));
		return;
	}
	const room = creep.room;
	const fill = room.find(FIND_MY_STRUCTURES, {
		filter: s =>
			(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
			freeCapacity(s.store, RESOURCE_ENERGY) > 0,
	});
	if (fill.length) {
		const t = creep.pos.findClosestByRange(fill);
		goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
		return;
	}
	const site = creep.pos.findClosestByRange(FIND_MY_CONSTRUCTION_SITES);
	if (site) {
		goDo(creep, site, 3, () => creep.build(site));
		return;
	}
	if (room.controller && room.controller.my) {
		goDo(creep, room.controller, 3, () => creep.upgradeController(room.controller));
	}
}

// ---------- Harvester (static) ----------
function harvester(creep) {
	const source = Game.getObjectById(creep.memory.sourceId);
	if (!source) {
		const taken = {};
		for (const n in Game.creeps) {
			const m = Game.creeps[n].memory;
			if (m && (m.role === 'harvester' || m.role === 'remoteHarvester') && m.sourceId) {
				taken[m.sourceId] = true;
			}
		}
		const free = creep.room.find(FIND_SOURCES).find(s => !taken[s.id]);
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
		creep.moveTo(seat, { reusePath: 40, ignoreCreeps: true });
		return;
	}
	if (!creep.pos.isNearTo(source)) {
		creep.moveTo(source, { reusePath: 40, ignoreCreeps: true });
		return;
	}

	creep.harvest(source);

	// Empty CARRY into container or drop.
	if (energyOf(creep.store) > 0) {
		const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER && freeCapacity(s.store) > 0,
		})[0];
		if (container && creep.pos.isNearTo(container)) {
			creep.transfer(container, RESOURCE_ENERGY);
			return;
		}
		const link = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
			filter: s => s.structureType === STRUCTURE_LINK && freeCapacity(s.store, RESOURCE_ENERGY) > 0,
		})[0];
		if (link && creep.pos.isNearTo(link)) {
			creep.transfer(link, RESOURCE_ENERGY);
			return;
		}
		// Drop for haulers (design intent when no container yet).
		creep.drop(RESOURCE_ENERGY);
	}
}

// ---------- Hauler ----------
function hauler(creep) {
	const home = creep.memory.home;
	const working = updateWorking(creep);

	if (!working) {
		// Prefer assigned source drop/container, else any pickup in home/remote.
		const source = creep.memory.sourceId && Game.getObjectById(creep.memory.sourceId);
		if (source) {
			if (creep.room.name !== source.pos.roomName) {
				moveToRoom(creep, source.pos.roomName);
				return;
			}
			const drop = source.pos.findInRange(FIND_DROPPED_RESOURCES, 2, {
				filter: r => r.resourceType === RESOURCE_ENERGY && r.amount > 10,
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
		const remote = creep.memory.remote || creep.memory.targetRoom;
		if (remote && creep.room.name !== remote && !source) {
			moveToRoom(creep, remote);
			return;
		}
		const pickup = nearestEnergyPickup(creep.pos, 50, { minAmount: 30 });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		return;
	}

	// Deliver: storage first, else containers near spawn, else spawn/extensions, else drop near spawn for fillers.
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const room = creep.room;
	if (room.storage && freeCapacity(room.storage.store, RESOURCE_ENERGY) > 0) {
		goDo(creep, room.storage, 1, () => creep.transfer(room.storage, RESOURCE_ENERGY));
		return;
	}
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (spawn) {
		// Prefer hungry spawn network if no storage yet.
		const hungry = room.find(FIND_MY_STRUCTURES, {
			filter: s =>
				(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
				freeCapacity(s.store, RESOURCE_ENERGY) > 0,
		});
		if (hungry.length && !room.storage) {
			const t = creep.pos.findClosestByRange(hungry);
			goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
			return;
		}
		// Dump into container near spawn for fillers, or drop next to spawn.
		const nearCont = spawn.pos.findInRange(FIND_STRUCTURES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER && freeCapacity(s.store) > 0,
		})[0];
		if (nearCont) {
			goDo(creep, nearCont, 1, () => creep.transfer(nearCont, RESOURCE_ENERGY));
			return;
		}
		if (!creep.pos.inRangeTo(spawn, 2)) {
			creep.moveTo(spawn, { reusePath: 20 });
			return;
		}
		// Drop for filler if we can't transfer (full spawn network).
		if (hungry.length) {
			const t = creep.pos.findClosestByRange(hungry);
			if (creep.pos.isNearTo(t)) {
				creep.transfer(t, RESOURCE_ENERGY);
				return;
			}
		}
		creep.drop(RESOURCE_ENERGY);
		return;
	}
}

// ---------- Filler (stationary logistics at spawn) ----------
function filler(creep) {
	const home = creep.memory.home || creep.room.name;
	if (creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const room = creep.room;
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (!spawn) return;

	// Park within fillerRange of spawn.
	if (!creep.pos.inRangeTo(spawn, config.fillerRange || 3)) {
		creep.moveTo(spawn, { reusePath: 30, range: 1 });
		return;
	}

	const working = updateWorking(creep);
	if (!working) {
		const pickup = nearestEnergyPickup(creep.pos, config.fillerRange + 2, { minAmount: 10 });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		// Idle — stay put.
		return;
	}

	// Fill spawn/extensions only (not storage — that's hauler's job).
	const targets = room.find(FIND_MY_STRUCTURES, {
		filter: s =>
			(s.structureType === STRUCTURE_SPAWN ||
				s.structureType === STRUCTURE_EXTENSION ||
				s.structureType === STRUCTURE_TOWER) &&
			freeCapacity(s.store, RESOURCE_ENERGY) > 0,
	});
	// Prioritize spawn/extension over tower.
	targets.sort((a, b) => {
		const pa = a.structureType === STRUCTURE_TOWER ? 2 : 1;
		const pb = b.structureType === STRUCTURE_TOWER ? 2 : 1;
		if (pa !== pb) return pa - pb;
		return energyOf(a.store) - energyOf(b.store);
	});
	if (targets.length) {
		const t = creep.pos.findClosestByRange(targets.slice(0, 12)) || targets[0];
		if (creep.pos.isNearTo(t)) creep.transfer(t, RESOURCE_ENERGY);
		else creep.moveTo(t, { reusePath: 5, maxRooms: 1 });
		return;
	}
	// Nowhere to put energy — drop next to spawn for later or hold.
	if (energyOf(creep.store) > 0 && freeCapacity(spawn.store, RESOURCE_ENERGY) === 0) {
		// Stay full; next tick extensions may free.
	}
}

// ---------- Upgrader (controller park) ----------
function upgrader(creep) {
	const home = creep.memory.home;
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const room = creep.room;
	const ctrl = room.controller;
	if (!ctrl || !ctrl.my) return;

	const park = config.upgraderParkRange || 3;
	const working = updateWorking(creep);

	if (!working) {
		// Only take energy near controller (or link/container by controller).
		const pickup = nearestEnergyPickup(ctrl.pos, park + 1, { minAmount: 10, links: true });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		// Fall back: storage if close enough path, but prefer moving energy via haulers.
		if (room.storage && energyOf(room.storage.store) > 1000) {
			goDo(creep, room.storage, 1, () => creep.withdraw(room.storage, RESOURCE_ENERGY));
			return;
		}
		// Wait at controller for haulers to drop.
		if (!creep.pos.inRangeTo(ctrl, park)) creep.moveTo(ctrl, { reusePath: 20, range: 2 });
		return;
	}

	goDo(creep, ctrl, 3, () => creep.upgradeController(ctrl));
}

// ---------- Builder ----------
function builder(creep) {
	const home = creep.memory.home;
	if (creep.memory.pioneer) {
		return pioneer(creep);
	}
	const working = updateWorking(creep);

	if (!working) {
		if (home && creep.room.name !== home) {
			moveToRoom(creep, home);
			return;
		}
		// Never harvest — only containers/drops/storage.
		const pickup = nearestEnergyPickup(creep.pos, 50, { minAmount: 30, links: true });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		if (creep.room.storage && energyOf(creep.room.storage.store) > 500) {
			goDo(creep, creep.room.storage, 1, () => creep.withdraw(creep.room.storage, RESOURCE_ENERGY));
		}
		return;
	}

	// Sites: prefer unseeded colony spawns, then any.
	let site = null;
	for (const rname in Game.rooms) {
		const r = Game.rooms[rname];
		if (!r.controller || !r.controller.my) continue;
		if (r.find(FIND_MY_SPAWNS).length) continue;
		site = r.find(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_SPAWN,
		})[0];
		if (site) break;
	}
	if (!site) site = creep.pos.findClosestByRange(FIND_MY_CONSTRUCTION_SITES);
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

	// Repair critical non-walls.
	const repair = creep.room.find(FIND_STRUCTURES, {
		filter: s =>
			s.structureType !== STRUCTURE_WALL &&
			s.structureType !== STRUCTURE_RAMPART &&
			s.hits < s.hitsMax * (config.repairThreshold || 0.7),
	}).sort((a, b) => a.hits - b.hits)[0];
	if (repair) {
		goDo(creep, repair, 3, () => creep.repair(repair));
		return;
	}

	// Idle as upgrader.
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
		for (let r = 2; r <= 4 && !spawnSite; r++) {
			for (let dx = -r; dx <= r && !spawnSite; dx++) {
				for (let dy = -r; dy <= r && !spawnSite; dy++) {
					if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
					const x = ctrl.pos.x + dx;
					const y = ctrl.pos.y + dy;
					if (x < 2 || x > 47 || y < 2 || y > 47) continue;
					if (room.getTerrain().get(x, y) === TERRAIN_MASK_WALL) continue;
					if (room.createConstructionSite(x, y, STRUCTURE_SPAWN) === OK) {
						spawnSite = true;
					}
				}
			}
		}
		spawnSite = room.find(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_SPAWN,
		})[0];
	}
	const working = updateWorking(creep);
	if (!working) {
		// Pioneer may harvest — exception for unseeded rooms with no logistics.
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

// ---------- Reserver / claimer / scout ----------
function reserver(creep) {
	const target = creep.memory.targetRoom;
	if (!target) return;
	if (creep.room.name !== target) {
		moveToRoom(creep, target);
		return;
	}
	const ctrl = creep.room.controller;
	if (!ctrl) return;
	const mine = creep.owner && creep.owner.username;
	if (ctrl.reservation && ctrl.reservation.username && mine && ctrl.reservation.username !== mine) {
		goDo(creep, ctrl, 1, () => creep.attackController(ctrl));
		return;
	}
	goDo(creep, ctrl, 1, () => creep.reserveController(ctrl));
	if (config.sign && creep.pos.isNearTo(ctrl)) {
		if (!ctrl.sign || ctrl.sign.text !== config.sign) creep.signController(ctrl, config.sign);
	}
}

function claimer(creep) {
	const target = creep.memory.targetRoom;
	if (!target) return;
	if (creep.room.name !== target) {
		moveToRoom(creep, target);
		return;
	}
	const ctrl = creep.room.controller;
	if (!ctrl || ctrl.my) {
		if (ctrl && ctrl.my) creep.suicide();
		return;
	}
	if (ctrl.owner && !ctrl.my) {
		goDo(creep, ctrl, 1, () => creep.attackController(ctrl));
		return;
	}
	goDo(creep, ctrl, 1, () => creep.claimController(ctrl));
}

function scout(creep) {
	const target = creep.memory.targetRoom;
	if (!target) {
		creep.suicide();
		return;
	}
	if (creep.room.name === target) {
		creep.memory.arrived = creep.memory.arrived || Game.time;
		if (Game.time - creep.memory.arrived > 5) {
			const q = creep.memory.queue;
			if (q && q.length) {
				creep.memory.targetRoom = q.shift();
				creep.memory.arrived = null;
			} else creep.suicide();
		}
		return;
	}
	moveToRoom(creep, target);
}

function defender(creep) {
	if (combat && combat.runDefender) return combat.runDefender(creep);
	const home = creep.memory.home || creep.room.name;
	if (creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const hostiles = creep.room.find(FIND_HOSTILE_CREEPS);
	if (!hostiles.length) {
		const spawn = creep.room.find(FIND_MY_SPAWNS)[0];
		if (spawn && !creep.pos.inRangeTo(spawn, 5)) creep.moveTo(spawn, { reusePath: 20 });
		return;
	}
	hostiles.sort((a, b) => militaryParts(b) - militaryParts(a));
	const t = hostiles[0];
	if (creep.pos.isNearTo(t) && creep.getActiveBodyparts(ATTACK)) creep.attack(t);
	else if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(t, 3)) creep.rangedAttack(t);
	else creep.moveTo(t, { reusePath: 3 });
}

const runners = {
	bootstrap,
	harvester,
	remoteHarvester: harvester,
	hauler,
	remoteHauler: hauler,
	filler,
	upgrader,
	builder,
	reserver,
	claimer,
	scout,
	defender,
	attacker: combat && combat.runAttacker,
	ranged: combat && combat.runRanged,
	healer: combat && combat.runHealer,
	dismantler: combat && combat.runDismantler,
};

function run(creep) {
	if (!creep.memory || !creep.memory.role) creep.memory.role = 'bootstrap';
	const fn = runners[creep.memory.role];
	if (!fn) return;
	try {
		fn(creep);
	} catch (err) {
		console.log(`Apex v3 role error ${creep.name} ${creep.memory.role}: ${err}`);
	}
}

module.exports = {
	run,
	runners,
	bootstrap,
	harvester,
	hauler,
	filler,
	upgrader,
	builder,
};
