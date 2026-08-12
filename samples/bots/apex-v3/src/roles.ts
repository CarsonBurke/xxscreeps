// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * Apex v3 — strict role delegation.
 * Creep state via CreepMem / Role enums only (see creepMem.ts).
 */
const config = require('./config');
const {
	energyOf, freeCapacity, updateWorking, goDo, goTo, moveToRoom,
	nearestEnergyPickup, doPickup, militaryParts,
} = require('./util');
const { markImmovable } = require('./trafficCore');
const {
	getRole, setRole, roleName, getHome, setHome, getSourceId, setSourceId,
	getTargetRoom, setTargetRoom, getRemote, getSeat, setSeat, getPioneer,
	setPioneer, getQueue, getArrived, setArrived, Role,
} = require('./creepMem');

let combat = null;
try {
	combat = require('./combat');
} catch {
	try {
		const war = require('./war');
		combat = war.runners || war.combat || null;
	} catch {
		combat = null;
	}
}

/**
 * Early multi-skill worker.
 * P0: spawn-fill → **upgrade** until RCL≥2 (and on downgrade risk) → then build.
 * Never let construction starve the controller at RCL1.
 */
function bootstrap(creep) {
	const home = getHome(creep);
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
	const ctrl = room.controller;
	const rcl = ctrl && ctrl.my ? (ctrl.level || 1) : 1;
	const downgradeRisk = ctrl && ctrl.my &&
		(ctrl.ticksToDowngrade || Infinity) < (config.upgradeSafeTicks || 8000);
	// RCL1 → RCL2 is 200 energy; only then does CONTROLLER_STRUCTURES unlock extensions
	const pushController = rcl < 2 || downgradeRisk;
	// Spawn capacity — 5 extensions = 550 unlocks 5W miners → 40 e/t remotes
	let spawnCap = 0;
	for (const s of room.find(FIND_MY_STRUCTURES)) {
		if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
			spawnCap += s.store.getCapacity(RESOURCE_ENERGY) || 0;
		}
	}
	const needExt = rcl >= 2 && spawnCap < 550;

	// 1) Keep spawn minimally fueled — during ext rush prefer building over full fill
	const fill = room.find(FIND_MY_STRUCTURES, {
		filter: s =>
			(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
			freeCapacity(s.store, RESOURCE_ENERGY) > 0,
	});
	if (fill.length) {
		const spawn = room.find(FIND_MY_SPAWNS)[0];
		const spawnEmpty = spawn && energyOf(spawn.store) < 100;
		if (!needExt || spawnEmpty) {
			const t = creep.pos.findClosestByRange(fill);
			goDo(creep, t, 1, () => creep.transfer(t, RESOURCE_ENERGY));
			return;
		}
	}

	// 2) RCL1→2 only (200 energy) — then extensions unlock
	if (pushController && ctrl && ctrl.my) {
		goDo(creep, ctrl, 3, () => creep.upgradeController(ctrl));
		return;
	}

	// 3) Extension rush — dump energy into sites (binding unlock for 5W)
	if (needExt) {
		const extSite = creep.pos.findClosestByRange(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_EXTENSION,
		}) || creep.pos.findClosestByRange(FIND_MY_CONSTRUCTION_SITES);
		if (extSite) {
			goDo(creep, extSite, 3, () => creep.build(extSite));
			return;
		}
	}

	// 4) Other infrastructure.
	const site = creep.pos.findClosestByRange(FIND_MY_CONSTRUCTION_SITES);
	if (site) {
		goDo(creep, site, 3, () => creep.build(site));
		return;
	}

	// 5) Idle upgrade / GCL push.
	if (ctrl && ctrl.my) {
		goDo(creep, ctrl, 3, () => creep.upgradeController(ctrl));
	}
}

function harvester(creep) {
	const sid = getSourceId(creep);
	let source = sid && Game.getObjectById(sid);
	const remoteRoom = getRemote(creep) || getTargetRoom(creep);

	// Remote miners: walk to target room before (re)binding a source.
	// getObjectById fails when the room is not visible.
	if (!source && remoteRoom && creep.room.name !== remoteRoom) {
		moveToRoom(creep, remoteRoom);
		return;
	}

	if (!source) {
		// Multiple harvesters per source OK — pick source with least WORK assigned
		// (or keep memory sid if it becomes visible).
		const workBy = {};
		for (const n in Game.creeps) {
			const c = Game.creeps[n];
			const role = getRole(c);
			const src = getSourceId(c);
			if ((role === Role.harvester || role === Role.remoteHarvester) && src) {
				workBy[src] = (workBy[src] || 0) + c.getActiveBodyparts(WORK);
			}
		}
		const sources = creep.room.find(FIND_SOURCES);
		if (sid) {
			const preferred = sources.find(s => s.id === sid);
			if (preferred) {
				setSourceId(creep, preferred.id);
				source = preferred;
			}
		}
		if (!source && sources.length) {
			sources.sort((a, b) => (workBy[a.id] || 0) - (workBy[b.id] || 0));
			setSourceId(creep, sources[0].id);
			source = sources[0];
		} else if (!source && remoteRoom && creep.room.name !== remoteRoom) {
			moveToRoom(creep, remoteRoom);
			return;
		} else if (!source) {
			return;
		}
	}

	if (creep.room.name !== source.pos.roomName) {
		moveToRoom(creep, source.pos.roomName);
		return;
	}

	if (!getSeat(creep)) {
		const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		})[0];
		if (container) setSeat(creep, container.id);
	}
	const seatId = getSeat(creep);
	const seat = seatId && Game.getObjectById(seatId);
	if (seat && !creep.pos.isEqualTo(seat.pos)) {
		goTo(creep, seat.pos, { range: 0, preferRoads: true });
		return;
	}
	if (!creep.pos.isNearTo(source)) {
		goTo(creep, source.pos, { range: 1, preferRoads: true });
		return;
	}
	markImmovable(creep);
	creep.harvest(source);

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
		creep.drop(RESOURCE_ENERGY);
	}
}

function hauler(creep) {
	const home = getHome(creep);
	const working = updateWorking(creep);

	if (!working) {
		const sid = getSourceId(creep);
		const source = sid && Game.getObjectById(sid);
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
		const remote = getRemote(creep) || getTargetRoom(creep);
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
		const nearCont = spawn.pos.findInRange(FIND_STRUCTURES, 3, {
			filter: s => s.structureType === STRUCTURE_CONTAINER && freeCapacity(s.store) > 0,
		})[0];
		if (nearCont) {
			goDo(creep, nearCont, 1, () => creep.transfer(nearCont, RESOURCE_ENERGY));
			return;
		}
		if (!creep.pos.inRangeTo(spawn, 2)) {
			goTo(creep, spawn.pos, { range: 2, preferRoads: true });
			return;
		}
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

function filler(creep) {
	const home = getHome(creep) || creep.room.name;
	if (creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const room = creep.room;
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (!spawn) return;

	if (!creep.pos.inRangeTo(spawn, config.fillerRange || 3)) {
		goTo(creep, spawn.pos, { range: config.fillerRange || 3, preferRoads: true });
		return;
	}

	const working = updateWorking(creep);
	if (!working) {
		const pickup = nearestEnergyPickup(creep.pos, config.fillerRange + 2, { minAmount: 10 });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		// Idle on pad — hold seat
		markImmovable(creep);
		return;
	}

	const targets = room.find(FIND_MY_STRUCTURES, {
		filter: s =>
			(s.structureType === STRUCTURE_SPAWN ||
				s.structureType === STRUCTURE_EXTENSION ||
				s.structureType === STRUCTURE_TOWER) &&
			freeCapacity(s.store, RESOURCE_ENERGY) > 0,
	});
	targets.sort((a, b) => {
		const pa = a.structureType === STRUCTURE_TOWER ? 2 : 1;
		const pb = b.structureType === STRUCTURE_TOWER ? 2 : 1;
		if (pa !== pb) return pa - pb;
		return energyOf(a.store) - energyOf(b.store);
	});
	if (targets.length) {
		const t = creep.pos.findClosestByRange(targets.slice(0, 12)) || targets[0];
		if (creep.pos.isNearTo(t)) {
			markImmovable(creep);
			creep.transfer(t, RESOURCE_ENERGY);
		} else goTo(creep, t.pos, { range: 1, preferRoads: true });
	} else {
		markImmovable(creep);
	}
}

function upgrader(creep) {
	const home = getHome(creep);
	if (home && creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const room = creep.room;
	const ctrl = room.controller;
	if (!ctrl || !ctrl.my) return;

	const park = config.upgraderParkRange || 3;
	const rcl = ctrl.level || 1;
	const working = updateWorking(creep);

	if (!working) {
		// Prefer energy near controller (containers / drops / links).
		const pickup = nearestEnergyPickup(ctrl.pos, park + 1, { minAmount: 10, links: true });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		if (room.storage && energyOf(room.storage.store) > 200) {
			goDo(creep, room.storage, 1, () => creep.withdraw(room.storage, RESOURCE_ENERGY));
			return;
		}
		// No controller logistics yet — pull spawn network or self-harvest (any RCL).
		const spawnNet = room.find(FIND_MY_STRUCTURES, {
			filter: s =>
				(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
				energyOf(s.store) > 0,
		});
		if (spawnNet.length && !room.storage) {
			const t = creep.pos.findClosestByRange(spawnNet);
			goDo(creep, t, 1, () => creep.withdraw(t, RESOURCE_ENERGY));
			return;
		}
		const source = creep.pos.findClosestByRange(FIND_SOURCES_ACTIVE)
			|| creep.pos.findClosestByRange(FIND_SOURCES);
		if (source) {
			goDo(creep, source, 1, () => creep.harvest(source));
			return;
		}
		if (!creep.pos.inRangeTo(ctrl, park)) {
			goTo(creep, ctrl.pos, { range: park, preferRoads: true });
		} else markImmovable(creep);
		return;
	}

	goDo(creep, ctrl, 3, () => creep.upgradeController(ctrl));
	if (creep.pos.inRangeTo(ctrl, 3)) markImmovable(creep);
}

function builder(creep) {
	const home = getHome(creep);
	if (getPioneer(creep)) {
		return pioneer(creep);
	}
	const working = updateWorking(creep);

	if (!working) {
		if (home && creep.room.name !== home) {
			moveToRoom(creep, home);
			return;
		}
		const pickup = nearestEnergyPickup(creep.pos, 50, { minAmount: 20, links: true });
		if (pickup) {
			doPickup(creep, pickup);
			return;
		}
		// Extension rush: withdraw from spawn network (otherwise only drops feed builders)
		const spawnNet = creep.room.find(FIND_MY_STRUCTURES, {
			filter: s =>
				(s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) &&
				energyOf(s.store) >= 50,
		});
		if (spawnNet.length) {
			const t = creep.pos.findClosestByRange(spawnNet);
			goDo(creep, t, 1, () => creep.withdraw(t, RESOURCE_ENERGY));
			return;
		}
		if (creep.room.storage && energyOf(creep.room.storage.store) > 200) {
			goDo(creep, creep.room.storage, 1, () => creep.withdraw(creep.room.storage, RESOURCE_ENERGY));
		}
		return;
	}

	// Prefer home sites (extensions unlock 5W) over remote containers
	let site = null;
	const homeRoom = home && Game.rooms[home];
	for (const rname in Game.rooms) {
		const r = Game.rooms[rname];
		if (!r.controller || !r.controller.my) continue;
		if (r.find(FIND_MY_SPAWNS).length) continue;
		site = r.find(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_SPAWN,
		})[0];
		if (site) break;
	}
	if (!site && homeRoom) {
		// Extensions first at home
		site = homeRoom.find(FIND_MY_CONSTRUCTION_SITES, {
			filter: s => s.structureType === STRUCTURE_EXTENSION,
		})[0] || homeRoom.find(FIND_MY_CONSTRUCTION_SITES)[0];
	}
	if (!site) site = creep.pos.findClosestByRange(FIND_MY_CONSTRUCTION_SITES);
	if (!site) {
		// Only leave home for remote sites when home has no sites
		for (const rname in Game.rooms) {
			if (home && rname === home) continue;
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

	upgrader(creep);
}

function pioneer(creep) {
	const target = getTargetRoom(creep);
	if (!target) {
		setPioneer(creep, false);
		return builder(creep);
	}
	if (creep.room.name !== target) {
		moveToRoom(creep, target);
		return;
	}
	const room = creep.room;
	if (room.find(FIND_MY_SPAWNS).length > 0) {
		setHome(creep, room.name);
		setRole(creep, Role.bootstrap);
		setPioneer(creep, false);
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

function reserver(creep) {
	const target = getTargetRoom(creep);
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
	const target = getTargetRoom(creep);
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
	const target = getTargetRoom(creep);
	if (!target) {
		creep.suicide();
		return;
	}
	if (creep.room.name === target) {
		const arrived = getArrived(creep) || Game.time;
		setArrived(creep, arrived);
		if (Game.time - arrived > 5) {
			const q = getQueue(creep);
			if (q && q.length) {
				setTargetRoom(creep, q.shift());
				setArrived(creep, undefined);
			} else creep.suicide();
		}
		return;
	}
	moveToRoom(creep, target);
}

function defender(creep) {
	if (combat && combat.runDefender) return combat.runDefender(creep);
	const home = getHome(creep) || creep.room.name;
	if (creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}
	const hostiles = creep.room.find(FIND_HOSTILE_CREEPS);
	if (!hostiles.length) {
		const spawn = creep.room.find(FIND_MY_SPAWNS)[0];
		if (spawn && !creep.pos.inRangeTo(spawn, 5)) goTo(creep, spawn.pos, { range: 5 });
		return;
	}
	hostiles.sort((a, b) => militaryParts(b) - militaryParts(a));
	const t = hostiles[0];
	if (creep.pos.isNearTo(t) && creep.getActiveBodyparts(ATTACK)) creep.attack(t);
	else if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(t, 3)) creep.rangedAttack(t);
	else goTo(creep, t.pos, { range: 1, preferRoads: false });
}

const runners = {
	[Role.bootstrap]: bootstrap,
	[Role.harvester]: harvester,
	[Role.remoteHarvester]: harvester,
	[Role.hauler]: hauler,
	[Role.remoteHauler]: hauler,
	[Role.filler]: filler,
	[Role.upgrader]: upgrader,
	[Role.builder]: builder,
	[Role.reserver]: reserver,
	[Role.claimer]: claimer,
	[Role.scout]: scout,
	[Role.defender]: defender,
	[Role.attacker]: combat && combat.runAttacker,
	[Role.ranged]: combat && combat.runRanged,
	[Role.healer]: combat && combat.runHealer,
	[Role.dismantler]: combat && combat.runDismantler,
};

function run(creep) {
	const role = getRole(creep);
	const fn = runners[role];
	if (!fn) return;
	try {
		fn(creep);
	} catch (err) {
		console.log(`Apex v3 role error ${creep.name} ${roleName(role)}: ${err}`);
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

export { run, runners, bootstrap, harvester, hauler, filler, upgrader, builder };
