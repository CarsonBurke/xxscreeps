/**
 * Offensive combat + defenders.
 * - Attack squads form at rally before march
 * - Healers stick to lowest-HP ally in squad
 * - Defenders only when threat; recycle when safe
 */
const config = require('config');
const { militaryParts } = require('util');
const { moveToRoom, moveCreep, goDo } = require('traffic');

function attackTarget(room) {
	if (!room) return null;
	const hostiles = room.find(FIND_HOSTILE_CREEPS);
	if (hostiles.length) {
		hostiles.sort((a, b) => {
			const ha = a.getActiveBodyparts(HEAL);
			const hb = b.getActiveBodyparts(HEAL);
			if (ha !== hb) return hb - ha;
			return a.hits - b.hits;
		});
		return hostiles[0];
	}
	const structures = room.find(FIND_HOSTILE_STRUCTURES, {
		filter: s => s.structureType !== STRUCTURE_CONTROLLER,
	});
	const prio = {
		[STRUCTURE_TOWER]: 0,
		[STRUCTURE_SPAWN]: 1,
		[STRUCTURE_RAMPART]: 2,
		[STRUCTURE_EXTENSION]: 3,
		[STRUCTURE_STORAGE]: 4,
	};
	structures.sort((a, b) =>
		(prio[a.structureType] ?? 9) - (prio[b.structureType] ?? 9) || a.hits - b.hits);
	return structures[0] || null;
}

function rallyPos(creep) {
	const empire = Memory.empire || {};
	if (empire.rally && empire.rally.pos) {
		const p = empire.rally.pos;
		return new RoomPosition(p.x, p.y, p.roomName || empire.rally.room);
	}
	// Default: near home spawn.
	const home = creep.memory.home;
	if (home && Game.rooms[home]) {
		const spawn = Game.rooms[home].find(FIND_MY_SPAWNS)[0];
		if (spawn) return spawn.pos;
	}
	return null;
}

/**
 * True when enough squad members are near rally (or rally disabled).
 */
function squadReadyToMarch(creep) {
	if (!config.squadRallyBeforeMarch) return true;
	if (creep.memory.marching) return true;

	const squadId = creep.memory.squad;
	const targetRoom = creep.memory.targetRoom;
	if (!squadId || !targetRoom) return true;

	const want = config.attackSquad || {};
	const need = (want.attackers || 0) + (want.healers || 0) + (want.ranged || 0);
	if (need <= 1) return true;

	const rally = rallyPos(creep);
	const range = config.squadRallyRange || 3;

	let members = 0;
	let atRally = 0;
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory || c.memory.squad !== squadId) continue;
		if (c.spawning) continue;
		members++;
		if (c.memory.marching) {
			// Once any marches, all may march.
			creep.memory.marching = true;
			return true;
		}
		if (rally && c.pos.inRangeTo(rally, range + 2)) atRally++;
		// Already in target room counts as ready.
		if (c.room.name === targetRoom) atRally++;
	}

	// Need majority of expected squad or all currently living members at rally.
	const expected = Math.min(need, Math.max(members, 1));
	if (atRally >= Math.ceil(expected * 0.75) && members >= Math.ceil(need * 0.5)) {
		// Mark whole squad marching.
		for (const name in Game.creeps) {
			const c = Game.creeps[name];
			if (c.memory && c.memory.squad === squadId) c.memory.marching = true;
		}
		return true;
	}
	return false;
}

function goToRally(creep) {
	const rally = rallyPos(creep);
	if (!rally) return false;
	if (creep.pos.inRangeTo(rally, config.squadRallyRange || 3)) return true;
	if (creep.room.name !== rally.roomName) {
		moveToRoom(creep, rally.roomName);
	} else {
		moveCreep(creep, rally, { range: config.squadRallyRange || 3, reusePath: 10 });
	}
	return creep.pos.inRangeTo(rally, config.squadRallyRange || 3);
}

function runAttacker(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return;

	if (!squadReadyToMarch(creep)) {
		goToRally(creep);
		return;
	}

	if (creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
		return;
	}
	const target = attackTarget(creep.room);
	if (!target) {
		const flag = creep.memory.flag && Game.flags[creep.memory.flag];
		if (flag && !creep.pos.inRangeTo(flag, 2)) moveCreep(creep, flag);
		return;
	}
	if (creep.pos.isNearTo(target)) {
		creep.attack(target);
	} else {
		moveCreep(creep, target, { reusePath: 5, ignoreCreeps: false });
		if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(target, 3)) {
			creep.rangedAttack(target);
		}
	}
}

function runRanged(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return;

	if (!squadReadyToMarch(creep)) {
		goToRally(creep);
		return;
	}

	if (creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
		return;
	}
	const target = attackTarget(creep.room);
	const hostiles = creep.room.find(FIND_HOSTILE_CREEPS);
	const nearEnemy = hostiles.find(h => creep.pos.inRangeTo(h, 2));

	if (nearEnemy && nearEnemy.getActiveBodyparts(ATTACK) > 0) {
		const path = PathFinder.search(creep.pos, { pos: nearEnemy.pos, range: 3 }, {
			flee: true,
			maxRooms: 1,
		});
		if (path.path.length) creep.move(creep.pos.getDirectionTo(path.path[0]));
	}

	if (target) {
		if (creep.pos.inRangeTo(target, 3)) {
			if (creep.pos.inRangeTo(target, 1) && hostiles.length > 1) creep.rangedMassAttack();
			else creep.rangedAttack(target);
		} else {
			moveCreep(creep, target, { reusePath: 5, range: 3 });
		}
	}
}

function runHealer(creep) {
	const targetRoom = creep.memory.targetRoom;
	const squadId = creep.memory.squad;

	// Stick to lowest HP ally in squad (or any nearby ally if alone).
	let patients = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.my) continue;
		if (squadId && c.memory && c.memory.squad === squadId) patients.push(c);
	}
	if (!patients.length) {
		// Fallback: nearest hurt ally or self.
		patients = creep.pos.findInRange(FIND_MY_CREEPS, 10);
	}
	if (!patients.length) patients = [ creep ];

	patients.sort((a, b) => (a.hits / a.hitsMax) - (b.hits / b.hitsMax));
	const patient = patients[0];

	// Heal priority: self if hurt, else lowest HP ally.
	if (creep.hits < creep.hitsMax * 0.9) {
		creep.heal(creep);
	} else if (creep.pos.isNearTo(patient)) {
		creep.heal(patient);
	} else if (creep.pos.inRangeTo(patient, 3)) {
		creep.rangedHeal(patient);
	}

	if (!squadReadyToMarch(creep)) {
		goToRally(creep);
		// Still keep healing while rallying.
		if (creep.pos.isNearTo(patient) && patient.hits < patient.hitsMax) creep.heal(patient);
		return;
	}

	// Follow lowest-HP patient; if all healthy, march with squad.
	if (patient.hits < patient.hitsMax) {
		if (!creep.pos.inRangeTo(patient, 1)) {
			moveCreep(creep, patient, { reusePath: 3, range: 1 });
		}
	} else if (targetRoom && creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
	} else if (patient && patient.id !== creep.id) {
		// Stick near squad leader / first attacker.
		if (!creep.pos.inRangeTo(patient, 2)) {
			moveCreep(creep, patient, { reusePath: 3, range: 2 });
		}
	}
}

function runDismantler(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return;

	if (!squadReadyToMarch(creep)) {
		goToRally(creep);
		return;
	}

	if (creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
		return;
	}
	const structs = creep.room.find(FIND_HOSTILE_STRUCTURES, {
		filter: s => s.structureType !== STRUCTURE_CONTROLLER,
	});
	structs.sort((a, b) => {
		const prio = {
			[STRUCTURE_RAMPART]: 0,
			[STRUCTURE_WALL]: 1,
			[STRUCTURE_TOWER]: 2,
			[STRUCTURE_SPAWN]: 3,
		};
		return (prio[a.structureType] ?? 5) - (prio[b.structureType] ?? 5) || a.hits - b.hits;
	});
	const target = structs[0];
	if (!target) return;
	goDo(creep, target, 1, () => creep.dismantle(target));
}

function runDefender(creep) {
	const home = creep.memory.home || creep.room.name;
	if (creep.room.name !== home) {
		moveToRoom(creep, home);
		return;
	}

	const hostiles = creep.room.find(FIND_HOSTILE_CREEPS);
	if (!hostiles.length) {
		// Track safe ticks; recycle when safe for long enough.
		creep.memory.safeTicks = (creep.memory.safeTicks || 0) + 1;
		const recycleAfter = config.defenderRecycleSafeTicks || 200;
		if (creep.memory.safeTicks >= recycleAfter) {
			const spawn = creep.room.find(FIND_MY_SPAWNS)[0];
			if (spawn) {
				if (creep.pos.isNearTo(spawn)) {
					spawn.recycleCreep(creep);
					return;
				}
				moveCreep(creep, spawn, { range: 1, reusePath: 20 });
				return;
			}
			// No spawn: suicide as last resort to free CPU.
			if (creep.memory.safeTicks > recycleAfter + 100) creep.suicide();
			return;
		}
		const spawn = creep.room.find(FIND_MY_SPAWNS)[0];
		if (spawn && !creep.pos.inRangeTo(spawn, 5)) {
			moveCreep(creep, spawn, { range: 5, reusePath: 20 });
		}
		return;
	}

	creep.memory.safeTicks = 0;
	hostiles.sort((a, b) => militaryParts(b) - militaryParts(a) || a.hits - b.hits);
	const target = hostiles[0];
	if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(target, 3)) {
		creep.rangedAttack(target);
	}
	if (creep.pos.isNearTo(target)) {
		if (creep.getActiveBodyparts(ATTACK)) creep.attack(target);
	} else {
		moveCreep(creep, target, { reusePath: 3 });
	}
	if (creep.getActiveBodyparts(HEAL) && creep.hits < creep.hitsMax) creep.heal(creep);
}

function squadCounts(targetRoom) {
	const counts = { attacker: 0, ranged: 0, healer: 0, dismantler: 0 };
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory || c.memory.targetRoom !== targetRoom) continue;
		if (counts[c.memory.role] != null) counts[c.memory.role]++;
	}
	return counts;
}

function attackSpawnRequests(homeRoom) {
	const attacks = (Memory.empire && Memory.empire.attacks) || [];
	const reqs = [];
	const want = config.attackSquad;
	for (const atk of attacks) {
		const c = squadCounts(atk.room);
		const squadId = `atk_${atk.room}`;
		const baseMem = {
			targetRoom: atk.room,
			squad: squadId,
			flag: atk.flag,
			home: homeRoom.name,
			marching: false,
		};
		if (c.attacker < want.attackers) {
			reqs.push({ role: 'attacker', priority: 40, memory: { ...baseMem, role: 'attacker' }, context: {} });
		}
		if (c.healer < want.healers) {
			reqs.push({ role: 'healer', priority: 41, memory: { ...baseMem, role: 'healer' }, context: {} });
		}
		if (c.ranged < (want.ranged || 0)) {
			reqs.push({ role: 'ranged', priority: 42, memory: { ...baseMem, role: 'ranged' }, context: {} });
		}
	}
	return reqs;
}

module.exports = {
	runAttacker,
	runRanged,
	runHealer,
	runDismantler,
	runDefender,
	attackSpawnRequests,
	attackTarget,
	squadCounts,
	squadReadyToMarch,
};
