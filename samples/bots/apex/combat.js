/**
 * Offensive combat: squad assembly from attack_* flags, target prioritization.
 * Defenders are handled in spawn + role.defender; this module orchestrates attack waves.
 */
const config = require('config');
const { moveToRoom, goDo, militaryParts } = require('util');

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
	// Prefer towers, spawns, then any hostile structure.
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
	structures.sort((a, b) => (prio[a.structureType] ?? 9) - (prio[b.structureType] ?? 9) || a.hits - b.hits);
	return structures[0] || null;
}

function runAttacker(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return;
	if (creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
		return;
	}
	const target = attackTarget(creep.room);
	if (!target) {
		// Recycle near flag or idle center.
		const flag = Game.flags[creep.memory.flag];
		if (flag && !creep.pos.inRangeTo(flag, 2)) creep.moveTo(flag);
		return;
	}
	if (creep.pos.isNearTo(target)) {
		creep.attack(target);
	} else {
		creep.moveTo(target, { reusePath: 5, ignoreCreeps: false });
		// Pre-heal / ranged if mixed body.
		if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(target, 3)) {
			creep.rangedAttack(target);
		}
	}
}

function runRanged(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return;
	if (creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
		return;
	}
	const target = attackTarget(creep.room);
	const hostiles = creep.room.find(FIND_HOSTILE_CREEPS);
	const nearEnemy = hostiles.find(h => creep.pos.inRangeTo(h, 2));

	// Kite: if enemy melee adjacent-ish, step away.
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
			creep.moveTo(target, { reusePath: 5, range: 3 });
		}
	}
}

function runHealer(creep) {
	const targetRoom = creep.memory.targetRoom;
	// Stick with most damaged allied attacker in squad or self.
	let patients = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.my) continue;
		if (c.memory && c.memory.squad === creep.memory.squad) patients.push(c);
	}
	if (!patients.length) patients = [ creep ];
	patients.sort((a, b) => (a.hits / a.hitsMax) - (b.hits / b.hitsMax));
	const patient = patients[0];

	if (creep.hits < creep.hitsMax) {
		creep.heal(creep);
	} else if (creep.pos.isNearTo(patient)) {
		creep.heal(patient);
	} else if (creep.pos.inRangeTo(patient, 3)) {
		creep.rangedHeal(patient);
		creep.moveTo(patient, { reusePath: 3 });
	} else {
		creep.moveTo(patient, { reusePath: 3 });
	}

	// Move toward target room once healthy.
	if (targetRoom && creep.room.name !== targetRoom && patient.hits > patient.hitsMax * 0.8) {
		moveToRoom(creep, targetRoom);
	}
}

function runDismantler(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return;
	if (creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
		return;
	}
	const structs = creep.room.find(FIND_HOSTILE_STRUCTURES, {
		filter: s => s.structureType !== STRUCTURE_CONTROLLER,
	});
	structs.sort((a, b) => {
		const prio = { [STRUCTURE_RAMPART]: 0, [STRUCTURE_WALL]: 1, [STRUCTURE_TOWER]: 2, [STRUCTURE_SPAWN]: 3 };
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
		const spawn = creep.room.find(FIND_MY_SPAWNS)[0];
		if (spawn && !creep.pos.inRangeTo(spawn, 5)) creep.moveTo(spawn, { reusePath: 20 });
		return;
	}
	hostiles.sort((a, b) => militaryParts(b) - militaryParts(a) || a.hits - b.hits);
	const target = hostiles[0];
	if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(target, 3)) {
		creep.rangedAttack(target);
	}
	if (creep.pos.isNearTo(target)) {
		if (creep.getActiveBodyparts(ATTACK)) creep.attack(target);
	} else {
		creep.moveTo(target, { reusePath: 3 });
	}
	if (creep.getActiveBodyparts(HEAL) && creep.hits < creep.hitsMax) creep.heal(creep);
}

/**
 * Count military creeps assigned to an attack room.
 */
function squadCounts(targetRoom) {
	const counts = { attacker: 0, ranged: 0, healer: 0, dismantler: 0 };
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory || c.memory.targetRoom !== targetRoom) continue;
		if (counts[c.memory.role] != null) counts[c.memory.role]++;
	}
	return counts;
}

/**
 * Spawn requests for active attack flags (consumed by spawn manager).
 */
function attackSpawnRequests(homeRoom) {
	const attacks = (Memory.empire && Memory.empire.attacks) || [];
	const reqs = [];
	const want = config.attackSquad;
	for (const atk of attacks) {
		const c = squadCounts(atk.room);
		const squadId = `atk_${atk.room}`;
		const baseMem = { targetRoom: atk.room, squad: squadId, flag: atk.flag, home: homeRoom.name };
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
};
