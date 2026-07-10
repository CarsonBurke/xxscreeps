/**
 * Apex v3 combat role runners + tower helpers.
 *
 * Roles: defender, attacker, ranged, healer, dismantler.
 * Squads rally (flag rally_* or home spawn) before marching into targetRoom.
 * Campaign-aware: memory.campaignId / targetRoom set by war.js.
 *
 * Exports runners for roles.js and attackTarget / tower helpers for defense.
 */
const config = require('config');
const { militaryParts, energyOf, hostileThreat, moveToRoom, goDo } = require('util');

/** Role-tuned moveTo wrapper (apex-v3 has no traffic.js). */
function moveCreep(creep, target, opts = {}) {
	if (!target) return ERR_INVALID_TARGET;
	return creep.moveTo(target, {
		reusePath: opts.reusePath != null ? opts.reusePath : 5,
		range: opts.range != null ? opts.range : 1,
		ignoreCreeps: opts.ignoreCreeps,
		maxOps: 4000,
	});
}

// ---------------------------------------------------------------------------
// Target selection
// ---------------------------------------------------------------------------

/**
 * Prefer healers, then low-HP hostiles, then high military threat.
 * Falls back to hostile structures (towers → spawns → ramparts → rest).
 * Also targets invader cores when no creeps.
 */
function attackTarget(room) {
	if (!room) return null;

	const hostiles = room.find(FIND_HOSTILE_CREEPS);
	if (hostiles.length) {
		hostiles.sort((a, b) => {
			const ha = a.getActiveBodyparts(HEAL);
			const hb = b.getActiveBodyparts(HEAL);
			if (ha !== hb) return hb - ha;
			const ta = militaryParts(a);
			const tb = militaryParts(b);
			if (ta !== tb) return tb - ta;
			return a.hits - b.hits;
		});
		return hostiles[0];
	}

	// Invader cores (classic NPC structure with invader flag).
	// Invader cores when the constant exists (xxscreeps / modern Screeps).
	const coreType = typeof STRUCTURE_INVADER_CORE !== 'undefined' ? STRUCTURE_INVADER_CORE : 'invaderCore';
	const cores = room.find(FIND_HOSTILE_STRUCTURES, {
		filter: s => s.structureType === coreType ||
			(s.structureType === STRUCTURE_TOWER && s.owner && s.owner.username === 'Invader'),
	});
	if (cores.length) {
		cores.sort((a, b) => a.hits - b.hits);
		return cores[0];
	}

	const structures = room.find(FIND_HOSTILE_STRUCTURES, {
		filter: s => s.structureType !== STRUCTURE_CONTROLLER,
	});
	const prio = {
		[STRUCTURE_TOWER]: 0,
		[STRUCTURE_SPAWN]: 1,
		[coreType]: 1,
		[STRUCTURE_RAMPART]: 2,
		[STRUCTURE_EXTENSION]: 3,
		[STRUCTURE_STORAGE]: 4,
		[STRUCTURE_TERMINAL]: 5,
	};
	structures.sort((a, b) =>
		(prio[a.structureType] ?? 9) - (prio[b.structureType] ?? 9) || a.hits - b.hits);
	return structures[0] || null;
}

/** Hostile military score for tower focus (healers first, rampart-aware). */
function hostileScore(creep) {
	let score = 0;
	const heal = creep.getActiveBodyparts(HEAL);
	const attack = creep.getActiveBodyparts(ATTACK);
	const ranged = creep.getActiveBodyparts(RANGED_ATTACK);
	const work = creep.getActiveBodyparts(WORK);
	const focusHealers = config.towerFocusHealers !== false;
	if (focusHealers) score += heal * 1000;
	score += ranged * 40 + attack * 30 + work * 10;
	score += Math.max(0, 500 - creep.hits);

	if (config.towerRampartAware !== false && isOnHostileRampart(creep)) {
		score -= 800;
	}
	if (creep.hits < creep.hitsMax) score += 50;
	return score;
}

function isOnHostileRampart(creep) {
	const structs = creep.pos.lookFor(LOOK_STRUCTURES);
	return structs.some(s => s.structureType === STRUCTURE_RAMPART && !s.my);
}

// ---------------------------------------------------------------------------
// Squad rally / march
// ---------------------------------------------------------------------------

function rallyPos(creep) {
	const empire = Memory.empire || {};
	if (empire.rally && empire.rally.pos) {
		const p = empire.rally.pos;
		return new RoomPosition(p.x, p.y, p.roomName || empire.rally.room);
	}
	const home = creep.memory.home;
	if (home && Game.rooms[home]) {
		const spawn = Game.rooms[home].find(FIND_MY_SPAWNS)[0];
		if (spawn) return spawn.pos;
	}
	return null;
}

/**
 * True when enough squad members are near rally (or rally disabled / already marching).
 */
function squadReadyToMarch(creep) {
	if (config.squadRallyBeforeMarch === false) return true;
	if (creep.memory.marching) return true;

	const squadId = creep.memory.squad;
	const targetRoom = creep.memory.targetRoom;
	if (!squadId || !targetRoom) return true;

	const want = config.attackSquad || { attackers: 2, healers: 2, ranged: 1 };
	const need = (want.attackers || 0) + (want.healers || 0) + (want.ranged || 0) +
		(want.dismantlers || 0);
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
			creep.memory.marching = true;
			return true;
		}
		if (rally && c.pos.inRangeTo(rally, range + 2)) atRally++;
		if (c.room.name === targetRoom) atRally++;
	}

	const expected = Math.min(need, Math.max(members, 1));
	if (atRally >= Math.ceil(expected * 0.75) && members >= Math.ceil(need * 0.5)) {
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
	const range = config.squadRallyRange || 3;
	if (creep.pos.inRangeTo(rally, range)) return true;
	if (creep.room.name !== rally.roomName) {
		moveToRoom(creep, rally.roomName);
	} else {
		moveCreep(creep, rally, { range, reusePath: 10 });
	}
	return creep.pos.inRangeTo(rally, range);
}

// ---------------------------------------------------------------------------
// Melee / ranged engagement helpers
// ---------------------------------------------------------------------------

function meleeEngage(creep, target) {
	if (!target) return;
	if (creep.pos.isNearTo(target)) {
		if (creep.getActiveBodyparts(ATTACK)) creep.attack(target);
	} else {
		moveCreep(creep, target, { reusePath: 5, ignoreCreeps: false });
		if (creep.getActiveBodyparts(RANGED_ATTACK) && creep.pos.inRangeTo(target, 3)) {
			creep.rangedAttack(target);
		}
	}
}

function kiteAway(creep, enemy, range = 3) {
	const path = PathFinder.search(creep.pos, { pos: enemy.pos, range }, {
		flee: true,
		maxRooms: 1,
	});
	if (path.path.length) {
		creep.move(creep.pos.getDirectionTo(path.path[0]));
	}
}

// ---------------------------------------------------------------------------
// Role runners
// ---------------------------------------------------------------------------

function runAttacker(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) {
		// No assignment: defend home.
		return runDefender(creep);
	}

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
	meleeEngage(creep, target);
	if (creep.getActiveBodyparts(HEAL) && creep.hits < creep.hitsMax) creep.heal(creep);
}

function runRanged(creep) {
	const targetRoom = creep.memory.targetRoom;
	if (!targetRoom) return runDefender(creep);

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
	const nearEnemy = hostiles.find(h =>
		creep.pos.inRangeTo(h, 2) && h.getActiveBodyparts(ATTACK) > 0);

	if (nearEnemy) kiteAway(creep, nearEnemy, 3);

	if (target) {
		if (creep.pos.inRangeTo(target, 3)) {
			if (creep.pos.inRangeTo(target, 1) && hostiles.length > 1) {
				creep.rangedMassAttack();
			} else {
				creep.rangedAttack(target);
			}
		} else {
			moveCreep(creep, target, { reusePath: 5, range: 3 });
		}
	}
	if (creep.getActiveBodyparts(HEAL) && creep.hits < creep.hitsMax) creep.heal(creep);
}

function runHealer(creep) {
	const targetRoom = creep.memory.targetRoom;
	const squadId = creep.memory.squad;

	// Stick to lowest-HP ally in squad (or any nearby ally).
	let patients = [];
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.my) continue;
		if (squadId && c.memory && c.memory.squad === squadId) patients.push(c);
	}
	if (!patients.length) {
		patients = creep.pos.findInRange(FIND_MY_CREEPS, 10);
	}
	if (!patients.length) patients = [ creep ];

	patients.sort((a, b) => (a.hits / a.hitsMax) - (b.hits / b.hitsMax));
	const patient = patients[0];

	// Heal priority: self if hurt, else lowest-HP ally.
	if (creep.hits < creep.hitsMax * 0.9) {
		creep.heal(creep);
	} else if (creep.pos.isNearTo(patient)) {
		creep.heal(patient);
	} else if (creep.pos.inRangeTo(patient, 3)) {
		creep.rangedHeal(patient);
	}

	if (!squadReadyToMarch(creep)) {
		goToRally(creep);
		if (creep.pos.isNearTo(patient) && patient.hits < patient.hitsMax) {
			creep.heal(patient);
		}
		return;
	}

	if (patient.hits < patient.hitsMax) {
		if (!creep.pos.inRangeTo(patient, 1)) {
			moveCreep(creep, patient, { reusePath: 3, range: 1 });
		}
	} else if (targetRoom && creep.room.name !== targetRoom) {
		moveToRoom(creep, targetRoom);
	} else if (patient && patient.id !== creep.id) {
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

	// Prefer walls/ramparts, then towers/spawns, then other hostile structs.
	const structs = creep.room.find(FIND_HOSTILE_STRUCTURES, {
		filter: s => s.structureType !== STRUCTURE_CONTROLLER,
	});
	const coreType = typeof STRUCTURE_INVADER_CORE !== 'undefined' ? STRUCTURE_INVADER_CORE : 'invaderCore';
	structs.sort((a, b) => {
		const prio = {
			[STRUCTURE_RAMPART]: 0,
			[STRUCTURE_WALL]: 1,
			[STRUCTURE_TOWER]: 2,
			[STRUCTURE_SPAWN]: 3,
			[coreType]: 3,
			[STRUCTURE_EXTENSION]: 4,
		};
		return (prio[a.structureType] ?? 5) - (prio[b.structureType] ?? 5) || a.hits - b.hits;
	});
	const target = structs[0];
	if (!target) {
		// Nothing left to dismantle — help with hostile creeps if present.
		const creepTarget = attackTarget(creep.room);
		if (creepTarget && creepTarget.body) {
			meleeEngage(creep, creepTarget);
		}
		return;
	}
	goDo(creep, target, 1, () => creep.dismantle(target));
}

/**
 * Home (or defend-flag) military. Recycles after long safe stretch.
 * memory.defendRoom / memory.targetRoom can station defender off-home.
 */
function runDefender(creep) {
	const home = creep.memory.home || creep.room.name;
	const station = creep.memory.defendRoom || creep.memory.targetRoom || home;

	// Hunt hostiles in current room first.
	let hostiles = creep.room.find(FIND_HOSTILE_CREEPS);
	if (!hostiles.length && creep.room.name !== station) {
		moveToRoom(creep, station);
		return;
	}

	if (!hostiles.length && station !== home && Game.rooms[station]) {
		// Check station room once visible; otherwise travel there.
		hostiles = Game.rooms[station].find(FIND_HOSTILE_CREEPS);
		if (hostiles.length && creep.room.name !== station) {
			moveToRoom(creep, station);
			return;
		}
	}

	if (!hostiles.length) {
		creep.memory.safeTicks = (creep.memory.safeTicks || 0) + 1;
		const recycleAfter = (config.defenderRecycleSafeTicks != null)
			? config.defenderRecycleSafeTicks
			: 200;

		// Campaign / flag defenders do not auto-recycle while assigned.
		const sticky = !!(creep.memory.campaignId || creep.memory.defendRoom);

		if (!sticky && creep.memory.safeTicks >= recycleAfter) {
			const homeRoom = Game.rooms[home];
			const spawn = homeRoom && homeRoom.find(FIND_MY_SPAWNS)[0];
			if (spawn) {
				if (creep.room.name !== home) {
					moveToRoom(creep, home);
					return;
				}
				if (creep.pos.isNearTo(spawn)) {
					spawn.recycleCreep(creep);
					return;
				}
				moveCreep(creep, spawn, { range: 1, reusePath: 20 });
				return;
			}
			if (creep.memory.safeTicks > recycleAfter + 100) creep.suicide();
			return;
		}

		// Idle near station spawn / center.
		const room = Game.rooms[station] || creep.room;
		if (creep.room.name !== station) {
			moveToRoom(creep, station);
			return;
		}
		const spawn = room.find(FIND_MY_SPAWNS)[0];
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

// ---------------------------------------------------------------------------
// Tower helpers (also usable from defense.js)
// ---------------------------------------------------------------------------

function runTowers(room) {
	const towers = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_TOWER && energyOf(s.store) > 0,
	});
	if (!towers.length) return;

	const hostiles = room.find(FIND_HOSTILE_CREEPS);
	if (hostiles.length) {
		hostiles.sort((a, b) => hostileScore(b) - hostileScore(a));
		let target = hostiles[0];
		if (config.towerRampartAware !== false && isOnHostileRampart(target) && hostiles.length > 1) {
			const exposed = hostiles.find(h => !isOnHostileRampart(h));
			if (exposed) target = exposed;
		}
		for (const tower of towers) {
			tower.attack(target);
		}
		return;
	}

	// Heal most damaged friendly.
	const hurt = room.find(FIND_MY_CREEPS, { filter: c => c.hits < c.hitsMax });
	if (hurt.length) {
		hurt.sort((a, b) => (a.hits / a.hitsMax) - (b.hits / b.hitsMax));
		for (const tower of towers) {
			if (energyOf(tower.store) < 200) continue;
			tower.heal(hurt[0]);
		}
		return;
	}

	// Light repair when energy comfortable.
	if (energyOf(towers[0].store) < 500) return;
	const threshold = config.repairThreshold != null ? config.repairThreshold : 0.75;
	const repairTarget = room.find(FIND_STRUCTURES, {
		filter: s => {
			if (s.structureType === STRUCTURE_WALL || s.structureType === STRUCTURE_RAMPART) {
				return false;
			}
			if (s.structureType === STRUCTURE_ROAD) {
				return s.hits < s.hitsMax * 0.5;
			}
			return s.hits < s.hitsMax * threshold;
		},
	}).sort((a, b) => a.hits - b.hits)[0];

	if (repairTarget) {
		for (const tower of towers) {
			if (energyOf(tower.store) > 600) tower.repair(repairTarget);
		}
	}
}

function maybeSafeMode(room) {
	if (config.safeModeOnBreach === false) return;
	const ctrl = room.controller;
	if (!ctrl || !ctrl.my) return;
	if (ctrl.safeMode || ctrl.safeModeCooldown) return;
	if ((ctrl.safeModeAvailable || 0) <= 0) return;

	const hostiles = room.find(FIND_HOSTILE_CREEPS).filter(c => militaryParts(c) >= 1);
	if (!hostiles.length) return;

	const threat = hostileThreat(room);
	const towers = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_TOWER,
	});
	const towerEnergy = towers.reduce((s, t) => s + energyOf(t.store), 0);
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	const core = room.storage || spawn;
	const onCore = core && hostiles.some(h => h.pos.inRangeTo(core, 5));
	const towerStarved = towers.length === 0 || towerEnergy < 200;

	if ((onCore && threat >= 2) || (towerStarved && threat >= 3)) {
		const r = ctrl.activateSafeMode();
		if (r === OK) console.log(`Apex v3 SAFE MODE in ${room.name} at ${Game.time}`);
	}
}

function runDefense(room) {
	runTowers(room);
	maybeSafeMode(room);
}

// ---------------------------------------------------------------------------
// Squad counting (used by war.js and legacy attack lists)
// ---------------------------------------------------------------------------

const MILITARY_ROLES = [ 'attacker', 'ranged', 'healer', 'dismantler', 'defender' ];

function squadCounts(targetRoom, campaignId) {
	const counts = { attacker: 0, ranged: 0, healer: 0, dismantler: 0, defender: 0 };
	for (const name in Game.creeps) {
		const c = Game.creeps[name];
		if (!c.memory) continue;
		if (campaignId && c.memory.campaignId !== campaignId) continue;
		if (!campaignId && c.memory.targetRoom !== targetRoom) continue;
		if (counts[c.memory.role] != null) counts[c.memory.role]++;
	}
	return counts;
}

/**
 * Body energy cost of a living creep (for campaign spentEnergy accounting).
 */
function creepBodyCost(creep) {
	if (!creep || !creep.body) return 0;
	const costs = {
		[MOVE]: 50,
		[WORK]: 100,
		[CARRY]: 50,
		[ATTACK]: 80,
		[RANGED_ATTACK]: 150,
		[HEAL]: 250,
		[CLAIM]: 600,
		[TOUGH]: 10,
	};
	let sum = 0;
	for (const p of creep.body) sum += costs[p.type] || 0;
	return sum;
}

module.exports = {
	runAttacker,
	runRanged,
	runHealer,
	runDismantler,
	runDefender,
	runTowers,
	maybeSafeMode,
	runDefense,
	attackTarget,
	hostileScore,
	squadCounts,
	squadReadyToMarch,
	rallyPos,
	creepBodyCost,
	MILITARY_ROLES,
};
