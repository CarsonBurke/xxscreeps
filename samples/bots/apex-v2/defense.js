/**
 * Tower defense: rampart-aware focus, heal priority, safe mode.
 */
const config = require('config');
const { energyOf, militaryParts, hostileThreat } = require('util');

function isOnHostileRampart(creep) {
	const structs = creep.pos.lookFor(LOOK_STRUCTURES);
	return structs.some(s =>
		s.structureType === STRUCTURE_RAMPART && !s.my);
}

function hostileScore(creep) {
	// Focus healers first, then highest DPS, then lowest hits.
	// Rampart-aware: deprioritize targets under enemy ramparts (harder to kill).
	let score = 0;
	const heal = creep.getActiveBodyparts(HEAL);
	const attack = creep.getActiveBodyparts(ATTACK);
	const ranged = creep.getActiveBodyparts(RANGED_ATTACK);
	const work = creep.getActiveBodyparts(WORK);
	if (config.towerFocusHealers) score += heal * 1000;
	score += ranged * 40 + attack * 30 + work * 10;
	score += Math.max(0, 500 - creep.hits);

	if (config.towerRampartAware && isOnHostileRampart(creep)) {
		// Still attack if only target, but prefer exposed hostiles.
		score -= 800;
	}
	// Prefer targets not at full health (finish them).
	if (creep.hits < creep.hitsMax) score += 50;
	return score;
}

function runTowers(room) {
	const towers = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_TOWER && energyOf(s.store) > 0,
	});
	if (!towers.length) return;

	const hostiles = room.find(FIND_HOSTILE_CREEPS);
	if (hostiles.length) {
		hostiles.sort((a, b) => hostileScore(b) - hostileScore(a));
		// If top target is rampart-tanking and others exist, prefer next exposed.
		let target = hostiles[0];
		if (config.towerRampartAware && isOnHostileRampart(target) && hostiles.length > 1) {
			const exposed = hostiles.find(h => !isOnHostileRampart(h));
			if (exposed) target = exposed;
		}
		for (const tower of towers) {
			tower.attack(target);
		}
		return;
	}

	// Heal friendlies (most damaged first).
	const hurt = room.find(FIND_MY_CREEPS, { filter: c => c.hits < c.hitsMax });
	if (hurt.length) {
		hurt.sort((a, b) => (a.hits / a.hitsMax) - (b.hits / b.hitsMax));
		for (const tower of towers) {
			if (energyOf(tower.store) < 200) continue;
			tower.heal(hurt[0]);
		}
		return;
	}

	// Repair critical structures when energy is comfortable.
	if (towers[0] && energyOf(towers[0].store) < 500) return;
	const repairTarget = room.find(FIND_STRUCTURES, {
		filter: s => {
			if (s.structureType === STRUCTURE_WALL || s.structureType === STRUCTURE_RAMPART) {
				return false;
			}
			if (s.structureType === STRUCTURE_ROAD) {
				return s.hits < s.hitsMax * 0.5;
			}
			return s.hits < s.hitsMax * config.repairThreshold;
		},
	}).sort((a, b) => a.hits - b.hits)[0];

	if (repairTarget) {
		for (const tower of towers) {
			if (energyOf(tower.store) > 600) tower.repair(repairTarget);
		}
	}
}

function maybeSafeMode(room) {
	if (!config.safeModeOnBreach) return;
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
		if (r === OK) console.log(`Apex v2 SAFE MODE in ${room.name} at ${Game.time}`);
	}
}

function run(room) {
	runTowers(room);
	maybeSafeMode(room);
}

module.exports = { run, runTowers, maybeSafeMode, hostileScore };
