/**
 * Tower defense + safe mode decisions.
 */
const config = require('config');
const { energyOf, militaryParts, hostileThreat } = require('util');

function hostileScore(creep) {
	// Focus healers first (if configured), then highest threat DPS, then lowest hits.
	let score = 0;
	const heal = creep.getActiveBodyparts(HEAL);
	const attack = creep.getActiveBodyparts(ATTACK);
	const ranged = creep.getActiveBodyparts(RANGED_ATTACK);
	const work = creep.getActiveBodyparts(WORK);
	if (config.towerFocusHealers) score += heal * 1000;
	score += ranged * 40 + attack * 30 + work * 10;
	score += Math.max(0, 500 - creep.hits); // finish wounded
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
		const target = hostiles[0];
		for (const tower of towers) {
			tower.attack(target);
		}
		return;
	}

	// Heal friendlies.
	const hurt = room.find(FIND_MY_CREEPS, { filter: c => c.hits < c.hitsMax });
	if (hurt.length) {
		hurt.sort((a, b) => (a.hits / a.hitsMax) - (b.hits / b.hitsMax));
		for (const tower of towers) {
			if (energyOf(tower.store) < 200) continue;
			tower.heal(hurt[0]);
		}
		return;
	}

	// Repair critical structures (not walls) when energy is comfortable.
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

	// Breach: enemy near spawn/storage or no tower energy vs serious threat.
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	const core = room.storage || spawn;
	const onCore = core && hostiles.some(h => h.pos.inRangeTo(core, 5));
	const towerStarved = towers.length === 0 || towerEnergy < 200;

	if ((onCore && threat >= 2) || (towerStarved && threat >= 3)) {
		const r = ctrl.activateSafeMode();
		if (r === OK) console.log(`Apex SAFE MODE in ${room.name} at ${Game.time}`);
	}
}

function run(room) {
	runTowers(room);
	maybeSafeMode(room);
}

module.exports = { run, runTowers, maybeSafeMode, hostileScore };
