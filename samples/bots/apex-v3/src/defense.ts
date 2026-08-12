import config = require('./config');
import { energyOf, hostileThreat, militaryParts } from './util';

function hostileScore(creep: Creep): number {
	let score = 0;
	score += creep.getActiveBodyparts(HEAL) * 1000;
	score += creep.getActiveBodyparts(RANGED_ATTACK) * 40;
	score += creep.getActiveBodyparts(ATTACK) * 30;
	score += Math.max(0, 500 - creep.hits);
	return score;
}

export function run(room: Room): void {
	const towers = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_TOWER && energyOf((s as StructureTower).store) > 0,
	}) as StructureTower[];

	if (towers.length) {
		const hostiles = room.find(FIND_HOSTILE_CREEPS);
		if (hostiles.length) {
			hostiles.sort((a, b) => hostileScore(b) - hostileScore(a));
			for (const tower of towers) tower.attack(hostiles[0]!);
		} else {
			const hurt = room.find(FIND_MY_CREEPS, { filter: c => c.hits < c.hitsMax });
			if (hurt.length) {
				hurt.sort((a, b) => a.hits / a.hitsMax - b.hits / b.hitsMax);
				for (const tower of towers) {
					if (energyOf(tower.store) > 200) tower.heal(hurt[0]!);
				}
			}
		}
	}

	if (!config.safeModeOnBreach) return;
	const ctrl = room.controller;
	if (!ctrl || !ctrl.my || ctrl.safeMode || ctrl.safeModeCooldown) return;
	if ((ctrl.safeModeAvailable || 0) <= 0) return;
	const hostiles = room.find(FIND_HOSTILE_CREEPS).filter(c => militaryParts(c) >= 1);
	if (!hostiles.length) return;
	const threat = hostileThreat(room);
	const towerE = towers.reduce((s, t) => s + energyOf(t.store), 0);
	const core = room.storage || room.find(FIND_MY_SPAWNS)[0];
	const onCore = core && hostiles.some(h => h.pos.inRangeTo(core, 5));
	if ((onCore && threat >= 2) || (towerE < 200 && threat >= 3)) {
		if (ctrl.activateSafeMode() === OK) console.log(`Apex v3 SAFE MODE ${room.name}`);
	}
}
