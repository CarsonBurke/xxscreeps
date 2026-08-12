/**
 * Movement priorities for traffic resolution.
 * Higher = more important to keep seat / complete move.
 * Only changes with role/task — not every step (keeps path cache valid).
 */
import { getRole, getWorking, Role } from './creepMem';

export const enum MovePrio {
	idle = 0,
	scout = 10,
	builder = 20,
	upgrader = 30,
	emptyHauler = 40,
	fullHauler = 50,
	filler = 60,
	/** Static seats — do not push */
	harvesterSeat = 100,
	fillerPad = 100,
	military = 70,
}

export function movePriority(creep: Creep): number {
	const role = getRole(creep);
	switch (role) {
		case Role.harvester:
		case Role.remoteHarvester:
			return MovePrio.harvesterSeat;
		case Role.filler:
			return MovePrio.fillerPad;
		case Role.hauler:
		case Role.remoteHauler:
			return getWorking(creep) ? MovePrio.fullHauler : MovePrio.emptyHauler;
		case Role.upgrader:
			return MovePrio.upgrader;
		case Role.builder:
		case Role.bootstrap:
			return MovePrio.builder;
		case Role.defender:
		case Role.attacker:
		case Role.ranged:
		case Role.healer:
		case Role.dismantler:
			return MovePrio.military;
		case Role.scout:
		case Role.reserver:
		case Role.claimer:
			return MovePrio.scout;
		default:
			return MovePrio.idle;
	}
}
