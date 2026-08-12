import config = require('./config');
import { bodyCost } from './util';
import { Role, parseRole } from './memoryKeys';

function repeat(part: BodyPartConstant, n: number): BodyPartConstant[] {
	return Array(Math.max(0, n)).fill(part);
}

function scaleParts(
	spec: Partial<Record<BodyPartConstant, number>>,
	energy: number,
	opts: { roads?: boolean } = {},
): BodyPartConstant[] {
	const order: BodyPartConstant[] = [
		TOUGH, WORK, CARRY, CLAIM, ATTACK, RANGED_ATTACK, HEAL, MOVE,
	];
	const ratios: Partial<Record<BodyPartConstant, number>> = {};
	let totalRatio = 0;
	for (const k of order) {
		if (spec[k]) {
			ratios[k] = spec[k];
			totalRatio += spec[k]!;
		}
	}
	if (!totalRatio) return [];

	let unitCost = 0;
	for (const k of Object.keys(ratios) as BodyPartConstant[]) {
		unitCost += (BODYPART_COST[k] || 0) * (ratios[k] || 0);
	}
	if (spec[MOVE] == null) {
		const nonMove = totalRatio;
		const moveRatio = opts.roads ? Math.ceil(nonMove / 2) : nonMove;
		unitCost += BODYPART_COST[MOVE] * moveRatio;
		ratios[MOVE] = moveRatio;
		totalRatio += moveRatio;
	}

	let mult = Math.floor(energy / unitCost);
	while (mult > 0) {
		let parts = 0;
		for (const k of Object.keys(ratios) as BodyPartConstant[]) parts += (ratios[k] || 0) * mult;
		if (parts <= MAX_CREEP_SIZE) break;
		mult--;
	}
	if (mult <= 0) {
		if (energy >= 200) return [WORK, CARRY, MOVE];
		if (energy >= 100) return [CARRY, MOVE];
		return [];
	}
	const body: BodyPartConstant[] = [];
	const buildOrder: BodyPartConstant[] = [
		TOUGH, WORK, CARRY, CLAIM, ATTACK, RANGED_ATTACK, HEAL, MOVE,
	];
	for (const k of buildOrder) {
		if (!ratios[k]) continue;
		for (let i = 0; i < ratios[k]! * mult; i++) body.push(k);
	}
	return body;
}

export function bootstrap(energy: number): BodyPartConstant[] {
	return scaleParts({ [WORK]: 1, [CARRY]: 1 }, Math.min(energy, 800), { roads: false });
}

export function harvester(
	energy: number,
	workParts = config.sourceWorkParts,
	opts: { roads?: boolean } = {},
): BodyPartConstant[] {
	const roads = opts.roads !== false;
	const work = Math.max(1, Math.min(workParts, 10));
	// Off-road remotes need ~1 MOVE per non-MOVE part to not crawl
	const moveNeed = roads ? Math.ceil(work / 2) + 1 : work + 1; // +1 for CARRY
	const need = BODYPART_COST[WORK] * work + BODYPART_COST[CARRY] + BODYPART_COST[MOVE] * Math.max(1, roads ? 1 : moveNeed);
	if (energy < 150) return scaleParts({ [WORK]: 1, [CARRY]: 1 }, energy, { roads: false });
	if (energy < need) {
		// Fit as many WORK as possible; keep MOVE parity for off-road
		if (!roads) {
			// Unit: W+C+M+M = 250, or W+M = 150 for pure walk-in
			const unit = [WORK, CARRY, MOVE, MOVE] as BodyPartConstant[];
			const unitCost = bodyCost(unit);
			let mult = Math.max(1, Math.floor(energy / unitCost));
			while (mult > 0 && mult * 4 > MAX_CREEP_SIZE) mult--;
			if (mult <= 0) return energy >= 200 ? [WORK, CARRY, MOVE, MOVE] : energy >= 150 ? [WORK, MOVE, MOVE] : [];
			const body: BodyPartConstant[] = [];
			for (let i = 0; i < mult && i < work; i++) body.push(...unit);
			// If energy left for more WORK with MOVE
			return body.slice(0, MAX_CREEP_SIZE);
		}
		const w = Math.max(1, Math.floor((energy - 100) / 100));
		return [...repeat(WORK, w), CARRY, MOVE];
	}
	const body: BodyPartConstant[] = [...repeat(WORK, work), CARRY];
	const remaining = energy - bodyCost(body);
	const maxMoves = roads
		? Math.min(Math.floor(remaining / 50), Math.ceil(work / 2) + 1)
		: Math.min(Math.floor(remaining / 50), work + 1);
	body.push(...repeat(MOVE, Math.max(1, maxMoves)));
	return body.slice(0, MAX_CREEP_SIZE);
}

export function hauler(
	energy: number,
	carryParts?: number,
	opts: { roads?: boolean } = {},
): BodyPartConstant[] {
	const roads = opts.roads !== false;
	// Cap by MAX_CREEP_SIZE packing, not an arbitrary config max
	const maxCarryBySize = roads
		? Math.floor((MAX_CREEP_SIZE * 2) / 3) // 2C1M pattern
		: Math.floor(MAX_CREEP_SIZE / 2); // 1C1M
	const carry = Math.max(
		config.minHaulerCarryParts || 2,
		Math.min(maxCarryBySize, carryParts || config.minHaulerCarryParts || 2),
	);
	// Roads: 2C1M; off-road remotes: 1C1M so they actually move at full speed
	const pattern: BodyPartConstant[] = roads ? [CARRY, CARRY, MOVE] : [CARRY, MOVE];
	const carryPerUnit = roads ? 2 : 1;
	const unitsWanted = Math.ceil(carry / carryPerUnit);
	const unitCost = bodyCost(pattern);
	const maxUnits = Math.min(
		unitsWanted,
		Math.floor(energy / unitCost),
		Math.floor(MAX_CREEP_SIZE / pattern.length),
	);
	if (maxUnits <= 0) return energy >= 100 ? [CARRY, MOVE] : [];
	const body: BodyPartConstant[] = [];
	for (let i = 0; i < maxUnits; i++) body.push(...pattern);
	return body;
}

export function filler(energy: number): BodyPartConstant[] {
	return scaleParts({ [CARRY]: 2, [MOVE]: 1 }, Math.min(energy, 1200), {});
}

export function upgrader(energy: number, rcl: number): BodyPartConstant[] {
	if (energy < 200) return energy >= 150 ? [WORK, CARRY, MOVE] : [];
	if (rcl >= 8) {
		const base: BodyPartConstant[] = [...repeat(WORK, 15), CARRY, ...repeat(MOVE, 8)];
		if (bodyCost(base) <= energy) return base;
	}
	// Prefer max WORK for controller e/t (each WORK ≈ 1 e/t when fed).
	return scaleParts({ [WORK]: 2, [CARRY]: 1 }, energy, { roads: true });
}

export function builder(energy: number): BodyPartConstant[] {
	return scaleParts({ [WORK]: 1, [CARRY]: 1 }, energy, { roads: true });
}

export function reserver(energy: number): BodyPartConstant[] {
	if (energy >= 1300) return [CLAIM, CLAIM, MOVE, MOVE];
	if (energy >= 650) return [CLAIM, MOVE];
	return [];
}

export function claimer(energy: number): BodyPartConstant[] {
	return energy >= 650 ? [CLAIM, MOVE] : [];
}

export function scout(energy: number): BodyPartConstant[] {
	return energy >= 50 ? [MOVE] : [];
}

export function defender(energy: number): BodyPartConstant[] {
	return scaleParts({ [TOUGH]: 1, [ATTACK]: 1, [RANGED_ATTACK]: 1, [MOVE]: 3 }, energy, {});
}

export function attacker(energy: number): BodyPartConstant[] {
	return scaleParts({ [TOUGH]: 1, [ATTACK]: 2, [MOVE]: 3 }, energy, {});
}

/**
 * Melee SK killer: max ATTACK with 1:1 MOVE (off-road).
 * Part order: MOVE first (damaged last for DPS), ATTACK last — standard Screeps.
 * Needs ~19A to win solo vs ~5000-hit keepers at ~400 dps incoming.
 */
export function skKiller(energy: number, minAttack = 0): BodyPartConstant[] {
	if (energy < 130) return energy >= 80 ? [MOVE, ATTACK] : [];
	const unitCost = BODYPART_COST[ATTACK] + BODYPART_COST[MOVE];
	let n = Math.floor(energy / unitCost);
	n = Math.min(n, Math.floor(MAX_CREEP_SIZE / 2));
	if (n < 1) return [];
	if (minAttack > 0 && n < minAttack) return []; // can't field a winning body yet
	const body: BodyPartConstant[] = [];
	for (let i = 0; i < n; i++) body.push(MOVE);
	for (let i = 0; i < n; i++) body.push(ATTACK);
	return body;
}

export function ranged(energy: number): BodyPartConstant[] {
	return scaleParts({ [TOUGH]: 1, [RANGED_ATTACK]: 2, [MOVE]: 3 }, energy, {});
}

export function healer(energy: number): BodyPartConstant[] {
	return scaleParts({ [HEAL]: 1, [MOVE]: 1 }, energy, {});
}

export function dismantler(energy: number): BodyPartConstant[] {
	return scaleParts({ [TOUGH]: 1, [WORK]: 2, [MOVE]: 3 }, energy, {});
}

export function build(
	role: Role | string | number,
	energy: number,
	context: {
		rcl?: number;
		workParts?: number;
		carryParts?: number;
		/** false for remotes (1:1 MOVE). Default true for local. */
		roads?: boolean;
		/** Melee SK killer body instead of generic attacker */
		skKiller?: boolean;
	} = {},
): BodyPartConstant[] {
	const r = typeof role === 'number' ? (role as Role) : parseRole(role);
	if (r == null) return [];
	const rcl = context.rcl || 1;
	// Remotes default off-road unless context.roads === true
	const remoteRoads = context.roads === true;
	switch (r) {
		case Role.bootstrap: return bootstrap(energy);
		case Role.harvester:
			return harvester(energy, context.workParts, { roads: true });
		case Role.remoteHarvester:
			return harvester(energy, context.workParts, { roads: remoteRoads });
		case Role.hauler:
			return hauler(energy, context.carryParts, { roads: true });
		case Role.remoteHauler:
			return hauler(energy, context.carryParts, { roads: remoteRoads });
		case Role.filler: return filler(energy);
		case Role.upgrader: return upgrader(energy, rcl);
		case Role.builder: return builder(energy);
		case Role.reserver: return reserver(energy);
		case Role.claimer: return claimer(energy);
		case Role.scout: return scout(energy);
		case Role.defender: return defender(energy);
		case Role.attacker:
			return context.skKiller
				? skKiller(energy, config.skKillerMinAttack || 0)
				: attacker(energy);
		case Role.ranged: return ranged(energy);
		case Role.healer: return healer(energy);
		case Role.dismantler: return dismantler(energy);
		default: return [];
	}
}

export { bodyCost };
