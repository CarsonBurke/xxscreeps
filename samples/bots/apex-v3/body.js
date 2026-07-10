/**
 * Body factories for delegated roles.
 */
const config = require('config');
const { bodyCost } = require('util');

function repeat(part, n) {
	return Array(Math.max(0, n)).fill(part);
}

function scaleParts(spec, energy, opts = {}) {
	const order = [ 'tough', 'work', 'carry', 'claim', 'attack', 'ranged_attack', 'heal', 'move' ];
	const ratios = {};
	let totalRatio = 0;
	for (const k of order) {
		if (spec[k]) {
			ratios[k] = spec[k];
			totalRatio += spec[k];
		}
	}
	if (!totalRatio) return [];

	let unitCost = 0;
	for (const k of Object.keys(ratios)) unitCost += (BODYPART_COST[k] || 0) * ratios[k];
	if (spec.move == null) {
		const nonMove = totalRatio;
		const moveRatio = opts.roads ? Math.ceil(nonMove / 2) : nonMove;
		unitCost += BODYPART_COST[MOVE] * moveRatio;
		ratios.move = moveRatio;
		totalRatio += moveRatio;
	}

	let mult = Math.floor(energy / unitCost);
	while (mult > 0) {
		let parts = 0;
		for (const k of Object.keys(ratios)) parts += ratios[k] * mult;
		if (parts <= MAX_CREEP_SIZE) break;
		mult--;
	}
	if (mult <= 0) {
		if (energy >= 200) return [ WORK, CARRY, MOVE ];
		if (energy >= 100) return [ CARRY, MOVE ];
		return [];
	}
	const body = [];
	const buildOrder = [ 'tough', 'work', 'carry', 'claim', 'attack', 'ranged_attack', 'heal', 'move' ];
	for (const k of buildOrder) {
		if (!ratios[k]) continue;
		for (let i = 0; i < ratios[k] * mult; i++) body.push(k);
	}
	return body;
}

function bootstrap(energy) {
	return scaleParts({ work: 1, carry: 1 }, Math.min(energy, 800), { roads: false });
}

/** Static harvester: WORK*N + CARRY(1) + MOVE enough to walk once. */
function harvester(energy, workParts = config.sourceWorkParts) {
	const work = Math.max(1, Math.min(workParts, 10));
	const need = BODYPART_COST[WORK] * work + BODYPART_COST[CARRY] + BODYPART_COST[MOVE];
	if (energy < 150) return scaleParts({ work: 1, carry: 1 }, energy, { roads: false });
	if (energy < need) {
		const w = Math.max(1, Math.floor((energy - 100) / 100));
		return [ ...repeat(WORK, w), CARRY, MOVE ];
	}
	const body = [ ...repeat(WORK, work), CARRY ];
	let remaining = energy - bodyCost(body);
	const moves = Math.min(Math.floor(remaining / 50), Math.ceil(work / 2) + 1);
	body.push(...repeat(MOVE, Math.max(1, moves)));
	return body.slice(0, MAX_CREEP_SIZE);
}

function hauler(energy, carryParts) {
	const carry = Math.max(
		config.minHaulerCarryParts,
		Math.min(config.maxHaulerCarryParts, carryParts || config.minHaulerCarryParts),
	);
	const pattern = [ CARRY, CARRY, MOVE ];
	const unitsWanted = Math.ceil(carry / 2);
	const unitCost = bodyCost(pattern);
	const maxUnits = Math.min(unitsWanted, Math.floor(energy / unitCost), Math.floor(MAX_CREEP_SIZE / 3));
	if (maxUnits <= 0) return energy >= 100 ? [ CARRY, MOVE ] : [];
	const body = [];
	for (let i = 0; i < maxUnits; i++) body.push(...pattern);
	return body;
}

/** Filler: mostly CARRY, low MOVE (parks next to spawn). */
function filler(energy) {
	return scaleParts({ carry: 2, move: 1 }, Math.min(energy, 1200), {});
}

function upgrader(energy, rcl) {
	if (rcl >= 8) {
		const base = [ ...repeat(WORK, 15), CARRY, ...repeat(MOVE, 8) ];
		if (bodyCost(base) <= energy) return base;
	}
	return scaleParts({ work: 2, carry: 1 }, energy, { roads: true });
}

function builder(energy) {
	return scaleParts({ work: 1, carry: 1 }, energy, { roads: true });
}

function reserver(energy) {
	if (energy >= 1300) return [ CLAIM, CLAIM, MOVE, MOVE ];
	if (energy >= 650) return [ CLAIM, MOVE ];
	return [];
}

function claimer(energy) {
	return energy >= 650 ? [ CLAIM, MOVE ] : [];
}

function scout(energy) {
	return energy >= 50 ? [ MOVE ] : [];
}

function defender(energy) {
	return scaleParts({ tough: 1, attack: 1, ranged_attack: 1, move: 3 }, energy, {});
}

function attacker(energy) {
	return scaleParts({ tough: 1, attack: 2, move: 3 }, energy, {});
}

function ranged(energy) {
	return scaleParts({ tough: 1, ranged_attack: 2, move: 3 }, energy, {});
}

function healer(energy) {
	return scaleParts({ heal: 1, move: 1 }, energy, {});
}

function dismantler(energy) {
	return scaleParts({ tough: 1, work: 2, move: 3 }, energy, {});
}

function build(role, energy, context = {}) {
	const rcl = context.rcl || 1;
	switch (role) {
		case 'bootstrap': return bootstrap(energy);
		case 'harvester':
		case 'remoteHarvester':
			return harvester(energy, context.workParts);
		case 'hauler':
		case 'remoteHauler':
			return hauler(energy, context.carryParts);
		case 'filler': return filler(energy);
		case 'upgrader': return upgrader(energy, rcl);
		case 'builder': return builder(energy);
		case 'reserver': return reserver(energy);
		case 'claimer': return claimer(energy);
		case 'scout': return scout(energy);
		case 'defender': return defender(energy);
		case 'attacker': return attacker(energy);
		case 'ranged': return ranged(energy);
		case 'healer': return healer(energy);
		case 'dismantler': return dismantler(energy);
		default: return [];
	}
}

module.exports = { bodyCost, build, harvester, hauler, filler };
