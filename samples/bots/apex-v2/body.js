/**
 * Body composition factories.
 * Prefer move efficiency: road-ratio 1 MOVE per 2 non-move on roads; full MOVE off-road.
 */
const { bodyCost, clamp } = require('util');
const config = require('config');

function repeat(part, n) {
	return Array(Math.max(0, n)).fill(part);
}

/** Pack body as [pattern...] capped by energy and MAX_CREEP_SIZE. */
function patternBody(pattern, energy, maxParts = MAX_CREEP_SIZE) {
	const unitCost = bodyCost(pattern);
	if (unitCost <= 0) return [];
	const maxUnitsByEnergy = Math.floor(energy / unitCost);
	const maxUnitsBySize = Math.floor(maxParts / pattern.length);
	const units = Math.max(0, Math.min(maxUnitsByEnergy, maxUnitsBySize));
	const body = [];
	for (let i = 0; i < units; i++) body.push(...pattern);
	return body;
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
	for (const k of Object.keys(ratios)) {
		unitCost += (BODYPART_COST[k] || 0) * ratios[k];
	}
	const moveSpecified = spec.move != null;
	if (!moveSpecified) {
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
		return minimalFallback(spec, energy, opts);
	}

	const body = [];
	const buildOrder = [ 'tough', 'work', 'carry', 'claim', 'attack', 'ranged_attack', 'heal', 'move' ];
	for (const k of buildOrder) {
		if (!ratios[k]) continue;
		for (let i = 0; i < ratios[k] * mult; i++) body.push(k);
	}
	return body;
}

function minimalFallback(spec, energy, opts) {
	if (spec.claim && energy >= 650) return [ CLAIM, MOVE ];
	if (spec.heal && energy >= 300) return [ HEAL, MOVE ];
	if (spec.attack && energy >= 130) return [ ATTACK, MOVE ];
	if (spec.work && spec.carry) {
		const e = Math.min(energy, 300);
		if (e >= 200) return [ WORK, CARRY, MOVE ];
		if (e >= 100) return [ CARRY, MOVE ];
	}
	if (spec.carry && energy >= 100) return [ CARRY, MOVE ];
	if (spec.work && energy >= 150) return [ WORK, MOVE ];
	if (energy >= 50) return [ MOVE ];
	return [];
}

function bootstrap(energy) {
	const e = Math.max(energy, 200);
	return scaleParts({ work: 1, carry: 1 }, Math.min(e, 800), { roads: false });
}

function miner(energy, workParts = config.sourceWorkParts) {
	const work = clamp(workParts, 1, 10);
	const need = BODYPART_COST[WORK] * work + BODYPART_COST[CARRY] + BODYPART_COST[MOVE];
	if (energy < 150) {
		return scaleParts({ work: 1, carry: 1 }, energy, { roads: false });
	}
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
	const carry = clamp(carryParts || config.minHaulerCarryParts, config.minHaulerCarryParts, config.maxHaulerCarryParts);
	const pattern = [ CARRY, CARRY, MOVE ];
	const unitCarry = 2;
	const unitsWanted = Math.ceil(carry / unitCarry);
	const unitCost = bodyCost(pattern);
	const maxUnits = Math.min(unitsWanted, Math.floor(energy / unitCost), Math.floor(MAX_CREEP_SIZE / 3));
	if (maxUnits <= 0) return energy >= 100 ? [ CARRY, MOVE ] : [];
	const body = [];
	for (let i = 0; i < maxUnits; i++) body.push(...pattern);
	return body;
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

function repairer(energy) {
	return scaleParts({ work: 1, carry: 1 }, Math.min(energy, 1200), { roads: true });
}

function reserver(energy) {
	if (energy >= 1300) return [ CLAIM, CLAIM, MOVE, MOVE ];
	if (energy >= 650) return [ CLAIM, MOVE ];
	return [];
}

function claimer(energy) {
	if (energy >= 650) return [ CLAIM, MOVE ];
	return [];
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

function mineralMiner(energy) {
	return scaleParts({ work: 4, carry: 1, move: 2 }, Math.min(energy, 2500), { roads: true });
}

function remoteMiner(energy) {
	const work = config.sourceWorkParts;
	const body = [ ...repeat(WORK, work), CARRY, ...repeat(MOVE, Math.ceil(work / 2) + 1) ];
	if (bodyCost(body) <= energy) return body;
	return miner(energy, work);
}

function remoteHauler(energy, pathLen) {
	const carry = clamp(
		Math.ceil((pathLen || 20) * config.haulerCarryPerPathTile / 50),
		4,
		config.maxHaulerCarryParts,
	);
	return hauler(energy, carry);
}

/**
 * Ideal (full) body cost at capacity — used for spawn hysteresis.
 */
function idealBody(role, capacity, context = {}) {
	return build(role, capacity, context);
}

const builders = {
	bootstrap,
	miner,
	hauler,
	upgrader,
	builder,
	repairer,
	reserver,
	claimer,
	scout,
	defender,
	attacker,
	ranged,
	healer,
	dismantler,
	mineralMiner,
	remoteMiner,
	remoteHauler,
};

function build(role, energy, context = {}) {
	const fn = builders[role];
	if (!fn) return [];
	if (role === 'hauler') return hauler(energy, context.carryParts);
	if (role === 'remoteHauler') return remoteHauler(energy, context.pathLen);
	if (role === 'upgrader') return upgrader(energy, context.rcl || 1);
	if (role === 'miner') return miner(energy, context.workParts);
	if (role === 'remoteMiner') return remoteMiner(energy);
	return fn(energy, context.rcl);
}

module.exports = {
	bodyCost,
	patternBody,
	scaleParts,
	build,
	builders,
	idealBody,
};
