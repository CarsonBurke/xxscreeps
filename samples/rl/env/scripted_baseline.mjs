/**
 * Scripted RCL1→remote-bootstrap economy teacher using the SAME action interface as the RL policy.
 * Used as: (1) competence floor, (2) BC teacher, (3) reward sanity check.
 *
 * Policy: cheap source lanes → five RCL2 extensions → RCL3/650-energy claim →
 * remote bootstrap. Every phase is reconstructed from current Game state.
 */
import * as C from 'xxscreeps/game/constants/index.js';
import { SCHEMA } from './encode.mjs';

const {
	intentTypes, intentSlots, maxActors,
	maxBodyParts, bodyPartTypes,
} = SCHEMA;
const INTENT = Object.fromEntries(intentTypes.map((n, i) => [n, i]));
const BODY_TOKEN = Object.fromEntries(bodyPartTypes.map((name, index) => [ name, index ]));
const CONSTRUCTION_TYPES = SCHEMA.constructionTypes.map(
	type => C[type.toUpperCase()] || type,
);
const CONSTRUCTION_MASK_BYTES = Math.ceil(SCHEMA.roomSize * SCHEMA.roomSize / 8);
const BOOTSTRAP_BODIES = Object.freeze({
	flexible: [ C.WORK, C.CARRY, C.MOVE ],
	// Two stationary miners per source sustain eight energy/tick at 300 capacity.
	// Haulers recover their output; spending CARRY on miners made most of their
	// lifetime disappear into source-to-spawn travel.
	miner: [ C.WORK, C.WORK, C.MOVE ],
	hauler: [ C.CARRY, C.CARRY, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
	builder: [ C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
	upgrader: [ C.WORK, C.WORK, C.CARRY, C.MOVE ],
});
const MATURE_BODIES = Object.freeze({
	flexible: [ C.WORK, C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
	miner: [ C.WORK, C.WORK, C.WORK, C.WORK, C.WORK, C.MOVE ],
	hauler: [ C.CARRY, C.CARRY, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
	builder: [ C.WORK, C.WORK, C.CARRY, C.CARRY, C.CARRY, C.MOVE, C.MOVE, C.MOVE ],
	upgrader: [
		C.WORK, C.WORK, C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE, C.MOVE,
	],
});
for (const [ tier, bodies ] of Object.entries({ bootstrap: BOOTSTRAP_BODIES, mature: MATURE_BODIES })) {
	const signatures = new Map();
	for (const [ archetype, body ] of Object.entries(bodies)) {
		const signature = body.join(',');
		const previous = signatures.get(signature);
		if (previous) {
			throw new Error(`${tier} teacher bodies collide: ${previous} and ${archetype}`);
		}
		signatures.set(signature, archetype);
	}
}
const REPLACEMENT_ROUTE_SLACK = Object.freeze({
	// One spawn must replace an entire economy, not just one body. Starting the
	// replacement wave roughly 500 ticks early prevents synchronized TTL deaths
	// from collapsing the miner -> hauler -> spawn loop.
	flexible: 600,
	miner: 600,
	hauler: 600,
	builder: 600,
	upgrader: 600,
	claimer: 150,
});
const TARGET_REPLACEMENT_PRESSURE = 0.6;
const CURRICULUM = process.env.RL_CURRICULUM || 'empty';
const SPAWN_CURRICULUM = /^spawn_(flexible|miner|hauler|builder|upgrader|claimer)_\d+$/
	.exec(CURRICULUM)?.[1] || null;
const EXPLICIT_CLAIM_CURRICULUM = CURRICULUM === 'seed_claimer'
	|| SPAWN_CURRICULUM === 'claimer';
const SPAWN_CURRICULUM_BODIES = Object.freeze({
	// Match the observable zero-workforce recovery decision. A hidden curriculum
	// tag must not demand a more expensive body from the same world state.
	flexible: [ C.WORK, C.CARRY, C.MOVE ],
	miner: [ C.WORK, C.WORK, C.WORK, C.WORK, C.MOVE ],
	hauler: Array.from({ length: 25 }, () => [ C.CARRY, C.MOVE ]).flat(),
	builder: Array.from({ length: 2 }, () => [
		C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE,
	]).flat(),
	upgrader: [ C.WORK, C.WORK, C.CARRY, C.MOVE, C.WORK, C.WORK, C.CARRY ],
	claimer: [ C.CLAIM, C.MOVE ],
});

/** Fixed bodies avoid turning each extension into a more expensive savings target. */
function teacherBody(role, energyCapacity) {
	if (SPAWN_CURRICULUM === role) return [ ...SPAWN_CURRICULUM_BODIES[role] ];
	if (role === 'claimer') return [ C.CLAIM, C.MOVE ];
	const bodies = energyCapacity < 550 ? BOOTSTRAP_BODIES : MATURE_BODIES;
	return [ ...(bodies[role] || bodies.flexible) ];
}

function replacementLead(role, energyCapacity) {
	return teacherBody(role, energyCapacity).length * C.CREEP_SPAWN_TIME
		+ (REPLACEMENT_ROUTE_SLACK[role] || 60);
}

function staffingCreep(creep, role, energyCapacity) {
	if (creep.spawning) return true;
	const lead = replacementLead(role, energyCapacity);
	const ttl = creep.ticksToLive ?? 1500;
	// Counting every creep below the conservative route bound as absent creates
	// hundreds of ticks of overlapping cohorts. Trigger at a measured fraction
	// of that bound: still proactive, without spending the economy on duplicates.
	return ttl >= Math.ceil(lead * TARGET_REPLACEMENT_PRESSURE);
}

function bodyEncoding(body) {
	const counts = Array(bodyPartTypes.length).fill(0);
	const order = [];
	for (const part of body.slice(0, maxBodyParts)) {
		const type = BODY_TOKEN[part];
		if (counts[type] === 0) order.push(type);
		counts[type] += 1;
	}
	for (let type = 0; type < counts.length; type++) {
		if (counts[type] === 0) order.push(type);
	}
	return { counts, order };
}

function bodyCost(body) {
	return body.reduce((sum, part) => sum + (C.BODYPART_COST[part] || 0), 0);
}

/** Fill a composition proportionally instead of exhausting one role first. */
function mostUnderfilledRole(roleCounts, plan) {
	return plan
		.map(([role, target], order) => ({
			role, order, ratio: (roleCounts.get(role) || 0) / Math.max(1, target),
		}))
		.filter(entry => (roleCounts.get(entry.role) || 0) < plan[entry.order][1])
		.sort((a, b) => a.ratio - b.ratio || a.order - b.order)[0]?.role || null;
}

function roomDemand(room) {
	const rcl = room.controller?.level || 1;
	const energyCapacity = room.energyCapacityAvailable || 300;
	const structures = room.find(C.FIND_MY_STRUCTURES);
	const extensions = structures.filter(s => s.structureType === C.STRUCTURE_EXTENSION).length;
	const sourceCount = Math.max(1, room.find(C.FIND_SOURCES).length);
	// Before 550 capacity, two 2-WORK miners/source approach the sustainable
	// source rate. At 550, one 5-WORK miner/source saturates it with half as many
	// bodies and frees spawn time for the long-horizon economy.
	const miners = sourceCount * (energyCapacity < 550 ? 2 : 1);
	// Two 200-capacity haulers/source cover both the source-to-spawn leg and the
	// spawn-to-worker leg without letting stationary-miner output accumulate.
	const haulers = sourceCount * 2;
	if (rcl <= 1) {
		return [
			[ 'miner', miners ], [ 'hauler', haulers ],
			[ 'builder', 1 ], [ 'upgrader', 2 ], [ 'flexible', 1 ],
		];
	}
	if (extensions < 5) {
		return [
			[ 'miner', miners ], [ 'hauler', haulers ],
			[ 'builder', 3 ], [ 'upgrader', 1 ], [ 'flexible', 1 ],
		];
	}
	if (rcl < 3) {
		return [
			[ 'miner', miners ], [ 'hauler', haulers ],
			[ 'builder', 1 ], [ 'upgrader', 8 ], [ 'flexible', 1 ],
		];
	}
	if ((room.energyCapacityAvailable || 300) < 650) {
		return [
			[ 'miner', miners ], [ 'hauler', haulers ],
			// RCL3 is not useful for expansion until the seventh extension raises
			// room capacity to the 650-energy CLAIM+MOVE body.  Keep only an
			// upgrade floor and finish that infrastructure before scaling control.
			[ 'builder', 5 ], [ 'upgrader', 1 ], [ 'flexible', 1 ],
		];
	}
	return [
		[ 'miner', sourceCount ], [ 'hauler', sourceCount + 2 ],
		[ 'builder', 3 ], [ 'upgrader', 12 ], [ 'flexible', 3 ],
	];
}

function sourceAssignments(room, creeps, creepRole) {
	const sources = room.find(C.FIND_SOURCES).sort((a, b) =>
		a.pos.x - b.pos.x || a.pos.y - b.pos.y || a.id.localeCompare(b.id));
	const assigned = new Map();
	const miners = creeps
		.filter(creep => creepRole(creep) === 'miner')
		.sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id));
	for (let index = 0; index < miners.length && sources.length; index++) {
		assigned.set(miners[index].id, sources[index % sources.length]);
	}
	return assigned;
}

function sourceLaneCount(room, source) {
	const terrain = room.getTerrain?.();
	if (!terrain) return 3;
	let count = 0;
	for (let dx = -1; dx <= 1; dx++) {
		for (let dy = -1; dy <= 1; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = source.pos.x + dx;
			const y = source.pos.y + dy;
			if (x > 0 && x < 49 && y > 0 && y < 49
				&& terrain.get(x, y) !== C.TERRAIN_MASK_WALL) count++;
		}
	}
	return Math.max(1, count);
}

function chebyshev(a, b) {
	return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
}

function dirIndex(from, to) {
	const dx = Math.sign(to.x - from.x);
	const dy = Math.sign(to.y - from.y);
	const map = {
		'0,-1': 0, '1,-1': 1, '1,0': 2, '1,1': 3,
		'0,1': 4, '-1,1': 5, '-1,0': 6, '-1,-1': 7,
	};
	return map[`${dx},${dy}`] ?? 0;
}

/** Prefer PathFinder one-step when available; else Chebyshev greedy. */
function stepDir(creep, dest) {
	if (creep.pos.isEqualTo(dest)) return 0;
	try {
		if (typeof PathFinder !== 'undefined' && PathFinder.search) {
			const res = PathFinder.search(creep.pos, { pos: dest, range: 1 }, {
				plainCost: 2, swampCost: 10, maxOps: 500, maxRooms: 1,
			});
			if (res.path?.length) {
				return dirIndex(creep.pos, res.path[0]);
			}
		}
	} catch {
		/* fall through */
	}
	return dirIndex(creep.pos, dest);
}

/**
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {{ actorMeta: any[], targetMeta: any[] }} meta
 */
export function scriptedActions(Game, meta) {
	const nA = Math.min(maxActors, meta.actorMeta?.length || 0);
	const types = Array.from({ length: nA }, () => Array(intentSlots).fill(INTENT.none));
	const dirs = Array.from({ length: nA }, () => Array(intentSlots).fill(0));
	const targets = Array.from({ length: nA }, () => Array(intentSlots).fill(0));
	const amounts = Array.from({ length: nA }, () => Array(intentSlots).fill(0));
	const bodyCounts = Array.from({ length: nA }, () =>
		Array.from({ length: intentSlots }, () => Array(bodyPartTypes.length).fill(0))
	);
	const bodyOrder = Array.from({ length: nA }, () =>
		Array.from({ length: intentSlots }, () =>
			Array.from({ length: bodyPartTypes.length }, (_, type) => type))
	);
	const constructionTypes = Array.from(
		{ length: nA }, () => Array(intentSlots).fill(0),
	);
	const constructionTiles = Array.from(
		{ length: nA }, () => Array(intentSlots).fill(0),
	);
	const legalConstructionTile = (room, typeIndex) => {
		const roomName = room.name;
		const roomIndex = meta.roomNames?.indexOf(roomName) ?? -1;
		if (roomIndex < 0 || typeIndex < 0 || !meta.constructionMask) return -1;
		const base = (roomIndex * CONSTRUCTION_TYPES.length + typeIndex)
			* CONSTRUCTION_MASK_BYTES;
		const legal = tile => Boolean(
			meta.constructionMask[base + (tile >> 3)] & (1 << (tile & 7)),
		);
		// The scripted corpus needs one stable, competent demonstration. This is a
		// declarative stamp order, not a hand-authored layout score: engine legality
		// remains the only filter and the learned policy owns the 2,500-way choice.
		const structureType = CONSTRUCTION_TYPES[typeIndex];
		const sources = room.find(C.FIND_SOURCES)
			.slice().sort((a, b) =>
				a.pos.x - b.pos.x || a.pos.y - b.pos.y || a.id.localeCompare(b.id));
		const spawns = room.find(C.FIND_MY_SPAWNS)
			.slice().sort((a, b) => a.id.localeCompare(b.id));
		const anchors = structureType === C.STRUCTURE_CONTAINER
			? [ ...sources, ...spawns, room.controller ]
			: [ ...spawns, room.controller, ...sources ];
		const minRing = structureType === C.STRUCTURE_CONTAINER ? 1 : 2;
		const maxRing = structureType === C.STRUCTURE_CONTAINER ? 2 : 7;
		for (const anchor of anchors.filter(Boolean)) {
			for (let ring = minRing; ring <= maxRing; ring++) {
				for (let dy = -ring; dy <= ring; dy++) {
					for (let dx = -ring; dx <= ring; dx++) {
						if (Math.max(Math.abs(dx), Math.abs(dy)) !== ring) continue;
						// Leave alternating lanes through dense extension stamps.
						if (structureType === C.STRUCTURE_EXTENSION && (dx + dy) % 2 !== 0) continue;
						const x = anchor.pos.x + dx;
						const y = anchor.pos.y + dy;
						if (x < 1 || x >= SCHEMA.roomSize - 1 || y < 1 || y >= SCHEMA.roomSize - 1) continue;
						const tile = y * SCHEMA.roomSize + x;
						if (legal(tile)) return tile;
					}
				}
			}
		}
		// A legal action must remain available even in an unusual fully-developed
		// room whose anchors have no free stamp position.
		for (let y = 1; y < SCHEMA.roomSize - 1; y++) {
			for (let x = 1; x < SCHEMA.roomSize - 1; x++) {
				const tile = y * SCHEMA.roomSize + x;
				if (legal(tile)) return tile;
			}
		}
		return -1;
	};

	const targetMeta = meta.targetMeta || [];
	const findTarget = (pred) => {
		for (let i = 0; i < targetMeta.length; i++) {
			if (pred(targetMeta[i], i)) return i;
		}
		return -1;
	};
	const findBestTarget = (pred, score) => {
		let best = -1;
		let bestScore = Infinity;
		for (let index = 0; index < targetMeta.length; index++) {
			const target = targetMeta[index];
			if (!pred(target, index)) continue;
			const candidateScore = score(target, index);
			if (candidateScore < bestScore || (candidateScore === bestScore && index < best)) {
				best = index;
				bestScore = candidateScore;
			}
		}
		return best;
	};

	// Room-local source claims keep multi-room workers independent. Specialists are
	// assigned only when they actually own WORK parts; haulers never receive harvest.
	/** @type {{ id: string, src: any, claimed: number, max: number }[]} */
	const sourceClaimsByRoom = new Map();
	for (let ai = 0; ai < nA; ai++) {
		const am = meta.actorMeta[ai];
		if (!am || am.kind !== 'creep') continue;
		const c0 = Game.creeps[am.id];
		if (!c0?.room || sourceClaimsByRoom.has(c0.room.name)) continue;
		const srcs = c0.room.find(C.FIND_SOURCES).filter(s => (s.energy || 0) > 0);
		srcs.sort((a, b) =>
			a.pos.x - b.pos.x || a.pos.y - b.pos.y || a.id.localeCompare(b.id));
		sourceClaimsByRoom.set(
			c0.room.name,
			srcs.map(s => ({
				id: s.id,
				src: s,
				claimed: 0,
				max: sourceLaneCount(c0.room, s),
			})),
		);
	}
	// Sticky: creeps already adjacent to a live source keep that claim
	/** @type {Map<number, any>} */
	const harvestAssign = new Map();
	let siteIssued = false;
	for (let ai = 0; ai < nA; ai++) {
		const am = meta.actorMeta[ai];
		if (!am || am.kind !== 'creep') continue;
		const creep = Game.creeps[am.id];
		if (!creep || creep.spawning) continue;
		const hasWork = creep.body?.some(part => part.type === C.WORK && part.hits > 0);
		if (!hasWork) continue;
		const energy = creep.store?.[C.RESOURCE_ENERGY] || 0;
		const free = creep.store?.getFreeCapacity?.() ?? 0;
		const cap = creep.store?.getCapacity?.() || 50;
		if (!(cap === 0 || (free > 0 && energy / cap < 0.8))) continue;
		for (const claim of sourceClaimsByRoom.get(creep.room.name) || []) {
			if (claim.claimed >= claim.max) continue;
			if (chebyshev(creep.pos, claim.src.pos) <= 1) {
				claim.claimed += 1;
				harvestAssign.set(ai, claim.src);
				break;
			}
		}
	}

	const objectForTarget = target => Game.getObjectById?.(target?.id) || null;
	const buildableNow = target => {
		const site = objectForTarget(target);
		if (!site || site.progress == null) return false;
		if ([ C.STRUCTURE_ROAD, C.STRUCTURE_CONTAINER, C.STRUCTURE_RAMPART ]
			.includes(site.structureType)) return true;
		return (site.room?.lookForAt?.(C.LOOK_CREEPS, site.pos.x, site.pos.y) || []).length === 0;
	};
	const isClaimableController = target => (
		target.structureType === C.STRUCTURE_CONTROLLER
		&& !target.my
		&& !target.owned
		&& (!target.reserved || target.myReservation)
	);
	const hasPart = (creep, part) => creep.body?.some(p => p.type === part && p.hits > 0);
	const partCount = (creep, part) => creep.body
		?.filter(p => p.type === part && p.hits > 0).length || 0;
	const creepRole = creep => {
		const work = partCount(creep, C.WORK);
		const carry = partCount(creep, C.CARRY);
		const claim = partCount(creep, C.CLAIM);
		const attack = partCount(creep, C.ATTACK);
		const ranged = partCount(creep, C.RANGED_ATTACK);
		const heal = partCount(creep, C.HEAL);
		if (claim > 0) return 'claimer';
		if (heal > 0) return 'healer';
		if (ranged > 0) return 'ranged';
		if (attack > 0) return 'melee';
		if (work > 0 && carry === 0) return 'miner';
		if (carry > 0 && work === 0) return 'hauler';
		if (work > 0 && carry > work) return 'builder';
		if (work > carry && carry > 0) return 'upgrader';
		if (work === 0 && carry === 0 && partCount(creep, C.MOVE) > 0) return 'scout';
		return 'flexible';
	};
	const ownedRooms = Object.values(Game.rooms).filter(room => room.controller?.my);
	const ownedSpawns = ownedRooms.flatMap(room => room.find(C.FIND_MY_SPAWNS));
	ownedSpawns.sort((a, b) => a.id.localeCompare(b.id));
	const primarySpawnId = ownedSpawns[0]?.id;
	const remoteBootstrapRooms = ownedRooms
		.filter(room => room.find(C.FIND_MY_SPAWNS).length === 0)
		.sort((a, b) => a.name.localeCompare(b.name));
	const viableForRole = (creep, role = creepRole(creep)) => staffingCreep(
		creep, role, creep.room?.energyCapacityAvailable || 300,
	);
	const capableFlexible = creep => creepRole(creep) === 'flexible'
		&& hasPart(creep, C.WORK)
		&& hasPart(creep, C.CARRY)
		&& hasPart(creep, C.MOVE);
	const viableFlexible = creep => capableFlexible(creep) && viableForRole(creep);
	const homeRooms = ownedRooms
		.filter(room => room.find(C.FIND_MY_SPAWNS).length > 0)
		.sort((a, b) => a.name.localeCompare(b.name));
	const homeRoomNames = new Set(homeRooms.map(room => room.name));
	const neutralOutpostNames = [ ...new Set(targetMeta
		.filter(isClaimableController)
		.map(target => target.room)
		.filter(roomName => targetMeta.some(target =>
			target.kind === 'source' && target.room === roomName))) ].sort();
	// Ordinary empty-world play earns an adjacent outpost only after reaching RCL2.
	// A seeded worker already standing there is itself an observable ready signal.
	const outpostReady = homeRooms.some(room => (room.controller?.level || 0) >= 2)
		|| homeRooms.some(room => room.find(C.FIND_MY_CREEPS).filter(viableFlexible).length > 1)
		|| Object.values(Game.creeps).some(creep =>
			capableFlexible(creep) && neutralOutpostNames.includes(creep.room?.name));
	const neutralOutpostRooms = outpostReady
		? neutralOutpostNames.map(name => ({ name, kind: 'outpost' }))
		: [];

	// Keep one flexible worker in every established home. Claimed bootstrap rooms
	// retain their three-worker setup, while a neutral outpost receives one viable
	// surplus WORK+CARRY worker for a harvest/haul round trip. An aging worker may
	// overlap its successor only long enough to finish its already-committed route.
	const retainedFlexible = new Set();
	for (const room of homeRooms) {
		const keeper = room.find(C.FIND_MY_CREEPS)
			.filter(creep => creepRole(creep) === 'flexible' && viableForRole(creep))
			// Keep an empty worker local when possible so a loaded outpost returner
			// remains surplus and can finish its deterministic delivery leg.
			.sort((a, b) => Number((a.store?.[C.RESOURCE_ENERGY] || 0) > 0)
				- Number((b.store?.[C.RESOURCE_ENERGY] || 0) > 0)
				|| (a.ticksToLive ?? Infinity) - (b.ticksToLive ?? Infinity)
				|| (a.name || a.id).localeCompare(b.name || b.id))[0];
		if (keeper) retainedFlexible.add(keeper.id);
	}
	const bootstrapPool = Object.values(Game.creeps)
		.filter(creep => creepRole(creep) === 'flexible'
			&& viableForRole(creep)
			&& !retainedFlexible.has(creep.id)
			&& !remoteBootstrapRooms.some(room => room.name === creep.room?.name))
		.sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id));
	const remoteAssignments = new Map();
	let dispatchIndex = 0;
	for (const remote of remoteBootstrapRooms) {
		const present = remote.find(C.FIND_MY_CREEPS)
			.filter(creep => creepRole(creep) === 'flexible' && viableForRole(creep)).length;
		for (let needed = present; needed < 3 && dispatchIndex < bootstrapPool.length; needed++) {
			const creep = bootstrapPool[dispatchIndex++];
			remoteAssignments.set(creep.id, { name: remote.name, kind: 'bootstrap' });
		}
	}
	const assigned = new Set(remoteAssignments.keys());
	for (const outpost of neutralOutpostRooms) {
		const available = Object.values(Game.creeps)
			.filter(creep => capableFlexible(creep)
				&& !retainedFlexible.has(creep.id)
				&& !assigned.has(creep.id));
		// The route-finisher cohort is independent from replacement staffing. A
		// nonviable worker stays committed while in the outpost and, once home, until
		// its loaded cargo is transferred. Home-loaded finishers go first because
		// they can complete their observable commitment immediately.
		const finisher = available
			.filter(creep => !viableForRole(creep)
				&& (creep.room?.name === outpost.name
					|| (homeRoomNames.has(creep.room?.name)
						&& (creep.store?.[C.RESOURCE_ENERGY] || 0) > 0)))
			.sort((a, b) => Number(
					homeRoomNames.has(b.room?.name) && (b.store?.[C.RESOURCE_ENERGY] || 0) > 0,
				) - Number(
					homeRoomNames.has(a.room?.name) && (a.store?.[C.RESOURCE_ENERGY] || 0) > 0,
				)
				|| Number((b.store?.[C.RESOURCE_ENERGY] || 0) > 0)
				- Number((a.store?.[C.RESOURCE_ENERGY] || 0) > 0)
				|| (a.ticksToLive ?? Infinity) - (b.ticksToLive ?? Infinity)
				|| (a.name || a.id).localeCompare(b.name || b.id))[0];
		if (finisher) {
			remoteAssignments.set(finisher.id, outpost);
			assigned.add(finisher.id);
		}
		// Independently choose exactly one viable successor. Deterministic observable
		// ordering prevents cohort labels from flipping as the finisher crosses home.
		const candidates = available
			.filter(creep => creep.id !== finisher?.id && viableForRole(creep))
			.sort((a, b) => Number(b.room?.name === outpost.name)
				- Number(a.room?.name === outpost.name)
				|| Number((b.store?.[C.RESOURCE_ENERGY] || 0) > 0)
				- Number((a.store?.[C.RESOURCE_ENERGY] || 0) > 0)
				|| (a.ticksToLive ?? Infinity) - (b.ticksToLive ?? Infinity)
				|| (a.name || a.id).localeCompare(b.name || b.id));
		if (candidates.length) {
			remoteAssignments.set(candidates[0].id, outpost);
			assigned.add(candidates[0].id);
		}
	}

	const reservedPickupEnergy = new Map();
	const reservedWithdrawals = new Set();
	const reservedRefills = new Set();
	const reservedWorkerDeliveries = new Set();
	const workerDeliveryOrderByHauler = new Map();
	const sourceByMiner = new Map();
	const sourceByHauler = new Map();
	for (const ownedRoom of ownedRooms) {
		for (const [ creepId, source ] of sourceAssignments(
			ownedRoom, ownedRoom.find(C.FIND_MY_CREEPS), creepRole,
		)) sourceByMiner.set(creepId, source);
		const haulers = ownedRoom.find(C.FIND_MY_CREEPS)
			.filter(creep => creepRole(creep) === 'hauler')
			.sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id));
		const sources = ownedRoom.find(C.FIND_SOURCES)
			.sort((a, b) =>
				a.pos.x - b.pos.x || a.pos.y - b.pos.y || a.id.localeCompare(b.id));
		for (let index = 0; index < haulers.length && sources.length; index++) {
			sourceByHauler.set(haulers[index].id, sources[index % sources.length]);
		}
		const siteActive = ownedRoom.find(C.FIND_MY_CONSTRUCTION_SITES).length > 0;
		const receivers = ownedRoom.find(C.FIND_MY_CREEPS)
			.filter(creep => !creep.spawning
				&& hasPart(creep, C.WORK) && hasPart(creep, C.CARRY)
				&& [ 'builder', 'upgrader', 'flexible' ].includes(creepRole(creep)))
			.sort((a, b) => {
				const priority = candidate => {
					const role = creepRole(candidate);
					if (siteActive && role === 'builder') return 0;
					if (role === 'upgrader') return 1;
					if (role === 'builder') return 2;
					return 3;
				};
				return priority(a) - priority(b)
					|| (a.name || a.id).localeCompare(b.name || b.id);
			});
		for (let index = 0; index < haulers.length && receivers.length; index++) {
			const offset = index % receivers.length;
			workerDeliveryOrderByHauler.set(
				haulers[index].id,
				[ ...receivers.slice(offset), ...receivers.slice(0, offset) ],
			);
		}
	}
	for (let ai = 0; ai < nA; ai++) {
		const am = meta.actorMeta[ai];
		if (!am) continue;
		if (am.kind === 'room') {
			const room = Game.rooms[am.room];
			const controller = room?.controller;
			if (!siteIssued && controller?.my
				&& room.find(C.FIND_MY_CONSTRUCTION_SITES).length === 0) {
				const myStructures = room.find(C.FIND_MY_STRUCTURES);
				const roomStructures = room.find(C.FIND_STRUCTURES);
				const count = structureType => myStructures
					.filter(structure => structure.structureType === structureType).length;
				const countIncludingPublic = structureType => roomStructures
					.filter(structure => structure.structureType === structureType).length;
				const belowLimit = (structureType, desired = Infinity) => {
					const allowed = C.CONTROLLER_STRUCTURES?.[structureType]?.[controller.level] ?? 0;
					return count(structureType) < Math.min(allowed, desired);
				};
				const initialExtensionTarget = controller.level >= 3 ? 7 : 5;
				const plan = [
					['extension', () => belowLimit(
						C.STRUCTURE_EXTENSION,
						ownedRooms.length > 1 ? Infinity : initialExtensionTarget,
					)],
					['container', () => {
						const allowed = C.CONTROLLER_STRUCTURES?.[C.STRUCTURE_CONTAINER]
							?.[controller.level] ?? 0;
						const desired = Math.min(allowed, room.find(C.FIND_SOURCES).length);
						return controller.level >= 3
							&& countIncludingPublic(C.STRUCTURE_CONTAINER) < desired;
					}],
					['tower', () => belowLimit(C.STRUCTURE_TOWER)],
					['storage', () => belowLimit(C.STRUCTURE_STORAGE)],
					['road', () => controller.level >= 3 && belowLimit(C.STRUCTURE_ROAD, 10)],
					['rampart', () => controller.level >= 3 && belowLimit(C.STRUCTURE_RAMPART, 4)],
				];
				const wanted = room.find(C.FIND_MY_SPAWNS).length === 0
					? 'spawn' : plan.find(([, allowed]) => allowed())?.[0] ?? null;
				const typeIndex = SCHEMA.constructionTypes.indexOf(wanted);
				if (typeIndex >= 0) {
					const tile = legalConstructionTile(room, typeIndex);
					if (tile >= 0) {
						types[ai][0] = INTENT.createConstructionSite;
						constructionTypes[ai][0] = typeIndex;
						constructionTiles[ai][0] = tile;
						siteIssued = true;
					}
				}
			}
			continue;
		}

		if (am.kind === 'structure') {
			const spawn = Game.getObjectById?.(am.id)
				|| Object.values(Game.spawns || {}).find(candidate => candidate.id === am.id);
			if (spawn?.structureType === C.STRUCTURE_SPAWN && !spawn.spawning) {
				const energyAvailable = spawn.room?.energyAvailable
					?? (spawn.store?.[C.RESOURCE_ENERGY] || 0);
				const energyCapacity = spawn.room?.energyCapacityAvailable ?? 300;
				const roomCreeps = spawn.room.find(C.FIND_MY_CREEPS);
				const healthyCreeps = roomCreeps.filter(creep => viableForRole(creep));
				const plan = roomDemand(spawn.room);
				const ownedRoomCount = Math.max(1, ownedRooms.length);
				const hasClaimer = Object.values(Game.creeps).some(creep => hasPart(creep, C.CLAIM));
				const needClaimer = EXPLICIT_CLAIM_CURRICULUM
					&& Game.gcl?.level > ownedRoomCount
					&& targetMeta.some(isClaimableController) && !hasClaimer;
				let wantedRole = SPAWN_CURRICULUM;
				const hasHealthyCarrier = healthyCreeps.some(creep =>
					['hauler', 'flexible'].includes(creepRole(creep)));
				const hasHealthyHarvester = healthyCreeps.some(creep =>
					hasPart(creep, C.WORK));
				let recoveryBody = null;
				if (wantedRole) {
					// Explicit engine-backed body-composition curriculum. The complete
					// body is still affordability-checked and executed by the real spawn.
				} else if (!hasHealthyCarrier || !hasHealthyHarvester) {
					// Attrition recovery takes precedence over strategic save-up; without
					// both WORK and CARRY, source energy cannot re-enter the spawn economy.
					wantedRole = 'flexible';
					recoveryBody = [ ...BOOTSTRAP_BODIES.flexible ];
				} else if (needClaimer && energyCapacity >= 650) {
					wantedRole = 'claimer';
				} else {
					const roleCounts = new Map();
					for (const creep of healthyCreeps) {
						const role = creepRole(creep);
						roleCounts.set(role, (roleCounts.get(role) || 0) + 1);
					}
				if (spawn.id === primarySpawnId
					&& (remoteBootstrapRooms.length || neutralOutpostRooms.length)) {
					const allHealthyFlexible = Object.values(Game.creeps).filter(creep =>
						creepRole(creep) === 'flexible' && viableForRole(creep)).length;
					roleCounts.set('flexible', allHealthyFlexible);
					const flexible = plan.find(entry => entry[0] === 'flexible');
					if (flexible) flexible[1] += remoteBootstrapRooms.length * 3
						+ neutralOutpostRooms.length;
					}
					wantedRole = mostUnderfilledRole(roleCounts, plan);
					const wantedCost = wantedRole
						? bodyCost(teacherBody(wantedRole, energyCapacity)) : Infinity;
					// Spawn a useful affordable deficit every tick. Claimer save-up is
					// the sole intentional reserve because no cheaper role can claim.
					const haulerTarget = plan.find(([ role ]) => role === 'hauler')?.[1] || 0;
					const haulerCount = roleCounts.get('hauler') || 0;
					const droppedEnergy = spawn.room.find(C.FIND_DROPPED_RESOURCES)
						.reduce((sum, resource) => sum + (
							resource.resourceType === C.RESOURCE_ENERGY ? resource.amount || 0 : 0
						), 0);
					const haulerCost = bodyCost(teacherBody('hauler', energyCapacity));
					if (wantedCost > energyAvailable && droppedEnergy > 0
						&& haulerCount < haulerTarget && haulerCost <= energyAvailable) {
						wantedRole = 'hauler';
					} else if (wantedCost > energyAvailable) {
						const affordableCounts = new Map(roleCounts);
						const affordablePlan = plan.filter(([ role ]) =>
							bodyCost(teacherBody(role, energyCapacity)) <= energyAvailable);
						wantedRole = mostUnderfilledRole(affordableCounts, affordablePlan);
					}
				}
				if (wantedRole) {
					const body = recoveryBody || teacherBody(wantedRole, energyCapacity);
					const cost = body ? bodyCost(body) : Infinity;
					// The body is the exact teacher target; no template substitution occurs.
					if (body && energyAvailable >= cost) {
						types[ai][0] = INTENT.spawnCreep;
						const encoding = bodyEncoding(body);
						bodyCounts[ai][0] = encoding.counts;
						bodyOrder[ai][0] = encoding.order;
					}
				}
			}
			continue;
		}

		const creep = Game.creeps[am.id];
		if (!creep || creep.spawning) continue;
		const room = creep.room;
		const controller = room.controller;
		const energy = creep.store?.[C.RESOURCE_ENERGY] || 0;
		const free = creep.store?.getFreeCapacity?.() ?? 0;
		const capacity = creep.store?.getCapacity?.() || 0;
		const work = hasPart(creep, C.WORK);
		const carry = hasPart(creep, C.CARRY);
		const claim = hasPart(creep, C.CLAIM);
		const role = creepRole(creep);
		const roomCreeps = room.find(C.FIND_MY_CREEPS);
		const roomHasHauler = roomCreeps.some(candidate => creepRole(candidate) === 'hauler');
		const bootstrapRefiller = roomCreeps
			.filter(candidate => creepRole(candidate) === 'flexible')
			.sort((a, b) => a.name.localeCompare(b.name))[0];

		if (claim && EXPLICIT_CLAIM_CURRICULUM) {
			const claimTarget = findTarget(isClaimableController);
			if (claimTarget >= 0) {
				types[ai][0] = INTENT.claimController;
				targets[ai][0] = claimTarget;
				continue;
			}
		}
		const dispatchRoom = remoteAssignments.get(creep.id);
		if (dispatchRoom?.kind === 'outpost') {
			const atHome = homeRoomNames.has(room.name);
			const loadFraction = capacity > 0 ? energy / capacity : 0;
			if (energy > 0 && (atHome || loadFraction >= 0.8 || free <= 0)) {
				// A remote returner's destination has to be stable across ticks. The
				// spawn's free capacity flips as it fills and drains, and the home
				// bank sits on the opposite side of the room, so ranking the spawn
				// first made the chosen sink alternate every tick and the executor
				// reversed its route each time, leaving the carrier oscillating in
				// the outpost forever. Remote cargo therefore belongs in a bank,
				// which local haulers then draw on to refill the spawn; distance
				// breaks ties deterministically.
				const homeSink = findBestTarget(target => {
					if (target.kind !== 'structure' || !homeRoomNames.has(target.room)) return false;
					if (![ C.STRUCTURE_SPAWN, C.STRUCTURE_EXTENSION, C.STRUCTURE_TOWER,
						C.STRUCTURE_STORAGE, C.STRUCTURE_CONTAINER ]
						.includes(target.structureType)) return false;
					const sink = objectForTarget(target);
					return !reservedRefills.has(target.id)
						&& (sink?.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) || 0) > 0;
				}, target => {
					const bank = [ C.STRUCTURE_STORAGE, C.STRUCTURE_CONTAINER ]
						.includes(target.structureType) ? 0
						: target.structureType === C.STRUCTURE_TOWER ? 2 : 1;
					return bank * 128 + Math.min(127, chebyshev(target, creep.pos));
				});
				if (homeSink >= 0) {
					reservedRefills.add(targetMeta[homeSink].id);
					types[ai][0] = INTENT.transfer;
					targets[ai][0] = homeSink;
					continue;
				}
				// A committed carrier waits for observable home capacity rather than
				// spending remote cargo on local work or issuing ERR_FULL. This also
				// keeps an aging route finisher committed until its delivery completes.
				continue;
			}
			if (energy === 0 || (!atHome && loadFraction < 0.8 && free > 0)) {
				const remoteSource = findBestTarget(target =>
					target.kind === 'source' && target.room === dispatchRoom.name,
					target => chebyshev(target, creep.pos));
				if (remoteSource >= 0) {
					types[ai][0] = INTENT.harvest;
					targets[ai][0] = remoteSource;
					continue;
				}
			}
		} else if (dispatchRoom && dispatchRoom.name !== room.name) {
			const remoteController = energy > 0 ? findTarget(target =>
				target.structureType === C.STRUCTURE_CONTROLLER
				&& target.room === dispatchRoom.name && target.my) : -1;
			if (remoteController >= 0) {
				types[ai][0] = INTENT.upgradeController;
				targets[ai][0] = remoteController;
				continue;
			}
			const remoteSource = findTarget(target =>
				target.kind === 'source' && target.room === dispatchRoom.name);
			if (remoteSource >= 0) {
				types[ai][0] = INTENT.harvest;
				targets[ai][0] = remoteSource;
				continue;
			}
		}

		const myStructures = room.find(C.FIND_MY_STRUCTURES);
		const refill = myStructures
			.filter(structure => [
				C.STRUCTURE_SPAWN,
				C.STRUCTURE_EXTENSION,
				C.STRUCTURE_TOWER,
			].includes(structure.structureType)
				&& (structure.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) || 0) > 0)
			.sort((a, b) => {
				const priority = structure => structure.structureType === C.STRUCTURE_TOWER ? 1 : 0;
				return priority(a) - priority(b)
					|| chebyshev(creep.pos, a.pos) - chebyshev(creep.pos, b.pos);
			});
		// Any empty carrier first consumes dropped energy. Workers may withdraw
		// for building/upgrading; pure haulers only withdraw stored energy when a
		// productive sink needs it, preventing storage withdraw/deposit churn.
		if (carry && energy === 0 && free > 0) {
			const pickupSource = sourceByHauler.get(creep.id);
			const pickup = (role === 'hauler' || !roomHasHauler)
				? findBestTarget(
					target => target.kind === 'resource'
						&& target.room === room.name
						&& (objectForTarget(target)?.amount || target.amount || 0)
							> (reservedPickupEnergy.get(target.id) || 0),
					target => pickupSource
						? chebyshev(target, pickupSource.pos)
						: chebyshev(target, creep.pos),
				)
				: -1;
			if (pickup >= 0) {
				const target = targetMeta[pickup];
				reservedPickupEnergy.set(
					target.id,
					(reservedPickupEnergy.get(target.id) || 0) + Math.max(1, free),
				);
				types[ai][0] = INTENT.pickup;
				targets[ai][0] = pickup;
				continue;
			}
			const withdraw = (work || refill.length > 0) ? findTarget(target => {
				if (target.kind !== 'structure' || target.room !== room.name) return false;
				const structure = objectForTarget(target);
				return [C.STRUCTURE_STORAGE, C.STRUCTURE_CONTAINER, C.STRUCTURE_LINK]
					.includes(target.structureType)
					&& !reservedWithdrawals.has(target.id)
					&& (structure?.store?.[C.RESOURCE_ENERGY] || 0) > 0;
			}) : -1;
			if (withdraw >= 0) {
				reservedWithdrawals.add(targetMeta[withdraw].id);
				types[ai][0] = INTENT.withdraw;
				targets[ai][0] = withdraw;
				continue;
			}
		}

		const sourceClaims = sourceClaimsByRoom.get(room.name) || [];
		const canHarvest = work
			&& (role === 'miner' || (carry && energy === 0))
			&& (capacity === 0 || free > 0);
		const assignedSource = sourceByMiner.get(creep.id);
		if (canHarvest && assignedSource && !harvestAssign.has(ai)) {
			const claim = sourceClaims.find(candidate => candidate.id === assignedSource.id);
			if (claim && claim.claimed < claim.max && (claim.src.energy || 0) > 0) {
				claim.claimed += 1;
				harvestAssign.set(ai, assignedSource);
			}
		}
		if (canHarvest && !harvestAssign.has(ai)) {
			const open = sourceClaims
				.filter(source => source.claimed < source.max)
				.sort((a, b) => (a.claimed - b.claimed)
					|| chebyshev(creep.pos, a.src.pos) - chebyshev(creep.pos, b.src.pos));
			if (open.length) {
				open[0].claimed += 1;
				harvestAssign.set(ai, open[0].src);
			}
		}
		const assigned = harvestAssign.get(ai);
		const shouldHarvest = assigned && (capacity === 0 || energy / Math.max(1, capacity) < 0.8);
		if (shouldHarvest) {
			const sourceTarget = findTarget(target => target.kind === 'source' && target.id === assigned.id);
			if (sourceTarget >= 0) {
				types[ai][0] = INTENT.harvest;
				targets[ai][0] = sourceTarget;
				continue;
			}
		}

		if (energy > 0) {
			// Spawns/extensions are always productive, and towers must stay fueled;
			// population targets must not suppress either logistics label.
			const refillTarget = refill.find(candidate => !reservedRefills.has(candidate.id));
			if (refillTarget && (role === 'hauler' || role === 'miner'
				|| (role === 'flexible' && !roomHasHauler
					&& bootstrapRefiller?.id === creep.id))) {
				const target = findTarget(candidate => candidate.id === refillTarget.id);
				if (target >= 0) {
					reservedRefills.add(refillTarget.id);
					types[ai][0] = INTENT.transfer;
					targets[ai][0] = target;
					continue;
				}
			}
			const siteTarget = findTarget(target =>
				target.kind === 'site' && target.room === room.name && buildableNow(target));
			const basicBuilds = role === 'flexible';
			if (siteTarget >= 0 && work && (role === 'builder' || basicBuilds)) {
				types[ai][0] = INTENT.build;
				targets[ai][0] = siteTarget;
				continue;
			}
			// A hauler may feed exactly one working creep per tick. Reserving receivers
			// keeps the teacher free of aliased same-target transfer labels.
			if (role === 'hauler') {
				const receiver = (workerDeliveryOrderByHauler.get(creep.id) || [])
					.find(candidate => !reservedWorkerDeliveries.has(candidate.id)
						&& (candidate.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) || 0) > 0);
				if (receiver) {
					const target = findTarget(candidate =>
						candidate.kind === 'creep' && candidate.id === receiver.id);
					if (target >= 0) {
						reservedWorkerDeliveries.add(receiver.id);
						types[ai][0] = INTENT.transfer;
						targets[ai][0] = target;
						continue;
					}
				}
				// Bank surplus in durable logistics stores. A later spawn or tower
				// demand makes the guarded withdrawal branch executable.
				const bank = myStructures
					.filter(structure => [C.STRUCTURE_STORAGE, C.STRUCTURE_CONTAINER]
						.includes(structure.structureType)
						&& (structure.store?.getFreeCapacity?.(C.RESOURCE_ENERGY) || 0) > 0)
					.sort((a, b) => chebyshev(creep.pos, a.pos) - chebyshev(creep.pos, b.pos));
				if (bank.length) {
					const target = findTarget(candidate => candidate.id === bank[0].id);
					if (target >= 0) {
						types[ai][0] = INTENT.transfer;
						targets[ai][0] = target;
						continue;
					}
				}
			}
			// An aging remote worker can fall below the staffing threshold while it is
			// still in a neutral outpost. Generic local work must never turn that state
			// into an upgrade intent against an unowned controller (ERR_NOT_OWNER).
			const controllerTarget = controller?.my
				? findTarget(target => target.id === controller.id && target.my)
				: -1;
			if (controllerTarget >= 0 && work && role !== 'miner') {
				types[ai][0] = INTENT.upgradeController;
				targets[ai][0] = controllerTarget;
			}
		}
	}

	while (types.length < maxActors) {
		types.push(Array(intentSlots).fill(INTENT.none));
		dirs.push(Array(intentSlots).fill(0));
		targets.push(Array(intentSlots).fill(0));
		amounts.push(Array(intentSlots).fill(0));
		bodyCounts.push(Array.from(
			{ length: intentSlots }, () => Array(bodyPartTypes.length).fill(0),
		));
		bodyOrder.push(Array.from(
			{ length: intentSlots }, () =>
				Array.from({ length: bodyPartTypes.length }, (_, type) => type),
		));
		constructionTypes.push(Array(intentSlots).fill(0));
		constructionTiles.push(Array(intentSlots).fill(0));
	}

	return {
		types, dirs, targets, amounts, bodyCounts, bodyOrder,
		constructionTypes, constructionTiles,
	};
}
