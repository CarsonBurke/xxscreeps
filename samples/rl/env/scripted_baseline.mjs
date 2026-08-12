/**
 * Scripted RCL1→RCL2 economy teacher using the SAME action interface as the RL policy.
 * Used as: (1) competence floor, (2) BC teacher, (3) reward sanity check.
 *
 * Policy: bootstrap workers → balanced source assignments → productive refill →
 * extensions/scaled bodies → build/upgrade. Target macros use approachOr.
 */
import * as C from 'xxscreeps/game/constants/index.js';
import { SCHEMA } from './encode.mjs';

const { intentTypes, intentSlots, maxActors, maxCreepActors, maxTargets } = SCHEMA;
const INTENT = Object.fromEntries(intentTypes.map((n, i) => [n, i]));

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

	const targetMeta = meta.targetMeta || [];
	const findTarget = (pred) => {
		for (let i = 0; i < targetMeta.length; i++) {
			if (pred(targetMeta[i], i)) return i;
		}
		return -1;
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
		srcs.sort((a, b) => a.id.localeCompare(b.id));
			sourceClaimsByRoom.set(
				c0.room.name,
				srcs.map(s => ({ id: s.id, src: s, claimed: 0, max: 6 })),
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
		if (!creep) continue;
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
		if (claim > 0) return 'claimer';
		if (work > 0 && carry === 0) return 'miner';
		if (carry > 0 && work === 0) return 'hauler';
		if (work === 1 && carry === 2) return 'builder';
		if (work === 2 && carry === 1) return 'upgrader';
		return 'worker_basic';
	};
	const templateIndex = name => SCHEMA.bodyTemplates.findIndex(template => template.name === name);
	for (let ai = 0; ai < nA; ai++) {
		const am = meta.actorMeta[ai];
		if (!am) continue;

		if (am.kind === 'structure') {
			const spawn = Game.getObjectById?.(am.id)
				|| Object.values(Game.spawns || {}).find(candidate => candidate.id === am.id);
			if (spawn?.structureType === C.STRUCTURE_SPAWN && !spawn.spawning) {
				const energyAvailable = spawn.room?.energyAvailable
					?? (spawn.store?.[C.RESOURCE_ENERGY] || 0);
				const roomCreeps = spawn.room.find(C.FIND_MY_CREEPS);
				// Spawn replacements before a cohort disappears. Body spawn time plus
				// route slack is well below this threshold for every current template.
				const healthyCreeps = roomCreeps.filter(creep =>
					creep.ticksToLive == null || creep.ticksToLive > 150);
				const rcl = spawn.room.controller?.level || 1;
				const extensions = spawn.room.find(C.FIND_MY_STRUCTURES)
					.filter(structure => structure.structureType === C.STRUCTURE_EXTENSION).length;
				const ownedRoomCount = Math.max(
					1,
					Object.values(Game.rooms).filter(room => room.controller?.my).length,
				);
				const roomActorBudget = Math.max(4, Math.floor(maxCreepActors / ownedRoomCount));
				const energyCapacity = spawn.room?.energyCapacityAvailable ?? 300;
				// Populate enough basic workers to build the RCL2 extension bank
				// promptly. Once 550 capacity is online, scale to the full per-room
				// budget and switch new bodies to specialists.
				const bootstrapPopulation = 12;
				const scaledPopulation = energyCapacity >= 550
					? Math.min(roomActorBudget, 24)
					: bootstrapPopulation;
				const desired = Math.min(
					roomActorBudget,
					Math.max(scaledPopulation, 4 + extensions * 4),
				);
				const ownedRooms = ownedRoomCount;
				const hasClaimer = Object.values(Game.creeps).some(creep => hasPart(creep, C.CLAIM));
				const needClaimer = Game.gcl?.level > ownedRooms
					&& targetMeta.some(isClaimableController) && !hasClaimer;
				let wantedTemplate = null;
				const hasHealthyCarrier = healthyCreeps.some(creep =>
					['hauler', 'worker_basic'].includes(creepRole(creep)));
				if (!hasHealthyCarrier) {
					// Attrition recovery takes precedence over strategic save-up; without
					// a carrier, dropped energy can never re-enter the spawn economy.
					wantedTemplate = 'worker_basic';
				} else if (needClaimer && energyCapacity >= 650) {
					wantedTemplate = 'claimer';
				} else if (healthyCreeps.length < desired) {
					const roleCounts = new Map();
					for (const creep of healthyCreeps) {
						const role = creepRole(creep);
						roleCounts.set(role, (roleCounts.get(role) || 0) + 1);
					}
					if (healthyCreeps.length === 0) {
						// A carrier-capable first worker bootstraps the first miner/hauler pair.
						wantedTemplate = 'worker_basic';
					} else if (energyCapacity < 550) {
						const plan = [
							['miner', 4],
							['hauler', 3],
							['upgrader', 2],
							['builder', 2],
							['worker_basic', 1],
						];
						wantedTemplate = plan.find(([role, count]) =>
							(roleCounts.get(role) || 0) < count)?.[0] || 'upgrader';
					} else {
						const plan = [
							['miner', 8],
							['hauler', 5],
							['builder', 3],
							['upgrader', Math.max(3, desired - 16)],
						];
						wantedTemplate = plan.find(([role, count]) =>
							(roleCounts.get(role) || 0) < count)?.[0] || 'upgrader';
					}
				}
				if (wantedTemplate) {
					const idx = templateIndex(wantedTemplate);
					const cost = SCHEMA.bodyTemplates[idx]?.cost ?? Infinity;
					// Do not fall back to a cheap body: retaining room energy until the
					// exact queued role is affordable is essential for phase progression.
					if (idx >= 0 && energyAvailable >= cost) {
						types[ai][0] = INTENT.spawnCreep;
						amounts[ai][0] = idx;
					}
				}
			}
			continue;
		}

		const creep = Game.creeps[am.id];
		if (!creep) continue;
		const room = creep.room;
		const controller = room.controller;
		const energy = creep.store?.[C.RESOURCE_ENERGY] || 0;
		const free = creep.store?.getFreeCapacity?.() ?? 0;
		const capacity = creep.store?.getCapacity?.() || 0;
		const work = hasPart(creep, C.WORK);
		const carry = hasPart(creep, C.CARRY);
		const claim = hasPart(creep, C.CLAIM);
		const role = creepRole(creep);
		const roomHasHauler = room.find(C.FIND_MY_CREEPS)
			.some(candidate => creepRole(candidate) === 'hauler');

		if (claim) {
			const claimTarget = findTarget(isClaimableController);
			if (claimTarget >= 0) {
				types[ai][0] = INTENT.claimController;
				targets[ai][0] = claimTarget;
				continue;
			}
		}

		const myStructures = room.find(C.FIND_MY_STRUCTURES);
		const localSites = room.find(C.FIND_MY_CONSTRUCTION_SITES);
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
		const countStructures = structureType => myStructures
			.filter(structure => structure.structureType === structureType).length;
		const extensions = myStructures.filter(s => s.structureType === C.STRUCTURE_EXTENSION);
		const spawn = room.find(C.FIND_MY_SPAWNS)[0];
		const buildPosition = findTarget(target => target.kind === 'position' && target.room === room.name);
		if (!siteIssued && controller?.my && localSites.length === 0 && buildPosition >= 0) {
			const limits = C.CONTROLLER_STRUCTURES || {};
			const belowLimit = (structureType, desired = Infinity) => {
				const allowed = limits[structureType]?.[controller.level] ?? 0;
				return countStructures(structureType) < Math.min(allowed, desired);
			};
			const constructionPlan = [
				['extension', () => belowLimit(C.STRUCTURE_EXTENSION)],
				['container', () => controller.level >= 2
					&& belowLimit(C.STRUCTURE_CONTAINER, room.find(C.FIND_SOURCES).length)],
				['tower', () => belowLimit(C.STRUCTURE_TOWER)],
				['storage', () => belowLimit(C.STRUCTURE_STORAGE)],
				['road', () => controller.level >= 3 && belowLimit(C.STRUCTURE_ROAD, 10)],
				['rampart', () => controller.level >= 3 && belowLimit(C.STRUCTURE_RAMPART, 4)],
			];
			const wanted = !spawn
				? 'spawn'
				: constructionPlan.find(([, allowed]) => allowed())?.[0] ?? null;
			if (wanted) {
				types[ai][0] = INTENT.createConstructionSite;
				targets[ai][0] = buildPosition;
				amounts[ai][0] = SCHEMA.constructionTypes.indexOf(wanted);
				siteIssued = true;
				continue;
			}
		}

		// Any empty carrier first consumes dropped energy. Workers may withdraw
		// for building/upgrading; pure haulers only withdraw stored energy when a
		// productive sink needs it, preventing storage withdraw/deposit churn.
		if (carry && energy === 0 && free > 0) {
			const pickup = findTarget(target => target.kind === 'resource' && target.room === room.name);
			if (pickup >= 0) {
				types[ai][0] = INTENT.pickup;
				targets[ai][0] = pickup;
				continue;
			}
			const withdraw = (work || refill.length > 0) ? findTarget(target => {
				if (target.kind !== 'structure' || target.room !== room.name) return false;
				const structure = objectForTarget(target);
				return [C.STRUCTURE_STORAGE, C.STRUCTURE_CONTAINER, C.STRUCTURE_LINK]
					.includes(target.structureType)
					&& (structure?.store?.[C.RESOURCE_ENERGY] || 0) > 0;
			}) : -1;
			if (withdraw >= 0) {
				types[ai][0] = INTENT.withdraw;
				targets[ai][0] = withdraw;
				continue;
			}
		}

		const sourceClaims = sourceClaimsByRoom.get(room.name) || [];
		const canHarvest = work && (capacity === 0 || free > 0);
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
			if (refill.length && (role === 'hauler'
				|| (role === 'worker_basic' && !roomHasHauler))) {
				const target = findTarget(candidate => candidate.id === refill[0].id);
				if (target >= 0) {
					types[ai][0] = INTENT.transfer;
					targets[ai][0] = target;
					continue;
				}
			}
			const siteTarget = findTarget(target => target.kind === 'site');
			const basicBuilds = role === 'worker_basic';
			if (siteTarget >= 0 && work && (role === 'builder' || basicBuilds)) {
				types[ai][0] = INTENT.build;
				targets[ai][0] = siteTarget;
				continue;
			}
			// Pure haulers bank surplus in durable logistics stores. A later spawn
			// or tower demand makes their guarded withdrawal branch executable.
			if (role === 'hauler') {
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
			const controllerTarget = findTarget(target => target.id === controller?.id);
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
	}

	return { types, dirs, targets, amounts };
}
