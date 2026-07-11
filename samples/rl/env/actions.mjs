/**
 * Apply discrete AR actions from the policy onto Screeps Game objects.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from 'xxscreeps/game/constants/index.js';

const schema = JSON.parse(
	fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '../schema.json'), 'utf8'),
);

const { intentTypes, amountBins, directions } = schema;
const DIR_CONST = [
	C.TOP, C.TOP_RIGHT, C.RIGHT, C.BOTTOM_RIGHT,
	C.BOTTOM, C.BOTTOM_LEFT, C.LEFT, C.TOP_LEFT,
];

/**
 * @param {import('xxscreeps/game/index.js').GameConstructor} Game
 * @param {{
 *   actorMeta: { kind: string, id: string }[],
 *   targetMeta: { kind: string, id: string, room: string }[],
 * }} meta
 * @param {{
 *   types: number[][],
 *   dirs: number[][],
 *   targets: number[][],
 *   amounts: number[][],
 * }} actions  // [actor][slot]
 */
export function applyActions(Game, meta, actions) {
	const results = [];
	const { actorMeta, targetMeta } = meta;
	const nActors = Math.min(actorMeta.length, actions.types.length);

	for (let ai = 0; ai < nActors; ai++) {
		const actor = actorMeta[ai];
		const slots = actions.types[ai]?.length ?? 0;
		for (let slot = 0; slot < slots; slot++) {
			const typeIdx = actions.types[ai][slot] | 0;
			const type = intentTypes[typeIdx] || 'none';
			if (type === 'none') continue;

			const dirIdx = actions.dirs[ai]?.[slot] | 0;
			const tgtIdx = actions.targets[ai]?.[slot] | 0;
			const amtIdx = actions.amounts[ai]?.[slot] | 0;
			const amountBin = amountBins[Math.min(amtIdx, amountBins.length - 1)] ?? 0;

			let code = -1;
			try {
				if (actor.kind === 'creep') {
					const creep = Game.creeps[actor.id];
					if (!creep) {
						results.push({ actor: actor.id, type, code: -1, err: 'missing creep' });
						continue;
					}
					const target = resolveTarget(Game, targetMeta[tgtIdx]);
					code = runCreepIntent(creep, type, dirIdx, target, amountBin);
				} else {
					const struct = Game.getObjectById?.(actor.id);
					// also search spawns by id from rooms
					const obj = struct || findStructure(Game, actor.id);
					if (!obj) {
						results.push({ actor: actor.id, type, code: -1, err: 'missing structure' });
						continue;
					}
					const target = resolveTarget(Game, targetMeta[tgtIdx]);
					code = runStructureIntent(Game, obj, type, target);
				}
			} catch (err) {
				results.push({ actor: actor.id, type, code: -99, err: String(err?.message || err) });
				continue;
			}
			results.push({ actor: actor.id, type, code });
		}
	}
	return results;
}

function findStructure(Game, id) {
	for (const roomName of Object.keys(Game.rooms)) {
		for (const st of Game.rooms[roomName].find(C.FIND_STRUCTURES)) {
			if (st.id === id) return st;
		}
	}
	return null;
}

function resolveTarget(Game, meta) {
	if (!meta) return null;
	if (meta.kind === 'creep') {
		for (const name of Object.keys(Game.creeps)) {
			if (Game.creeps[name].id === meta.id) return Game.creeps[name];
		}
		// name as id fallback
		return Game.creeps[meta.id] || null;
	}
	if (Game.getObjectById) {
		const o = Game.getObjectById(meta.id);
		if (o) return o;
	}
	const room = Game.rooms[meta.room];
	if (!room) return null;
	const lists = [
		room.find(C.FIND_SOURCES),
		room.find(C.FIND_MINERALS),
		room.find(C.FIND_STRUCTURES),
		room.find(C.FIND_DROPPED_RESOURCES),
		room.find(C.FIND_CONSTRUCTION_SITES),
		room.find(C.FIND_CREEPS),
	];
	for (const list of lists) {
		for (const o of list) if (o.id === meta.id) return o;
	}
	return null;
}

function amountOrAll(creep, bin, resource = C.RESOURCE_ENERGY) {
	if (!bin || bin <= 0) return creep.store?.[resource] || creep.store?.getUsedCapacity?.(resource) || undefined;
	return bin;
}

function runCreepIntent(creep, type, dirIdx, target, amountBin) {
	switch (type) {
		case 'move':
			return creep.move(DIR_CONST[dirIdx % 8]);
		case 'harvest':
			return target ? creep.harvest(target) : C.ERR_INVALID_TARGET;
		case 'transfer': {
			const amt = amountOrAll(creep, amountBin);
			return target ? creep.transfer(target, C.RESOURCE_ENERGY, amt) : C.ERR_INVALID_TARGET;
		}
		case 'withdraw': {
			const amt = amountBin > 0 ? amountBin : undefined;
			return target ? creep.withdraw(target, C.RESOURCE_ENERGY, amt) : C.ERR_INVALID_TARGET;
		}
		case 'pickup':
			return target ? creep.pickup(target) : C.ERR_INVALID_TARGET;
		case 'drop': {
			const amt = amountOrAll(creep, amountBin);
			return creep.drop(C.RESOURCE_ENERGY, amt);
		}
		case 'upgradeController':
			return target ? creep.upgradeController(target) : (
				creep.room.controller ? creep.upgradeController(creep.room.controller) : C.ERR_INVALID_TARGET
			);
		case 'build':
			return target ? creep.build(target) : C.ERR_INVALID_TARGET;
		case 'repair':
			return target ? creep.repair(target) : C.ERR_INVALID_TARGET;
		case 'attack':
			return target ? creep.attack(target) : C.ERR_INVALID_TARGET;
		case 'rangedAttack':
			return target ? creep.rangedAttack(target) : C.ERR_INVALID_TARGET;
		case 'heal':
			return target ? creep.heal(target) : C.ERR_INVALID_TARGET;
		case 'rangedHeal':
			return target ? creep.rangedHeal(target) : C.ERR_INVALID_TARGET;
		case 'claimController':
			return target ? creep.claimController(target) : C.ERR_INVALID_TARGET;
		case 'reserveController':
			return target ? creep.reserveController(target) : C.ERR_INVALID_TARGET;
		case 'attackController':
			return target ? creep.attackController(target) : C.ERR_INVALID_TARGET;
		case 'dismantle':
			return target ? creep.dismantle(target) : C.ERR_INVALID_TARGET;
		case 'generateSafeMode':
			return creep.room.controller ? creep.generateSafeMode(creep.room.controller) : C.ERR_INVALID_TARGET;
		default:
			return C.ERR_NO_BODYPART;
	}
}

function runStructureIntent(Game, struct, type, target) {
	if (struct.structureType === C.STRUCTURE_SPAWN && type === 'spawnCreep') {
		// Minimal bootstrap body when RL asks to spawn
		const body = [ C.WORK, C.CARRY, C.MOVE, C.MOVE ];
		const name = `rl_${Game.time}_${Math.floor(Math.random() * 1e4)}`;
		return struct.spawnCreep(body, name);
	}
	if (struct.structureType === C.STRUCTURE_TOWER) {
		if (type === 'attack' && target) return struct.attack(target);
		if (type === 'repair' && target) return struct.repair(target);
		if (type === 'heal' && target) return struct.heal(target);
	}
	return C.ERR_INVALID_ARGS;
}
