import assert from 'node:assert/strict';
import test from 'node:test';
import * as C from 'xxscreeps/game/constants/index.js';
import { actionOutcome, applyActions } from './actions.mjs';
import { encodeObservationFromRooms, SCHEMA } from './encode.mjs';
import { makeEncodeFixture } from './encode_fixture.mjs';
import { scriptedActions } from './scripted_baseline.mjs';

const TYPE = Object.freeze({
	none: 0,
	move: 1,
	harvest: 2,
	transfer: 3,
	withdraw: 4,
	spawnCreep: 18,
});

test('schema body-part costs stay aligned with engine constants', () => {
	assert.deepEqual(
		SCHEMA.bodyPartCosts,
		SCHEMA.bodyPartTypes.map(type => C.BODYPART_COST[C[type.toUpperCase()]]),
	);
});

function directionTo(from, to) {
	const dx = Math.sign(to.x - from.x);
	const dy = Math.sign(to.y - from.y);
	return new Map([
		[ '0,-1', C.TOP ], [ '1,-1', C.TOP_RIGHT ], [ '1,0', C.RIGHT ],
		[ '1,1', C.BOTTOM_RIGHT ], [ '0,1', C.BOTTOM ], [ '-1,1', C.BOTTOM_LEFT ],
		[ '-1,0', C.LEFT ], [ '-1,-1', C.TOP_LEFT ],
	]).get(`${dx},${dy}`) ?? C.TOP;
}

function position(x, y, roomName = 'W0N0') {
	return {
		x, y, roomName,
		inRangeTo(other, range) {
			return roomName === other.roomName
				&& Math.max(Math.abs(x - other.x), Math.abs(y - other.y)) <= range;
		},
		getDirectionTo(other) {
			return directionTo(this, other);
		},
	};
}

function store(energy, capacity) {
	return {
		[C.RESOURCE_ENERGY]: energy,
		getFreeCapacity: () => Math.max(0, capacity - energy),
	};
}

function makeWorld({ creeps = [], targets = [], time = 1 } = {}) {
	const byId = new Map([ ...creeps, ...targets ].map(object => [ object.id, object ]));
	let objectLookups = 0;
	const room = {
		name: 'W0N0',
		getTerrain: () => ({ get: () => 0 }),
		lookForAt: () => [],
		find: () => [],
	};
	for (const creep of creeps) creep.room = room;
	for (const target of targets) target.room ??= room;
	const Game = {
		time,
		creeps: Object.fromEntries(creeps.map(creep => [ creep.name, creep ])),
		rooms: { W0N0: room },
		getObjectById(id) {
			objectLookups++;
			return byId.get(id) ?? null;
		},
	};
	return { Game, objectLookups: () => objectLookups };
}

function makeCreep(name, x, y, energy = 0, capacity = 100) {
	const moves = [];
	const calls = [];
	return {
		id: `id-${name}`,
		name,
		pos: position(x, y),
		fatigue: 0,
		store: store(energy, capacity),
		moves,
		calls,
		move(direction) {
			moves.push(direction);
			return C.OK;
		},
		harvest(target) {
			calls.push([ 'harvest', target.id ]);
			return C.OK;
		},
		transfer(target, resource, amount) {
			calls.push([ 'transfer', target.id, resource, amount ]);
			return C.OK;
		},
		withdraw(target, resource, amount) {
			calls.push([ 'withdraw', target.id, resource, amount ]);
			return C.OK;
		},
	};
}

function actorMeta(creeps) {
	return creeps.map(creep => ({ kind: 'creep', id: creep.name }));
}

function targetMeta(target, kind = 'structure') {
	return {
		kind,
		id: target.id,
		room: target.pos.roomName,
		x: target.pos.x,
		y: target.pos.y,
		structureType: target.structureType,
	};
}

function selected(types, { dirs, targets, amounts } = {}) {
	const rows = types.length;
	return {
		types: types.map(type => Array.isArray(type) ? type : [ type ]),
		dirs: dirs ?? types.map(() => [ 0 ]),
		targets: targets ?? types.map(() => [ 0 ]),
		amounts: amounts ?? types.map(() => [ 0 ]),
		bodyCounts: Array.from({ length: rows }, () => [ Array(SCHEMA.bodyPartTypes.length).fill(0) ]),
		bodyOrder: Array.from({ length: rows }, () => [
			Array.from({ length: SCHEMA.bodyPartTypes.length }, (_, type) => type),
		]),
	};
}

function setBody(actions, body, actor = 0) {
	const counts = Array(SCHEMA.bodyPartTypes.length).fill(0);
	const order = [];
	for (const type of body) {
		if (counts[type] === 0) order.push(type);
		counts[type] += 1;
	}
	for (let type = 0; type < counts.length; type++) if (counts[type] === 0) order.push(type);
	actions.bodyCounts[actor][0] = counts;
	actions.bodyOrder[actor][0] = order;
}

test('same-tick transfers reserve target capacity and resolve a shared target once', () => {
	const a = makeCreep('a', 10, 10, 50);
	const b = makeCreep('b', 10, 11, 50);
	const spawn = {
		id: 'spawn', structureType: C.STRUCTURE_SPAWN, pos: position(11, 10), store: store(250, 300),
	};
	const world = makeWorld({ creeps: [ a, b ], targets: [ spawn ] });
	const results = applyActions(
		world.Game,
		{ actorMeta: actorMeta([ a, b ]), targetMeta: [ targetMeta(spawn) ] },
		selected([ TYPE.transfer, TYPE.transfer ], { amounts: [ [ 5 ], [ 5 ] ] }),
	);

	assert.deepEqual(results.map(result => [ result.code, result.executed ]), [
		[ C.OK, true ], [ C.ERR_BUSY, false ],
	]);
	assert.deepEqual(a.calls[0], [ 'transfer', 'spawn', C.RESOURCE_ENERGY, 50 ]);
	assert.equal(b.calls.length, 0);
	assert.equal(world.objectLookups(), 1);
});

test('same-tick withdrawals reserve source energy', () => {
	const a = makeCreep('a', 10, 10);
	const b = makeCreep('b', 10, 11);
	const container = {
		id: 'container', structureType: C.STRUCTURE_CONTAINER,
		pos: position(11, 10), store: store(50, 2000),
	};
	const { Game } = makeWorld({ creeps: [ a, b ], targets: [ container ], time: 2 });
	const results = applyActions(
		Game,
		{ actorMeta: actorMeta([ a, b ]), targetMeta: [ targetMeta(container) ] },
		selected([ TYPE.withdraw, TYPE.withdraw ], { amounts: [ [ 5 ], [ 5 ] ] }),
	);

	assert.deepEqual(results.map(result => result.code), [ C.OK, C.ERR_BUSY ]);
	assert.equal(a.calls[0][3], 50);
	assert.equal(b.calls.length, 0);
});

test('shared-target resolution remains one object lookup for a 64-creep fleet', () => {
	const creeps = Array.from({ length: 64 }, (_, index) =>
		makeCreep(`fleet-${index}`, 10 + (index % 2), 10 + (index % 2), 1)
	);
	const storage = {
		id: 'fleet-storage', structureType: C.STRUCTURE_STORAGE,
		pos: position(11, 11), store: store(0, 64),
	};
	const world = makeWorld({ creeps, targets: [ storage ], time: 2 });
	const results = applyActions(
		world.Game,
		{ actorMeta: actorMeta(creeps), targetMeta: [ targetMeta(storage) ] },
		selected(
			Array(creeps.length).fill(TYPE.transfer),
			{ amounts: Array.from({ length: creeps.length }, () => [ 1 ]) },
		),
	);
	assert.equal(world.objectLookups(), 1);
	assert.equal(results.length, creeps.length);
	assert.ok(results.every(result => result.code === C.OK));
});

test('transfer and withdraw projections do not create same-tick energy or capacity', () => {
	const deliverer = makeCreep('deliverer', 10, 10, 50);
	const taker = makeCreep('taker', 10, 11);
	const empty = {
		id: 'empty', structureType: C.STRUCTURE_CONTAINER,
		pos: position(11, 10), store: store(0, 50),
	};
	const { Game } = makeWorld({ creeps: [ deliverer, taker ], targets: [ empty ], time: 3 });
	const results = applyActions(
		Game,
		{ actorMeta: actorMeta([ deliverer, taker ]), targetMeta: [ targetMeta(empty) ] },
		selected([ TYPE.transfer, TYPE.withdraw ], { amounts: [ [ 5 ], [ 5 ] ] }),
	);

	assert.deepEqual(results.map(result => result.code), [ C.OK, C.ERR_NOT_ENOUGH_RESOURCES ]);
	assert.equal(taker.calls.length, 0);
});

test('direct movement supports convoys and swaps but reserves duplicate destinations', () => {
	const lead = makeCreep('lead', 11, 10);
	const tail = makeCreep('tail', 10, 10);
	let world = makeWorld({ creeps: [ lead, tail ], time: 4 });
	let results = applyActions(
		world.Game,
		{ actorMeta: actorMeta([ lead, tail ]), targetMeta: [] },
		selected([ TYPE.move, TYPE.move ], { dirs: [ [ 2 ], [ 2 ] ] }),
	);
	assert.deepEqual(results.map(result => result.code), [ C.OK, C.OK ]);

	const left = makeCreep('left', 10, 10);
	const right = makeCreep('right', 11, 10);
	world = makeWorld({ creeps: [ left, right ], time: 5 });
	results = applyActions(
		world.Game,
		{ actorMeta: actorMeta([ left, right ]), targetMeta: [] },
		selected([ TYPE.move, TYPE.move ], { dirs: [ [ 2 ], [ 6 ] ] }),
	);
	assert.deepEqual(results.map(result => result.code), [ C.OK, C.OK ]);

	const a = makeCreep('a', 10, 10);
	const b = makeCreep('b', 12, 10);
	world = makeWorld({ creeps: [ a, b ], time: 6 });
	results = applyActions(
		world.Game,
		{ actorMeta: actorMeta([ a, b ]), targetMeta: [] },
		selected([ TYPE.move, TYPE.move ], { dirs: [ [ 2 ], [ 6 ] ] }),
	);
	assert.deepEqual(results.map(result => result.code), [ C.OK, C.ERR_BUSY ]);
	assert.equal(b.moves.length, 0);
});

test('stationary creeps block reserved movement and each actor gets at most one intent', () => {
	const actor = makeCreep('actor', 10, 10);
	const blocker = makeCreep('blocker', 11, 10);
	const { Game } = makeWorld({ creeps: [ actor, blocker ], time: 7 });
	const results = applyActions(
		Game,
		{ actorMeta: actorMeta([ actor ]), targetMeta: [] },
		selected([ [ TYPE.move, TYPE.move ] ], { dirs: [ [ 2, 6 ] ] }),
	);

	assert.equal(results.length, 1);
	assert.equal(results[0].code, C.ERR_BUSY);
	assert.equal(actor.moves.length, 0);
});

test('navigation success is not reported as primitive execution', () => {
	const previous = process.env.RL_NAV;
	process.env.RL_NAV = 'cheap';
	try {
		const actor = makeCreep('navigator', 10, 10);
		const source = { id: 'source', pos: position(20, 10), energy: 3000 };
		let world = makeWorld({ creeps: [ actor ], targets: [ source ], time: 8 });
		let results = applyActions(
			world.Game,
			{ actorMeta: actorMeta([ actor ]), targetMeta: [ targetMeta(source, 'source') ] },
			selected([ TYPE.harvest ]),
		);
		assert.equal(results[0].code, C.OK);
		assert.equal(results[0].executed, false);
		assert.equal(actor.calls.length, 0);
		assert.equal(actor.moves.length, 1);

		actor.pos = position(19, 10);
		world = makeWorld({ creeps: [ actor ], targets: [ source ], time: 9 });
		results = applyActions(
			world.Game,
			{ actorMeta: actorMeta([ actor ]), targetMeta: [ targetMeta(source, 'source') ] },
			selected([ TYPE.harvest ]),
		);
		assert.equal(results[0].executed, true);
		assert.deepEqual(actor.calls.at(-1), [ 'harvest', 'source' ]);
	} finally {
		if (previous == null) delete process.env.RL_NAV;
		else process.env.RL_NAV = previous;
	}
});

test('path cache retries an unconfirmed first step rather than skipping ahead', () => {
	const previousMode = process.env.RL_NAV;
	const previousPathFinder = globalThis.PathFinder;
	process.env.RL_NAV = 'pathfinder';
	let searches = 0;
	globalThis.PathFinder = {
		search: () => {
			searches++;
			return { path: [ position(11, 9), position(12, 9) ] };
		},
	};
	try {
		const actor = makeCreep('cache', 10, 10);
		const source = { id: 'far-source', pos: position(20, 10), energy: 3000 };
		for (const time of [ 20, 21 ]) {
			const { Game } = makeWorld({ creeps: [ actor ], targets: [ source ], time });
			const [ result ] = applyActions(
				Game,
				{ actorMeta: actorMeta([ actor ]), targetMeta: [ targetMeta(source, 'source') ] },
				selected([ TYPE.harvest ]),
			);
			assert.equal(result.code, C.OK);
		}
		assert.equal(searches, 1);
		assert.deepEqual(actor.moves, [ C.TOP_RIGHT, C.TOP_RIGHT ]);
	} finally {
		if (previousMode == null) delete process.env.RL_NAV;
		else process.env.RL_NAV = previousMode;
		globalThis.PathFinder = previousPathFinder;
	}
});

test('same-room macro congestion fallback never steps onto an exit', () => {
	const previousMode = process.env.RL_NAV;
	const previousPathFinder = globalThis.PathFinder;
	process.env.RL_NAV = 'pathfinder';
	globalThis.PathFinder = {
		search: () => ({ path: [ position(0, 1) ] }),
	};
	try {
		const actor = makeCreep('local-exit-guard', 1, 1);
		const source = { id: 'local-source', pos: position(10, 10), energy: 3000 };
		const { Game } = makeWorld({ creeps: [ actor ], targets: [ source ], time: 22 });
		const [ result ] = applyActions(
			Game,
			{ actorMeta: actorMeta([ actor ]), targetMeta: [ targetMeta(source, 'source') ] },
			selected([ TYPE.harvest ]),
		);
		assert.equal(result.code, C.OK);
		assert.equal(result.executed, false);
		assert.notEqual(actor.moves[0], C.LEFT);
		assert.notEqual(actor.moves[0], C.TOP_LEFT);
		assert.notEqual(actor.moves[0], C.TOP);
	} finally {
		if (previousMode == null) delete process.env.RL_NAV;
		else process.env.RL_NAV = previousMode;
		globalThis.PathFinder = previousPathFinder;
	}
});

test('cross-room approach falls back to bounded PathFinder when native routing has no path', () => {
	const previousPathFinder = globalThis.PathFinder;
	let searches = 0;
	globalThis.PathFinder = {
		search: () => {
			searches++;
			return { path: [ position(11, 10) ] };
		},
	};
	try {
		const actor = makeCreep('cross-room', 10, 10);
		actor.moveTo = () => C.ERR_NO_PATH;
		const source = { id: 'remote-source', pos: position(20, 20, 'W0N1'), energy: 3000 };
		const { Game } = makeWorld({ creeps: [ actor ], targets: [ source ], time: 22 });
		const [ result ] = applyActions(
			Game,
			{ actorMeta: actorMeta([ actor ]), targetMeta: [ targetMeta(source, 'source') ] },
			selected([ TYPE.harvest ]),
		);
		assert.equal(result.code, C.OK);
		assert.equal(result.executed, false);
		assert.equal(searches, 1);
		assert.deepEqual(actor.moves, [ C.RIGHT ]);
	} finally {
		globalThis.PathFinder = previousPathFinder;
	}
});

test('factor indices reject malformed actions instead of aliasing valid categories', () => {
	const actor = makeCreep('invalid', 10, 10);
	const { Game } = makeWorld({ creeps: [ actor ], time: 30 });
	const [ typeResult ] = applyActions(
		Game,
		{ actorMeta: actorMeta([ actor ]), targetMeta: [] },
		selected([ -1 ]),
	);
	const [ directionResult ] = applyActions(
		{ ...Game, time: 31 },
		{ actorMeta: actorMeta([ actor ]), targetMeta: [] },
		selected([ TYPE.move ], { dirs: [ [ -1 ] ] }),
	);
	assert.equal(typeResult.code, C.ERR_INVALID_ARGS);
	assert.equal(directionResult.code, C.ERR_INVALID_ARGS);
	assert.equal(actor.moves.length, 0);
});

test('spawn forwards an ordered free-form body to the engine without affordability gating', () => {
	const calls = [];
	const spawn = {
		id: 'spawn-free-form',
		structureType: C.STRUCTURE_SPAWN,
		spawning: null,
		store: store(0, 300),
		spawnCreep(body, name) {
			calls.push({ body, name });
			return C.ERR_NOT_ENOUGH_ENERGY;
		},
	};
	const { Game } = makeWorld({ targets: [ spawn ], time: 40 });
	Game.rooms.W0N0.energyAvailable = 0;
	Game.rooms.W0N0.energyCapacityAvailable = 300;
	const actions = selected([ TYPE.spawnCreep ]);
	setBody(actions, [ 1, 2, 0, 2 ]); // grouped as work, carry×2, move
	const [ result ] = applyActions(
		Game,
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);

	assert.equal(result.code, C.ERR_NOT_ENOUGH_ENERGY);
	assert.equal(result.executed, false);
	assert.deepEqual(calls[0].body, [ C.WORK, C.CARRY, C.CARRY, C.MOVE ]);
	assert.match(calls[0].name, /^rl_40_/);
	assert.doesNotMatch(calls[0].name, /hauler|rlr/);
	assert.equal(actionOutcome(result.code, result.type), SCHEMA.actionOutcomes.indexOf('not_enough_energy'));
	assert.equal(
		actionOutcome(C.OK, 'harvest', 'creep', false),
		SCHEMA.actionOutcomes.indexOf('approaching'),
	);
	assert.equal(actionOutcome(C.OK, 'move', 'creep', false), SCHEMA.actionOutcomes.indexOf('ok'));
});

test('spawn counts and canonical type order define one grouped body representation', () => {
	const calls = [];
	const spawn = {
		id: 'spawn-length',
		structureType: C.STRUCTURE_SPAWN,
		spawning: null,
		spawnCreep(body) {
			calls.push(body);
			return C.OK;
		},
	};
	const world = makeWorld({ targets: [ spawn ], time: 41 });
	let actions = selected([ TYPE.spawnCreep ]);
	setBody(actions, [ 7 ]);
	let [ result ] = applyActions(
		world.Game,
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);
	assert.equal(result.code, C.OK);
	assert.deepEqual(calls[0], [ C.TOUGH ]);
	assert.ok(Array.isArray(result.spawnBodyParts));
	assert.deepEqual(result.spawnBodyParts, [ 7 ]);
	assert.deepEqual(JSON.parse(JSON.stringify(result)).spawnBodyParts, [ 7 ]);

	actions = selected([ TYPE.spawnCreep ]);
	[ result ] = applyActions(
		{ ...world.Game, time: 42 },
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);
	assert.equal(result.code, C.ERR_INVALID_ARGS);
	assert.equal(calls.length, 1);

	actions = selected([ TYPE.spawnCreep ]);
	actions.bodyCounts[0][0][0] = 30;
	actions.bodyCounts[0][0][1] = 30;
	actions.bodyOrder[0][0] = [ 0, 1, 2, 3, 4, 5, 6, 7 ];
	[ result ] = applyActions(
		{ ...world.Game, time: 43 },
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);
	assert.equal(result.code, C.ERR_INVALID_ARGS);
	assert.equal(calls.length, 1);

	actions = selected([ TYPE.spawnCreep ]);
	setBody(actions, [ 1, 2, 2, 0 ]);
	actions.bodyOrder[0][0] = [ 1, 2, 0, 3, 4, 5, 7, 6 ];
	[ result ] = applyActions(
		{ ...world.Game, time: 44 },
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);
	assert.equal(result.code, C.ERR_INVALID_ARGS);
	assert.equal(calls.length, 1);
});

test('encoder exposes exact active body composition and next actor outcome', () => {
	const { Game, roomNames } = makeEncodeFixture({ roomCount: 1, creepsPerRoom: 1 });
	const [ creepName, creep ] = Object.entries(Game.creeps)[0];
	creep.body = [
		{ type: C.TOUGH, hits: 100 }, { type: C.MOVE, hits: 100 },
		{ type: C.WORK, hits: 100 }, { type: C.CARRY, hits: 100 },
		{ type: C.ATTACK, hits: 100 }, { type: C.RANGED_ATTACK, hits: 100 },
		{ type: C.HEAL, hits: 100 }, { type: C.CLAIM, hits: 100 },
		{ type: C.WORK, hits: 0 },
	];
	const outcomes = new Map([
		[ creepName, SCHEMA.actionOutcomes.indexOf('busy') ],
	]);
	const observation = encodeObservationFromRooms(Game, '100', roomNames, outcomes);
	const actorIndex = observation.actorMeta.findIndex(actor => actor.id === creepName);
	const feature = Object.fromEntries(SCHEMA.actorFeatures.map((name, index) => [ name, index ]));

	assert.ok(actorIndex >= 0);
	for (const name of [
		'activeMove', 'activeWork', 'activeCarry', 'activeAttack',
		'activeRangedAttack', 'activeHeal', 'activeClaim', 'activeTough',
	]) {
		assert.ok(Math.abs(
			observation._raw.actors[actorIndex * SCHEMA.actorFeat + feature[name]]
			- 1 / SCHEMA.maxBodyParts,
		) < 1e-8);
	}
	for (const name of [
		'totalMove', 'totalCarry', 'totalAttack', 'totalRangedAttack',
		'totalHeal', 'totalClaim', 'totalTough',
	]) {
		assert.ok(Math.abs(
			observation._raw.actors[actorIndex * SCHEMA.actorFeat + feature[name]]
			- 1 / SCHEMA.maxBodyParts,
		) < 1e-8);
	}
	assert.ok(Math.abs(
		observation._raw.actors[actorIndex * SCHEMA.actorFeat + feature.totalWork]
		- 2 / SCHEMA.maxBodyParts,
	) < 1e-8);
	assert.equal(observation._raw.actorOutcome[actorIndex], SCHEMA.actionOutcomes.indexOf('busy'));
	assert.ok(!Object.hasOwn(observation._raw, 'actor' + 'Role'));
	assert.ok(!Object.hasOwn(observation.shapes, 'actor' + 'Role'));
	assert.equal(observation.shapes.actorOutcome[0], SCHEMA.maxActors);
});

test('spawn intent mask matches immediate affordability and busy engine execution', () => {
	const { Game, roomNames } = makeEncodeFixture({ roomCount: 1, creepsPerRoom: 1 });
	const room = Game.rooms[roomNames[0]];
	room.energyAvailable = 0;
	room.energyCapacityAvailable = 0;
	const spawn = room.find(C.FIND_MY_STRUCTURES)
		.find(structure => structure.structureType === C.STRUCTURE_SPAWN);
	spawn.spawning = { remainingTime: 5 };
	let observation = encodeObservationFromRooms(Game, '100', roomNames);
	const actorIndex = observation.actorMeta.findIndex(actor => actor.id === `${room.name}:spawn`);
	const spawnIntent = SCHEMA.intentTypes.indexOf('spawnCreep');
	const intentOffset = actorIndex * SCHEMA.intentSlots * SCHEMA.intentTypes.length + spawnIntent;
	const amountBase = (
		(actorIndex * SCHEMA.intentSlots * SCHEMA.intentTypes.length + spawnIntent)
		* SCHEMA.amountBins.length
	);

	assert.ok(actorIndex >= 0);
	assert.equal(observation._raw.intentMask[intentOffset], 0);
	assert.ok(observation._raw.amountMask.subarray(
		amountBase, amountBase + SCHEMA.amountBins.length,
	).every(value => value === 0));

	room.energyAvailable = 10;
	observation = encodeObservationFromRooms(Game, '100', roomNames);
	assert.equal(observation._raw.intentMask[intentOffset], 0);
	spawn.spawnCreep = function spawnCreep() {
		return this.spawning ? C.ERR_BUSY : C.OK;
	};
	Game.getObjectById = id => id === spawn.id ? spawn : null;
	const actions = selected([ TYPE.spawnCreep ]);
	setBody(actions, [ 7 ]);
	let [ result ] = applyActions(
		Game,
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);
	assert.equal(result.code, C.ERR_BUSY);

	spawn.spawning = null;
	observation = encodeObservationFromRooms(Game, '100', roomNames);
	assert.equal(observation._raw.intentMask[intentOffset], 1);
	[ result ] = applyActions(
		{ ...Game, time: Game.time + 1 },
		{ actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] },
		actions,
	);
	assert.equal(result.code, C.OK);
	const roomEnergyAvailable = SCHEMA.actorFeatures.indexOf('roomEnergyAvailable');
	assert.ok(Math.abs(
		observation._raw.actors[actorIndex * SCHEMA.actorFeat + roomEnergyAvailable]
		- 10 / SCHEMA.maxRoomEnergy,
	) < 1e-8);
});

test('teacher emits the exact fixed bootstrap body as soon as its role is affordable', () => {
	const creep = (name, parts) => ({
		id: name,
		name,
		body: parts.map(type => ({ type, hits: 100 })),
		ticksToLive: 1200,
	});
	const population = [
		...Array.from({ length: 4 }, (_, index) => creep(`miner-${index}`, [ C.WORK, C.WORK, C.MOVE ])),
		creep('flexible', [ C.WORK, C.CARRY, C.MOVE, C.MOVE ]),
	];
	const spawn = { id: 'teacher-spawn', structureType: C.STRUCTURE_SPAWN, spawning: null };
	const room = {
		name: 'W0N0',
		energyAvailable: 250,
		energyCapacityAvailable: 300,
		controller: { my: true, level: 1 },
		find(type) {
			if (type === C.FIND_MY_CREEPS) return population;
			if (type === C.FIND_MY_STRUCTURES) return [ spawn ];
			return [];
		},
	};
	spawn.room = room;
	for (const member of population) member.room = room;
	const Game = {
		rooms: { W0N0: room },
		creeps: Object.fromEntries(population.map(member => [ member.name, member ])),
		gcl: { level: 1 },
		getObjectById: id => id === spawn.id ? spawn : null,
	};
	const meta = { actorMeta: [ { kind: 'structure', id: spawn.id } ], targetMeta: [] };

	room.energyAvailable = 299;
	let actions = scriptedActions(Game, meta);
	assert.equal(actions.types[0][0], TYPE.none);
	room.energyAvailable = 300;
	actions = scriptedActions(Game, meta);
	assert.equal(actions.types[0][0], TYPE.spawnCreep);
	assert.deepEqual(actions.bodyCounts[0][0], [ 2, 0, 4, 0, 0, 0, 0, 0 ]);
	assert.deepEqual(actions.bodyOrder[0][0], [ 2, 0, 1, 3, 4, 5, 6, 7 ]);
	assert.deepEqual(
		Object.keys(actions).sort(),
		[
			'amounts', 'bodyCounts', 'bodyOrder', 'constructionTiles',
			'constructionTypes', 'dirs', 'targets', 'types',
		],
	);
});
