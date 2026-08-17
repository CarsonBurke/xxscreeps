import assert from 'node:assert/strict';
import test from 'node:test';
import * as C from 'xxscreeps/game/constants/index.js';
import { SCHEMA } from './encode.mjs';
import { scriptedActions } from './scripted_baseline.mjs';

const INTENT = Object.fromEntries(SCHEMA.intentTypes.map((name, index) => [ name, index ]));

function encodedBody(actions, actor = 0) {
	const counts = actions.bodyCounts[actor][0];
	return actions.bodyOrder[actor][0].flatMap(type => Array(counts[type]).fill(type));
}

function pos(x, y, roomName = 'W0N0') {
	return {
		x, y, roomName,
		isEqualTo(other) {
			return x === other.x && y === other.y && roomName === other.roomName;
		},
	};
}

function energyStore(energy, capacity = 50) {
	return {
		[C.RESOURCE_ENERGY]: energy,
		getCapacity: () => capacity,
		getFreeCapacity: () => Math.max(0, capacity - energy),
	};
}

const ROLE_BODIES = {
	flexible: [ C.WORK, C.CARRY, C.MOVE ],
	miner: [ C.WORK, C.WORK, C.MOVE ],
	hauler: [ C.CARRY, C.CARRY, C.MOVE ],
	builder: [ C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
	upgrader: [ C.WORK, C.WORK, C.CARRY, C.MOVE ],
};

function makeCreep(role, serial, {
	energy = 0, ttl = 1000, roomName = 'W0N0', body = ROLE_BODIES[role],
} = {}) {
	const id = `${role}-${serial}`;
	return {
		id,
		name: id,
		body: body.map(type => ({ type, hits: 100 })),
		ticksToLive: ttl,
		store: energyStore(energy),
		pos: pos(10 + serial, 10, roomName),
	};
}

function makeRoom(name, {
	level = 1,
	owned = true,
	energyAvailable = 300,
	energyCapacity = 300,
	creeps = [],
	extensions = 0,
	spawn = true,
	sources = 1,
	sites = [],
} = {}) {
	const structures = [];
	if (spawn) {
		structures.push({
			id: `${name}:spawn`,
			structureType: C.STRUCTURE_SPAWN,
			spawning: null,
			store: energyStore(energyAvailable, 300),
			pos: pos(10, 10, name),
		});
	}
	for (let index = 0; index < extensions; index++) {
		structures.push({
			id: `${name}:extension:${index}`,
			structureType: C.STRUCTURE_EXTENSION,
			store: energyStore(0, 50),
			pos: pos(12 + index, 12, name),
		});
	}
	const roomSources = Array.from({ length: sources }, (_, index) => ({
		id: `${name}:source:${index}`,
		energy: 3000,
		pos: pos(20 + index * 5, 20, name),
	}));
	const room = {
		name,
		energyAvailable,
		energyCapacityAvailable: energyCapacity,
		controller: {
			id: `${name}:controller`, my: owned, level, pos: pos(25, 25, name),
		},
		getTerrain: () => ({ get: () => 0 }),
		find(type) {
			if (type === C.FIND_MY_CREEPS) return creeps;
			if (type === C.FIND_MY_STRUCTURES) return structures;
			if (type === C.FIND_MY_SPAWNS) {
				return structures.filter(item => item.structureType === C.STRUCTURE_SPAWN);
			}
			if (type === C.FIND_SOURCES) return roomSources;
			if (type === C.FIND_MY_CONSTRUCTION_SITES) return sites;
			return [];
		},
	};
	for (const object of [ ...creeps, ...structures, ...roomSources, ...sites ]) object.room = room;
	return { room, structures, sources: roomSources };
}

function makeGame(rooms, gcl = 1) {
	const objects = [];
	for (const room of rooms) {
		objects.push(
			...room.find(C.FIND_MY_STRUCTURES),
			...room.find(C.FIND_MY_CREEPS),
			...room.find(C.FIND_SOURCES),
			room.controller,
		);
	}
	return {
		rooms: Object.fromEntries(rooms.map(room => [ room.name, room ])),
		creeps: Object.fromEntries(rooms.flatMap(room => room.find(C.FIND_MY_CREEPS))
			.map(creep => [ creep.name, creep ])),
		spawns: Object.fromEntries(rooms.flatMap(room => room.find(C.FIND_MY_SPAWNS))
			.map(spawn => [ spawn.id, spawn ])),
		gcl: { level: gcl },
		getObjectById: id => objects.find(object => object.id === id) || null,
	};
}

function spawnDecision(room, Game, targetMeta = []) {
	const spawn = room.find(C.FIND_MY_SPAWNS)[0];
	return scriptedActions(Game, {
		actorMeta: [ { kind: 'structure', id: spawn.id } ],
		targetMeta,
	});
}

test('bootstrap demand establishes a stationary miner before growing general workers', () => {
	const flexible = makeCreep('flexible', 0);
	const haulers = Array.from({ length: 4 }, (_, index) => makeCreep('hauler', index + 1));
	const { room } = makeRoom('W0N0', {
		creeps: [ flexible, ...haulers ], energyAvailable: 249,
	});
	const Game = makeGame([ room ]);
	let actions = spawnDecision(room, Game);
	assert.equal(actions.types[0][0], INTENT.none);

	room.energyAvailable = 250;
	actions = spawnDecision(room, Game);
	assert.equal(actions.types[0][0], INTENT.spawnCreep);
	assert.deepEqual(encodedBody(actions), [ 1, 1, 0 ]);
});

test('pre-550 body cost stays fixed as extension capacity rises', () => {
	const creeps = [ makeCreep('flexible', 0), makeCreep('miner', 1) ];
	const { room } = makeRoom('W0N0', {
		level: 2, extensions: 4, energyCapacity: 500, energyAvailable: 300, creeps,
	});
	const actions = spawnDecision(room, makeGame([ room ]));
	assert.deepEqual(encodedBody(actions), [ 2, 2, 2, 2, 0, 0 ]);
});

test('economy-wide replacement lead replaces an expiring hauler before collapse', () => {
	const creeps = [
		makeCreep('flexible', 0),
		makeCreep('miner', 1, { ttl: 700 }),
		makeCreep('hauler', 2, { ttl: 300 }),
		makeCreep('builder', 3),
		makeCreep('upgrader', 4),
		makeCreep('upgrader', 5),
	];
	const { room } = makeRoom('W0N0', { energyAvailable: 300, creeps });
	const actions = spawnDecision(room, makeGame([ room ]));
	assert.equal(actions.types[0][0], INTENT.spawnCreep);
	assert.deepEqual(encodedBody(actions), [ 2, 2, 2, 2, 0, 0 ]);
});

test('the RCL2 demand phase builds all five extensions before the upgrader surge', () => {
	const mature = {
		flexible: [ C.WORK, C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
		miner: [ C.WORK, C.WORK, C.WORK, C.WORK, C.WORK, C.MOVE ],
		hauler: [ C.CARRY, C.CARRY, C.CARRY, C.CARRY, C.MOVE, C.MOVE ],
		builder: [
			C.WORK, C.WORK, C.CARRY, C.CARRY, C.CARRY, C.MOVE, C.MOVE, C.MOVE,
		],
		upgrader: [
			C.WORK, C.WORK, C.WORK, C.CARRY, C.CARRY, C.MOVE, C.MOVE, C.MOVE,
		],
	};
	const creeps = [
		makeCreep('flexible', 0, { body: mature.flexible }),
		makeCreep('miner', 1, { body: mature.miner }),
		makeCreep('miner', 2, { body: mature.miner }),
		makeCreep('miner', 3, { body: mature.miner }),
		makeCreep('miner', 4, { body: mature.miner }),
		makeCreep('hauler', 5, { body: mature.hauler }),
		makeCreep('builder', 6, { body: mature.builder }),
		makeCreep('upgrader', 7, { body: mature.upgrader }),
		makeCreep('upgrader', 8, { body: mature.upgrader }),
	];
	const { room: buildingRoom } = makeRoom('W0N0', {
		level: 2, extensions: 4, energyCapacity: 500, energyAvailable: 300, creeps,
	});
	let actions = spawnDecision(buildingRoom, makeGame([ buildingRoom ]));
	assert.deepEqual(encodedBody(actions), [ 1, 2, 2, 0, 0 ]);

	const { room: upgradingRoom } = makeRoom('W0N0', {
		level: 2, extensions: 5, energyCapacity: 550, energyAvailable: 550, creeps,
	});
	actions = spawnDecision(upgradingRoom, makeGame([ upgradingRoom ]));
	assert.deepEqual(encodedBody(actions), [ 1, 1, 1, 2, 2, 0, 0, 0 ]);
});

test('empty RCL3 does not automatically spawn a claimer for the neutral neighbor', () => {
	const flexible = makeCreep('flexible', 0);
	const { room } = makeRoom('W0N0', {
		level: 3, extensions: 7, energyCapacity: 650, energyAvailable: 650,
		creeps: [ flexible ],
	});
	const claimable = {
		kind: 'structure', id: 'W0N1:controller', room: 'W0N1',
		structureType: C.STRUCTURE_CONTROLLER, my: false, owned: false,
		reserved: false, myReservation: false,
	};
	const remoteSource = {
		kind: 'source', id: 'W0N1:source:0', room: 'W0N1', x: 20, y: 20,
	};
	const Game = makeGame([ room ], 2);
	const actions = spawnDecision(room, Game, [ claimable, remoteSource ]);
	assert.equal(actions.types[0][0], INTENT.spawnCreep);
	assert.ok(!encodedBody(actions).includes(SCHEMA.bodyPartTypes.indexOf('claim')));
});

test('surplus empty flexible worker targets a source in the neutral outpost', () => {
	const homeCreeps = Array.from({ length: 2 }, (_, index) => makeCreep('flexible', index));
	const { room: home } = makeRoom('W0N0', { level: 2, creeps: homeCreeps });
	const { room: remote, sources } = makeRoom('W0N1', {
		owned: false, level: 0, spawn: false,
	});
	const controller = {
		kind: 'structure', id: remote.controller.id, room: remote.name,
		structureType: C.STRUCTURE_CONTROLLER, my: false, owned: false,
		reserved: false, myReservation: false,
	};
	const source = {
		kind: 'source', id: sources[0].id, room: remote.name,
		x: sources[0].pos.x, y: sources[0].pos.y,
	};
	const traveler = homeCreeps[1];
	const actions = scriptedActions(makeGame([ home, remote ], 2), {
		actorMeta: [ { kind: 'creep', id: traveler.name } ],
		targetMeta: [ controller, source ],
	});
	assert.equal(actions.types[0][0], INTENT.harvest);
	assert.equal(actions.targets[0][0], 1);
});

test('loaded neutral-outpost worker targets an observable home sink', () => {
	const keeper = makeCreep('flexible', 0);
	const traveler = makeCreep('flexible', 1, { energy: 50, roomName: 'W0N1' });
	const { room: home, structures } = makeRoom('W0N0', {
		level: 2, energyAvailable: 0, creeps: [ keeper ],
	});
	const { room: remote, sources } = makeRoom('W0N1', {
		owned: false, level: 0, spawn: false, creeps: [ traveler ],
	});
	const targets = [
		{
			kind: 'structure', id: remote.controller.id, room: remote.name,
			structureType: C.STRUCTURE_CONTROLLER, my: false, owned: false,
			reserved: false, myReservation: false,
		},
		{
			kind: 'source', id: sources[0].id, room: remote.name,
			x: sources[0].pos.x, y: sources[0].pos.y,
		},
		{
			kind: 'structure', id: structures[0].id, room: home.name,
			structureType: C.STRUCTURE_SPAWN, my: true, owned: true,
			x: structures[0].pos.x, y: structures[0].pos.y,
		},
	];
	const actions = scriptedActions(makeGame([ home, remote ], 2), {
		actorMeta: [ { kind: 'creep', id: traveler.name } ],
		targetMeta: targets,
	});
	assert.equal(actions.types[0][0], INTENT.transfer);
	assert.equal(actions.targets[0][0], 2);
});

test('post-claim surplus flexible workers travel to the remote source', () => {
	const homeCreeps = Array.from({ length: 4 }, (_, index) => makeCreep('flexible', index));
	const { room: home } = makeRoom('W0N0', { level: 3, creeps: homeCreeps });
	const { room: remote, sources } = makeRoom('W0N1', { level: 1, spawn: false });
	const Game = makeGame([ home, remote ], 2);
	const traveler = homeCreeps[1];
	const actions = scriptedActions(Game, {
		actorMeta: [ { kind: 'creep', id: traveler.name } ],
		targetMeta: [ {
			kind: 'source', id: sources[0].id, room: remote.name,
			x: sources[0].pos.x, y: sources[0].pos.y,
		} ],
	});
	assert.equal(actions.types[0][0], INTENT.harvest);
	assert.equal(actions.targets[0][0], 0);
});

test('a flexible worker in a claimed spawnless room starts the remote spawn site', () => {
	const homeWorker = makeCreep('flexible', 0);
	const remoteWorker = makeCreep('flexible', 1, { roomName: 'W0N1' });
	const { room: home } = makeRoom('W0N0', { level: 3, creeps: [ homeWorker ] });
	const { room: remote } = makeRoom('W0N1', {
		level: 1, spawn: false, creeps: [ remoteWorker ],
	});
	const maskBytes = Math.ceil(SCHEMA.roomSize * SCHEMA.roomSize / 8);
	const constructionMask = new Uint8Array(
		2 * SCHEMA.constructionTypes.length * maskBytes,
	);
	const type = SCHEMA.constructionTypes.indexOf('spawn');
	const tile = 12 * SCHEMA.roomSize + 12;
	const base = (SCHEMA.constructionTypes.length + type) * maskBytes;
	constructionMask[base + (tile >> 3)] |= 1 << (tile & 7);
	const actions = scriptedActions(makeGame([ home, remote ], 2), {
		actorMeta: [ { kind: 'room', id: `room:${remote.name}`, room: remote.name } ],
		targetMeta: [],
		roomNames: [ home.name, remote.name ],
		constructionMask,
	});
	assert.equal(actions.types[0][0], INTENT.createConstructionSite);
	assert.equal(actions.constructionTypes[0][0], type);
	assert.equal(actions.constructionTiles[0][0], tile);
});

test('hauler-to-worker delivery targets are unique within a teacher tick', () => {
	const haulers = [
		makeCreep('hauler', 0, { energy: 50 }),
		makeCreep('hauler', 1, { energy: 50 }),
	];
	const workers = [ makeCreep('builder', 2), makeCreep('upgrader', 3) ];
	workers[0].pos = pos(10, 11);
	workers[1].pos = pos(11, 11);
	const { room } = makeRoom('W0N0', {
		spawn: false, sources: 0, creeps: [ ...haulers, ...workers ],
	});
	const Game = makeGame([ room ]);
	const targets = workers.map(creep => ({
		kind: 'creep', id: creep.id, room: room.name, x: creep.pos.x, y: creep.pos.y,
	}));
	const actions = scriptedActions(Game, {
		actorMeta: haulers.map(creep => ({ kind: 'creep', id: creep.name })),
		targetMeta: targets,
	});
	assert.deepEqual(actions.types.slice(0, 2).map(row => row[0]), [
		INTENT.transfer, INTENT.transfer,
	]);
	assert.notEqual(actions.targets[0][0], actions.targets[1][0]);
});
