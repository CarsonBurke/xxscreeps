import * as C from 'xxscreeps/game/constants/index.js';

function position(roomName, x, y) {
	return { roomName, x, y };
}

function store(energy, capacity) {
	return {
		[C.RESOURCE_ENERGY]: energy,
		getCapacity: () => capacity,
		getFreeCapacity: () => Math.max(0, capacity - energy),
	};
}

function body(parts) {
	return parts.map(type => ({ type, hits: 100 }));
}

function makeRoom(roomName, roomIndex, creepsPerRoom) {
	const structures = [];
	const sites = [];
	const sources = [];
	const minerals = [];
	const dropped = [];
	const creeps = [];
	const addStructure = (structureType, id, x, y, options = {}) => {
		const structure = {
			id: `${roomName}:${id}`,
			structureType,
			pos: position(roomName, x, y),
			hits: options.hits ?? 1000,
			hitsMax: options.hitsMax ?? 1000,
			my: options.my ?? true,
			owner: options.my === false ? { username: 'enemy' } : { username: 'me' },
			store: options.capacity == null ? undefined : store(options.energy ?? 0, options.capacity),
			spawning: options.spawning ?? null,
		};
		structures.push(structure);
		return structure;
	};

	const controller = addStructure(C.STRUCTURE_CONTROLLER, 'controller', 25, 25, {
		hits: 1, hitsMax: 1,
	});
	Object.assign(controller, {
		level: 4,
		progress: 13579 + roomIndex,
		progressTotal: 40500,
		room: null,
	});
	addStructure(C.STRUCTURE_SPAWN, 'spawn', 20, 20, { energy: 240, capacity: 300 });
	addStructure(C.STRUCTURE_TOWER, 'tower', 22, 20, { energy: 680, capacity: 1000 });
	addStructure(C.STRUCTURE_CONTAINER, 'container', 10, 11, { energy: 1275, capacity: 2000 });
	for (let i = 0; i < 18; i++) {
		addStructure(C.STRUCTURE_EXTENSION, `extension:${i}`, 6 + (i % 9), 16 + Math.floor(i / 9), {
			energy: i % 3 ? 50 : 0, capacity: 50,
		});
	}
	for (let i = 0; i < 28; i++) {
		addStructure(C.STRUCTURE_ROAD, `road:${i}`, 5 + i, 30, { my: false, hits: 2500, hitsMax: 5000 });
	}
	for (let i = 0; i < 2; i++) {
		sources.push({
			id: `${roomName}:source:${i}`,
			pos: position(roomName, 10 + i * 28, 10 + i * 27),
			energy: 1800 + i * 500,
			energyCapacity: 3000,
			ticksToRegeneration: 91 + i,
		});
	}
	for (let i = 0; i < 3; i++) {
		sites.push({
			id: `${roomName}:site:${i}`,
			pos: position(roomName, 28 + i, 22),
			structureType: C.STRUCTURE_EXTENSION,
			progress: 30 + i * 17,
			progressTotal: 200,
			my: true,
		});
	}
	minerals.push({
		id: `${roomName}:mineral`, pos: position(roomName, 40, 12),
		mineralAmount: 14000,
	});
	dropped.push({
		id: `${roomName}:drop`, pos: position(roomName, 17, 18), amount: 73,
	});
	for (let i = 0; i < creepsPerRoom; i++) {
		const energy = (i * 13) % 101;
		creeps.push({
			id: `${roomName}:creep:${i}`,
			name: `${roomName}:creep:${i}`,
			pos: position(roomName, 4 + (i % 38), 4 + (Math.floor(i / 38) % 38)),
			my: true,
			room: null,
			body: body([ C.WORK, C.CARRY, C.MOVE, C.MOVE ]),
			store: store(energy, 100),
			fatigue: i % 7 === 0 ? 2 : 0,
			hits: 400,
			hitsMax: 400,
			ticksToLive: 1200 - i,
		});
	}

	const terrain = {
		get(x, y) {
			if (x === 0 || y === 0 || x === 49 || y === 49 || (x === 35 && y > 4 && y < 44)) {
				return C.TERRAIN_MASK_WALL;
			}
			if ((x + y + roomIndex) % 17 === 0) return C.TERRAIN_MASK_SWAMP;
			return 0;
		},
	};
	const room = {
		name: roomName,
		controller,
		energyAvailable: 800,
		energyCapacityAvailable: 1300,
		getTerrain: () => terrain,
		lookForAt(type, x, y) {
			if (type !== C.LOOK_STRUCTURES) return [];
			return structures.filter(item => item.pos.x === x && item.pos.y === y);
		},
		find(type) {
			if (type === C.FIND_STRUCTURES) return structures;
			if (type === C.FIND_MY_STRUCTURES) return structures.filter(item => item.my);
			if (type === C.FIND_SOURCES) return sources;
			if (type === C.FIND_MINERALS) return minerals;
			if (type === C.FIND_DROPPED_RESOURCES) return dropped;
			if (type === C.FIND_CONSTRUCTION_SITES || type === C.FIND_MY_CONSTRUCTION_SITES) return sites;
			if (type === C.FIND_CREEPS || type === C.FIND_MY_CREEPS) return creeps;
			return [];
		},
	};
	controller.room = room;
	for (const structure of structures) structure.room = room;
	for (const creep of creeps) creep.room = room;
	return { room, creeps, sites };
}

export function makeEncodeFixture({ roomCount = 4, creepsPerRoom = 24 } = {}) {
	const roomNames = [ 'W7N3', 'W7N4', 'W8N3', 'W8N4' ].slice(0, roomCount);
	const built = roomNames.map((name, index) => makeRoom(name, index, creepsPerRoom));
	const rooms = Object.fromEntries(built.map(({ room }) => [ room.name, room ]));
	const creeps = Object.fromEntries(built.flatMap(item => item.creeps).map(creep => [ creep.name, creep ]));
	const constructionSites = Object.fromEntries(
		built.flatMap(item => item.sites).map(site => [ site.id, site ]),
	);
	return {
		Game: {
			time: 12345,
			rooms,
			creeps,
			constructionSites,
			gcl: { level: 3 },
			cpu: { bucket: 9876 },
		},
		roomNames,
	};
}
