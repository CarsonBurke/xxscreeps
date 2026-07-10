/**
 * Apex v3 — delegated multi-room empire.
 *
 * Economy roles:
 *   harvester → static at source (drop / container)
 *   hauler    → move energy to base
 *   filler    → park at spawn/extensions, fill them
 *   upgrader  → park at controller energy
 *   builder   → build from containers/drops (no harvest)
 *
 * Modules war.js / economy.js (optional) add campaigns + projections.
 *
 * Install:
 *   npx xxscreeps manage bot add apex-v3 samples/bots/apex-v3 --spawn W5N5
 *
 * Metrics: RawMemory segment 87 → TensorBoard watcher (see ../PROTOCOL.md)
 */
const config = require('config');
const intel = require('intel');
const roomManager = require('room');
const roles = require('roles');
const metrics = require('metrics');

function memhack() {
	if (Memory._apexTick === Game.time) return;
	if (Memory.creeps) {
		for (const name in Memory.creeps) {
			if (!Game.creeps[name]) delete Memory.creeps[name];
		}
	}
	if (Memory.flags) {
		for (const name in Memory.flags) {
			if (!Game.flags[name]) delete Memory.flags[name];
		}
	}
	Memory._apexTick = Game.time;
	Memory._apexVisuals = config.visuals;
}

function initEmpire() {
	Memory.empire ||= {
		attacks: [],
		claims: [],
		forcedRemotes: {},
		ignoreRooms: {},
		campaigns: {},
		economy: {},
		version: 3,
	};
	Memory.empire.version = 3;
	Memory.intel ||= {};
	Memory.apex ||= { boot: Game.time, version: 3 };
}

function runCreeps() {
	for (const name in Game.creeps) {
		const creep = Game.creeps[name];
		if (creep.spawning) continue;
		roles.run(creep);
	}
}

function cpuBanner() {
	if (Game.time % (config.consoleSummaryInterval || 50) !== 0) return;
	const used = Game.cpu.getUsed();
	const creeps = Object.keys(Game.creeps).length;
	let colonies = 0;
	for (const n in Game.rooms) {
		if (Game.rooms[n].controller && Game.rooms[n].controller.my) colonies++;
	}
	const plan = Memory.empire && Memory.empire.plan;
	const freeGcl = plan ? plan.freeGcl : '?';
	const expand = plan && plan.expansions ? plan.expansions.map(e => e.room).join(',') : '';
	const lead = plan && plan.leadColony ? plan.leadColony : '?';
	console.log(
		`Apex v3 t=${Game.time} cpu=${used.toFixed(1)}/${Game.cpu.limit} bucket=${Game.cpu.bucket}` +
		` creeps=${creeps} colonies=${colonies} freeGCL=${freeGcl} lead=${lead}` +
		(expand ? ` expand→${expand}` : ''),
	);
}

module.exports.loop = function() {
	try {
		initEmpire();
		memhack();
		intel.tick();
		roomManager.runAll();
		runCreeps();
		metrics.tick();
		cpuBanner();
	} catch (err) {
		console.log('Apex v3 FATAL', err && err.stack || err);
	}
};
