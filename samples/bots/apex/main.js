/**
 * Apex — multi-room empire bot for xxscreeps / Screeps.
 *
 * Loop phases:
 *  1. Memory hygiene + memhack-friendly raw parse
 *  2. Intel scan (visible rooms + flag orders)
 *  3. Colony managers (defense, logistics, construction, spawn)
 *  4. Creep role runners
 *  5. Optional CPU diagnostics
 *
 * Install:
 *   npx xxscreeps manage bot add apex samples/bots/apex --spawn W5N5
 *
 * Flag orders (place in world):
 *   attack_W1N1   — siege squad to room
 *   claim_W2N2    — send claimer
 *   remote_W3N3   — force remote mining
 *   ignore_W4N4   — never remote/expand there
 *   rally_W5N5    — squad rally point
 */
const config = require('config');
const intel = require('intel');
const roomManager = require('room');
const roles = require('roles');
const metrics = require('metrics');

// Memhack: keep parsed Memory across ticks inside the same isolate when possible.
function memhack() {
	if (Memory._apexTick === Game.time) return;
	// Clear dead creep memory.
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
		version: 1,
	};
	Memory.intel ||= {};
	Memory.apex ||= { boot: Game.time };
}

function runCreeps() {
	for (const name in Game.creeps) {
		const creep = Game.creeps[name];
		if (creep.spawning) continue;
		roles.run(creep);
	}
}

function cpuBanner() {
	if (Game.time % 50 !== 0) return;
	const used = Game.cpu.getUsed();
	const creeps = Object.keys(Game.creeps).length;
	const rooms = Object.keys(Game.rooms).filter(r => Game.rooms[r].controller && Game.rooms[r].controller.my).length;
	console.log(
		`Apex t=${Game.time} cpu=${used.toFixed(1)}/${Game.cpu.limit} bucket=${Game.cpu.bucket}` +
		` creeps=${creeps} colonies=${rooms}`,
	);
}

module.exports.loop = function() {
	try {
		initEmpire();
		memhack();
		intel.tick();
		roomManager.runAll();
		runCreeps();
		// Metrics last so event logs + spawn costs from this tick are included.
		// Writes RawMemory.segments[87] for the TensorBoard watcher (see PROTOCOL.md).
		metrics.tick();
		cpuBanner();
	} catch (err) {
		console.log('Apex FATAL', err && err.stack || err);
	}
};
