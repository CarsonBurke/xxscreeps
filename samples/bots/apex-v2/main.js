/**
 * Apex v2 — multi-room empire bot for xxscreeps / Screeps.
 *
 * Loop phases:
 *  1. Memory hygiene
 *  2. Intel scan (visible rooms + flag orders)
 *  3. Colony managers (defense, logistics, construction, spawn, remotes FSM)
 *  4. Creep role runners (+ traffic heat)
 *  5. Stats / console summary
 *  6. CPU bucket awareness (skip remotes/scouts when low — handled in modules)
 *
 * Install:
 *   npx xxscreeps manage bot add apex-v2 samples/bots/apex-v2 --spawn W5N5
 *
 * Flag orders (place in world):
 *   attack_W1N1   — siege squad to room (rally first)
 *   claim_W2N2    — send claimer
 *   remote_W3N3   — force remote mining
 *   ignore_W4N4   — never remote/expand there
 *   rally_W5N5    — squad rally point
 */
const config = require('config');
const intel = require('intel');
const roomManager = require('room');
const roles = require('roles');
const stats = require('stats');
const metrics = require('metrics');
const { lowBucket } = require('util');

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
	// Prune stale path caches occasionally.
	if (Game.time % 1000 === 0 && Memory.apex) {
		if (Memory.apex.pathCache) {
			for (const k of Object.keys(Memory.apex.pathCache)) {
				if (Game.time - (Memory.apex.pathCache[k].t || 0) > 5000) {
					delete Memory.apex.pathCache[k];
				}
			}
		}
		if (Memory.apex.localPathCache) {
			for (const room of Object.keys(Memory.apex.localPathCache)) {
				const rc = Memory.apex.localPathCache[room];
				for (const k of Object.keys(rc)) {
					if (Game.time - (rc[k].t || 0) > 2000) delete rc[k];
				}
			}
		}
	}
	Memory._apexTick = Game.time;
	Memory._apexVisuals = config.visuals && !lowBucket();
}

function initEmpire() {
	Memory.empire ||= {
		attacks: [],
		claims: [],
		forcedRemotes: {},
		ignoreRooms: {},
		version: config.version || 2,
	};
	Memory.empire.version = config.version || 2;
	Memory.intel ||= {};
	Memory.apex ||= { boot: Game.time, version: 2 };
	stats.ensure();
}

function runCreeps() {
	for (const name in Game.creeps) {
		const creep = Game.creeps[name];
		if (creep.spawning) continue;
		roles.run(creep);
	}
}

module.exports.loop = function() {
	try {
		initEmpire();
		memhack();
		intel.tick();
		roomManager.runAll();
		runCreeps();
		stats.tick();
		// Segment 87 for TensorBoard watcher (samples/bots/PROTOCOL.md).
		metrics.tick();
	} catch (err) {
		console.log('Apex v2 FATAL', err && err.stack || err);
	}
};
