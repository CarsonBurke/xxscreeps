/**
 * Benchmark Apex for N ticks inside the xxscreeps test shard simulator.
 *
 * Usage (from repo root, Node ≥ 22 with SuppressedError):
 *   mise exec node@24 -- node --import xxscreeps/loader samples/bots/apex/bench.mjs [ticks=20000] [botDir=.]
 *
 * Also dumps RawMemory segment 87 (bot metrics) to:
 *   samples/bots/apex/runs/<stamp>/metrics.jsonl
 * for the TensorBoard watcher:
 *   python3 samples/bots/metrics-watcher/watch.py --jsonl … --logdir …/tb
 *
 * Reports: control points, harvested, build, creep energy, RCL, …
 */
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from 'xxscreeps/config/index.js';
import * as Code from 'xxscreeps/engine/db/user/code.js';
import { userToIntentRoomsSetKey, userToVisibleRoomsSetKey } from 'xxscreeps/engine/processor/model.js';
import { PlayerInstance } from 'xxscreeps/engine/runner/instance.js';
import { getConsoleChannel } from 'xxscreeps/engine/runner/model.js';
import * as C from 'xxscreeps/game/constants/index.js';
import { RoomPosition } from 'xxscreeps/game/position.js';
import { loadMemorySegmentBlob } from 'xxscreeps/mods/memory/model.js';
import { create as createSpawn } from 'xxscreeps/mods/spawn/spawn.js';
import { simulate } from 'xxscreeps/test/index.js';
import { typedArrayToString } from 'xxscreeps/utility/string.js';

// Runner connectors (flags, memory, …) — same side-effect import as `xxscreeps test`.
import 'xxscreeps:mods/driver';

// Isolate from a live `xxscreeps start` process that holds ./screeps/.lock.
// Must run before any local storage connect (isSiblingProcess is memoized).
config.database.lock = null;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TICKS = Number(process.argv[2] || 20_000);
const BOT_DIR = path.resolve(__dirname, process.argv[3] || '.');
const USER = '100';
const ROOM = 'W1N1';
/** Must match samples/bots/apex/metrics.js SEGMENT_ID */
const METRICS_SEGMENT = 87;

const bodyCost = body => body.reduce((s, p) => s + (C.BODYPART_COST[p] || 0), 0);

async function loadApexModules() {
	const names = await fs.readdir(BOT_DIR);
	const modules = new Map();
	for (const name of names) {
		if (!name.endsWith('.js')) continue;
		const full = path.join(BOT_DIR, name);
		const text = await fs.readFile(full, 'utf8');
		modules.set(name, text);
	}
	if (!modules.has('main.js')) {
		throw new Error(`main.js missing in ${BOT_DIR}`);
	}
	return modules;
}

function decodeSegmentBlob(blob) {
	if (!blob || blob.byteLength < 2) return null;
	const u16 = new Uint16Array(blob.buffer, blob.byteOffset, blob.byteLength >>> 1);
	return typedArrayToString(u16);
}

const simulation = simulate({
	[ROOM]: room => {
		// Own the room and place spawn near the first source (fair early-game start).
		room['#user'] = USER;
		room.controller['#user'] = USER;
		room['#level'] = 1;
		const sources = room.find(C.FIND_SOURCES);
		const src = sources[0];
		let placed = false;
		if (src) {
			for (let r = 1; r <= 3 && !placed; r++) {
				for (let dx = -r; dx <= r && !placed; dx++) {
					for (let dy = -r; dy <= r && !placed; dy++) {
						if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
						const x = src.pos.x + dx;
						const y = src.pos.y + dy;
						if (x < 2 || x > 47 || y < 2 || y > 47) continue;
						if (room.getTerrain().get(x, y) === C.TERRAIN_MASK_WALL) continue;
						room['#insertObject'](createSpawn(new RoomPosition(x, y, ROOM), USER, 'Spawn1'));
						placed = true;
					}
				}
			}
		}
		if (!placed) {
			room['#insertObject'](createSpawn(new RoomPosition(25, 25, ROOM), USER, 'Spawn1'));
		}
	},
});

const metrics = {
	ticks: 0,
	energyHarvested: 0,
	energyBuild: 0,
	energyUpgrade: 0,
	controlPoints: 0,
	energyCreeps: 0,
	energyRepair: 0,
	spawns: 0,
	errors: [],
	console: [],
	rcl: 1,
	controllerProgress: 0,
	creepCount: 0,
	roleCounts: {},
	timeline: [],
};

await simulation(async ({ db, shard, world, tick, player, peekRoom }) => {
	const modules = await loadApexModules();
	// Prefer isolated-vm (default). 'unsafe' needs --experimental-vm-modules.
	if (!config.runner.sandbox) {
		config.runner.sandbox = 'isolated';
	}

	// User 100 is pre-created by instantiateTestShard as "Player 1".
	await Code.saveContent(db, USER, 'main', modules);

	const instance = await PlayerInstance.create(shard, world, USER);

	// Run directory for TensorBoard / JSONL (watcher reads metrics.jsonl).
	const stamp = new Date().toISOString().replace(/[:.]/g, '-');
	const runDir = path.join(__dirname, 'runs', stamp);
	await fs.mkdir(runDir, { recursive: true });
	const jsonlPath = path.join(runDir, 'metrics.jsonl');
	const jsonl = await fs.open(jsonlPath, 'a');
	// Convenience symlink/latest pointer
	const latestLink = path.join(__dirname, 'runs', 'latest');
	await fs.rm(latestLink, { recursive: true, force: true }).catch(() => {});
	await fs.symlink(runDir, latestLink).catch(async() => {
		// Windows / no symlink: copy a marker file
		await fs.writeFile(path.join(__dirname, 'runs', 'LATEST'), runDir + '\n');
	});

	// Capture console (stdout + stderr)
	const unlisten = await getConsoleChannel(shard, USER).listen(payload => {
		try {
			const frames = JSON.parse(payload);
			for (const frame of frames) {
				const { data, fd } = frame;
				if (fd === 2) metrics.errors.push(String(data));
				else metrics.console.push(String(data));
			}
		} catch {
			// ignore
		}
	});

	// Track known creeps for spawn cost accounting
	const knownCreeps = new Map(); // name -> bodyCost
	let lastSegmentSeq = -1;
	let lastBotSample = null;

	const t0 = Date.now();
	const reportEvery = Math.max(1, Math.floor(TICKS / 20));

	try {
		for (let i = 0; i < TICKS; i++) {
			const time = shard.time;
			const [ ir, vr ] = await Promise.all([
				shard.scratch.sMembers(userToIntentRoomsSetKey(USER)),
				shard.scratch.sMembers(userToVisibleRoomsSetKey(USER)),
			]);
			await instance.run(time, vr, ir);
			await tick(1);

			// Read event logs from all visible/active rooms we care about
			await peekRoom(ROOM, room => {
				const log = room.getEventLog();
				for (const ev of log) {
					const d = ev.data || {};
					if (ev.event === C.EVENT_HARVEST) {
						metrics.energyHarvested += d.amount || 0;
					} else if (ev.event === C.EVENT_BUILD) {
						metrics.energyBuild += d.energySpent || 0;
					} else if (ev.event === C.EVENT_UPGRADE_CONTROLLER) {
						metrics.energyUpgrade += d.energySpent || 0;
						metrics.controlPoints += d.amount || 0;
					} else if (ev.event === C.EVENT_REPAIR) {
						metrics.energyRepair += d.energySpent || 0;
					}
				}
			});

			// Creep spawn accounting + snapshot via player view
			await player(USER, Game => {
				const room = Game.rooms[ROOM];
				const seen = new Set();
				for (const name in Game.creeps) {
					seen.add(name);
					const creep = Game.creeps[name];
					if (!knownCreeps.has(name)) {
						const cost = bodyCost(creep.body.map(p => p.type));
						knownCreeps.set(name, cost);
						metrics.energyCreeps += cost;
						metrics.spawns++;
					}
				}
				// Drop dead
				for (const name of knownCreeps.keys()) {
					if (!seen.has(name) && !Game.creeps[name]) {
						knownCreeps.delete(name);
					}
				}

				if (room?.controller) {
					metrics.rcl = room.controller.level;
					metrics.controllerProgress = room.controller.progress || 0;
				}
				metrics.creepCount = Object.keys(Game.creeps).length;
				const roles = {};
				for (const name in Game.creeps) {
					const role = Game.creeps[name].memory?.role || '?';
					roles[role] = (roles[role] || 0) + 1;
				}
				metrics.roleCounts = roles;
			});

			// Pull bot metrics segment → JSONL only when seq advances (bot writes every 5 ticks).
			try {
				const blob = await loadMemorySegmentBlob(shard, USER, METRICS_SEGMENT);
				const text = decodeSegmentBlob(blob);
				if (text) {
					const payload = JSON.parse(text);
					if (payload.seq != null && payload.seq !== lastSegmentSeq) {
						lastSegmentSeq = payload.seq;
						const botSample = payload.sample
							|| (Array.isArray(payload.ring) && payload.ring.length
								? payload.ring[payload.ring.length - 1]
								: null);
						if (botSample) {
							lastBotSample = botSample;
							await jsonl.write(JSON.stringify({
								seq: payload.seq,
								written: payload.written,
								source: 'segment',
								sample: botSample,
							}) + '\n');
						}
					}
				}
			} catch {
				// Segment may not exist until setActiveSegments settles.
			}

			metrics.ticks = i + 1;

			if ((i + 1) % reportEvery === 0 || i === 0) {
				const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
				const line = {
					tick: i + 1,
					rcl: metrics.rcl,
					progress: metrics.controllerProgress,
					harvested: metrics.energyHarvested,
					build: metrics.energyBuild,
					upgrade: metrics.energyUpgrade,
					controlPoints: metrics.controlPoints,
					creepEnergy: metrics.energyCreeps,
					creeps: metrics.creepCount,
					cpu: lastBotSample && lastBotSample.cpu,
					bucket: lastBotSample && lastBotSample.bucket,
					errors: metrics.errors.length,
					elapsedSec: Number(elapsed),
				};
				metrics.timeline.push(line);
				console.log(JSON.stringify(line));
			}
		}
	} finally {
		unlisten();
		instance.disconnect();
		await jsonl.close();
	}

	const elapsed = (Date.now() - t0) / 1000;
	const summary = {
		ticks: metrics.ticks,
		elapsedSec: elapsed,
		ticksPerSec: metrics.ticks / elapsed,
		rcl: metrics.rcl,
		controllerProgress: metrics.controllerProgress,
		controlPoints: metrics.controlPoints,
		energyHarvested: metrics.energyHarvested,
		energySpentBuilding: metrics.energyBuild,
		energySpentUpgrading: metrics.energyUpgrade,
		energySpentRepair: metrics.energyRepair,
		energySpentCreeps: metrics.energyCreeps,
		spawnEvents: metrics.spawns,
		finalCreeps: metrics.creepCount,
		finalRoles: metrics.roleCounts,
		errorCount: metrics.errors.length,
		lastErrors: metrics.errors.slice(-10),
		lastConsole: metrics.console.slice(-15),
		runDir,
		jsonlPath,
		botDir: BOT_DIR,
	};

	const outPath = path.join(runDir, 'bench-results.json');
	await fs.writeFile(outPath, JSON.stringify({ summary, timeline: metrics.timeline }, null, 2));
	await fs.writeFile(path.join(__dirname, 'bench-results.json'), JSON.stringify({ summary, timeline: metrics.timeline }, null, 2));
	console.log('\n=== APEX BENCH SUMMARY ===');
	console.log(JSON.stringify(summary, null, 2));
	console.log(`\nWrote ${outPath}`);
	console.log(`Metrics JSONL: ${jsonlPath}`);
	console.log(`TensorBoard:\n  python3 samples/bots/metrics-watcher/watch.py --jsonl ${jsonlPath} --logdir ${path.join(runDir, 'tb')}\n  tensorboard --logdir ${path.join(runDir, 'tb')}`);
});

// Force exit — open handles from db
process.exit(0);
