/**
 * Apex bench implementation — loaded by bench.mjs after CLI flag stripping.
 * See bench.mjs for usage / flags.
 */
import * as fs from 'node:fs/promises';
import * as fsSync from 'node:fs';
import * as path from 'node:path';
import { createRequire } from 'node:module';
import { spawn as spawnChild } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { BackendContext } from 'xxscreeps/backend/context.js';
import { listenBackend } from 'xxscreeps/backend/listen.js';
import { config } from 'xxscreeps/config/index.js';
import * as Code from 'xxscreeps/engine/db/user/code.js';
import * as Badge from 'xxscreeps/engine/db/user/badge.js';
import * as User from 'xxscreeps/engine/db/user/index.js';
import { userToIntentRoomsSetKey, userToVisibleRoomsSetKey } from 'xxscreeps/engine/processor/model.js';
import { PlayerInstance } from 'xxscreeps/engine/runner/instance.js';
import { getConsoleChannel } from 'xxscreeps/engine/runner/model.js';
import * as C from 'xxscreeps/game/constants/index.js';
import { RoomPosition } from 'xxscreeps/game/position.js';
import { loadMemorySegmentBlob, loadUserMemoryString } from 'xxscreeps/mods/memory/model.js';
import { setPassword } from 'xxscreeps/mods/backend/password/model.js';
import { create as createSpawn } from 'xxscreeps/mods/spawn/spawn.js';
import { simulate } from 'xxscreeps/test/index.js';
import { typedArrayToString } from 'xxscreeps/utility/string.js';

// Runner connectors (flags, memory, …) — same side-effect import as `xxscreeps test`.
import 'xxscreeps:mods/driver';

// Isolate from a live `xxscreeps start` process that holds ./screeps/.lock.
// Must run before any local storage connect (isSiblingProcess is memoized).
config.database.lock = null;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Positionals only (flags stripped by bench.mjs launcher)
const positionals = process.argv.slice(2).filter(a => !a.startsWith('--'));
const TICKS = Number(positionals[0] || 20_000);
const BOT_DIR = path.resolve(__dirname, positionals[1] || '.');
/** Set by bench.mjs launcher; default ON */
const TENSORBOARD = process.env.APEX_BENCH_TB !== '0' && process.env.APEX_BENCH_TB !== 'false';
/** Open TensorBoard UI in browser (--tensorboard / APEX_BENCH_TENSORBOARD) */
const OPEN_TENSORBOARD = process.env.APEX_BENCH_TENSORBOARD === '1' ||
	process.env.APEX_BENCH_TENSORBOARD === 'true';
/** Serve BM world on Screeps client HTTP (--headful / --client) */
const HEADFUL = process.env.APEX_BENCH_HEADFUL === '1' ||
	process.env.APEX_BENCH_HEADFUL === 'true';
const TB_PORT = Number(process.env.APEX_BENCH_TB_PORT || 6006);
const HEADFUL_PASSWORD = process.env.APEX_BENCH_PASSWORD || 'apexbench';
// Headful defaults to 100ms so the client room socket can keep up (loadRoom only
// retains time / time-1). Explicit 0 = max speed (view may lag/stall).
const HEADFUL_TICK_MS = process.env.APEX_BENCH_TICK_MS != null && process.env.APEX_BENCH_TICK_MS !== ''
	? Math.max(0, Number(process.env.APEX_BENCH_TICK_MS))
	: (HEADFUL ? 100 : 0);
const AUTO_OPEN = process.env.APEX_BENCH_NO_OPEN !== '1' &&
	process.env.APEX_BENCH_NO_OPEN !== 'true';
const USER = '100';
const ROOM = 'W7N3';
const WATCH_PY = path.join(__dirname, '../metrics-watcher/watch.py');

/**
 * Open a URL in the default browser (xdg-open / open / start).
 */
function openInBrowser(url) {
	const plat = process.platform;
	let cmd;
	let args;
	if (plat === 'darwin') {
		cmd = 'open';
		args = [ url ];
	} else if (plat === 'win32') {
		cmd = 'cmd';
		args = [ '/c', 'start', '', url ];
	} else {
		cmd = 'xdg-open';
		args = [ url ];
	}
	try {
		const child = spawnChild(cmd, args, { stdio: 'ignore', detached: true });
		child.unref();
		return true;
	} catch (err) {
		console.error(`[browser] could not open: ${err.message}`);
		return false;
	}
}

/**
 * Start the Screeps backend HTTP/SockJS server against the *in-process* BM shard.
 * Same stack as `xxscreeps backend`, but shares simulate()'s db/world (no second process).
 */
async function startHeadfulClient(db, shard, world) {
	// Load backend mods (client package.nw, password, steam, cookie, classic renderers).
	await import('xxscreeps:mods/backend');

	const backendContext = await BackendContext.attach(db, shard, world);

	// Player 1 can log in; badge so map shows ownership; start room = BM room.
	await setPassword(db, USER, HEADFUL_PASSWORD);
	const existingBadge = await db.data.hGet(User.infoKey(USER), 'badge');
	if (existingBadge == null) {
		await Badge.save(db, USER, JSON.stringify(Badge.generateRandom()));
	}
	await db.data.hSet(User.infoKey(USER), 'lastViewedRoom', ROOM);

	// Localhost only — BM is not a multi-user world; keep password off the LAN.
	const prevBind = config.backend.bind;
	config.backend.bind = '127.0.0.1:21025';
	let handle;
	try {
		handle = await listenBackend(backendContext);
	} finally {
		config.backend.bind = prevBind;
	}

	console.error(`[headful] Screeps client → ${handle.url}`);
	console.error(`[headful] BM room ${ROOM}; login "Player 1" / "${HEADFUL_PASSWORD}" (or guest)`);
	console.error('[headful] view-only: client intents/code can mutate the BM — do not interact');
	if (HEADFUL_TICK_MS > 0) {
		console.error(`[headful] tick delay ${HEADFUL_TICK_MS}ms (set APEX_BENCH_TICK_MS=0 for max speed)`);
	} else {
		console.error('[headful] full-speed ticks — room stream may lag; prefer APEX_BENCH_TICK_MS=100+');
	}
	if (AUTO_OPEN) {
		setTimeout(() => {
			if (openInBrowser(handle.url)) console.error(`[headful] opened ${handle.url}`);
			else console.error(`[headful] open manually: ${handle.url}`);
		}, 400);
	}

	return {
		url: handle.url,
		port: handle.port,
		async stop() {
			try { await handle.stop(); } catch { /* */ }
			try { await backendContext.disposeAsync(); } catch { /* */ }
		},
	};
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Ensure TensorBoard is serving multi-run logdir, then open the UI.
 * Returns a stop() that kills TB if we started it.
 */
function openTensorBoardUi(tbIndexRoot) {
	const port = TB_PORT;
	const url = `http://127.0.0.1:${port}/`;
	let tbProc = null;

	function startTb() {
		try {
			tbProc = spawnChild(
				'tensorboard',
				[ '--logdir', tbIndexRoot, '--port', String(port), '--bind_all' ],
				{ stdio: [ 'ignore', 'pipe', 'pipe' ] },
			);
			tbProc.stderr?.on('data', buf => {
				const s = String(buf);
				if (/error|Error|EADDRINUSE/i.test(s)) console.error(`[tb-server] ${s.trim()}`);
			});
			tbProc.on('error', err => {
				console.error(`[tb-server] failed to start tensorboard: ${err.message}`);
				console.error('[tb-server] install: pip install tensorboard   then re-run with --tensorboard');
				tbProc = null;
			});
			console.error(`[tensorboard] serving ${url}  (logdir ${tbIndexRoot})`);
			setTimeout(() => {
				if (openInBrowser(url)) console.error(`[tensorboard] opened ${url}`);
				else console.error(`[tensorboard] open manually: ${url}`);
			}, 1500);
		} catch (err) {
			console.error(`[tensorboard] ${err.message}`);
		}
	}

	import('node:net').then(({ default: net }) => {
		const sock = net.connect({ port, host: '127.0.0.1' }, () => {
			sock.end();
			console.error(`[tensorboard] already on :${port}`);
			if (openInBrowser(url)) console.error(`[tensorboard] opened ${url}`);
		});
		sock.on('error', () => {
			startTb();
		});
	}).catch(() => startTb());

	return {
		url,
		async stop() {
			if (tbProc && !tbProc.killed) {
				try { tbProc.kill('SIGTERM'); } catch { /* */ }
				tbProc = null;
			}
		},
	};
}

/**
 * Load MetricKey / SegmentKey enums from apex-v3 (or BOT_DIR) — never hardcode wire shorts.
 * Wire values live only in the enum definitions.
 */
function loadMetricEnums() {
	const candidates = [
		path.join(BOT_DIR, 'memoryKeys.js'),
		path.join(__dirname, '../apex-v3/dist/memoryKeys.js'),
		path.join(__dirname, 'memoryKeys.js'),
	];
	for (const p of candidates) {
		try {
			if (!fsSync.existsSync(p)) continue;
			const mod = require(p);
			const MetricKey = mod.MetricKey || mod.default?.MetricKey;
			const SegmentKey = mod.SegmentKey || mod.default?.SegmentKey;
			const expand = mod.expandMetricSample || mod.default?.expandMetricSample;
			const segId = mod.METRICS_SEGMENT ?? mod.default?.METRICS_SEGMENT ?? 87;
			if (MetricKey && SegmentKey) {
				return { MetricKey, SegmentKey, expandMetricSample: expand, METRICS_SEGMENT: segId };
			}
		} catch {
			/* try next */
		}
	}
	throw new Error(
		'Cannot load MetricKey/SegmentKey enums from memoryKeys.js — build apex-v3 first',
	);
}

const {
	MetricKey,
	SegmentKey,
	expandMetricSample,
	METRICS_SEGMENT,
} = loadMetricEnums();

/** Expand dense segment sample using MetricKey enum only (no raw 'hr'/'s' literals). */
function expandDenseSample(botSample) {
	if (typeof expandMetricSample === 'function') {
		return expandMetricSample(botSample);
	}
	const out = {};
	for (const longName of Object.keys(MetricKey)) {
		const wire = MetricKey[longName];
		if (typeof wire !== 'string') continue;
		const v = botSample[wire] ?? botSample[longName];
		if (v != null && typeof v === 'number') out[longName] = v;
	}
	return out;
}

/**
 * Live TensorBoard for one stamped run.
 *
 * Layout (no overwrites — every BM is a separate TB run name):
 *   samples/bots/apex/runs/<stamp>/metrics.jsonl
 *   samples/bots/apex/runs/<stamp>/tb/          ← event files
 *   samples/bots/apex/runs/tb/<stamp> → ../<stamp>/tb   ← multi-run index
 *
 * Open ALL runs (compare without losing previous):
 *   tensorboard --logdir samples/bots/apex/runs/tb
 *
 * `latest` only retargets a convenience pointer; it never deletes prior event files.
 */
function createTensorBoard(runDir, stamp) {
	const tbDir = path.join(runDir, 'tb');
	fsSync.mkdirSync(tbDir, { recursive: true });

	// Multi-run index: parent logdir so TB shows every stamp side-by-side
	const tbIndexRoot = path.join(__dirname, 'runs', 'tb');
	const tbIndexLink = path.join(tbIndexRoot, stamp);
	try {
		fsSync.mkdirSync(tbIndexRoot, { recursive: true });
		// Relative symlink so the tree is portable
		const rel = path.relative(tbIndexRoot, tbDir);
		try { fsSync.rmSync(tbIndexLink, { recursive: true, force: true }); } catch { /* */ }
		fsSync.symlinkSync(rel, tbIndexLink);
	} catch (err) {
		// Fallback: if symlink fails, still write events under tbDir only
		console.error(`[tb] multi-run index link failed: ${err.message}`);
	}

	// Backfill index for any older runs that lack a link (idempotent)
	try {
		const runsRoot = path.join(__dirname, 'runs');
		for (const name of fsSync.readdirSync(runsRoot)) {
			if (name === 'tb' || name === 'latest' || name === 'LATEST') continue;
			const oldTb = path.join(runsRoot, name, 'tb');
			const link = path.join(tbIndexRoot, name);
			if (!fsSync.existsSync(oldTb)) continue;
			if (fsSync.existsSync(link)) continue;
			try {
				fsSync.symlinkSync(path.relative(tbIndexRoot, oldTb), link);
			} catch { /* ignore */ }
		}
	} catch { /* ignore */ }

	let child = null;
	let childFailed = false;

	function startFollow(jsonlPath) {
		try {
			child = spawnChild(
				'python3',
				[ WATCH_PY, '--jsonl', jsonlPath, '--logdir', tbDir, '--follow', '--poll', '0.2' ],
				{ stdio: [ 'ignore', 'pipe', 'pipe' ] },
			);
			child.stderr?.on('data', buf => {
				const s = String(buf).trim();
				if (s) console.error(`[tb] ${s}`);
			});
			child.on('error', err => {
				childFailed = true;
				console.error(`[tb] watch.py failed to start: ${err.message}`);
			});
			child.on('exit', (code, signal) => {
				if (code && code !== 0 && signal !== 'SIGTERM' && signal !== 'SIGINT') {
					childFailed = true;
					console.error(`[tb] watch.py exited code=${code} signal=${signal}`);
				}
				child = null;
			});
			console.error(`[tb] live events → ${tbDir}`);
			console.error(`[tb] all runs → tensorboard --logdir ${tbIndexRoot}`);
		} catch (err) {
			childFailed = true;
			console.error(`[tb] could not spawn watch.py: ${err.message}`);
		}
	}

	function hasEventFiles() {
		try {
			return fsSync.readdirSync(tbDir).some(f => f.startsWith('events.'));
		} catch {
			return false;
		}
	}

	async function stop() {
		if (child && !child.killed) {
			child.kill('SIGTERM');
			await new Promise(resolve => {
				const t = setTimeout(resolve, 2000);
				child?.once('exit', () => {
					clearTimeout(t);
					resolve();
				});
			});
			if (child && !child.killed) {
				try { child.kill('SIGKILL'); } catch { /* */ }
			}
			child = null;
		}
		// Offline one-shot if follow never wrote event files
		if (childFailed || !hasEventFiles()) {
			try {
				const r = spawnChild(
					'python3',
					[ WATCH_PY, '--jsonl', path.join(runDir, 'metrics.jsonl'), '--logdir', tbDir ],
					{ stdio: [ 'ignore', 'pipe', 'pipe' ] },
				);
				await new Promise(resolve => r.on('exit', resolve));
			} catch {
				/* ignore */
			}
		}
	}

	return { tbDir, tbIndexRoot, startFollow, stop };
}

const bodyCost = body => body.reduce((s, p) => s + (C.BODYPART_COST[p] || 0), 0);

/**
 * Load bot modules for Code.saveContent.
 * Matches `xxscreeps manage bot add`: plain `.js` as strings; `.wasm` as binary
 * keyed without extension (require('commiebot_wasm_bg') → file commiebot_wasm_bg.wasm).
 * Skips source maps / junk so a rollup dist/ works as-is.
 */
async function loadApexModules() {
	const names = await fs.readdir(BOT_DIR);
	const modules = new Map();
	for (const name of names) {
		if (name.endsWith('.map') || name.endsWith('.map.js')) continue;
		if (name.startsWith('.')) continue;
		const full = path.join(BOT_DIR, name);
		const st = await fs.stat(full).catch(() => null);
		if (!st?.isFile()) continue;

		const isWasm = name.endsWith('.wasm');
		const isJs = name.endsWith('.js') || name.endsWith('.mjs') || name === 'main';
		if (!isWasm && !isJs) continue;

		const content = isWasm
			? new Uint8Array(await fs.readFile(full))
			: await fs.readFile(full, 'utf8');
		// Screeps require('foo') for wasm → module key `foo` (no .wasm), except main.wasm
		const key = isWasm && name !== 'main.wasm'
			? path.basename(name, '.wasm')
			: name;
		modules.set(key, content);
	}
	if (![ 'main.js', 'main', 'main.mjs', 'main.wasm' ].some(n => modules.has(n))) {
		throw new Error(`main.js (or main/main.mjs/main.wasm) missing in ${BOT_DIR}`);
	}
	return modules;
}

function decodeSegmentBlob(blob) {
	if (!blob || blob.byteLength < 2) return null;
	const u16 = new Uint16Array(blob.buffer, blob.byteOffset, blob.byteLength >>> 1);
	return typedArrayToString(u16);
}

/**
 * The International (and Grafana/pandascreeps) stats live in Memory.stats with
 * short room keys (RoomStatsKeys: eih, eou, eob, cl, …). Flatten to long names
 * that watch.py already emits, plus ti/* room tags.
 */
function sampleFromTiMemoryStats(stats, gameTime) {
	if (!stats || typeof stats !== 'object') return null;
	const rooms = stats.rooms && typeof stats.rooms === 'object' ? stats.rooms : {};
	let eih = 0;
	let eou = 0;
	let eob = 0;
	let reih = 0;
	let eoro = 0;
	let eorwr = 0;
	let eosp = 0;
	let es = 0;
	let su = 0;
	let clMax = 0;
	let roomCount = 0;
	const perRoom = {};
	for (const [roomName, rs] of Object.entries(rooms)) {
		if (!rs || typeof rs !== 'object') continue;
		roomCount++;
		const r = /** @type {Record<string, number>} */ (rs);
		eih += Number(r.eih) || 0;
		eou += Number(r.eou) || 0;
		eob += Number(r.eob) || 0;
		reih += Number(r.reih) || 0;
		eoro += Number(r.eoro) || 0;
		eorwr += Number(r.eorwr) || 0;
		eosp += Number(r.eosp) || 0;
		es += Number(r.es) || 0;
		su += Number(r.su) || 0;
		const cl = Number(r.cl) || 0;
		if (cl > clMax) clMax = cl;
		perRoom[roomName] = {
			eih: Number(r.eih) || 0,
			eou: Number(r.eou) || 0,
			eob: Number(r.eob) || 0,
			reih: Number(r.reih) || 0,
			es: Number(r.es) || 0,
			su: Number(r.su) || 0,
			cl,
			cc: Number(r.cc) || 0,
			cpu: Number(r.cpu) || 0,
		};
	}
	const gcl = stats.gcl || {};
	const gclLevel = Number(gcl.level) || 0;
	const gclProg = Number(gcl.progress) || 0;
	const gclTotal = Number(gcl.progressTotal) || 0;
	const gclCont = gclTotal > 0 ? gclLevel + gclProg / gclTotal : gclLevel;
	const cpu = stats.cpu || {};
	return {
		t: Number(stats.lastTick) || gameTime,
		// watch.py long names
		harvestRate: eih,
		claimedHarvestRate: eih,
		remoteHarvestRate: reih,
		upgradeRate: eou,
		buildRate: eob,
		cpu: Number(cpu.usage) || 0,
		bucket: Number(cpu.bucket) || 0,
		creeps: Number(stats.creeps) || 0,
		rclMax: clMax,
		colonies: roomCount,
		storedEnergy: es,
		gcl: gclCont,
		// TI aggregate shorts (also emitted under ti/ by watch.py)
		eih,
		eou,
		eob,
		reih,
		eoro,
		eorwr,
		eosp,
		es,
		su: roomCount > 0 ? su / roomCount : 0,
		cl: clMax,
		// empire
		tickLength: Number(stats.tickLength) || 0,
		heapUsage: Number(stats.heapUsage) || 0,
		memoryUsage: Number(stats.memory?.usage) || 0,
		constructionSites: Number(stats.constructionSites) || 0,
		rooms: perRoom,
		source: 'ti-memory-stats',
	};
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

	// --headful: serve this BM shard on the Screeps client (same process).
	let headful = null;
	if (HEADFUL) {
		try {
			headful = await startHeadfulClient(db, shard, world);
		} catch (err) {
			console.error(`[headful] failed to start client server: ${err.message}`);
			if (err.code === 'EADDRINUSE') {
				console.error('[headful] port busy — stop `xxscreeps start` or free :21025');
			}
			throw err;
		}
	}

	// Stamped run dir — never reuse / overwrite prior runs.
	const stamp = new Date().toISOString().replace(/[:.]/g, '-');
	const runDir = path.join(__dirname, 'runs', stamp);
	await fs.mkdir(runDir, { recursive: true });
	const jsonlPath = path.join(runDir, 'metrics.jsonl');
	// Create empty jsonl so watch.py --follow can open it immediately
	await fs.writeFile(jsonlPath, '');
	const jsonl = await fs.open(jsonlPath, 'a');
	// Convenience pointer only (does not delete old run data)
	const latestLink = path.join(__dirname, 'runs', 'latest');
	await fs.rm(latestLink, { recursive: true, force: true }).catch(() => {});
	await fs.symlink(runDir, latestLink).catch(async() => {
		// Windows / no symlink: copy a marker file
		await fs.writeFile(path.join(__dirname, 'runs', 'LATEST'), runDir + '\n');
	});

	const tb = TENSORBOARD ? createTensorBoard(runDir, stamp) : null;
	if (tb) tb.startFollow(jsonlPath);
	else console.error('[tb] disabled (--no-tb)');

	// --tensorboard: open TB metrics UI (orthogonal to --headful Screeps map)
	const tbIndexRoot = path.join(__dirname, 'runs', 'tb');
	let tbUi = null;
	if (OPEN_TENSORBOARD) {
		if (!TENSORBOARD) {
			console.error('[tensorboard] needs event generation (omit --no-tb)');
		} else {
			if (!HEADFUL) {
				console.error('[tensorboard] Metrics UI only. For Screeps map: re-run with --headful');
			}
			tbUi = openTensorBoardUi(tbIndexRoot);
		}
	}

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
	let lastTiStatsTick = -1;
	let lastHarnessJsonlTick = -1;
	let prevHarness = null;

	const t0 = Date.now();
	const reportEvery = Math.max(1, Math.floor(TICKS / 20));
	/** How often to sample Memory.stats / harness into JSONL (TB feed) */
	const sampleEvery = Math.max(1, Number(process.env.APEX_BENCH_SAMPLE_EVERY || 5));

	try {
		for (let i = 0; i < TICKS; i++) {
			const time = shard.time;
			const [ ir, vr ] = await Promise.all([
				shard.scratch.sMembers(userToIntentRoomsSetKey(USER)),
				shard.scratch.sMembers(userToVisibleRoomsSetKey(USER)),
			]);
			await instance.run(time, vr, ir);
			await tick(1);
			if (HEADFUL && HEADFUL_TICK_MS > 0) await sleep(HEADFUL_TICK_MS);

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
				// Apex: Role at memory[0] (CreepMem.role). Names are `${role}_${tick}_…`.
				const ROLE_NAMES = [
					'bootstrap', 'harvester', 'remoteHarvester', 'hauler', 'remoteHauler',
					'filler', 'upgrader', 'builder', 'reserver', 'claimer', 'scout',
					'defender', 'attacker', 'ranged', 'healer', 'dismantler',
				];
				const roles = {};
				for (const name in Game.creeps) {
					const mem = Game.creeps[name].memory || {};
					let role = mem[0] ?? mem['0'] ?? mem.role;
					if (typeof role === 'string' && /^\d+$/.test(role)) role = Number(role);
					if (typeof role === 'number') role = ROLE_NAMES[role] ?? `role_${role}`;
					else if (typeof role !== 'string' || !role) {
						// Fallback: spawn names encode role (harvester_k3j_a)
						const prefix = String(name).split('_')[0];
						role = ROLE_NAMES.includes(prefix) ? prefix : '?';
					}
					roles[role] = (roles[role] || 0) + 1;
				}
				metrics.roleCounts = roles;
			});

			// Pull Apex metrics segment → JSONL only when seq advances (bot writes every 5 ticks).
			try {
				const blob = await loadMemorySegmentBlob(shard, USER, METRICS_SEGMENT);
				const text = decodeSegmentBlob(blob);
				if (text) {
					const payload = JSON.parse(text);
					// Envelope via SegmentKey enum only (never raw 's'/'w'/'d')
					const seq = payload[SegmentKey.seq] ?? payload.seq;
					const written = payload[SegmentKey.written] ?? payload.written;
					if (seq != null && seq !== lastSegmentSeq) {
						lastSegmentSeq = seq;
						let botSample = payload[SegmentKey.sample] ?? payload.sample;
						if (!botSample && Array.isArray(payload.ring) && payload.ring.length) {
							botSample = payload.ring[payload.ring.length - 1];
						}
						if (botSample) {
							// Expand via MetricKey enum (wire → long member names)
							const expanded = expandDenseSample(botSample);
							lastBotSample = expanded;
							const row = {
								seq,
								written,
								source: 'segment',
								sample: expanded,
								dense: botSample,
							};
							await jsonl.write(JSON.stringify(row) + '\n');
							// watch.py --follow consumes jsonl → TB events
						}
					}
				}
			} catch {
				// Segment may not exist until setActiveSegments settles.
			}

			// The International: Memory.stats (Grafana / pandascreeps shape) → JSONL → TB
			if ((i + 1) % sampleEvery === 0) {
				try {
					const memText = await loadUserMemoryString(shard, USER);
					if (memText) {
						const mem = JSON.parse(memText);
						const tiSample = sampleFromTiMemoryStats(mem.stats, shard.time);
						if (tiSample && tiSample.t !== lastTiStatsTick) {
							lastTiStatsTick = tiSample.t;
							lastBotSample = { ...lastBotSample, ...tiSample };
							await jsonl.write(JSON.stringify({
								seq: `ti-${tiSample.t}`,
								written: Date.now(),
								source: 'ti-memory-stats',
								sample: tiSample,
							}) + '\n');
						}
					}
				} catch {
					// Memory may be empty early / non-JSON
				}

				// Always emit harness event totals so TB has curves even without bot metrics
				if (shard.time !== lastHarnessJsonlTick) {
					lastHarnessJsonlTick = shard.time;
					const h = {
						t: shard.time,
						harvested: metrics.energyHarvested,
						build: metrics.energyBuild,
						upgrade: metrics.energyUpgrade,
						controlPoints: metrics.controlPoints,
						creepEnergy: metrics.energyCreeps,
						creeps: metrics.creepCount,
						rclMax: metrics.rcl,
						progress: metrics.controllerProgress,
						source: 'harness',
					};
					if (prevHarness) {
						const dt = Math.max(1, h.t - prevHarness.t);
						h.harvestRate = (h.harvested - prevHarness.harvested) / dt;
						h.buildRate = (h.build - prevHarness.build) / dt;
						h.upgradeRate = (h.upgrade - prevHarness.upgrade) / dt;
						h.controlRate = (h.controlPoints - prevHarness.controlPoints) / dt;
						// Prefer TI rates when present
						if (lastBotSample?.source === 'ti-memory-stats') {
							if (lastBotSample.harvestRate != null) h.harvestRate = lastBotSample.harvestRate;
							if (lastBotSample.claimedHarvestRate != null) h.claimedHarvestRate = lastBotSample.claimedHarvestRate;
							if (lastBotSample.remoteHarvestRate != null) h.remoteHarvestRate = lastBotSample.remoteHarvestRate;
							if (lastBotSample.upgradeRate != null) h.upgradeRate = lastBotSample.upgradeRate;
							if (lastBotSample.buildRate != null) h.buildRate = lastBotSample.buildRate;
							if (lastBotSample.cpu != null) h.cpu = lastBotSample.cpu;
							if (lastBotSample.bucket != null) h.bucket = lastBotSample.bucket;
							if (lastBotSample.gcl != null) h.gcl = lastBotSample.gcl;
						}
					}
					prevHarness = { t: h.t, harvested: h.harvested, build: h.build, upgrade: h.upgrade, controlPoints: h.controlPoints };
					if (!lastBotSample || lastBotSample.source === 'ti-memory-stats' || lastBotSample.source === 'harness') {
						lastBotSample = { ...lastBotSample, ...h };
					}
					await jsonl.write(JSON.stringify({
						seq: `harness-${h.t}`,
						written: Date.now(),
						source: 'harness',
						sample: h,
					}) + '\n');
				}
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
					/** e/t harvested (bot window or harness delta) */
					harvestPerTick: lastBotSample && lastBotSample.harvestRate,
					claimedHarvestPerTick: lastBotSample && lastBotSample.claimedHarvestRate,
					remoteHarvestPerTick: lastBotSample && lastBotSample.remoteHarvestRate,
					/** control points / tick */
					controlPerTick: lastBotSample && lastBotSample.controlRate,
					upgradePerTick: lastBotSample && lastBotSample.upgradeRate,
					/** spawn-bound ceiling e/t */
					maxHarvestEt: lastBotSample && lastBotSample.maxHarvestEt,
					maxHarvestPhysics: lastBotSample && lastBotSample.maxHarvestPhysics,
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
		if (tb) await tb.stop();
		if (headful) {
			await headful.stop();
			console.error(`[headful] stopped ${headful.url}`);
		}
		// Leave TB server running after BM unless APEX_BENCH_TB_KILL=1
		if (tbUi && process.env.APEX_BENCH_TB_KILL === '1') {
			await tbUi.stop();
		} else if (tbUi) {
			console.error(`[tensorboard] left running at ${tbUi.url} (set APEX_BENCH_TB_KILL=1 to stop with BM)`);
		}
	}

	const elapsed = (Date.now() - t0) / 1000;
	const tbDir = path.join(runDir, 'tb');
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
		tbDir: TENSORBOARD ? tbDir : null,
		tensorboard: TENSORBOARD,
		headful: HEADFUL,
		botDir: BOT_DIR,
	};

	const outPath = path.join(runDir, 'bench-results.json');
	await fs.writeFile(outPath, JSON.stringify({ summary, timeline: metrics.timeline }, null, 2));
	await fs.writeFile(path.join(__dirname, 'bench-results.json'), JSON.stringify({ summary, timeline: metrics.timeline }, null, 2));
	// Full console dump (TI screeps-profiler prints its table here on the last profile tick)
	if (metrics.console.length) {
		const consolePath = path.join(runDir, 'console.txt');
		await fs.writeFile(consolePath, metrics.console.join('\n') + '\n');
		const profileLines = metrics.console.filter(l =>
			/calls\s+time\s+avg\s+function|Profiling for |APEX_BM_PROFILE|Per tick:/.test(l));
		if (profileLines.length) {
			const profPath = path.join(runDir, 'ti-profiler.txt');
			// Include full profiler tables (multi-line blobs that contain the header)
			const blobs = metrics.console.filter(l =>
				l.includes('calls') && l.includes('function') ||
				l.includes('Per tick:') ||
				l.includes('APEX_BM_PROFILE') ||
				l.includes('Profiling for'));
			await fs.writeFile(profPath, (blobs.length ? blobs : profileLines).join('\n\n') + '\n');
			console.log(`TI profiler dump: ${profPath}`);
		}
		console.log(`Console dump: ${consolePath} (${metrics.console.length} lines)`);
	}
	console.log('\n=== APEX BENCH SUMMARY ===');
	console.log(JSON.stringify(summary, null, 2));
	console.log(`\nWrote ${outPath}`);
	console.log(`Metrics JSONL: ${jsonlPath}`);
	if (HEADFUL) {
		console.log('Headful: Screeps client was served during the run (stopped with BM).');
		console.log('  Re-run with --headful to watch live; login Player 1 / apexbench → W7N3');
	}
	if (TENSORBOARD) {
		const multi = path.join(__dirname, 'runs', 'tb');
		console.log('TensorBoard (all runs — does not overwrite):');
		console.log(`  tensorboard --logdir ${multi}`);
		console.log(`This run only: tensorboard --logdir ${tbDir}`);
		if (OPEN_TENSORBOARD) {
			console.log(`TensorBoard UI (--tensorboard): http://127.0.0.1:${TB_PORT}/`);
		}
	} else {
		console.log('TensorBoard: disabled (--no-tb). Re-run without flag, or:');
		console.log(`  python3 samples/bots/metrics-watcher/watch.py --jsonl ${jsonlPath} --logdir ${tbDir}`);
	}
});

// Force exit — open handles from db
process.exit(0);
