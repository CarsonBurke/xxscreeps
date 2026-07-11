/**
 * RL environment server — JSON lines on stdin/stdout.
 *
 * Modes:
 *   reset / step          — external AR policy actions
 *   reset_expert / step_expert — fixed bot (The International) for critic pretraining
 *
 * Usage (from repo root):
 *   mise exec node@24 -- node --import xxscreeps/loader samples/rl/env/server.mjs
 */
import * as fs from 'node:fs/promises';
import * as fsSync from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { spawn as spawnChild } from 'node:child_process';
import { BackendContext } from 'xxscreeps/backend/context.js';
import { listenBackend } from 'xxscreeps/backend/listen.js';
import { config } from 'xxscreeps/config/index.js';
import * as Code from 'xxscreeps/engine/db/user/code.js';
import * as Badge from 'xxscreeps/engine/db/user/badge.js';
import * as User from 'xxscreeps/engine/db/user/index.js';
import { PlayerInstance } from 'xxscreeps/engine/runner/instance.js';
import { userToIntentRoomsSetKey, userToVisibleRoomsSetKey } from 'xxscreeps/engine/processor/model.js';
import * as C from 'xxscreeps/game/constants/index.js';
import { RoomPosition } from 'xxscreeps/game/position.js';
import { setPassword } from 'xxscreeps/mods/backend/password/model.js';
import { create as createSpawn } from 'xxscreeps/mods/spawn/spawn.js';
import { simulate } from 'xxscreeps/test/index.js';
import 'xxscreeps:mods/driver';
import { SCHEMA, encodeObservation } from './encode.mjs';
import { applyActions } from './actions.mjs';

config.database.lock = null;
if (!config.runner.sandbox) {
	config.runner.sandbox = 'isolated';
}

const USER = '100';
const ROOM = process.env.RL_ROOM || 'W7N3';
const MAX_EPISODE = Number(process.env.RL_MAX_EPISODE || 2000);
const REWARD_CFG = SCHEMA.reward;
const DEFAULT_EXPERT_BOT = process.env.RL_EXPERT_BOT ||
	path.resolve(path.dirname(fileURLToPath(import.meta.url)),
		'../../../The-International-Open-Source/dist');
/** Serve Screeps client against this BM session (watch mode). */
const HEADFUL = process.env.RL_HEADFUL === '1' || process.env.RL_HEADFUL === 'true';
const HEADFUL_PASSWORD = process.env.RL_HEADFUL_PASSWORD || 'rlwatch';
const HEADFUL_TICK_MS = process.env.RL_TICK_MS != null && process.env.RL_TICK_MS !== ''
	? Math.max(0, Number(process.env.RL_TICK_MS))
	: (HEADFUL ? 150 : 0);
const AUTO_OPEN = process.env.RL_NO_OPEN !== '1' && process.env.RL_NO_OPEN !== 'true';

function reply(obj) {
	process.stdout.write(JSON.stringify(obj) + '\n');
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

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

/** Attach Screeps client to the in-process BM (same pattern as apex --headful). */
async function startHeadfulClient(db, shard, world) {
	await import('xxscreeps:mods/backend');
	const backendContext = await BackendContext.attach(db, shard, world);
	await setPassword(db, USER, HEADFUL_PASSWORD);
	const existingBadge = await db.data.hGet(User.infoKey(USER), 'badge');
	if (existingBadge == null) {
		await Badge.save(db, USER, JSON.stringify(Badge.generateRandom()));
	}
	await db.data.hSet(User.infoKey(USER), 'lastViewedRoom', ROOM);

	const prevBind = config.backend.bind;
	config.backend.bind = process.env.RL_HEADFUL_BIND || '127.0.0.1:21025';
	let handle;
	try {
		handle = await listenBackend(backendContext);
	} finally {
		config.backend.bind = prevBind;
	}

	console.error(`[headful] Screeps client → ${handle.url}`);
	console.error(`[headful] room ${ROOM}; login "Player 1" / "${HEADFUL_PASSWORD}" (or guest)`);
	console.error('[headful] view-only preferred — client intents can mutate the BM');
	if (HEADFUL_TICK_MS > 0) {
		console.error(`[headful] tick delay ${HEADFUL_TICK_MS}ms (RL_TICK_MS=0 for max speed)`);
	}
	if (AUTO_OPEN) {
		setTimeout(() => {
			if (openInBrowser(handle.url)) console.error(`[headful] opened ${handle.url}`);
			else console.error(`[headful] open manually: ${handle.url}`);
		}, 400);
	}
	return {
		url: handle.url,
		async stop() {
			try { await handle.stop(); } catch { /* */ }
			try { await backendContext.disposeAsync(); } catch { /* */ }
		},
	};
}

function placeSpawn(room) {
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
}

async function loadBotModules(botDir) {
	const names = await fs.readdir(botDir);
	const modules = new Map();
	for (const name of names) {
		if (name.endsWith('.map') || name.endsWith('.map.js') || name.startsWith('.')) continue;
		const full = path.join(botDir, name);
		const st = await fs.stat(full).catch(() => null);
		if (!st?.isFile()) continue;
		const isWasm = name.endsWith('.wasm');
		const isJs = name.endsWith('.js') || name.endsWith('.mjs') || name === 'main';
		if (!isWasm && !isJs) continue;
		const content = isWasm
			? new Uint8Array(await fs.readFile(full))
			: await fs.readFile(full, 'utf8');
		const key = isWasm && name !== 'main.wasm' ? path.basename(name, '.wasm') : name;
		modules.set(key, content);
	}
	if (![ 'main.js', 'main', 'main.mjs', 'main.wasm' ].some(n => modules.has(n))) {
		throw new Error(`main.js missing in ${botDir}`);
	}
	return modules;
}

const factory = simulate({ [ROOM]: placeSpawn });

/** @type {any} */
let session = null;

async function bootShard({ expert = false, botDir = DEFAULT_EXPERT_BOT } = {}) {
	if (session?.close) {
		try { session.instance?.disconnect?.(); } catch { /* */ }
		await session.close();
	}

	let resolveReady;
	const ready = new Promise(r => { resolveReady = r; });
	const handle = {
		tick: null,
		player: null,
		peekRoom: null,
		db: null,
		shard: null,
		world: null,
		close: null,
		step: 0,
		lastControl: 0,
		meta: null,
		expert: Boolean(expert),
		instance: null,
		botDir,
		headful: null,
	};

	let releaseBody;
	const bodyDone = new Promise(r => { releaseBody = r; });

	const simPromise = factory(async refs => {
		handle.tick = refs.tick;
		handle.player = refs.player;
		handle.peekRoom = refs.peekRoom;
		handle.db = refs.db;
		handle.shard = refs.shard;
		handle.world = refs.world;
		handle.close = async() => {
			try { handle.instance?.disconnect?.(); } catch { /* */ }
			releaseBody();
			try { await simPromise; } catch { /* disposed */ }
		};
		resolveReady();
		await bodyDone;
	});

	await ready;
	simPromise.catch(() => {});

	if (expert) {
		const modules = await loadBotModules(botDir);
		await Code.saveContent(handle.db, USER, 'main', modules);
		handle.instance = await PlayerInstance.create(handle.shard, handle.world, USER);
		console.error(`[rl-env] expert bot loaded from ${botDir}`);
	}

	await handle.tick(1);
	let obs = null;
	await handle.player(USER, Game => {
		obs = encodeObservation(Game, USER);
		handle.meta = { actorMeta: obs.actorMeta, targetMeta: obs.targetMeta };
		handle.lastControl = Game.rooms[ROOM]?.controller?.progress || 0;
	});
	await handle.tick(1);

	if (HEADFUL) {
		try {
			handle.headful = await startHeadfulClient(handle.db, handle.shard, handle.world);
		} catch (err) {
			console.error(`[headful] failed: ${err?.message || err}`);
			console.error('[headful] is port 21025 free? stop `xxscreeps start` if needed');
		}
	}

	session = handle;
	return {
		ok: true,
		obs,
		info: {
			room: ROOM,
			time: obs.time,
			step: 0,
			expert: handle.expert,
			botDir: handle.botDir,
			headful: Boolean(handle.headful),
			headfulUrl: handle.headful?.url || null,
		},
	};
}

async function measureReward() {
	let harvestDelta = 0;
	let upgradeDelta = 0;
	let controlNow = session.lastControl;
	await session.peekRoom(ROOM, room => {
		for (const ev of room.getEventLog()) {
			if (ev.event === C.EVENT_HARVEST) harvestDelta += ev.data?.amount || 0;
			if (ev.event === C.EVENT_UPGRADE_CONTROLLER) upgradeDelta += ev.data?.amount || 0;
		}
		if (room.controller) controlNow = room.controller.progress || controlNow;
	});
	const controlDelta = Math.max(0, controlNow - session.lastControl);
	const cp = upgradeDelta > 0 ? upgradeDelta : controlDelta;
	session.lastControl = controlNow;
	const reward = REWARD_CFG.controlPoints * cp + REWARD_CFG.energyHarvested * harvestDelta;
	return { reward, harvestDelta, controlDelta: cp, controlNow };
}

async function stepRl(actions) {
	if (!session) throw new Error('call reset first');
	if (session.expert) throw new Error('session is expert mode; use step_expert');

	let intentResults = [];
	let obs = null;
	await session.player(USER, Game => {
		intentResults = applyActions(Game, session.meta, actions || {
			types: [], dirs: [], targets: [], amounts: [],
		});
		obs = encodeObservation(Game, USER);
		session.meta = { actorMeta: obs.actorMeta, targetMeta: obs.targetMeta };
	});
	await session.tick(1);
	session.step += 1;
	if (HEADFUL_TICK_MS > 0) await sleep(HEADFUL_TICK_MS);
	const { reward, harvestDelta, controlDelta } = await measureReward();
	const done = session.step >= MAX_EPISODE;
	return {
		ok: true,
		obs,
		reward,
		done,
		info: {
			step: session.step,
			time: obs.time,
			controlDelta,
			harvestDelta,
			intentResults,
			globals: obs.globals,
			expert: false,
		},
	};
}

/**
 * Expert step: TI (or other bot) runs via PlayerInstance; no external actions.
 * Obs is pre-bot state; reward is from the subsequent tick.
 */
async function stepExpert() {
	if (!session) throw new Error('call reset_expert first');
	if (!session.expert || !session.instance) throw new Error('not in expert mode');

	let obs = null;
	await session.player(USER, Game => {
		obs = encodeObservation(Game, USER);
		session.meta = { actorMeta: obs.actorMeta, targetMeta: obs.targetMeta };
	});

	const time = session.shard.time;
	const [ ir, vr ] = await Promise.all([
		session.shard.scratch.sMembers(userToIntentRoomsSetKey(USER)),
		session.shard.scratch.sMembers(userToVisibleRoomsSetKey(USER)),
	]);
	await session.instance.run(time, vr, ir);
	await session.tick(1);
	session.step += 1;

	const { reward, harvestDelta, controlDelta } = await measureReward();
	const done = session.step >= MAX_EPISODE;
	return {
		ok: true,
		obs,
		reward,
		done,
		info: {
			step: session.step,
			time: obs.time,
			controlDelta,
			harvestDelta,
			globals: obs.globals,
			expert: true,
			botDir: session.botDir,
		},
	};
}

async function closeEpisode() {
	if (session?.headful?.stop) {
		try { await session.headful.stop(); } catch { /* */ }
	}
	if (session?.close) await session.close();
	session = null;
	return { ok: true };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let chain = Promise.resolve();

rl.on('line', line => {
	const raw = line.trim();
	if (!raw) return;
	chain = chain.then(async() => {
		let msg;
		try {
			msg = JSON.parse(raw);
		} catch (err) {
			reply({ ok: false, error: `invalid json: ${err.message}` });
			return;
		}
		try {
			if (msg.cmd === 'reset') {
				reply(await bootShard({ expert: false }));
			} else if (msg.cmd === 'reset_expert') {
				const botDir = msg.botDir || DEFAULT_EXPERT_BOT;
				if (!fsSync.existsSync(botDir)) {
					reply({ ok: false, error: `expert bot dir missing: ${botDir}` });
					return;
				}
				reply(await bootShard({ expert: true, botDir }));
			} else if (msg.cmd === 'step') {
				reply(await stepRl(msg.actions));
			} else if (msg.cmd === 'step_expert') {
				reply(await stepExpert());
			} else if (msg.cmd === 'schema') {
				reply({ ok: true, schema: SCHEMA });
			} else if (msg.cmd === 'close') {
				reply(await closeEpisode());
				rl.close();
				setTimeout(() => process.exit(0), 50);
			} else {
				reply({ ok: false, error: `unknown cmd ${msg.cmd}` });
			}
		} catch (err) {
			reply({ ok: false, error: String(err?.stack || err) });
		}
	});
});

console.error(
	`[rl-env] ready room=${ROOM} maxEpisode=${MAX_EPISODE} headful=${HEADFUL} `
	+ `tickMs=${HEADFUL_TICK_MS} expertDefault=${DEFAULT_EXPERT_BOT}`,
);
