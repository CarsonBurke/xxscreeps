/**
 * RL environment server — JSON lines on stdin/stdout.
 *
 * Modes:
 *   reset / step          — external AR policy actions
 *   reset_expert / step_expert — fixed bot (The International) for critic pretraining
 *   step_scripted         — RCL1 scripted baseline (same action interface as RL)
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
import { GameState, runForUser } from 'xxscreeps/game/index.js';
import { RoomPosition } from 'xxscreeps/game/position.js';
import { makeRoomName, parseRoomName } from 'xxscreeps/game/room/name.js';
import { setPassword } from 'xxscreeps/mods/backend/password/model.js';
import { create as createSpawn } from 'xxscreeps/mods/spawn/spawn.js';
import { create as createCreep } from 'xxscreeps/mods/creep/creep.js';
import { simulate } from 'xxscreeps/test/index.js';
import 'xxscreeps:mods/driver';
import { SCHEMA, encodeObservation } from './encode.mjs';
import { applyActions } from './actions.mjs';
import { scriptedActions } from './scripted_baseline.mjs';

config.database.lock = null;
if (!config.runner.sandbox) {
	config.runner.sandbox = 'isolated';
}

const USER = '100';
const ROOM = process.env.RL_ROOM || 'W7N3';
const roomCoord = parseRoomName(ROOM);
// The imported W7N3 test terrain has a real north exit shared with W7N4; its
// south edge is sealed. Custom scenarios can name another connected neighbor.
const EXPANSION_ROOM = process.env.RL_EXPANSION_ROOM
	|| makeRoomName(roomCoord.rx, roomCoord.ry - 1);
const MAX_EPISODE = Number(process.env.RL_MAX_EPISODE || 2000);
/** Curriculum: empty | seed_creep | seed_full (creep with energy near source) */
const CURRICULUM = process.env.RL_CURRICULUM || 'empty';
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
/** bin = length-prefixed raw frames; pack/b64/json = JSONL */
const OBS_FMT = process.env.RL_OBS_FMT || 'bin';
const BIN_MODE = OBS_FMT === 'bin';
/**
 * Lean train meta (default on): omit actorMeta/targetMeta/intentResults from the
 * wire frame. Server still keeps session.meta for apply/scripted. Watch/debug:
 * RL_LEAN_META=0.
 */
const LEAN_META = process.env.RL_LEAN_META !== '0' && process.env.RL_LEAN_META !== 'false';

// Bin mode: stdout is protocol-only. Engine notify / debug console.log would
// desync XRL1 frames — force them to stderr.
if (BIN_MODE) {
	const _err = console.error.bind(console);
	console.log = (...args) => _err(...args);
}

function bufFromTyped(a) {
	if (a instanceof Uint8Array || a instanceof Float32Array) {
		return Buffer.from(a.buffer, a.byteOffset, a.byteLength);
	}
	if (ArrayBuffer.isView(a)) {
		return Buffer.from(a.buffer, a.byteOffset, a.byteLength);
	}
	return Buffer.from(a);
}

/**
 * Binary response frame (RL_OBS_FMT=bin):
 *   magic "XRL1" | ver u8 | flags u8 | schema_ver u16 LE
 *   | meta_len u32 LE | blob_len u32 LE | meta JSON | tensor blob
 * flags: bit0=ok bit1=done bit2=has_obs
 */
function writeBinFrame(obj) {
	const ok = !!obj.ok;
	const obs = obj.obs;
	const hasObs = ok && obs != null && (obs._raw != null || obs.encoding === 'bin');
	// Strip undefined info keys so JSON is smaller.
	const rawInfo = obj.info ?? {};
	const info = {};
	for (const k of Object.keys(rawInfo)) {
		if (rawInfo[k] !== undefined) info[k] = rawInfo[k];
	}
	const meta = {
		ok,
		reward: obj.reward ?? 0,
		done: !!obj.done,
		error: obj.error,
		info,
	};
	let parts = null;
	let blobLen = 0;
	if (hasObs) {
		meta.time = obs.time;
		meta.roomsUsed = obs.roomsUsed ?? obs.shapes?.patches?.[0] ?? 1;
		meta.roomNames = obs.roomNames;
		meta.globals = obs.globals;
		meta.shapes = obs.shapes;
		// Full metas only when not lean (watch / debugging).
		if (!LEAN_META) {
			meta.actorMeta = obs.actorMeta;
			meta.targetMeta = obs.targetMeta;
		}
		const raw = obs._raw;
		if (raw) {
			parts = [
				bufFromTyped(raw.patches),
				bufFromTyped(raw.actors),
				bufFromTyped(raw.targets),
				bufFromTyped(raw.roomCoords),
				bufFromTyped(raw.roomMask),
				bufFromTyped(raw.actorMask),
				bufFromTyped(raw.targetMask),
				bufFromTyped(raw.intentMask),
				bufFromTyped(raw.dirMask),
				bufFromTyped(raw.targetSelectMask),
				bufFromTyped(raw.amountMask),
			];
			blobLen = parts.reduce((n, p) => n + p.byteLength, 0);
		}
	}
	const metaBuf = Buffer.from(JSON.stringify(meta), 'utf8');
	const total = 16 + metaBuf.length + blobLen;
	const out = Buffer.allocUnsafe(total);
	out.write('XRL1', 0, 4, 'ascii');
	out[4] = 1; // version
	out[5] = (ok ? 1 : 0) | (obj.done ? 2 : 0) | (hasObs ? 4 : 0);
	out.writeUInt16LE(Number(SCHEMA.version) || 1, 6);
	out.writeUInt32LE(metaBuf.length, 8);
	out.writeUInt32LE(blobLen, 12);
	metaBuf.copy(out, 16);
	if (parts) {
		let o = 16 + metaBuf.length;
		for (const p of parts) {
			p.copy(out, o);
			o += p.byteLength;
		}
	}
	process.stdout.write(out);
}

function reply(obj) {
	if (BIN_MODE) {
		writeBinFrame(obj);
		return;
	}
	// Strip internal TypedArray payload if present (should not reach JSON path)
	if (obj?.obs?._raw) {
		const { _raw, ...rest } = obj.obs;
		obj = { ...obj, obs: rest };
	}
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

function freeTileNear(room, pos, maxR = 3) {
	for (let r = 1; r <= maxR; r++) {
		for (let dx = -r; dx <= r; dx++) {
			for (let dy = -r; dy <= r; dy++) {
				if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
				const x = pos.x + dx;
				const y = pos.y + dy;
				if (x < 2 || x > 47 || y < 2 || y > 47) continue;
				if (room.getTerrain().get(x, y) === C.TERRAIN_MASK_WALL) continue;
				if (room.lookForAt(C.LOOK_STRUCTURES, x, y).length) continue;
				if (room.lookForAt(C.LOOK_CREEPS, x, y).length) continue;
				if (room.lookForAt(C.LOOK_CONSTRUCTION_SITES, x, y).length) continue;
				return new RoomPosition(x, y, ROOM);
			}
		}
	}
	return null;
}

function placeSpawn(room) {
	room['#user'] = USER;
	room.controller['#user'] = USER;
	room['#level'] = 1;
	const sources = room.find(C.FIND_SOURCES);
	const src = sources[0];
	let placed = false;
	if (src) {
		const tile = freeTileNear(room, src.pos, 3);
		if (tile) {
			room['#insertObject'](createSpawn(tile, USER, 'Spawn1'));
			placed = true;
		}
	}
	if (!placed) {
		room['#insertObject'](createSpawn(new RoomPosition(25, 25, ROOM), USER, 'Spawn1'));
	}

	// Curriculum: seed a bootstrap worker so BC/PPO can train logistics without first-spawn lottery.
	if (CURRICULUM === 'seed_creep' || CURRICULUM === 'seed_full' || CURRICULUM === 'seed_claimer') {
		const seat = src
			? freeTileNear(room, src.pos, 3)
			: freeTileNear(room, new RoomPosition(25, 25, ROOM), 3);
		if (seat) {
			const body = CURRICULUM === 'seed_claimer'
				? [ C.CLAIM, C.MOVE ]
				: [ C.WORK, C.CARRY, C.MOVE, C.MOVE ];
			const name = CURRICULUM === 'seed_claimer' ? 'seed_claimer' : 'seed_worker';
			const creep = createCreep(
				seat,
				body,
				name,
				USER,
			);
			if (CURRICULUM === 'seed_full' && creep.store) {
				try {
					// Prefer OpenStore.#add if present
					const cap = creep.store.getCapacity?.(C.RESOURCE_ENERGY) || 50;
					const amt = Math.min(50, cap);
					if (typeof creep.store['#add'] === 'function') {
						creep.store['#add'](C.RESOURCE_ENERGY, amt);
					}
				} catch { /* best-effort */ }
			}
			room['#insertObject'](creep);
		} else {
			console.error('[rl-env] curriculum seed failed: no free tile near source');
		}
	}
}

function placeExpansionRoom(room) {
	// Keep the adjacent room neutral but materialized in the simulation so exit
	// movement, scouting, reservation, and claiming are real engine operations.
	room['#user'] = null;
	if (room.controller) room.controller['#user'] = null;
	room['#level'] = 0;
}

/** `simulate.player()` has no runner payload, so install its missing GCL view. */
function installSimulationGcl(Game) {
	const controlled = Object.values(Game.rooms)
		.filter(room => room.controller?.my).length;
	Game.gcl = {
		level: 2,
		progress: 0,
		progressTotal: Math.floor(2 ** C.GCL_POW * C.GCL_MULTIPLY),
		'#roomCount': controlled,
	};
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

const factory = simulate({
	[ROOM]: placeSpawn,
	[EXPANSION_ROOM]: placeExpansionRoom,
});

/** @type {any} */
let session = null;

async function bootShard({ expert = false, botDir = DEFAULT_EXPERT_BOT } = {}) {
	// Tear down prior headful client first so :21025 can rebind on reset.
	if (session?.headful?.stop) {
		try { await session.headful.stop(); } catch { /* */ }
	}
	if (session?.close) {
		try { session.instance?.disconnect?.(); } catch { /* */ }
		await session.close();
	}
	session = null;

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
		lastOwnedRooms: 1,
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
	// The expansion scenario starts with one owned room and must have capacity
	// for a second. Runner-backed expert sessions read this persisted GCL2 value;
	// direct simulate.player calls use installSimulationGcl below.
	await handle.db.data.hSet(User.infoKey(USER), 'gcl', String(C.GCL_MULTIPLY));

	if (expert) {
		const modules = await loadBotModules(botDir);
		await Code.saveContent(handle.db, USER, 'main', modules);
		handle.instance = await PlayerInstance.create(handle.shard, handle.world, USER);
		console.error(`[rl-env] expert bot loaded from ${botDir}`);
	}

	// One settle tick, then post-tick obs (matches step contract: agent sees true state).
	// Encode read-only — do not consume player() so the first step can apply intents.
	await handle.tick(1);
	session = handle;
	const obs = await encodeAfterTick();
	handle.lastControl = 0;
	handle.lastRcl = 1;
	handle.lastCreeps = 0;
	// Seed reward baselines from globals / peek
	await handle.peekRoom(ROOM, room => {
		handle.lastControl = room.controller?.progress || 0;
		handle.lastRcl = room.controller?.level || 1;
		handle.lastCreeps = room.find(C.FIND_MY_CREEPS).length;
	});

	if (HEADFUL) {
		try {
			handle.headful = await startHeadfulClient(handle.db, handle.shard, handle.world);
		} catch (err) {
			console.error(`[headful] failed: ${err?.message || err}`);
			console.error('[headful] is port 21025 free? stop `xxscreeps start` if needed');
		}
	}

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

/**
 * Post-tick: ONE room load + ONE runForUser for encode + H+C reward (SPS win).
 * `player()` is reserved for intents; this path is read-only.
 */
async function encodeAndRewardAfterTick() {
	const visibleRooms = await session.shard.scratch.sMembers(userToVisibleRoomsSetKey(USER));
	const roomNames = visibleRooms.length ? visibleRooms : [ ROOM ];
	// The expansion curriculum exposes the connected neutral neighbor as a known
	// strategic room. Primitive intents still run through the player's normal
	// visible-room state; actions.mjs uses positional proxies until a creep scouts it.
	if (!roomNames.includes(ROOM)) roomNames.push(ROOM);
	if (!roomNames.includes(EXPANSION_ROOM)) roomNames.push(EXPANSION_ROOM);
	const rooms = await Promise.all(roomNames.map(n => session.shard.loadRoom(n)));
	const state = new GameState(session.world, session.shard.time, rooms);

	let obs = null;
	let harvestDelta = 0;
	let upgradeDelta = 0;
	let transferToSpawn = 0;
	let advancedDepositDelta = 0;
	let advancedWithdrawDelta = 0;
	let towerRefillDelta = 0;
	let buildDelta = 0;
	let controlNow = session.lastControl;
	let rclNow = session.lastRcl || 1;
	let creeps = 0;
	let spawnEnergy = 0;
	let energyAvailable = 0;
	let sitesNow = 0;
	let ownedRooms = 0;

	runForUser(USER, state, Game => {
		installSimulationGcl(Game);
		obs = encodeObservation(Game, USER);
		controlNow = 0;
		rclNow = 0;
		for (const room of Object.values(Game.rooms)) {
			const productiveSinkIds = new Set();
			const bankIds = new Set();
			const towerIds = new Set();
			for (const s of room.find(C.FIND_MY_STRUCTURES)) {
				if (
					s.structureType === C.STRUCTURE_SPAWN
					|| s.structureType === C.STRUCTURE_EXTENSION
					|| s.structureType === C.STRUCTURE_TOWER
				) {
					productiveSinkIds.add(s.id);
				}
				if (s.structureType === C.STRUCTURE_STORAGE
					|| s.structureType === C.STRUCTURE_CONTAINER) {
					bankIds.add(s.id);
				}
				if (s.structureType === C.STRUCTURE_TOWER) towerIds.add(s.id);
			}
			for (const ev of room.getEventLog()) {
				const amt = ev.data?.amount ?? ev.amount ?? 0;
				if (ev.event === C.EVENT_HARVEST) harvestDelta += amt;
				if (ev.event === C.EVENT_UPGRADE_CONTROLLER) upgradeDelta += amt;
				if (ev.event === C.EVENT_BUILD) buildDelta += amt;
				if (ev.event === C.EVENT_TRANSFER) {
					const res = ev.resourceType ?? ev.data?.resourceType;
					if (res != null && res !== C.RESOURCE_ENERGY) continue;
					const tid = ev.targetId ?? ev.data?.targetId;
					const oid = ev.objectId ?? ev.data?.objectId;
					if (tid && productiveSinkIds.has(tid)) transferToSpawn += amt;
					if (tid && bankIds.has(tid)) advancedDepositDelta += amt;
					if (oid && bankIds.has(oid)) advancedWithdrawDelta += amt;
					if (tid && towerIds.has(tid)) towerRefillDelta += amt;
				}
			}
			if (room.controller?.my) {
				ownedRooms += 1;
				controlNow += room.controller.progress || 0;
				rclNow = Math.max(rclNow, room.controller.level || 0);
			}
			creeps += room.find(C.FIND_MY_CREEPS).length;
			sitesNow += room.find(C.FIND_MY_CONSTRUCTION_SITES).length;
			for (const s of room.find(C.FIND_MY_SPAWNS)) {
				spawnEnergy += s.store?.[C.RESOURCE_ENERGY] || 0;
				if (s.store) energyAvailable += s.store[C.RESOURCE_ENERGY] || 0;
			}
			for (const s of room.find(C.FIND_MY_STRUCTURES)) {
				if (s.structureType === C.STRUCTURE_EXTENSION) {
					energyAvailable += s.store?.[C.RESOURCE_ENERGY] || 0;
				}
			}
		}
	});

	session.meta = { actorMeta: obs.actorMeta, targetMeta: obs.targetMeta };

	const controlDelta = Math.max(0, controlNow - session.lastControl);
	const cp = upgradeDelta > 0 ? upgradeDelta : controlDelta;
	session.lastControl = controlNow;
	const prevCreeps = session.lastCreeps ?? 0;
	const spawnSuccess = Math.max(0, creeps - prevCreeps);
	session.lastCreeps = creeps;
	const prevRcl = session.lastRcl ?? 1;
	const rclUp = Math.max(0, rclNow - prevRcl);
	session.lastRcl = rclNow;
	session.lastSites = sitesNow;
	const claimDelta = Math.max(0, ownedRooms - (session.lastOwnedRooms ?? 1));
	session.lastOwnedRooms = ownedRooms;

	const wH = REWARD_CFG.energyHarvested ?? 1.0;
	const wC = REWARD_CFG.controlPoints ?? 1.0;
	const wD = REWARD_CFG.energyDelivered ?? 0.0;
	const wB = REWARD_CFG.buildProgress ?? 0.0;
	const wClaim = REWARD_CFG.roomClaim ?? 0.0;
	const reward = wH * harvestDelta + wC * cp
		+ wD * transferToSpawn + wB * buildDelta + wClaim * claimDelta;
	const extras = {
		transferDelta: transferToSpawn,
		advancedDepositDelta,
		advancedWithdrawDelta,
		towerRefillDelta,
		buildDelta,
		spawnSuccess,
		rclUp,
		creeps,
		energyAvailable,
		ownedRooms,
		claimDelta,
		spawnEnergy,
		sites: sitesNow,
	};
	return {
		obs,
		reward,
		harvestDelta,
		controlDelta: cp,
		controlNow,
		extras,
	};
}

/** @deprecated use encodeAndRewardAfterTick — kept for reset boot path only */
async function encodeAfterTick() {
	const { obs } = await encodeAndRewardAfterTick();
	return obs;
}

function summarizeIntentResults(results) {
	const byType = {};
	let invalid = 0;
	for (const result of results) {
		const type = result.type || 'unknown';
		const row = byType[type] || { issued: 0, invalid: 0 };
		row.issued += 1;
		const code = Number(result.code);
		if (code !== C.OK && code !== C.ERR_BUSY && code !== C.ERR_TIRED) {
			row.invalid += 1;
			invalid += 1;
		}
		byType[type] = row;
	}
	return { intentIssued: results.length, intentInvalid: invalid, intentByType: byType };
}

async function stepRl(actions) {
	if (!session) throw new Error('call reset first');
	if (session.expert) throw new Error('session is expert mode; use step_expert');

	// Apply intents against the world matching the last returned obs (session.meta).
	// Encode ONLY after tick so the agent conditions on the true next state (H1 fix).
	let intentResults = [];
	await session.player(USER, Game => {
		installSimulationGcl(Game);
		intentResults = applyActions(Game, session.meta, actions || {
			types: [], dirs: [], targets: [], amounts: [],
		});
	});
	await session.tick(1);
	session.step += 1;
	if (HEADFUL_TICK_MS > 0) await sleep(HEADFUL_TICK_MS);

	const { obs, reward, harvestDelta, controlDelta, extras } = await encodeAndRewardAfterTick();
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
				// intentResults omitted on lean wire (still in session if needed)
				intentResults: LEAN_META ? undefined : intentResults,
				...summarizeIntentResults(intentResults),
				globals: obs.globals,
				curriculum: CURRICULUM,
				expert: false,
			...extras,
		},
	};
}

/**
 * Scripted RCL1 baseline: decide+apply in a single player() call, then tick+encode.
 */
async function stepScripted() {
	if (!session) throw new Error('call reset first');
	if (session.expert) throw new Error('session is expert mode; use step_expert');

	let intentResults = [];
	let actions = null;
	await session.player(USER, Game => {
		installSimulationGcl(Game);
		actions = scriptedActions(Game, session.meta || { actorMeta: [], targetMeta: [] });
		intentResults = applyActions(Game, session.meta, actions);
	});
	await session.tick(1);
	session.step += 1;
	if (HEADFUL_TICK_MS > 0) await sleep(HEADFUL_TICK_MS);

	const { obs, reward, harvestDelta, controlDelta, extras } = await encodeAndRewardAfterTick();
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
				intentResults: LEAN_META ? undefined : intentResults,
				...summarizeIntentResults(intentResults),
				globals: obs.globals,
				curriculum: CURRICULUM,
				expert: false,
			scripted: true,
			actions,
			...extras,
		},
	};
}

/**
 * Expert step: TI (or other bot) runs via PlayerInstance; no external actions.
 * Obs is post-tick state (matches RL step contract).
 */
async function stepExpert() {
	if (!session) throw new Error('call reset_expert first');
	if (!session.expert || !session.instance) throw new Error('not in expert mode');

	const time = session.shard.time;
	const [ ir, vr ] = await Promise.all([
		session.shard.scratch.sMembers(userToIntentRoomsSetKey(USER)),
		session.shard.scratch.sMembers(userToVisibleRoomsSetKey(USER)),
	]);
	await session.instance.run(time, vr, ir);
	await session.tick(1);
	session.step += 1;

	const { obs, reward, harvestDelta, controlDelta, extras } = await encodeAndRewardAfterTick();
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
			...extras,
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
			} else if (msg.cmd === 'step_scripted') {
				// One player() only (simulate forbids two per tick). Decide + apply together.
				reply(await stepScripted());
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
