/**
 * RL environment server — framed binary commands, framed observations.
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
import { hooks as runnerHooks } from 'xxscreeps/engine/runner/index.js';
import { getConsoleChannel } from 'xxscreeps/engine/runner/model.js';
import { userToIntentRoomsSetKey, userToVisibleRoomsSetKey } from 'xxscreeps/engine/processor/model.js';
import { controlledRoomsKey } from 'xxscreeps/mods/controller/model.js';
import * as C from 'xxscreeps/game/constants/index.js';
import { GameState, runForUser } from 'xxscreeps/game/index.js';
import { RoomPosition } from 'xxscreeps/game/position.js';
import { makeRoomName, parseRoomName } from 'xxscreeps/game/room/name.js';
import { setPassword } from 'xxscreeps/mods/backend/password/model.js';
import { create as createSpawn } from 'xxscreeps/mods/spawn/spawn.js';
import { create as createExtension } from 'xxscreeps/mods/spawn/extension.js';
import { create as createCreep } from 'xxscreeps/mods/creep/creep.js';
import { create as createConstructionSite } from 'xxscreeps/mods/construction/construction-site.js';
import { create as createResource } from 'xxscreeps/mods/resource/resource.js';
import { create as createContainer } from 'xxscreeps/mods/resource/container.js';
import { Source } from 'xxscreeps/mods/source/source.js';
import { createRoomObject } from 'xxscreeps/game/object.js';
import { simulate } from 'xxscreeps/test/index.js';
import 'xxscreeps:mods/driver';
import { SCHEMA, encodeObservation } from './encode.mjs';
import { actionOutcome, applyActions, resetNavigationCaches } from './actions.mjs';
import { scriptedActions } from './scripted_baseline.mjs';
import {
	applyShardState, captureShardState, decodeSnapshotFile, encodeSnapshotFile,
} from './snapshot.mjs';

config.database.lock = null;
if (!config.runner.sandbox) {
	config.runner.sandbox = 'isolated';
}
// The expert bot runs in its own sandbox realm, so the scenario stream installed
// below cannot reach it. Without a seeded generator inside that realm a fixed bot
// on a fixed world diverges between runs, which makes a teacher corpus
// unreproducible and a matched comparison meaningless.
config.runner.randomSeed = Number(process.env.RL_SEED || 0) >>> 0;
config.runner.deterministicCpu = true;
// NPC invasion waves are scheduled by a room's cumulative harvest, so every
// productive 20k-tick episode meets one in its second half. This ABI supervises
// no defense: the invasion took the teacher's spawn at tick 6,997 of a healthy
// seed_full world and left 13 creeps idling with no way to rebuild, which fails
// collection as a stalled teacher. Economy runs are invasion-free; RL_INVADERS=1
// restores stock engine behavior for defense work.
const INVADERS = process.env.RL_INVADERS === '1';
config.game.invaders = INVADERS;
/** Mirror the expert bot's own console logging, not only its faults. */
const BOT_CONSOLE = process.env.RL_BOT_CONSOLE === '1';

const USER = '100';
const ROOM = process.env.RL_ROOM || 'W7N3';
const roomCoord = parseRoomName(ROOM);
// The imported W7N3 test terrain has a real north exit shared with W7N4; its
// south edge is sealed. Custom scenarios can name another connected neighbor.
const EXPANSION_ROOM = process.env.RL_EXPANSION_ROOM
	|| makeRoomName(roomCoord.rx, roomCoord.ry - 1);
const MAX_EPISODE = Number(process.env.RL_MAX_EPISODE || 2000);
/** Economy curricula plus spawn_<role>_<energy-capacity> expert scenarios. */
const CURRICULUM = process.env.RL_CURRICULUM || 'empty';
const SCENARIO_SEED = Number(process.env.RL_SEED || 0) >>> 0;
const SOURCE_COUNT = process.env.RL_SOURCE_COUNT == null
	? null
	: Math.max(1, Math.min(4, Number(process.env.RL_SOURCE_COUNT)));
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
/** bin = XAC1 framed commands (default); json = deliberate JSONL debug mode */
const CMD_FMT = process.env.RL_CMD_FMT || 'bin';
if (CMD_FMT !== 'bin' && CMD_FMT !== 'json') {
	throw new Error(`RL_CMD_FMT=${JSON.stringify(CMD_FMT)} unsupported; use bin|json`);
}
/**
 * Lean train meta (default on): omit actorMeta/targetMeta/intentResults from the
 * wire frame. Server still keeps session.meta for apply/scripted. Watch/debug:
 * RL_LEAN_META=0.
 */
const LEAN_META = process.env.RL_LEAN_META !== '0' && process.env.RL_LEAN_META !== 'false';
/** Rich post-tick economy snapshots are opt-in on the lean training path. */
const ECONOMY_TELEMETRY = process.env.RL_ECONOMY_TELEMETRY != null
	? process.env.RL_ECONOMY_TELEMETRY !== '0' && process.env.RL_ECONOMY_TELEMETRY !== 'false'
	: !LEAN_META;
const CAPTURE_EXPERT_INTENTS = process.env.RL_EXPERT_INTENTS === '1'
	|| process.env.RL_EXPERT_INTENTS === 'true';
let capturedExpertIntents = null;
/**
 * Event-trigger thresholds for start-state snapshots. `pre_spawn` uses the
 * cheapest useful worker body rather than the minimum legal one so the tag
 * marks a real composition decision, not a forced single-part spawn.
 */
const PRE_SPAWN_ENERGY = Number(process.env.RL_SNAPSHOT_PRE_SPAWN_ENERGY || 250);
const REPLACEMENT_TTL = Number(process.env.RL_SNAPSHOT_REPLACEMENT_TTL || 120);
const RECOVERY_AFTER_STEP = Number(process.env.RL_SNAPSHOT_RECOVERY_AFTER || 200);

// The scenario stream drives generated ids and movement tie-breaks, so a state
// snapshot that omitted it would replay a different world from the same tick.
// It is installed and seeded at module load: a zero state is a fixed point of
// this xorshift, so an unseeded window would make Math.random return 0 for every
// consumer that runs during import.
function scenarioRandomSeed(seed) {
	const state = (Number(seed) >>> 0) ^ 0x9e3779b9;
	return state === 0 ? 0x6d2b79f5 : state;
}

let scenarioRandomState = scenarioRandomSeed(SCENARIO_SEED);

// A divergent draw count is the cheapest observable symptom of environment
// nondeterminism: identical replicas must consume this stream identically.
let scenarioRandomDraws = 0;
Math.random = () => {
	++scenarioRandomDraws;
	scenarioRandomState ^= scenarioRandomState << 13;
	scenarioRandomState ^= scenarioRandomState >>> 17;
	scenarioRandomState ^= scenarioRandomState << 5;
	return (scenarioRandomState >>> 0) / 0x100000000;
};

/** Reset the dedicated environment process to a reproducible random stream. */
function resetScenarioRandom(seed) {
	scenarioRandomState = scenarioRandomSeed(seed);
}

function setScenarioRandomState(state) {
	const next = Number(state) >>> 0;
	scenarioRandomState = next === 0 ? 0x6d2b79f5 : next;
}

// PlayerInstance otherwise publishes and discards the exact payload generated by
// user code.  Capture it at the runner boundary before engine processing so TI
// samples can be translated without guessing from events or next-state deltas.
// The connector is inert unless explicitly requested by the expert collector.
if (CAPTURE_EXPERT_INTENTS) {
	runnerHooks.register('runnerConnector', player => [ undefined, {
		save(result) {
			if (player.userId === USER) capturedExpertIntents = result.intentPayloads || {};
		},
	} ]);
}

function energyIn(store) {
	return Number(store?.[C.RESOURCE_ENERGY] || 0);
}

function energyCapacity(store) {
	return Number(store?.getCapacity?.(C.RESOURCE_ENERGY) || 0);
}

// StructureController.reservation resolves the reserving username through the
// runtime user registry. Long synthetic episodes can retain a valid reservation
// expiry after that registry entry is gone, so telemetry must use authoritative
// controller state instead of invoking the presentation getter.
function controllerHasActiveReservation(Game, controller) {
	return Boolean(controller)
		&& Number(controller['#reservationEndTime'] || 0) > Number(Game.time || 0);
}

function isNeutralController(Game, controller) {
	return Boolean(controller)
		&& !controller.my
		&& Number(controller.level || 0) === 0
		&& !controllerHasActiveReservation(Game, controller);
}

function creepRole(creep) {
	const count = part => creep.body
		?.filter(bodyPart => bodyPart.type === part && bodyPart.hits > 0).length || 0;
	const work = count(C.WORK);
	const carry = count(C.CARRY);
	if (count(C.CLAIM) > 0) return 'claimer';
	if (count(C.HEAL) > 0) return 'healer';
	if (count(C.RANGED_ATTACK) > 0) return 'ranged';
	if (count(C.ATTACK) > 0) return 'melee';
	if (work > 0 && carry === 0) return 'miner';
	if (carry > 0 && work === 0) return 'hauler';
	if (work > 0 && carry > work) return 'builder';
	if (work > carry && carry > 0) return 'upgrader';
	if (work === 0 && carry === 0 && count(C.MOVE) > 0) return 'scout';
	return work > 0 || carry > 0 ? 'flexible' : 'unknown';
}

function addCount(record, key, amount = 1) {
	record[key] = (record[key] || 0) + amount;
}

function spawnFailureCategory(code) {
	switch (Number(code)) {
		case C.OK: return 'accepted';
		case C.ERR_BUSY: return 'busy';
		case C.ERR_NOT_ENOUGH_ENERGY: return 'notEnoughEnergy';
		case C.ERR_NAME_EXISTS: return 'nameExists';
		case C.ERR_INVALID_ARGS: return 'invalidArgs';
		case C.ERR_RCL_NOT_ENOUGH: return 'rclNotEnough';
		case C.ERR_NOT_OWNER: return 'notOwner';
		default: return 'other';
	}
}

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
 * flags: bit0=ok bit1=done bit2=has_obs bit3=has_action_tail
 *
 * The action tail carries the scripted baseline's chosen planes and is set only
 * by `step_scripted`. Behaviour-cloning labels never travel here: the expert's
 * own intents are captured inside the runner and returned as `expertIntents`
 * metadata by `step_expert`.
 */
function encodeScriptedActionPlanes(actions) {
	const scalarNames = [
		'types', 'dirs', 'targets', 'amounts', 'constructionTypes',
	];
	const scalarLimits = [
		SCHEMA.intentTypes.length, SCHEMA.directions.length, SCHEMA.maxTargets,
		SCHEMA.amountBins.length, SCHEMA.constructionTypes.length,
	];
	const rows = actions?.types?.length ?? 0;
	const slots = SCHEMA.intentSlots;
	if (rows < 1 || rows > SCHEMA.maxActors) {
		throw new Error(`scripted action rows=${rows}, expected [1, ${SCHEMA.maxActors}]`);
	}
	const cells = rows * slots;
	const scalarBytes = cells * scalarNames.length;
	const tileBytes = cells * 2;
	const bodyBytes = cells * SCHEMA.bodyPartTypes.length;
	const payload = Buffer.allocUnsafe(scalarBytes + tileBytes + bodyBytes * 2);
	for (let planeIndex = 0; planeIndex < scalarNames.length; planeIndex++) {
		const name = scalarNames[planeIndex];
		const plane = actions[name];
		if (!Array.isArray(plane) || plane.length !== rows) {
			throw new Error(`scripted actions.${name} rows differ from types`);
		}
		for (let actor = 0; actor < rows; actor++) {
			if (!Array.isArray(plane[actor]) || plane[actor].length !== slots) {
				throw new Error(`scripted actions.${name}[${actor}] slots differ from schema`);
			}
			for (let slot = 0; slot < slots; slot++) {
				const value = Number(plane[actor][slot]);
				if (!Number.isInteger(value) || value < 0 || value >= scalarLimits[planeIndex]) {
					throw new Error(`scripted actions.${name}[${actor}][${slot}]=${value} out of range`);
				}
				payload[planeIndex * cells + actor * slots + slot] = value;
			}
		}
	}
	const tiles = actions.constructionTiles;
	const counts = actions.bodyCounts;
	const order = actions.bodyOrder;
	for (let actor = 0; actor < rows; actor++) {
		if (!Array.isArray(tiles?.[actor]) || tiles[actor].length !== slots
			|| !Array.isArray(counts?.[actor]) || counts[actor].length !== slots
			|| !Array.isArray(order?.[actor]) || order[actor].length !== slots) {
			throw new Error(`scripted vector action shape differs at actor ${actor}`);
		}
		for (let slot = 0; slot < slots; slot++) {
			const cell = actor * slots + slot;
			const tile = Number(tiles[actor][slot]);
			if (!Number.isInteger(tile) || tile < 0 || tile >= SCHEMA.roomSize * SCHEMA.roomSize) {
				throw new Error(`scripted construction tile ${tile} out of range`);
			}
			payload.writeUInt16LE(tile, scalarBytes + cell * 2);
				if (counts[actor][slot]?.length !== SCHEMA.bodyPartTypes.length
					|| order[actor][slot]?.length !== SCHEMA.bodyPartTypes.length) {
					throw new Error(`scripted body vector length differs at actor ${actor} slot ${slot}`);
				}
				if (!decodeSpawnBodyForValidation(
					counts[actor][slot], order[actor][slot],
					actions.types[actor][slot] === SCHEMA.intentTypes.indexOf('spawnCreep'),
				)) {
					throw new Error(`invalid scripted body counts/order at actor ${actor} slot ${slot}`);
				}
				for (let type = 0; type < SCHEMA.bodyPartTypes.length; type++) {
				const count = Number(counts[actor][slot][type]);
				const orderedType = Number(order[actor][slot][type]);
				if (!Number.isInteger(count) || count < 0 || count > SCHEMA.maxBodyParts) {
					throw new Error(`scripted body count ${count} out of range`);
				}
				if (!Number.isInteger(orderedType)
					|| orderedType < 0 || orderedType >= SCHEMA.bodyPartTypes.length) {
					throw new Error(`scripted body order ${orderedType} out of range`);
				}
				payload[scalarBytes + tileBytes + cell * SCHEMA.bodyPartTypes.length + type] = count;
				payload[scalarBytes + tileBytes + bodyBytes
					+ cell * SCHEMA.bodyPartTypes.length + type] = orderedType;
			}
		}
	}
	return { payload, rows, slots };
}

function writeBinFrame(obj) {
	const ok = !!obj.ok;
	const obs = obj.obs;
	const hasObs = ok && obs != null && (obs._raw != null || obs.encoding === 'bin');
	// Strip undefined info keys so JSON is smaller.
	const rawInfo = obj.info ?? {};
	const info = {};
	for (const k of Object.keys(rawInfo)) {
		// Scripted teacher labels use compact binary response planes. Keeping the
		// nested arrays in JSON made Python parse and repad ~2,300 scalar values per
		// environment tick under the GIL.
		if (k !== 'actions' && rawInfo[k] !== undefined) info[k] = rawInfo[k];
	}
	const teacher = rawInfo.actions == null ? null : encodeScriptedActionPlanes(rawInfo.actions);
	const meta = {
		ok,
		reward: obj.reward ?? 0,
		done: !!obj.done,
		error: obj.error,
		info,
		schema: obj.schema,
		teacherActions: teacher == null ? undefined : {
			rows: teacher.rows,
			slots: teacher.slots,
			byteLength: teacher.payload.byteLength,
		},
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
				bufFromTyped(raw.actorOutcome),
				bufFromTyped(raw.targetMask),
				bufFromTyped(raw.intentMask),
				bufFromTyped(raw.dirMask),
				bufFromTyped(raw.targetSelectMask),
				bufFromTyped(raw.amountMask),
				bufFromTyped(raw.constructionMask),
			];
		}
	}
	if (teacher != null) {
		parts = parts == null ? [ teacher.payload ] : [ ...parts, teacher.payload ];
	}
	blobLen = parts == null ? 0 : parts.reduce((n, p) => n + p.byteLength, 0);
	const metaBuf = Buffer.from(JSON.stringify(meta), 'utf8');
	const total = 16 + metaBuf.length + blobLen;
	const out = Buffer.allocUnsafe(total);
	out.write('XRL1', 0, 4, 'ascii');
	out[4] = 4; // response payload version: compact scripted-action tail
	out[5] = (ok ? 1 : 0) | (obj.done ? 2 : 0) | (hasObs ? 4 : 0)
		| (teacher != null ? 8 : 0);
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

function freeTileNear(room, pos, maxR = 3, reserved = null) {
	for (let r = 1; r <= maxR; r++) {
		for (let dx = -r; dx <= r; dx++) {
			for (let dy = -r; dy <= r; dy++) {
				if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
				const x = pos.x + dx;
				const y = pos.y + dy;
				if (x < 2 || x > 47 || y < 2 || y > 47) continue;
				if (reserved?.has(`${x},${y}`)) continue;
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

function outpostStagingTile(room) {
	const targetCoord = parseRoomName(EXPANSION_ROOM);
	const dx = Math.sign(targetCoord.rx - roomCoord.rx);
	const dy = Math.sign(targetCoord.ry - roomCoord.ry);
	const offsets = Array.from({ length: 48 }, (_, index) => index + 1)
		.sort((a, b) => Math.abs(a - 25) - Math.abs(b - 25));
	for (const offset of offsets) {
		const x = dx < 0 ? 1 : dx > 0 ? 48 : offset;
		const y = dy < 0 ? 1 : dy > 0 ? 48 : offset;
		const borderX = dx < 0 ? 0 : dx > 0 ? 49 : offset;
		const borderY = dy < 0 ? 0 : dy > 0 ? 49 : offset;
		if (room.getTerrain().get(borderX, borderY) !== C.TERRAIN_MASK_WALL
			&& room.getTerrain().get(x, y) !== C.TERRAIN_MASK_WALL) {
			const tile = freeTileNear(room, new RoomPosition(x, y, ROOM), 2);
			if (tile) return tile;
		}
	}
	return null;
}

function placeSpawn(room) {
	room['#user'] = USER;
	room.controller['#user'] = USER;
	room['#level'] = 1;
	let sources = room.find(C.FIND_SOURCES);
	// Optional scenario diversity can add deterministic source lanes. The default
	// preserves the imported room's ordinary two-source Screeps economy.
	const sourceReserved = new Set(sources.map(source => `${source.pos.x},${source.pos.y}`));
	for (let index = sources.length; index < (SOURCE_COUNT ?? sources.length); index++) {
		const anchor = room.controller?.pos || new RoomPosition(25, 25, ROOM);
		const tile = freeTileNear(room, anchor, 18, sourceReserved);
		if (!tile) throw new Error(`no source tile for configured source count ${SOURCE_COUNT}`);
		sourceReserved.add(`${tile.x},${tile.y}`);
		const source = createRoomObject(new Source(), tile);
		source.energy = source.energyCapacity = C.SOURCE_ENERGY_NEUTRAL_CAPACITY;
		room['#insertObject'](source);
	}
	// Stable scenario geometry: object IDs are randomized by the engine and must
	// not decide which source anchors the only initial spawn.
	sources = room.find(C.FIND_SOURCES).sort((a, b) =>
		b.pos.x - a.pos.x || a.pos.y - b.pos.y || a.id.localeCompare(b.id));
	const src = sources[0];
	let placedSpawn = null;
	if (src) {
		const tile = freeTileNear(room, src.pos, 3);
		if (tile) {
			placedSpawn = createSpawn(tile, USER, 'Spawn1');
			room['#insertObject'](placedSpawn);
		}
	}
	if (!placedSpawn) {
		placedSpawn = createSpawn(new RoomPosition(25, 25, ROOM), USER, 'Spawn1');
		room['#insertObject'](placedSpawn);
	}
	if (CURRICULUM === 'seed_outpost') {
		const containerTile = outpostStagingTile(room);
		if (!containerTile) throw new Error('curriculum seed_outpost has no home-sink tile');
		room['#insertObject'](createContainer(containerTile));
	}

	// Spawn-composition pretraining uses real engine structures and energy. It
	// varies the decision context without bypassing spawn legality or execution.
	const spawnCurriculum = /^spawn_(?:flexible|miner|hauler|builder|upgrader|claimer)_(\d+)$/.exec(
		CURRICULUM,
	);
	if (spawnCurriculum) {
		const archetype = CURRICULUM.split('_')[1];
		const requestedCapacity = Number(spawnCurriculum[1]);
		const level = requestedCapacity <= 550 ? 2
			: requestedCapacity <= 800 ? 3
				: requestedCapacity <= 1300 ? 4 : 8;
		room['#level'] = level;
		const spawn = placedSpawn;
		spawn.store['#add'](
			C.RESOURCE_ENERGY,
			spawn.store.getFreeCapacity(C.RESOURCE_ENERGY),
		);
		const reserved = new Set([ `${spawn.pos.x},${spawn.pos.y}` ]);
		let remaining = Math.max(0, requestedCapacity - 300);
		while (remaining > 0) {
			const tile = freeTileNear(room, spawn.pos, 12, reserved);
			if (!tile) throw new Error(`no extension tile for ${CURRICULUM}`);
			reserved.add(`${tile.x},${tile.y}`);
			const extension = createExtension(tile, level, USER);
			const fill = Math.min(
				remaining,
				extension.store.getCapacity(C.RESOURCE_ENERGY),
			);
			extension.store['#add'](C.RESOURCE_ENERGY, fill);
			room['#insertObject'](extension);
			remaining -= fill;
		}

		// Body-composition labels must follow observable demand, never a hidden
		// curriculum tag. Builder and claimer both cost 650; without a real site
		// their initial observations are permutation-equivalent but their labels
		// are mutually exclusive.
		if (archetype === 'builder') {
			const tile = freeTileNear(room, spawn.pos, 12, reserved);
			if (!tile) throw new Error(`no construction-site tile for ${CURRICULUM}`);
			room['#insertObject'](createConstructionSite(
				tile,
				C.STRUCTURE_EXTENSION,
				USER,
				C.CONSTRUCTION_COST[C.STRUCTURE_EXTENSION],
			));
		}

		const seedCreep = (body, name, anchor = src?.pos || spawn.pos) => {
			const tile = freeTileNear(room, anchor, 12, reserved);
			if (!tile) throw new Error(`no context-creep tile for ${CURRICULUM}`);
			reserved.add(`${tile.x},${tile.y}`);
			const creep = createCreep(tile, body, name, USER);
			room['#insertObject'](creep);
			return creep;
		};
		if (archetype === 'miner') {
			// Transport exists but source extraction does not.
			seedCreep([ C.CARRY, C.CARRY, C.MOVE, C.MOVE ], 'context_carrier');
		} else if (archetype === 'hauler') {
			// Extraction exists and has produced a visible logistics backlog.
			seedCreep([ C.WORK, C.WORK, C.WORK, C.WORK, C.MOVE ], 'context_miner');
			const pileTile = freeTileNear(room, src?.pos || spawn.pos, 12, reserved);
			if (!pileTile) throw new Error(`no dropped-energy tile for ${CURRICULUM}`);
			room['#insertObject'](createResource(pileTile, C.RESOURCE_ENERGY, 2000));
		} else if (archetype === 'upgrader' || archetype === 'claimer') {
			// Mining and transport are covered; controller work is the missing lane.
			seedCreep([ C.WORK, C.WORK, C.WORK, C.WORK, C.MOVE ], 'context_miner');
			seedCreep([ C.CARRY, C.CARRY, C.MOVE, C.MOVE ], 'context_carrier');
		}
	}

	// simulate() creates a neutral room before invoking this initializer. Directly
	// changing #user/#level is not enough: sources and level-sensitive structures
	// cache room-status-dependent capacities. Mirror the controller processor's
	// status notification after every initial object is present. A newly created
	// owned room starts with full sources; preserve partial energy if a future
	// curriculum deliberately mutates a source before this point.
	for (const object of room['#immediateObjects']()) {
		const wasFullSource = object.structureType == null
			&& typeof object.energy === 'number'
			&& typeof object.energyCapacity === 'number'
			&& object.energy === object.energyCapacity;
		object['#roomStatusDidChange'](room['#level'], USER);
		if (wasFullSource && typeof object.energyCapacity === 'number') {
			object.energy = object.energyCapacity;
		}
	}

	// Curriculum: seed a bootstrap worker so BC/PPO can train logistics without first-spawn lottery.
	if (CURRICULUM === 'seed_creep' || CURRICULUM === 'seed_full'
		|| CURRICULUM === 'seed_claimer' || CURRICULUM === 'seed_outpost') {
		const seedCount = CURRICULUM === 'seed_outpost' ? 2 : 1;
		for (let seedIndex = 0; seedIndex < seedCount; seedIndex++) {
			const staging = CURRICULUM === 'seed_outpost' && seedIndex === 0
				? outpostStagingTile(room) : null;
			const seat = staging || (src
				? freeTileNear(room, src.pos, 3)
				: freeTileNear(room, new RoomPosition(25, 25, ROOM), 3));
			if (!seat) throw new Error(`curriculum ${CURRICULUM} has no seed-creep tile`);
			const body = CURRICULUM === 'seed_claimer'
				? [ C.CLAIM, C.MOVE ]
				: [ C.WORK, C.CARRY, C.MOVE, C.MOVE ];
			const name = CURRICULUM === 'seed_claimer'
				? 'seed_claimer'
				: seedIndex === 0 ? 'seed_worker' : 'seed_outpost';
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
// Restores rebuild every room from the snapshot, so the scenario initializers
// must not also seed a spawn or creep into the fresh shard.
const restoreFactory = simulate({});

/** @type {any} */
let session = null;

async function bootShard({
	expert = false, botDir = DEFAULT_EXPERT_BOT, restore = null,
} = {}) {
	// Tear down prior headful client first so :21025 can rebind on reset.
	if (session?.headful?.stop) {
		try { await session.headful.stop(); } catch { /* */ }
	}
	if (session?.close) {
		try { session.console?.(); } catch { /* */ }
		try { session.instance?.disconnect?.(); } catch { /* */ }
		await session.close();
	}
	session = null;
	// The executor's route cache is per-process operational state, never part of a
	// snapshot. Every fresh episode and every restored segment starts it cold.
	resetNavigationCaches();
	if (!restore) {
		// Each reset is a complete episode. Rewinding the per-process stream makes
		// movement tie-breaks, generated IDs, fresh runs and resumed runs replay the
		// same declared scenario rather than qualifying by a lucky dense-traffic draw.
		// A restore installs the captured stream state instead, but only after shard
		// instantiation has finished consuming randomness for users and ids.
		resetScenarioRandom(SCENARIO_SEED);
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
		lastOwnedRooms: 1,
		remoteCargoByCreep: new Map(),
		remoteRoomsStaffedPeak: 0,
		remoteProductiveCreepsPeak: 0,
		remoteOwnedRoomsPeak: 0,
		meta: null,
		actorOutcomes: new Map(),
		economyPrimed: false,
		lastEconomyCreeps: new Map(),
		lastConstructionSites: new Set(),
		expert: Boolean(expert),
		// A restored world belongs to the scenario that produced it, not to this
		// process's `RL_CURRICULUM`. Reservoir lanes restore snapshots from several
		// curricula into one process, so every per-tick decision that depends on the
		// scenario reads this field instead of the module constant.
		curriculum: restore ? String(restore.meta?.curriculum ?? CURRICULUM) : CURRICULUM,
		instance: null,
		botDir,
		headful: null,
	};

	let releaseBody;
	const bodyDone = new Promise(r => { releaseBody = r; });

	const simPromise = (restore ? restoreFactory : factory)(async refs => {
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
		// A teacher that throws inside its own tick publishes no intents and reports
		// only on this channel, so leaving it unread turns a broken expert into a
		// silently empty demonstration. Faults always reach stderr; the bot's own
		// logging is voluminous and needs RL_BOT_CONSOLE=1.
		handle.console = await getConsoleChannel(handle.shard, USER).listen(message => {
			for (const line of JSON.parse(message)) {
				const text = String(line.data);
				if (line.fd === 2 || /error|exception/i.test(text) || BOT_CONSOLE) {
					process.stderr.write(`[bot fd${line.fd}] ${text.slice(0, 400)}\n`);
				}
			}
		});
		console.error(`[rl-env] expert bot loaded from ${botDir}`);
	}

	if (restore) {
		await applyShardState(handle.shard, restore.meta.world, restore.blobs);
		// The observation joins previous-action outcomes by entity identity, so
		// they belong to the restored state and must precede encoding.
		handle.actorOutcomes = new Map(restore.meta.session.actorOutcomes);
		session = handle;
		const restored = await encodeAfterTick();
		applySessionSnapshot(handle, restore.meta.session);
		// Last: the next simulated tick must consume exactly the stream the
		// snapshot tick left behind, so shard boot and encoding cannot shift it.
		setScenarioRandomState(restore.meta.randomState);
		return {
			ok: true,
			obs: restored,
			info: {
				room: ROOM,
				seed: SCENARIO_SEED,
				// The restored world's own scenario, so a reservoir lane can attribute
				// its segment to the stage that produced the state, not to this
				// process's `RL_CURRICULUM`.
				curriculum: handle.curriculum,
				time: restored.time,
				step: handle.step,
				expert: handle.expert,
				botDir: handle.botDir,
				headful: false,
				headfulUrl: null,
				restored: true,
				snapshotTick: restore.meta.world.time,
				// Overflow flags gate whether a start state is representable at
				// all, so every boot reply reports them without an extra step.
				globals: restored.globals,
				// The claim budget is scratch state, not room state: reporting it on
				// every boot is what makes a restored expansion budget auditable.
				controlledRooms: await handle.shard.scratch.sCard(
					controlledRoomsKey(USER),
				),
			},
		};
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
			seed: SCENARIO_SEED,
			curriculum: handle.curriculum,
			time: obs.time,
			step: 0,
			expert: handle.expert,
			botDir: handle.botDir,
			headful: Boolean(handle.headful),
			headfulUrl: handle.headful?.url || null,
			globals: obs.globals,
			controlledRooms: await handle.shard.scratch.sCard(controlledRoomsKey(USER)),
		},
	};
}

/**
 * Session bookkeeping that is not engine state: reward baselines, remote-cargo
 * attribution, previous-action outcomes, and economy priming. Restoring it is
 * what makes the first post-restore reward a true delta instead of a spike.
 */
function captureSessionSnapshot() {
	return {
		step: session.step,
		lastControl: session.lastControl ?? 0,
		lastRcl: session.lastRcl ?? 1,
		lastCreeps: session.lastCreeps ?? 0,
		lastSites: session.lastSites ?? 0,
		lastOwnedRooms: session.lastOwnedRooms ?? 1,
		remoteCargoByCreep: [ ...session.remoteCargoByCreep ],
		remoteRoomsStaffedPeak: session.remoteRoomsStaffedPeak,
		remoteProductiveCreepsPeak: session.remoteProductiveCreepsPeak,
		remoteOwnedRoomsPeak: session.remoteOwnedRoomsPeak,
		actorOutcomes: [ ...session.actorOutcomes ],
		economyPrimed: Boolean(session.economyPrimed),
		lastEconomyCreeps: [ ...session.lastEconomyCreeps ],
		lastConstructionSites: [ ...session.lastConstructionSites ],
	};
}

function applySessionSnapshot(handle, state) {
	handle.step = Number(state.step) || 0;
	handle.lastControl = Number(state.lastControl) || 0;
	handle.lastRcl = Number(state.lastRcl) || 1;
	handle.lastCreeps = Number(state.lastCreeps) || 0;
	handle.lastSites = Number(state.lastSites) || 0;
	handle.lastOwnedRooms = Number(state.lastOwnedRooms) || 0;
	handle.remoteCargoByCreep = new Map(state.remoteCargoByCreep || []);
	handle.remoteRoomsStaffedPeak = Number(state.remoteRoomsStaffedPeak) || 0;
	handle.remoteProductiveCreepsPeak = Number(state.remoteProductiveCreepsPeak) || 0;
	handle.remoteOwnedRoomsPeak = Number(state.remoteOwnedRoomsPeak) || 0;
	handle.actorOutcomes = new Map(state.actorOutcomes || []);
	handle.economyPrimed = Boolean(state.economyPrimed);
	handle.lastEconomyCreeps = new Map(state.lastEconomyCreeps || []);
	handle.lastConstructionSites = new Set(state.lastConstructionSites || []);
}

/**
 * A snapshot is only interchangeable between processes that share terrain,
 * room geometry, and the observation/action ABI. Mismatches must fail loudly
 * rather than silently training on a different world.
 */
function snapshotWorldIdentity() {
	return {
		room: ROOM,
		expansionRoom: EXPANSION_ROOM,
		sourceCount: SOURCE_COUNT,
		schemaVersion: Number(SCHEMA.version),
		schemaObservation: SCHEMA.artifact?.observationAbi ?? null,
		schemaAction: SCHEMA.artifact?.actionAbi ?? null,
	};
}

function assertSnapshotCompatible(meta) {
	const current = snapshotWorldIdentity();
	for (const [ key, value ] of Object.entries(current)) {
		const other = meta.world_identity?.[key] ?? null;
		if (JSON.stringify(other) !== JSON.stringify(value ?? null)) {
			throw new Error(
				`snapshot ${key}=${JSON.stringify(other)} does not match environment `
				+ `${JSON.stringify(value ?? null)}`,
			);
		}
	}
}

async function writeSnapshot(filePath, { events = [] } = {}) {
	if (!session) throw new Error('call reset first');
	if (typeof filePath !== 'string' || !filePath) {
		throw new Error('snapshot requires a destination path');
	}
	const world = await captureShardState(session.shard, {
		extraRooms: [ ROOM, EXPANSION_ROOM ],
		users: [ USER ],
	});
	const blobs = [];
	let offset = 0;
	const rooms = world.rooms.map(room => {
		blobs.push(room.blob);
		const entry = { name: room.name, offset, length: room.blob.byteLength };
		offset += room.blob.byteLength;
		return entry;
	});
	const meta = {
		schemaVersion: Number(SCHEMA.version),
		world_identity: snapshotWorldIdentity(),
		curriculum: session.curriculum,
		seed: SCENARIO_SEED,
		expert: Boolean(session.expert),
		randomState: scenarioRandomState >>> 0,
		events: [ ...events ],
		rooms,
		world: {
			time: world.time,
			rooms: rooms.map(room => ({ name: room.name })),
			scratch: world.scratch,
		},
		session: captureSessionSnapshot(),
	};
	const payload = encodeSnapshotFile(meta, blobs);
	await fs.mkdir(path.dirname(filePath), { recursive: true });
	const temporary = `${filePath}.tmp`;
	await fs.writeFile(temporary, payload);
	await fs.rename(temporary, filePath);
	return {
		ok: true,
		info: {
			snapshot: {
				path: filePath,
				bytes: payload.byteLength,
				tick: world.time,
				step: session.step,
				rooms: rooms.map(room => room.name),
				curriculum: session.curriculum,
				expert: Boolean(session.expert),
				events: [ ...events ],
			},
		},
	};
}

async function restoreSnapshot(filePath) {
	if (typeof filePath !== 'string' || !filePath) {
		throw new Error('restore requires a snapshot path');
	}
	const payload = await fs.readFile(filePath);
	const { meta, blobs } = decodeSnapshotFile(payload);
	assertSnapshotCompatible(meta);
	// Snapshot rooms carry only names in `world`; the blob map is authoritative.
	meta.world.rooms = meta.rooms.map(room => ({ name: room.name }));
	return bootShard({ expert: false, restore: { meta, blobs } });
}

/**
 * Detailed post-tick state for long-horizon economy diagnosis. This is kept off
 * the lean training wire unless RL_ECONOMY_TELEMETRY=1.
 */
function collectEconomyTelemetry(Game, observation, intentResults = null) {
	const primed = session.economyPrimed;
	const currentCreeps = new Map();
	for (const creep of Object.values(Game.creeps)) {
		if (creep?.my) currentCreeps.set(creep.name, Boolean(creep.spawning));
	}
	const startedNames = primed
		? [ ...currentCreeps ].filter(([ name, spawning ]) => (
			spawning && !session.lastEconomyCreeps.has(name)
		)).map(([ name ]) => name)
		: [];
	const completedNames = primed
		? [ ...currentCreeps ].filter(([ name, spawning ]) => (
			!spawning && session.lastEconomyCreeps.get(name) === true
		)).map(([ name ]) => name)
		: [];
	const cancelledNames = primed
		? [ ...session.lastEconomyCreeps ].filter(([ name, spawning ]) => (
			spawning && !currentCreeps.has(name)
		)).map(([ name ]) => name)
		: [];

	const hasIntentResults = Array.isArray(intentResults);
	const spawnResults = hasIntentResults
		? intentResults.filter(result => result.type === 'spawnCreep') : [];
	const spawnOutcomes = {};
	for (const result of spawnResults) addCount(spawnOutcomes, spawnFailureCategory(result.code));
	const accepted = spawnOutcomes.accepted || 0;
	const spawnFailures = { ...spawnOutcomes };
	delete spawnFailures.accepted;

	const rooms = {};
	const currentSites = new Set();
	const totals = {
		ownedRooms: 0,
		neutralOutpostRooms: 0,
		remoteOwnedRooms: 0,
		remoteRoomsStaffed: 0,
		remoteEconomyRoomsStaffed: 0,
		remoteCreeps: 0,
		remoteProductiveCreeps: 0,
		creeps: {
			total: 0, active: 0, spawning: 0, productive: 0,
			carriedEnergy: 0, byRole: {},
		},
		droppedEnergy: 0,
		sources: {
			count: 0, remaining: 0, capacity: 0, harvestedThisTick: 0,
			sustainableRate: 0, utilization: 0, depleted: 0,
		},
		sinks: {
			energy: 0, capacity: 0, free: 0, starvation: 0,
			starvedStructures: 0, spawnExtensionFree: 0, towerFree: 0,
			spawnExtensionFillFraction: 1, towerFillFraction: 1, byType: {},
		},
		construction: {
			sites: 0, progress: 0, progressTotal: 0, remaining: 0,
			createdThisTick: 0, completedThisTick: 0, disappearedThisTick: 0,
			buildProgressThisTick: 0, byType: {},
		},
		controller: { rclMax: 0, progress: 0, progressTotal: 0 },
		// NPC invasions are generated from cumulative harvest, so a productive
		// episode meets an adversary. Losing a spawn ends the economy, and that
		// loss is invisible in sinks/sources: record it explicitly.
		hostility: { hostileCreeps: 0, attackDamage: 0, destroyed: {} },
	};

	for (const room of Object.values(Game.rooms)) {
		const events = room.getEventLog();
		const harvestedBySource = {};
		const destroyed = {};
		let buildProgressThisTick = 0;
		let completedThisTick = 0;
		let attackDamage = 0;
		for (const event of events) {
			const amount = Number(event.data?.amount ?? event.amount ?? 0);
			if (event.event === C.EVENT_HARVEST) {
				const sourceId = event.targetId ?? event.data?.targetId;
				if (sourceId) addCount(harvestedBySource, sourceId, amount);
			}
			if (event.event === C.EVENT_BUILD) {
				buildProgressThisTick += amount;
				if ((event.incomplete ?? event.data?.incomplete) === false) completedThisTick += 1;
			}
			if (event.event === C.EVENT_ATTACK) {
				attackDamage += Number(event.data?.damage ?? event.damage ?? 0);
			}
			if (event.event === C.EVENT_OBJECT_DESTROYED) {
				addCount(destroyed, String(event.data?.type ?? event.type ?? 'unknown'));
			}
		}

		const roomCreeps = room.find(C.FIND_MY_CREEPS);
		const byRole = {};
		let activeCreeps = 0;
		let spawningCreeps = 0;
		let productiveCreeps = 0;
		let carriedEnergy = 0;
		for (const creep of roomCreeps) {
			const role = creepRole(creep);
			addCount(byRole, role);
			addCount(totals.creeps.byRole, role);
			if (creep.spawning) spawningCreeps += 1;
			else {
				activeCreeps += 1;
				if (creep.body.some(part => part.hits > 0
					&& (part.type === C.WORK || part.type === C.CARRY))) productiveCreeps += 1;
			}
			carriedEnergy += energyIn(creep.store);
		}

		const sourceItems = room.find(C.FIND_SOURCES).map(source => {
			const capacity = Number(source.energyCapacity || 0);
			const remaining = Number(source.energy || 0);
			const harvestedThisTick = Number(harvestedBySource[source.id] || 0);
			const sustainableRate = capacity / Number(C.ENERGY_REGEN_TIME || 300);
			return {
				id: source.id,
				x: source.pos.x,
				y: source.pos.y,
				remaining,
				capacity,
				ticksToRegeneration: source.ticksToRegeneration ?? null,
				harvestedThisTick,
				sustainableRate,
				utilization: sustainableRate > 0 ? harvestedThisTick / sustainableRate : 0,
				depletedFraction: capacity > 0 ? (capacity - remaining) / capacity : 0,
			};
		});
		const sources = {
			count: sourceItems.length,
			remaining: sourceItems.reduce((sum, source) => sum + source.remaining, 0),
			capacity: sourceItems.reduce((sum, source) => sum + source.capacity, 0),
			harvestedThisTick: sourceItems.reduce(
				(sum, source) => sum + source.harvestedThisTick, 0,
			),
			sustainableRate: sourceItems.reduce((sum, source) => sum + source.sustainableRate, 0),
			depleted: sourceItems.filter(source => source.remaining === 0).length,
			items: sourceItems,
		};
		sources.utilization = sources.sustainableRate > 0
			? sources.harvestedThisTick / sources.sustainableRate : 0;

		let droppedEnergy = 0;
		for (const resource of room.find(C.FIND_DROPPED_RESOURCES)) {
			if (resource.resourceType === C.RESOURCE_ENERGY) droppedEnergy += Number(resource.amount || 0);
		}

		const sinks = {
			energy: 0, capacity: 0, free: 0, starvation: 0,
			starvedStructures: 0, spawnExtensionFree: 0, towerFree: 0,
			spawnExtensionFillFraction: 1, towerFillFraction: 1, byType: {},
		};
		for (const structure of room.find(C.FIND_STRUCTURES)) {
			// Public containers are valid economy buffers; hostile owned stores are not.
			if (structure.my === false) continue;
			const capacity = energyCapacity(structure.store);
			if (capacity <= 0) continue;
			const energy = energyIn(structure.store);
			const free = Math.max(0, capacity - energy);
			const type = structure.structureType || 'unknown';
			const byType = sinks.byType[type] || { count: 0, energy: 0, capacity: 0, free: 0 };
			byType.count += 1;
			byType.energy += energy;
			byType.capacity += capacity;
			byType.free += free;
			sinks.byType[type] = byType;
			sinks.energy += energy;
			sinks.capacity += capacity;
			sinks.free += free;
			if (
				structure.structureType === C.STRUCTURE_SPAWN
				|| structure.structureType === C.STRUCTURE_EXTENSION
			) {
				sinks.spawnExtensionFree += free;
				sinks.starvation += free;
				if (free > 0) sinks.starvedStructures += 1;
			} else if (structure.structureType === C.STRUCTURE_TOWER) {
				sinks.towerFree += free;
				sinks.starvation += free;
				if (free > 0) sinks.starvedStructures += 1;
			}
		}
		const spawnExtension = [ C.STRUCTURE_SPAWN, C.STRUCTURE_EXTENSION ]
			.flatMap(type => sinks.byType[type] ? [ sinks.byType[type] ] : []);
		const spawnExtensionCapacity = spawnExtension.reduce((sum, row) => sum + row.capacity, 0);
		const spawnExtensionEnergy = spawnExtension.reduce((sum, row) => sum + row.energy, 0);
		sinks.spawnExtensionFillFraction = spawnExtensionCapacity > 0
			? spawnExtensionEnergy / spawnExtensionCapacity : 1;
		const tower = sinks.byType[C.STRUCTURE_TOWER];
		sinks.towerFillFraction = tower?.capacity > 0 ? tower.energy / tower.capacity : 1;

		const constructionByType = {};
		let siteProgress = 0;
		let siteProgressTotal = 0;
		const sites = room.find(C.FIND_MY_CONSTRUCTION_SITES);
		let createdThisTick = 0;
		for (const site of sites) {
			currentSites.add(site.id);
			if (primed && !session.lastConstructionSites.has(site.id)) createdThisTick += 1;
			const type = site.structureType || 'unknown';
			const byType = constructionByType[type] || {
				sites: 0, progress: 0, progressTotal: 0, remaining: 0,
			};
			byType.sites += 1;
			byType.progress += Number(site.progress || 0);
			byType.progressTotal += Number(site.progressTotal || 0);
			byType.remaining += Math.max(0, Number(site.progressTotal || 0) - Number(site.progress || 0));
			constructionByType[type] = byType;
			siteProgress += Number(site.progress || 0);
			siteProgressTotal += Number(site.progressTotal || 0);
		}

		let reservation = null;
		try {
			reservation = room.controller?.reservation || null;
		} catch {
			// The engine can retain a reservation expiry for one ownership-transition
			// tick after clearing its reservation user. Telemetry must never make a
			// legal claim step fatal.
			reservation = null;
		}
		const rawReservationTicks = Math.max(
			0,
			Number(room.controller?.['#reservationEndTime'] || 0) - Number(Game.time || 0),
		);
		const controller = room.controller ? {
			my: Boolean(room.controller.my),
			level: Number(room.controller.level || 0),
			progress: Number(room.controller.progress || 0),
			progressTotal: Number(room.controller.progressTotal || 0),
			ticksToDowngrade: room.controller.ticksToDowngrade ?? null,
			reservationUser: reservation?.username ?? null,
			reservationTicks: reservation?.ticksToEnd ?? rawReservationTicks,
		} : null;
		const isHome = room.name === ROOM;
		const owned = Boolean(room.controller?.my);
		const neutralOutpost = !isHome && isNeutralController(Game, room.controller)
			&& sources.count > 0;
		// Totals describe the economy the policy currently operates, not every
		// source/sink in privileged visible rooms. A staffed outpost counts before
		// ownership; an empty neutral expansion room remains available per-room.
		const managedEconomy = owned || roomCreeps.length > 0;
		const construction = {
			sites: sites.length,
			progress: siteProgress,
			progressTotal: siteProgressTotal,
			remaining: Math.max(0, siteProgressTotal - siteProgress),
			createdThisTick,
			completedThisTick,
			buildProgressThisTick,
			byType: constructionByType,
		};
		const hostility = {
			hostileCreeps: room.find(C.FIND_HOSTILE_CREEPS).length,
			attackDamage,
			destroyed,
		};
		rooms[room.name] = {
			isHome,
			owned,
			neutralOutpost,
			controller,
			creeps: {
				total: roomCreeps.length,
				active: activeCreeps,
				spawning: spawningCreeps,
				productive: productiveCreeps,
				carriedEnergy,
				byRole,
			},
			droppedEnergy,
			sources,
			sinks,
			construction,
			hostility,
		};

		if (owned) totals.ownedRooms += 1;
		if (neutralOutpost) totals.neutralOutpostRooms += 1;
		if (!isHome && owned) totals.remoteOwnedRooms += 1;
		if (!isHome && activeCreeps > 0) totals.remoteRoomsStaffed += 1;
		if (!isHome && productiveCreeps > 0) totals.remoteEconomyRoomsStaffed += 1;
		if (!isHome) {
			totals.remoteCreeps += activeCreeps;
			totals.remoteProductiveCreeps += productiveCreeps;
		}
		totals.creeps.total += roomCreeps.length;
		totals.creeps.active += activeCreeps;
		totals.creeps.spawning += spawningCreeps;
		totals.creeps.productive += productiveCreeps;
		totals.creeps.carriedEnergy += carriedEnergy;
		totals.hostility.hostileCreeps += hostility.hostileCreeps;
		totals.hostility.attackDamage += attackDamage;
		for (const [ type, count ] of Object.entries(destroyed)) {
			addCount(totals.hostility.destroyed, type, count);
		}
		if (managedEconomy) {
			totals.droppedEnergy += droppedEnergy;
			for (const key of [ 'count', 'remaining', 'capacity', 'harvestedThisTick',
				'sustainableRate', 'depleted' ]) {
				totals.sources[key] += sources[key];
			}
			for (const key of [
				'energy', 'capacity', 'free', 'starvation', 'starvedStructures',
				'spawnExtensionFree', 'towerFree',
			]) {
				totals.sinks[key] += sinks[key];
			}
			for (const [ type, values ] of Object.entries(sinks.byType)) {
				const row = totals.sinks.byType[type]
					|| { count: 0, energy: 0, capacity: 0, free: 0 };
				for (const key of [ 'count', 'energy', 'capacity', 'free' ]) row[key] += values[key];
				totals.sinks.byType[type] = row;
			}
			for (const key of [ 'sites', 'progress', 'progressTotal', 'remaining',
				'createdThisTick', 'completedThisTick', 'buildProgressThisTick' ]) {
				totals.construction[key] += construction[key];
			}
			for (const [ type, values ] of Object.entries(construction.byType)) {
				const row = totals.construction.byType[type] || {
					sites: 0, progress: 0, progressTotal: 0, remaining: 0,
				};
				for (const key of [ 'sites', 'progress', 'progressTotal', 'remaining' ]) {
					row[key] += values[key];
				}
				totals.construction.byType[type] = row;
			}
		}
		if (controller?.my) {
			totals.controller.rclMax = Math.max(totals.controller.rclMax, controller.level);
			totals.controller.progress += controller.progress;
			totals.controller.progressTotal += controller.progressTotal;
		}
	}
	const totalSpawnExtension = [ C.STRUCTURE_SPAWN, C.STRUCTURE_EXTENSION ]
		.flatMap(type => totals.sinks.byType[type] ? [ totals.sinks.byType[type] ] : []);
	const totalSpawnExtensionCapacity = totalSpawnExtension.reduce(
		(sum, row) => sum + row.capacity, 0,
	);
	const totalSpawnExtensionEnergy = totalSpawnExtension.reduce(
		(sum, row) => sum + row.energy, 0,
	);
	totals.sinks.spawnExtensionFillFraction = totalSpawnExtensionCapacity > 0
		? totalSpawnExtensionEnergy / totalSpawnExtensionCapacity : 1;
	const totalTower = totals.sinks.byType[C.STRUCTURE_TOWER];
	totals.sinks.towerFillFraction = totalTower?.capacity > 0
		? totalTower.energy / totalTower.capacity : 1;

	if (primed) {
		let removedSites = 0;
		for (const siteId of session.lastConstructionSites) {
			if (!currentSites.has(siteId)) removedSites += 1;
		}
		totals.construction.disappearedThisTick = removedSites;
	}
	totals.sources.utilization = totals.sources.sustainableRate > 0
		? totals.sources.harvestedThisTick / totals.sources.sustainableRate : 0;
	session.lastEconomyCreeps = currentCreeps;
	session.lastConstructionSites = currentSites;
	session.economyPrimed = true;

	const globals = observation.globals || {};
	return {
		version: 1,
		time: Game.time,
		homeRoom: ROOM,
		rooms,
		totals,
		spawn: {
			intentMetricsAvailable: hasIntentResults,
			attempts: hasIntentResults ? spawnResults.length : null,
			accepted: hasIntentResults ? accepted : null,
			failures: hasIntentResults ? spawnFailures : null,
			outcomes: hasIntentResults ? spawnOutcomes : null,
			startedThisTick: startedNames.length,
			completedThisTick: completedNames.length,
			cancelledThisTick: cancelledNames.length,
			acceptedWithoutObservedStart: hasIntentResults
				? Math.max(0, accepted - startedNames.length) : null,
			startedNames,
			completedNames,
			cancelledNames,
		},
		overflow: {
			rooms: {
				overflow: Number(globals.roomOverflow || 0),
				count: Number(globals.visibleRooms || 0),
				limit: Number(SCHEMA.maxRooms),
			},
			actors: {
				overflow: Number(globals.actorOverflow || 0),
				count: Number(globals.actorCount || 0),
				limit: Number(SCHEMA.maxActors),
			},
			targets: {
				overflow: Number(globals.targetOverflow || 0),
				count: Number(globals.targetCount || 0),
				limit: Number(SCHEMA.maxTargets),
			},
		},
	};
}

/**
 * Post-tick: ONE room load + ONE runForUser for encode + H+C reward (SPS win).
 * `player()` is reserved for intents; this path is read-only.
 */
async function encodeAndRewardAfterTick(intentResults = []) {
	const visibleRooms = await session.shard.scratch.sMembers(userToVisibleRoomsSetKey(USER));
	const roomNames = visibleRooms.length ? visibleRooms : [ ROOM ];
	// The expansion curriculum exposes the connected neutral neighbor as a known
	// strategic room. Primitive intents still run through the player's normal
	// visible-room state; actions.mjs uses positional proxies until a creep scouts it.
	if (!roomNames.includes(ROOM)) roomNames.push(ROOM);
	const spawnArchetype = /^spawn_([^_]+)_/.exec(session.curriculum)?.[1] || null;
	const exposeExpansion = spawnArchetype == null || spawnArchetype === 'claimer';
	if (exposeExpansion && !roomNames.includes(EXPANSION_ROOM)) roomNames.push(EXPANSION_ROOM);
	const rooms = await Promise.all(roomNames.map(n => session.shard.loadRoom(n)));
	const state = new GameState(session.world, session.shard.time, rooms);

	let obs = null;
	let harvestDelta = 0;
	let remoteHarvestDelta = 0;
	let remoteHomeDeliveryDelta = 0;
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
	let neutralOutpostRooms = 0;
	let remoteOwnedRooms = 0;
	let remoteRoomsStaffed = 0;
	let remoteProductiveCreeps = 0;
	let economy = null;
	// Event-stratified snapshot triggers. Pre-decision tags describe a state in
	// which a strategic choice is open, so a policy restored there must make the
	// decision itself instead of inheriting the teacher's.
	const eventFlags = Object.create(null);
	let homeSpawnIdle = false;
	let homeEnergyAvailable = 0;
	let claimerReady = false;

	runForUser(USER, state, Game => {
		installSimulationGcl(Game);
		obs = encodeObservation(Game, USER, session.actorOutcomes);
		controlNow = 0;
		rclNow = 0;
		for (const room of Object.values(Game.rooms)) {
			const isHome = room.name === ROOM;
			// One cached lookup shared by the outpost check and the remote event
			// tags; the home room never needs it.
			let remoteSources = null;
			const sourcesOf = () => {
				if (remoteSources === null) remoteSources = room.find(C.FIND_SOURCES);
				return remoteSources;
			};
			if (!isHome && isNeutralController(Game, room.controller)
				&& sourcesOf().length > 0) neutralOutpostRooms += 1;
			const productiveSinkIds = new Set();
			const bankIds = new Set();
			const towerIds = new Set();
			for (const s of room.find(C.FIND_STRUCTURES)) {
				const owned = s.my !== false;
				if (
					owned && (
						s.structureType === C.STRUCTURE_SPAWN
						|| s.structureType === C.STRUCTURE_EXTENSION
						|| s.structureType === C.STRUCTURE_TOWER
					)
				) {
					productiveSinkIds.add(s.id);
				}
				if ((owned && s.structureType === C.STRUCTURE_STORAGE)
					|| s.structureType === C.STRUCTURE_CONTAINER) {
					bankIds.add(s.id);
				}
				if (owned && s.structureType === C.STRUCTURE_TOWER) towerIds.add(s.id);
			}
			for (const ev of room.getEventLog()) {
				const amt = ev.data?.amount ?? ev.amount ?? 0;
				const oid = ev.objectId ?? ev.data?.objectId;
				if (ev.event === C.EVENT_HARVEST) {
					harvestDelta += amt;
					if (!isHome) {
						remoteHarvestDelta += amt;
						if (oid) {
							session.remoteCargoByCreep.set(
								oid, (session.remoteCargoByCreep.get(oid) || 0) + amt,
							);
						}
					}
				}
				if (ev.event === C.EVENT_UPGRADE_CONTROLLER) upgradeDelta += amt;
				if (ev.event === C.EVENT_BUILD) buildDelta += amt;
				if (ev.event === C.EVENT_TRANSFER) {
					const res = ev.resourceType ?? ev.data?.resourceType;
					if (res != null && res !== C.RESOURCE_ENERGY) continue;
					const tid = ev.targetId ?? ev.data?.targetId;
					if (tid && productiveSinkIds.has(tid)) transferToSpawn += amt;
					const remoteCargo = oid ? session.remoteCargoByCreep.get(oid) || 0 : 0;
					if (remoteCargo > 0) {
					if (isHome && tid && (productiveSinkIds.has(tid) || bankIds.has(tid))) {
							remoteHomeDeliveryDelta += Math.min(remoteCargo, amt);
						}
						const remaining = Math.max(0, remoteCargo - amt);
						if (remaining > 0) session.remoteCargoByCreep.set(oid, remaining);
						else session.remoteCargoByCreep.delete(oid);
					}
					if (tid && bankIds.has(tid)) advancedDepositDelta += amt;
					if (oid && bankIds.has(oid)) advancedWithdrawDelta += amt;
					if (tid && towerIds.has(tid)) towerRefillDelta += amt;
				}
			}
			if (room.controller?.my) {
				ownedRooms += 1;
				if (!isHome) remoteOwnedRooms += 1;
				controlNow += room.controller.progress || 0;
				rclNow = Math.max(rclNow, room.controller.level || 0);
			}
			const roomCreeps = room.find(C.FIND_MY_CREEPS);
			creeps += roomCreeps.length;
			for (const creep of roomCreeps) {
				if (creep.spawning) continue;
				const ttl = creep.ticksToLive;
				if (typeof ttl === 'number' && ttl > 0 && ttl <= REPLACEMENT_TTL) {
					eventFlags.replacement_due = true;
				}
				const carried = energyIn(creep.store);
				const live = part => part.hits > 0;
				if (!isHome) {
					// An outbound hauler needs cargo space; only a WORK creep can be
					// at a source for extraction. A claimer or scout passing through
					// is not a remote-economy decision state.
					if (carried === 0 && creep.body.some(part =>
						live(part) && part.type === C.CARRY)) {
						eventFlags.remote_outbound = true;
					}
					if (creep.body.some(part => live(part) && part.type === C.WORK)
						&& sourcesOf().some(source => creep.pos.inRangeTo(source.pos, 1))) {
						eventFlags.remote_at_source = true;
					}
				} else if (carried > 0 && session.remoteCargoByCreep.has(creep.id)) {
					// The delivery decision exists once remotely mined cargo is home;
					// while the creep is still away this is the outbound half of the
					// same round trip.
					eventFlags.remote_loaded_home = true;
				}
				if (!claimerReady && creep.body.some(part =>
					live(part) && part.type === C.CLAIM)) claimerReady = true;
			}
			if (!isHome) {
				const active = roomCreeps.filter(creep => !creep.spawning);
				if (active.length) remoteRoomsStaffed += 1;
				remoteProductiveCreeps += active.filter(creep => creep.body.some(part =>
					part.hits > 0 && (part.type === C.WORK || part.type === C.CARRY))).length;
			}
			sitesNow += room.find(C.FIND_MY_CONSTRUCTION_SITES).length;
			for (const s of room.find(C.FIND_MY_SPAWNS)) {
				const stored = s.store?.[C.RESOURCE_ENERGY] || 0;
				spawnEnergy += stored;
				energyAvailable += stored;
				if (isHome) homeEnergyAvailable += stored;
				if (isHome) {
					if (s.spawning) {
						// Post-tick, the engine has already burned the first tick of
						// the spawn, so the freshly committed body reads one below
						// its declared need time.
						if (s.spawning.remainingTime >= s.spawning.needTime - 1) {
							eventFlags.post_spawn = true;
						}
					} else {
						// One free spawn is one open spawn decision, even when a
						// sibling spawn in the same room is busy.
						homeSpawnIdle = true;
					}
				}
			}
			for (const s of room.find(C.FIND_MY_STRUCTURES)) {
				if (s.structureType === C.STRUCTURE_EXTENSION) {
					const stored = s.store?.[C.RESOURCE_ENERGY] || 0;
					energyAvailable += stored;
					if (isHome) homeEnergyAvailable += stored;
				}
			}
		}
		const liveCreepsById = new Map(
			Object.values(Game.creeps).map(creep => [ creep.id, creep ]),
		);
		for (const [ creepId, credited ] of session.remoteCargoByCreep) {
			const carried = energyIn(liveCreepsById.get(creepId)?.store);
			const remaining = Math.min(credited, carried);
			if (remaining > 0) session.remoteCargoByCreep.set(creepId, remaining);
			else session.remoteCargoByCreep.delete(creepId);
		}
		if (ECONOMY_TELEMETRY) economy = collectEconomyTelemetry(Game, obs, intentResults);
	});

	let constructionMask = obs._raw?.constructionMask;
	if (!constructionMask && obs.encoding === 'json' && Array.isArray(obs.constructionMask)) {
		constructionMask = Uint8Array.from(obs.constructionMask);
	} else if (!constructionMask && typeof obs.constructionMask === 'string') {
		constructionMask = new Uint8Array(Buffer.from(obs.constructionMask, 'base64'));
	} else if (!constructionMask && obs.encoding === 'pack' && typeof obs.blob === 'string') {
		const bytes = Buffer.from(obs.blob, 'base64');
		const length = SCHEMA.maxRooms * SCHEMA.constructionTypes.length
			* Math.ceil(SCHEMA.roomSize * SCHEMA.roomSize / 8);
		constructionMask = new Uint8Array(bytes.subarray(bytes.length - length));
	}
	session.meta = {
		actorMeta: obs.actorMeta,
		targetMeta: obs.targetMeta,
		roomNames: obs.roomNames,
		constructionMask,
	};

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
	session.remoteRoomsStaffedPeak = Math.max(
		session.remoteRoomsStaffedPeak, remoteRoomsStaffed,
	);
	session.remoteProductiveCreepsPeak = Math.max(
		session.remoteProductiveCreepsPeak, remoteProductiveCreeps,
	);
	session.remoteOwnedRoomsPeak = Math.max(session.remoteOwnedRoomsPeak, remoteOwnedRooms);

	// A spawn decision is open when the home room can afford a body and at least
	// one home spawn is free. Remote-room energy cannot fund it, so the sum has to
	// be home-local even though the telemetry total stays fleet-wide.
	if (homeSpawnIdle && homeEnergyAvailable >= PRE_SPAWN_ENERGY) {
		eventFlags.pre_spawn = true;
	}
	if (claimerReady && neutralOutpostRooms > 0) eventFlags.pre_claim = true;
	if (claimDelta > 0) eventFlags.post_claim = true;
	if (rclUp > 0) eventFlags.rcl_up = true;
	if (creeps <= 1 && session.step > RECOVERY_AFTER_STEP) eventFlags.recovery = true;
	const events = Object.keys(eventFlags);

	const wH = REWARD_CFG.energyHarvested ?? 1.0;
	const wC = REWARD_CFG.controlPoints ?? 1.0;
	// Delivery, construction, and claims are diagnostics and release gates, not
	// scalar reward. Their gross deltas are reversible or proxy-gameable.
	const reward = wH * harvestDelta + wC * cp;
	const extras = {
		transferDelta: transferToSpawn,
		remoteHarvestDelta,
		remoteHomeDeliveryDelta,
		remoteRoomsStaffed,
		remoteRoomsStaffedPeak: session.remoteRoomsStaffedPeak,
		remoteProductiveCreeps,
		remoteProductiveCreepsPeak: session.remoteProductiveCreepsPeak,
		remoteOwnedRooms,
		remoteOwnedRoomsPeak: session.remoteOwnedRoomsPeak,
		advancedDepositDelta,
		advancedWithdrawDelta,
		towerRefillDelta,
		buildDelta,
		spawnSuccess,
		rclUp,
		creeps,
		energyAvailable,
		ownedRooms,
		neutralOutpostRooms,
		claimDelta,
		spawnEnergy,
		sites: sitesNow,
		economy,
		events,
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
		if (code !== C.OK && code !== C.ERR_BUSY && code !== C.ERR_TIRED
			&& code !== C.ERR_NO_PATH) {
			row.invalid += 1;
			invalid += 1;
		}
		byType[type] = row;
	}
	return { intentIssued: results.length, intentInvalid: invalid, intentByType: byType };
}

function rememberActorOutcomes(results) {
	const actorKinds = new Map((session.meta?.actorMeta || []).map(actor => [ actor.id, actor.kind ]));
	session.actorOutcomes = new Map(results.map(result => [
		result.actor,
		actionOutcome(
			result.code, result.type, actorKinds.get(result.actor), result.executed,
		),
	]));
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
	rememberActorOutcomes(intentResults);
	await session.tick(1);
	session.step += 1;
	if (HEADFUL_TICK_MS > 0) await sleep(HEADFUL_TICK_MS);

	const { obs, reward, harvestDelta, controlDelta, extras } = await encodeAndRewardAfterTick(
		intentResults,
	);
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
			curriculum: session.curriculum,
			expert: false,
			...extras,
		},
	};
}

function validateStepActions(actions) {
	if (!actions || typeof actions !== 'object') throw new Error('step actions must be an object');
	const names = [
		[ 'types', SCHEMA.intentTypes.length ], [ 'dirs', SCHEMA.directions.length ],
		[ 'targets', SCHEMA.maxTargets ], [ 'amounts', SCHEMA.amountBins.length ],
		[ 'constructionTypes', SCHEMA.constructionTypes.length ],
	];
	const rows = actions.types?.length;
	if (!Number.isInteger(rows) || rows < 0 || rows > SCHEMA.maxActors) {
		throw new Error(`actions.types row count ${rows} is invalid`);
	}
	for (const [ name, limit ] of names) {
		if (actions[name]?.length !== rows) throw new Error(`actions.${name} row count mismatch`);
		for (let actor = 0; actor < rows; actor++) {
			if (actions[name][actor]?.length !== SCHEMA.intentSlots) {
				throw new Error(`actions.${name}[${actor}] slot count mismatch`);
			}
			for (const value of actions[name][actor]) {
				if (!Number.isInteger(Number(value)) || Number(value) < 0 || Number(value) >= limit) {
					throw new Error(`actions.${name} contains value ${value} outside [0, ${limit})`);
				}
			}
		}
	}
	if (actions.constructionTiles?.length !== rows
		|| actions.bodyCounts?.length !== rows || actions.bodyOrder?.length !== rows) {
		throw new Error('step action row count mismatch');
	}
	for (let actor = 0; actor < rows; actor++) {
		if (actions.constructionTiles[actor]?.length !== SCHEMA.intentSlots
			|| actions.bodyCounts[actor]?.length !== SCHEMA.intentSlots
			|| actions.bodyOrder[actor]?.length !== SCHEMA.intentSlots) {
			throw new Error(`step action slot count mismatch at actor ${actor}`);
		}
		for (let slot = 0; slot < SCHEMA.intentSlots; slot++) {
			const tile = Number(actions.constructionTiles[actor][slot]);
			if (!Number.isInteger(tile) || tile < 0 || tile >= SCHEMA.roomSize * SCHEMA.roomSize) {
				throw new Error(`actions.constructionTiles contains invalid tile ${tile}`);
			}
			const counts = actions.bodyCounts[actor][slot];
			const order = actions.bodyOrder[actor][slot];
			if (counts?.length !== SCHEMA.bodyPartTypes.length
				|| order?.length !== SCHEMA.bodyPartTypes.length) {
				throw new Error(`body vector width mismatch at actor ${actor}`);
			}
			if (!decodeSpawnBodyForValidation(counts, order,
				actions.types[actor][slot] === SCHEMA.intentTypes.indexOf('spawnCreep'))) {
				throw new Error(`invalid body counts/order at actor ${actor}`);
			}
		}
	}
}

function decodeSpawnBodyForValidation(counts, order, requireBody) {
	const seen = new Set();
	let total = 0;
	let nonzero = 0;
	for (let type = 0; type < SCHEMA.bodyPartTypes.length; type++) {
		const count = Number(counts[type]);
		const ordered = Number(order[type]);
		if (!Number.isInteger(count) || count < 0 || count > SCHEMA.maxBodyParts
			|| !Number.isInteger(ordered) || ordered < 0
			|| ordered >= SCHEMA.bodyPartTypes.length || seen.has(ordered)) return false;
		seen.add(ordered);
		total += count;
		if (count > 0) nonzero += 1;
	}
	if (total > SCHEMA.maxBodyParts || (requireBody && total < 1)) return false;
	for (let index = 0; index < order.length; index++) {
		const type = order[index];
		if ((index < nonzero) !== (counts[type] > 0)) return false;
		if (index > nonzero && type < order[index - 1]) return false;
	}
	return true;
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
	rememberActorOutcomes(intentResults);
	await session.tick(1);
	session.step += 1;
	if (HEADFUL_TICK_MS > 0) await sleep(HEADFUL_TICK_MS);

	const { obs, reward, harvestDelta, controlDelta, extras } = await encodeAndRewardAfterTick(
		intentResults,
	);
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
				curriculum: session.curriculum,
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
	const expertPreMeta = session.meta;
	const [ ir, vr ] = await Promise.all([
		session.shard.scratch.sMembers(userToIntentRoomsSetKey(USER)),
		session.shard.scratch.sMembers(userToVisibleRoomsSetKey(USER)),
	]);
	// Set membership order follows insertion, and insertion order follows the
	// completion order of concurrent room writes. A bot that iterates its rooms
	// inherits that order, so an unsorted set makes a fixed bot replay a
	// different game. Sorting costs nothing at four rooms.
	ir.sort();
	vr.sort();
	capturedExpertIntents = null;
	await session.instance.run(time, vr, ir);
	const expertIntents = capturedExpertIntents;
	session.actorOutcomes = new Map();
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
			// Stage attribution is per environment, and the teacher lane is the only
			// collector now, so this key must exist on the expert wire too.
			curriculum: session.curriculum,
			expert: true,
			botDir: session.botDir,
			expertIntents: CAPTURE_EXPERT_INTENTS ? expertIntents : undefined,
			// Exact translation belongs to s_t, not the post-tick s_(t+1).
			// Preserve the corresponding pre-action lookup tables alongside the
			// raw payload so collectors cannot accidentally align against new IDs.
			expertActorMeta: CAPTURE_EXPERT_INTENTS ? expertPreMeta?.actorMeta : undefined,
			expertTargetMeta: CAPTURE_EXPERT_INTENTS ? expertPreMeta?.targetMeta : undefined,
			expertRoomNames: CAPTURE_EXPERT_INTENTS ? expertPreMeta?.roomNames : undefined,
			// The runner only loads room blobs for `visibleRooms` and only publishes
			// intents for `intentRooms`. A teacher that keeps deciding from cached
			// Memory while these sets are empty issues nothing, which is otherwise
			// indistinguishable from a teacher that chose to idle.
			expertVisibleRooms: CAPTURE_EXPERT_INTENTS ? vr : undefined,
			expertIntentRooms: CAPTURE_EXPERT_INTENTS ? ir : undefined,
			// Identical replicas must consume the shared scenario stream identically.
			// A divergent draw count is the earliest observable symptom of engine
			// nondeterminism, well before the world visibly differs.
			expertRandomDraws: scenarioRandomDraws,
			expertRandomState: scenarioRandomState,
			...extras,
		},
	};
}

async function closeEpisode() {
	if (session?.headful?.stop) {
		try { await session.headful.stop(); } catch { /* */ }
	}
	try { session?.console?.(); } catch { /* */ }
	if (session?.close) await session.close();
	session = null;
	return { ok: true };
}

let chain = Promise.resolve();

async function dispatch(msg) {
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
		validateStepActions(msg.actions);
		reply(await stepRl(msg.actions));
	} else if (msg.cmd === 'step_scripted') {
		// One player() only (simulate forbids two per tick). Decide + apply together.
		reply(await stepScripted());
	} else if (msg.cmd === 'step_expert') {
		reply(await stepExpert());
	} else if (msg.cmd === 'snapshot') {
		reply(await writeSnapshot(msg.path, { events: msg.events || [] }));
	} else if (msg.cmd === 'restore') {
		reply(await restoreSnapshot(msg.path));
	} else if (msg.cmd === 'schema') {
		reply({ ok: true, schema: SCHEMA });
	} else if (msg.cmd === 'close') {
		reply(await closeEpisode());
		process.stdin.pause();
		setTimeout(() => process.exit(0), 50);
	} else {
		reply({ ok: false, error: `unknown cmd ${msg.cmd}` });
	}
}

function enqueue(msg) {
	chain = chain.then(async () => {
		try {
			await dispatch(msg);
		} catch (err) {
			reply({ ok: false, error: String(err?.stack || err) });
		}
	});
}

function enqueueProtocolError(message) {
	chain = chain.then(() => reply({ ok: false, error: message }));
}

const COMMAND_MAGIC = Buffer.from('XAC1', 'ascii');
const COMMAND_VERSION = 7;
const COMMAND_HEADER_BYTES = 16;
const COMMAND_MAX_PAYLOAD = 1024 * 1024;
const COMMANDS = Object.freeze({
	1: 'reset',
	2: 'reset_expert',
	3: 'step',
	4: 'step_scripted',
	5: 'step_expert',
	6: 'schema',
	7: 'close',
	8: 'snapshot',
	9: 'restore',
});
const EMPTY_COMMANDS = new Set([ 'reset', 'step_scripted', 'step_expert', 'schema', 'close' ]);
const ACTION_NAMES = [
	'types', 'dirs', 'targets', 'amounts', 'constructionTypes',
];
const ACTION_LIMITS = [ SCHEMA.intentTypes.length, SCHEMA.directions.length,
	SCHEMA.maxTargets, SCHEMA.amountBins.length, SCHEMA.constructionTypes.length ];
for (const [ name, cardinality ] of ACTION_NAMES.map((name, index) => [ name, ACTION_LIMITS[index] ])) {
	if (cardinality > 256) throw new Error(`XAC1 cannot encode actions.${name} cardinality ${cardinality}`);
}
const ACTION_ROW_POOL = Object.fromEntries(ACTION_NAMES.map(name => [ name,
	Array.from({ length: SCHEMA.maxActors }, () => new Uint8Array(SCHEMA.intentSlots)),
]));
const BODY_COUNT_ROW_POOL = Array.from({ length: SCHEMA.maxActors }, () =>
	Array.from({ length: SCHEMA.intentSlots }, () => new Uint8Array(SCHEMA.bodyPartTypes.length))
);
const BODY_ORDER_ROW_POOL = Array.from({ length: SCHEMA.maxActors }, () =>
	Array.from({ length: SCHEMA.intentSlots }, () => new Uint8Array(SCHEMA.bodyPartTypes.length))
);
const CONSTRUCTION_TILE_ROW_POOL = Array.from({ length: SCHEMA.maxActors }, () =>
	new Uint16Array(SCHEMA.intentSlots)
);

/** Decode a complete XAC1 frame with exact schema and categorical bounds. */
// Returns leases into reusable row pools. Keep private and dispatch immediately;
// callers must never retain a decoded command across the next frame.
function decodeBinaryCommand(frame) {
	if (frame.length < COMMAND_HEADER_BYTES) throw new Error('command frame shorter than header');
	if (!frame.subarray(0, 4).equals(COMMAND_MAGIC)) throw new Error('bad command magic');
	const version = frame.readUInt8(4);
	const opcode = frame.readUInt8(5);
	const schemaVersion = frame.readUInt16LE(6);
	const payloadLength = frame.readUInt32LE(8);
	const actors = frame.readUInt16LE(12);
	const slots = frame.readUInt8(14);
	const flags = frame.readUInt8(15);
	if (version !== COMMAND_VERSION) throw new Error(`unsupported command protocol version ${version}`);
	if (schemaVersion !== Number(SCHEMA.version)) {
		throw new Error(`command schema=${schemaVersion} does not match server schema=${SCHEMA.version}`);
	}
	if (flags !== 0) throw new Error(`unsupported command flags ${flags}`);
	if (payloadLength !== frame.length - COMMAND_HEADER_BYTES) {
		throw new Error(`command payload length=${payloadLength}, actual=${frame.length - COMMAND_HEADER_BYTES}`);
	}
	const cmd = COMMANDS[opcode];
	if (!cmd) throw new Error(`unknown command opcode ${opcode}`);
	const payload = frame.subarray(COMMAND_HEADER_BYTES);
	if (cmd === 'step') {
		if (actors > SCHEMA.maxActors) {
			throw new Error(`step actors=${actors} exceeds maxActors=${SCHEMA.maxActors}`);
		}
		if (slots !== SCHEMA.intentSlots) {
			throw new Error(`step slots=${slots} does not match schema=${SCHEMA.intentSlots}`);
		}
		const cells = actors * slots;
		const scalarBytes = cells * ACTION_NAMES.length;
		const constructionTileBytes = cells * 2;
		const bodyPlaneBytes = cells * SCHEMA.bodyPartTypes.length;
		const expectedBytes = scalarBytes + constructionTileBytes + bodyPlaneBytes * 2;
		if (payload.length !== expectedBytes) {
			throw new Error(`step payload=${payload.length}, expected=${expectedBytes}`);
		}
		const actions = {};
		for (let planeIndex = 0; planeIndex < ACTION_NAMES.length; planeIndex++) {
			const start = planeIndex * cells;
			const plane = payload.subarray(start, start + cells);
			const name = ACTION_NAMES[planeIndex];
			const limit = ACTION_LIMITS[planeIndex];
			for (let cellIndex = 0; cellIndex < plane.length; cellIndex++) {
				if (plane[cellIndex] >= limit) {
					throw new Error(`actions.${name}[${cellIndex}]=${plane[cellIndex]} outside [0, ${limit})`);
				}
			}
			const rows = ACTION_ROW_POOL[name].slice(0, actors);
			for (let actor = 0; actor < actors; actor++) {
				rows[actor].set(plane.subarray(actor * slots, (actor + 1) * slots));
			}
			actions[name] = rows;
		}
		const constructionTilePlane = payload.subarray(
			scalarBytes, scalarBytes + constructionTileBytes,
		);
		const constructionTileRows = CONSTRUCTION_TILE_ROW_POOL.slice(0, actors);
		for (let cell = 0; cell < cells; cell++) {
			const tile = constructionTilePlane.readUInt16LE(cell * 2);
			if (tile >= SCHEMA.roomSize * SCHEMA.roomSize) {
				throw new Error(
					`actions.constructionTiles[${cell}]=${tile} outside `
					+ `[0, ${SCHEMA.roomSize * SCHEMA.roomSize})`,
				);
			}
			constructionTileRows[Math.floor(cell / slots)][cell % slots] = tile;
		}
		actions.constructionTiles = constructionTileRows;
		const bodyCountPlane = payload.subarray(
			scalarBytes + constructionTileBytes,
			scalarBytes + constructionTileBytes + bodyPlaneBytes,
		);
		const bodyOrderPlane = payload.subarray(scalarBytes + constructionTileBytes + bodyPlaneBytes);
		for (let cell = 0; cell < cells; cell++) {
			const actor = Math.floor(cell / slots);
			const slot = cell % slots;
			const start = cell * SCHEMA.bodyPartTypes.length;
			const seen = new Uint8Array(SCHEMA.bodyPartTypes.length);
			let total = 0;
			let nonzeroTypes = 0;
			for (let type = 0; type < SCHEMA.bodyPartTypes.length; type++) {
				const count = bodyCountPlane[start + type];
				const orderedType = bodyOrderPlane[start + type];
				if (count > SCHEMA.maxBodyParts) {
					throw new Error(`actions.bodyCounts[${start + type}]=${count} exceeds ${SCHEMA.maxBodyParts}`);
				}
				if (orderedType >= SCHEMA.bodyPartTypes.length || seen[orderedType]) {
					throw new Error(`actions.bodyOrder cell ${cell} is not a permutation`);
				}
				seen[orderedType] = 1;
				total += count;
				if (count > 0) nonzeroTypes += 1;
			}
			for (let index = 0; index < SCHEMA.bodyPartTypes.length; index++) {
				const type = bodyOrderPlane[start + index];
				if ((index < nonzeroTypes) !== (bodyCountPlane[start + type] > 0)) {
					throw new Error(`actions.bodyOrder cell ${cell} does not place nonzero types first`);
				}
				if (index > nonzeroTypes && type < bodyOrderPlane[start + index - 1]) {
					throw new Error(`actions.bodyOrder cell ${cell} has noncanonical zero-count suffix`);
				}
			}
			if (actions.types[actor][slot] === SCHEMA.intentTypes.indexOf('spawnCreep')
				&& (total < 1 || total > SCHEMA.maxBodyParts)) {
				throw new Error(`actions.bodyCounts cell ${cell} sums to ${total}, expected [1, ${SCHEMA.maxBodyParts}]`);
			}
		}
		const countRows = BODY_COUNT_ROW_POOL.slice(0, actors);
		const orderRows = BODY_ORDER_ROW_POOL.slice(0, actors);
		for (let actor = 0; actor < actors; actor++) {
			for (let slot = 0; slot < slots; slot++) {
				const cell = actor * slots + slot;
				const start = cell * SCHEMA.bodyPartTypes.length;
				countRows[actor][slot].set(bodyCountPlane.subarray(start, start + SCHEMA.bodyPartTypes.length));
				orderRows[actor][slot].set(bodyOrderPlane.subarray(start, start + SCHEMA.bodyPartTypes.length));
			}
		}
		actions.bodyCounts = countRows;
		actions.bodyOrder = orderRows;
		return { cmd, actions };
	}
	if (actors !== 0 || slots !== 0) {
		throw new Error(`${cmd} must have actors=0 and slots=0`);
	}
	if (cmd === 'reset_expert') {
		if (payload.includes(0)) throw new Error('reset_expert botDir contains NUL');
		const botDir = new TextDecoder('utf-8', { fatal: true }).decode(payload);
		return botDir ? { cmd, botDir } : { cmd };
	}
	if (cmd === 'snapshot' || cmd === 'restore') {
		const request = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(payload));
		if (typeof request?.path !== 'string' || !request.path) {
			throw new Error(`${cmd} requires a non-empty path`);
		}
		const events = request.events ?? [];
		if (!Array.isArray(events) || events.some(tag => typeof tag !== 'string')) {
			throw new Error(`${cmd} events must be an array of tags`);
		}
		return { cmd, path: request.path, events };
	}
	if (EMPTY_COMMANDS.has(cmd) && payload.length !== 0) {
		throw new Error(`${cmd} payload must be empty`);
	}
	return { cmd };
}

if (CMD_FMT === 'json') {
	const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
	rl.on('line', line => {
		const raw = line.trim();
		if (!raw) return;
		try {
			enqueue(JSON.parse(raw));
		} catch (err) {
			enqueueProtocolError(`invalid json: ${err.message}`);
		}
	});
} else {
	let pending = Buffer.alloc(0);
	let failed = false;
	function failBinaryStream(message) {
		failed = true;
		process.stdin.pause();
		chain = chain.then(async () => {
			reply({ ok: false, error: message });
			try {
				await closeEpisode();
			} catch (err) {
				console.error(`[rl-env] protocol teardown failed: ${err?.stack || err}`);
			} finally {
				setTimeout(() => process.exit(1), 50);
			}
		});
	}
	process.stdin.on('data', chunk => {
		if (failed) return;
		pending = pending.length ? Buffer.concat([ pending, chunk ]) : chunk;
		while (pending.length >= COMMAND_HEADER_BYTES) {
			if (!pending.subarray(0, 4).equals(COMMAND_MAGIC)) {
				failBinaryStream('bad command magic; refusing unsafe resynchronization');
				return;
			}
			const payloadLength = pending.readUInt32LE(8);
			if (payloadLength > COMMAND_MAX_PAYLOAD) {
				failBinaryStream(`command payload too large: ${payloadLength}`);
				return;
			}
			const frameLength = COMMAND_HEADER_BYTES + payloadLength;
			if (pending.length < frameLength) return;
			const frame = pending.subarray(0, frameLength);
			pending = pending.subarray(frameLength);
			chain = chain.then(async () => {
				try {
					await dispatch(decodeBinaryCommand(frame));
				} catch (err) {
					reply({ ok: false, error: `invalid binary command: ${err?.stack || err}` });
				}
			});
		}
	});
	process.stdin.on('end', () => {
		if (!failed && pending.length !== 0) {
			failBinaryStream(`truncated command frame: ${pending.length} trailing bytes`);
		}
	});
}

console.error(
	`[rl-env] ready room=${ROOM} maxEpisode=${MAX_EPISODE} headful=${HEADFUL} ` +
	`tickMs=${HEADFUL_TICK_MS} cmdFmt=${CMD_FMT} obsFmt=${OBS_FMT} ` +
	`invaders=${INVADERS} expertDefault=${DEFAULT_EXPERT_BOT}`,
);
