/**
 * Simulator state capture/restore for the RL start-state reservoir.
 *
 * A snapshot is the exact post-tick world the environment just observed:
 * every live room blob plus the cross-tick scratch state that a room blob does
 * not contain. Derived scratch state (user room relationships, active-user set)
 * is rebuilt from the restored rooms rather than copied, so a restore can never
 * resurrect a stale relationship from the process that captured it.
 *
 * Restores target a freshly instantiated shard. The caller owns session-level
 * bookkeeping (reward baselines, actor outcomes, RNG stream); this module owns
 * engine state only.
 */
import { Buffer } from 'node:buffer';
import {
	activeRoomsKey,
	finalIntentsListForRoomKey,
	intentsListForRoomKey,
	sleepingRoomsKey,
	updateUserRoomRelationships,
	userToIntentRoomsSetKey,
	userToPresenceRoomsSetKey,
	userToVisibleRoomsSetKey,
} from 'xxscreeps/engine/processor/model.js';
import {
	controlledRoomsKey, reservedRoomsKey,
} from 'xxscreeps/mods/controller/model.js';

export const SNAPSHOT_MAGIC = 'XSNP';
export const SNAPSHOT_VERSION = 1;
const HEADER_BYTES = 16;

/**
 * Container: 16-byte header, UTF-8 JSON metadata, then concatenated room blobs
 * addressed by `meta.rooms[i].offset/length`. Room blobs are the engine's own
 * serialization, copied verbatim.
 */
export function encodeSnapshotFile(meta, blobs) {
	if (!Array.isArray(blobs)) throw new Error('snapshot blobs must be an array');
	const parts = blobs.map(blob => Buffer.from(blob.buffer, blob.byteOffset, blob.byteLength));
	const blobLength = parts.reduce((total, part) => total + part.byteLength, 0);
	const metaBuffer = Buffer.from(JSON.stringify(meta), 'utf8');
	const out = Buffer.allocUnsafe(HEADER_BYTES + metaBuffer.length + blobLength);
	out.write(SNAPSHOT_MAGIC, 0, 4, 'ascii');
	out[4] = SNAPSHOT_VERSION;
	out[5] = 0;
	out.writeUInt16LE(Number(meta.schemaVersion) || 0, 6);
	out.writeUInt32LE(metaBuffer.length, 8);
	out.writeUInt32LE(blobLength, 12);
	metaBuffer.copy(out, HEADER_BYTES);
	let offset = HEADER_BYTES + metaBuffer.length;
	for (const part of parts) {
		part.copy(out, offset);
		offset += part.byteLength;
	}
	return out;
}

export function decodeSnapshotFile(buffer) {
	if (buffer.length < HEADER_BYTES) throw new Error('snapshot shorter than header');
	if (buffer.subarray(0, 4).toString('ascii') !== SNAPSHOT_MAGIC) {
		throw new Error('bad snapshot magic');
	}
	const version = buffer[4];
	if (version !== SNAPSHOT_VERSION) {
		throw new Error(`unsupported snapshot version ${version}`);
	}
	const headerSchema = buffer.readUInt16LE(6);
	const metaLength = buffer.readUInt32LE(8);
	const blobLength = buffer.readUInt32LE(12);
	if (buffer.length !== HEADER_BYTES + metaLength + blobLength) {
		throw new Error(
			`snapshot length=${buffer.length}, expected=${HEADER_BYTES + metaLength + blobLength}`,
		);
	}
	const meta = JSON.parse(buffer.subarray(HEADER_BYTES, HEADER_BYTES + metaLength).toString('utf8'));
	if (headerSchema !== Number(meta.schemaVersion)) {
		throw new Error(
			`snapshot header schema=${headerSchema} disagrees with metadata `
			+ `${meta.schemaVersion}`,
		);
	}
	const base = HEADER_BYTES + metaLength;
	const blobs = new Map();
	for (const room of meta.rooms || []) {
		const start = base + Number(room.offset);
		const end = start + Number(room.length);
		if (!(start >= base && end <= buffer.length && end >= start)) {
			throw new Error(`snapshot room ${room.name} has an out-of-range blob span`);
		}
		blobs.set(String(room.name), new Uint8Array(buffer.subarray(start, end)));
	}
	return { meta, blobs };
}

/**
 * Rooms whose state an RL episode can have modified.
 *
 * The processor queues alone are not sufficient: `sleepRoomUntil` with an
 * infinite wake time removes a room from both sets, so a room holding only
 * player residue that schedules no wake-up — a construction site, for example —
 * would be dropped silently and restored from the pristine imported blob. The
 * player relationship sets cover exactly that residue, because `flushUsers`
 * records presence for every owned object.
 */
async function liveRoomNames(shard, extraRooms, users) {
	const [ active, sleeping, ...relationships ] = await Promise.all([
		shard.scratch.zRangeWithScores(activeRoomsKey, 0, -1),
		shard.scratch.zRangeWithScores(sleepingRoomsKey, 0, -1),
		...users.flatMap(userId => [
			shard.scratch.sMembers(userToPresenceRoomsSetKey(userId)),
			shard.scratch.sMembers(userToVisibleRoomsSetKey(userId)),
			shard.scratch.sMembers(userToIntentRoomsSetKey(userId)),
		]),
	]);
	const names = new Set();
	for (const [ , name ] of active) names.add(name);
	for (const [ , name ] of sleeping) names.add(name);
	for (const members of relationships) for (const name of members) names.add(name);
	for (const name of extraRooms) if (name) names.add(name);
	return { names: [ ...names ].sort(), active, sleeping };
}

/**
 * Capture engine state for the current tick. Returns room blobs plus the
 * scratch state that is not derivable from them.
 */
export async function captureShardState(shard, { extraRooms = [], users = [] } = {}) {
	const { names, active, sleeping } = await liveRoomNames(shard, extraRooms, users);
	const rooms = [];
	for (const name of names) {
		// `req` may hand back a view into the store; copy before it can be reused.
		const blob = await shard.loadRoomBlob(name, shard.time);
		rooms.push({ name, blob: new Uint8Array(blob) });
	}
	const roomIntents = {};
	const roomFinalIntents = {};
	for (const name of names) {
		const [ pending, final ] = await Promise.all([
			shard.scratch.lRange(intentsListForRoomKey(name), 0, -1),
			shard.scratch.lRange(finalIntentsListForRoomKey(name), 0, -1),
		]);
		if (pending.length) roomIntents[name] = pending;
		if (final.length) roomFinalIntents[name] = final;
	}
	// The controlled/reserved room sets are the sole enforcement of the global
	// control level's room cap, and a fresh shard flushes scratch. Rebuilding them
	// from restored ownership is not equivalent: it would insert the seeded home
	// room, which a claim intent never inserts, so they are captured verbatim.
	const controlledRooms = {};
	const reservedRooms = {};
	for (const userId of users) {
		const [ controlled, reserved ] = await Promise.all([
			shard.scratch.sMembers(controlledRoomsKey(userId)),
			shard.scratch.sMembers(reservedRoomsKey(userId)),
		]);
		controlledRooms[userId] = controlled;
		reservedRooms[userId] = reserved;
	}
	return {
		time: shard.time,
		rooms,
		scratch: {
			activeRooms: active,
			sleepingRooms: sleeping,
			roomIntents,
			roomFinalIntents,
			controlledRooms,
			reservedRooms,
		},
	};
}

/**
 * Write captured engine state into a freshly instantiated shard.
 *
 * The shard must be new: this rebuilds user room relationships without a
 * previous-state diff, so pre-existing relationships would never be removed.
 */
export async function applyShardState(shard, state, blobs) {
	const time = Number(state.time);
	if (!Number.isInteger(time) || time < 0) {
		throw new Error(`snapshot time ${state.time} is not a tick index`);
	}
	await shard.data.set('time', time);
	shard.time = time;
	for (const room of state.rooms) {
		const blob = blobs.get(room.name);
		if (!blob) throw new Error(`snapshot is missing the blob for room ${room.name}`);
		await shard.saveRoomBlob(room.name, time, blob);
		// Boot parity: both double-buffer slots hold the restored state so a room
		// that sleeps immediately cannot roll back to imported terrain-only state.
		await shard.copyRoomFromPreviousTick(room.name, time + 1);
	}
	for (const room of state.rooms) {
		const loaded = await shard.loadRoom(room.name, time);
		await updateUserRoomRelationships(shard, loaded);
	}
	// Relationship rebuilding marks every restored room active. Replace the
	// processor queues with the captured ones so sleeping rooms stay asleep.
	await Promise.all([
		shard.scratch.vDel(activeRoomsKey),
		shard.scratch.vDel(sleepingRoomsKey),
	]);
	const active = state.scratch?.activeRooms || [];
	const sleeping = state.scratch?.sleepingRooms || [];
	if (active.length) {
		await shard.scratch.zAdd(activeRoomsKey, active.map(([ score, name ]) => [ score, name ]));
	}
	if (sleeping.length) {
		await shard.scratch.zAdd(sleepingRoomsKey, sleeping.map(([ score, name ]) => [ score, name ]));
	}
	for (const [ name, payloads ] of Object.entries(state.scratch?.roomIntents || {})) {
		if (payloads.length) await shard.scratch.rPush(intentsListForRoomKey(name), payloads);
	}
	for (const [ name, payloads ] of Object.entries(state.scratch?.roomFinalIntents || {})) {
		if (payloads.length) await shard.scratch.rPush(finalIntentsListForRoomKey(name), payloads);
	}
	// Restoring the claim budget is load-bearing: without it every restored
	// segment would see zero controlled rooms and could claim past the global
	// control level's cap, which a fresh evaluation world enforces.
	for (const [ userId, rooms ] of Object.entries(state.scratch?.controlledRooms || {})) {
		if (rooms.length) await shard.scratch.sAdd(controlledRoomsKey(userId), rooms);
	}
	for (const [ userId, rooms ] of Object.entries(state.scratch?.reservedRooms || {})) {
		if (rooms.length) await shard.scratch.sAdd(reservedRoomsKey(userId), rooms);
	}
}
