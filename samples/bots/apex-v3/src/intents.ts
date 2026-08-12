/**
 * Room intents — enum status on room memory, set from flag *colors* (not names).
 *
 * Usage in logic:
 *   if (getRoomIntent(roomName) === RoomIntent.attack) { ... }
 *
 * Flags: put any-named flag in the room; primary color selects intent (see FLAG_COLOR_INTENT).
 */
import {
	FLAG_COLOR_INTENT,
	ROOM_INTENT_PRIORITY,
	RoomApexMem,
	RoomIntent,
} from './memoryKeys';
import type { Coord } from './codec';
import { asCoord } from './codec';

export { RoomIntent };

function readApex(roomName: string): Record<number, unknown> | undefined {
	const rm = Memory.rooms?.[roomName] as (RoomMemory & { a?: Record<number, unknown> }) | undefined;
	return rm?.a;
}

function writeApex(roomName: string): Record<number, unknown> {
	if (!Memory.rooms) Memory.rooms = {};
	if (!Memory.rooms[roomName]) Memory.rooms[roomName] = {};
	const rm = Memory.rooms[roomName] as RoomMemory & { a?: Record<number, unknown> };
	if (!rm.a) rm.a = {};
	return rm.a;
}

/** Read intent enum for a room (defaults to none). Does not create Memory bags. */
export function getRoomIntent(roomName: string): RoomIntent {
	const a = readApex(roomName);
	if (!a) return RoomIntent.none;
	const v = a[RoomApexMem.intent];
	return (v as RoomIntent) ?? RoomIntent.none;
}

export function setRoomIntent(roomName: string, intent: RoomIntent): void {
	const a = writeApex(roomName);
	a[RoomApexMem.intent] = intent;
	a[RoomApexMem.intentTick] = Game.time;
}

/** All rooms currently marked with a given intent. */
export function roomsWithIntent(intent: RoomIntent): string[] {
	const out: string[] = [];
	if (!Memory.rooms) return out;
	for (const name in Memory.rooms) {
		if (getRoomIntent(name) === intent) out.push(name);
	}
	return out;
}

export interface IntentSnapshot {
	/** roomName → intent */
	byRoom: Record<string, RoomIntent>;
	attacks: string[];
	defends: string[];
	claims: string[];
	remotes: string[];
	ignores: string[];
	/** rally room + packed coord of flag */
	rally: { room: string; pos: Coord } | null;
}

/**
 * Scan flags once per tick: color → RoomIntent on the flag's room.
 * Flags are the live source of truth for player orders this tick.
 * Rooms that lose their intent-colored flag return to RoomIntent.none.
 */
export function syncIntentsFromFlags(): IntentSnapshot {
	const byRoom: Record<string, RoomIntent> = {};
	let rally: IntentSnapshot['rally'] = null;
	const flaggedRooms = new Set<string>();

	for (const name in Game.flags) {
		const flag = Game.flags[name]!;
		const intent = FLAG_COLOR_INTENT[flag.color];
		if (intent === undefined) continue;

		const roomName = flag.pos.roomName;
		flaggedRooms.add(roomName);

		const prev = byRoom[roomName];
		if (prev === undefined || ROOM_INTENT_PRIORITY[intent] >= ROOM_INTENT_PRIORITY[prev]) {
			byRoom[roomName] = intent;
		}

		if (intent === RoomIntent.rally) {
			// Last yellow flag wins for rally point (coord).
			rally = { room: roomName, pos: asCoord(flag.pos) };
		}
	}

	// Write flag-derived intents; clear orders when the flag is gone.
	if (Memory.rooms) {
		for (const roomName in Memory.rooms) {
			if (flaggedRooms.has(roomName)) continue;
			const a = readApex(roomName);
			if (!a) continue;
			const intent = a[RoomApexMem.intent] as RoomIntent | undefined;
			if (intent != null && intent !== RoomIntent.none) {
				a[RoomApexMem.intent] = RoomIntent.none;
			}
		}
	}

	for (const roomName of Object.keys(byRoom)) {
		setRoomIntent(roomName, byRoom[roomName]!);
	}

	// Empire may reassert claim/remote without a flag (after this sync).
	// Those are written via setRoomIntent by empire/plan code.

	const snap: IntentSnapshot = {
		byRoom: {},
		attacks: [],
		defends: [],
		claims: [],
		remotes: [],
		ignores: [],
		rally,
	};

	// Snapshot from live room bags (flag + any empire reassert still present)
	if (Memory.rooms) {
		for (const roomName in Memory.rooms) {
			const intent = getRoomIntent(roomName);
			if (intent === RoomIntent.none) continue;
			snap.byRoom[roomName] = intent;
			switch (intent) {
				case RoomIntent.attack: snap.attacks.push(roomName); break;
				case RoomIntent.defend: snap.defends.push(roomName); break;
				case RoomIntent.claim: snap.claims.push(roomName); break;
				case RoomIntent.remote: snap.remotes.push(roomName); break;
				case RoomIntent.ignore: snap.ignores.push(roomName); break;
				default: break;
			}
		}
	}
	// Flag rooms not yet in Memory.rooms (getRoomIntent miss) — include from byRoom
	for (const roomName of Object.keys(byRoom)) {
		if (snap.byRoom[roomName] != null) continue;
		const intent = byRoom[roomName]!;
		snap.byRoom[roomName] = intent;
		switch (intent) {
			case RoomIntent.attack: snap.attacks.push(roomName); break;
			case RoomIntent.defend: snap.defends.push(roomName); break;
			case RoomIntent.claim: snap.claims.push(roomName); break;
			case RoomIntent.remote: snap.remotes.push(roomName); break;
			case RoomIntent.ignore: snap.ignores.push(roomName); break;
			default: break;
		}
	}

	if (!Memory.empire) Memory.empire = {};
	(Memory.empire as { intents?: IntentSnapshot }).intents = snap;

	return snap;
}

export function getIntentSnapshot(): IntentSnapshot | undefined {
	return Memory.empire && (Memory.empire as { intents?: IntentSnapshot }).intents;
}
