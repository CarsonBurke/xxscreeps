/**
 * Room apex bag — RoomApexMem numeric keys only under Memory.rooms[name].a
 */
import { RoomApexMem, RoomIntent } from './memoryKeys';
import type { Coord } from './codec';

function bag(roomName: string): Record<number, unknown> {
	if (!Memory.rooms) Memory.rooms = {};
	if (!Memory.rooms[roomName]) Memory.rooms[roomName] = {};
	const rm = Memory.rooms[roomName]!;
	if (!rm.a) rm.a = {};
	return rm.a!;
}

export function getRoomBag(roomName: string): Record<number, unknown> {
	return bag(roomName);
}

export function getRemotes(roomName: string): string[] {
	return (bag(roomName)[RoomApexMem.remotes] as string[]) || [];
}

export function setRemotes(roomName: string, remotes: string[]): void {
	const b = bag(roomName);
	b[RoomApexMem.remotes] = remotes;
	b[RoomApexMem.remotesAt] = Game.time;
}

export function getRemotesAt(roomName: string): number {
	return (bag(roomName)[RoomApexMem.remotesAt] as number) || 0;
}

export function getFillerPads(roomName: string): Coord[] {
	const a = Memory.rooms?.[roomName]?.a;
	if (!a) return [];
	return (a[RoomApexMem.fillerPads] as Coord[]) || [];
}

export function setFillerPads(roomName: string, pads: Coord[]): void {
	bag(roomName)[RoomApexMem.fillerPads] = pads;
}

export function getIntent(roomName: string): RoomIntent {
	// Read-only: do not create Memory bags for unscanned rooms.
	const rm = Memory.rooms?.[roomName];
	const a = rm?.a;
	if (!a) return RoomIntent.none;
	return (a[RoomApexMem.intent] as RoomIntent) ?? RoomIntent.none;
}

export function setIntent(roomName: string, intent: RoomIntent): void {
	const b = bag(roomName);
	b[RoomApexMem.intent] = intent;
	b[RoomApexMem.intentTick] = Game.time;
}

export function setPlanDebug(
	roomName: string,
	top: string,
	placed: number,
	tags?: string[],
): void {
	const b = bag(roomName);
	b[RoomApexMem.planTop] = top;
	b[RoomApexMem.planPlaced] = placed;
	b[RoomApexMem.planAt] = Game.time;
	if (tags) b[RoomApexMem.planTags] = tags;
}

export function getPlanTop(roomName: string): string | undefined {
	const a = Memory.rooms?.[roomName]?.a;
	if (!a) return undefined;
	return a[RoomApexMem.planTop] as string | undefined;
}

export function getPlanAt(roomName: string): number {
	const a = Memory.rooms?.[roomName]?.a;
	if (!a) return 0;
	return (a[RoomApexMem.planAt] as number) || 0;
}
