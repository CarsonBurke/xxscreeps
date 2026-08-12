/**
 * Compact position / id helpers (International-style).
 * Store Coord in Memory; expand to RoomPosition only in-process.
 */

export interface Coord {
	x: number;
	y: number;
}

/** Room-local coord only (50×50). */
export function asCoord(pos: RoomPosition | Coord): Coord {
	return { x: pos.x, y: pos.y };
}

export function asPos(c: Coord, roomName: string): RoomPosition {
	return new RoomPosition(c.x, c.y, roomName);
}

/** Pack x,y into a single int 0..2499 (x + y*50). */
export function packXY(x: number, y: number): number {
	return x + y * 50;
}

export function unpackXY(n: number): Coord {
	return { x: n % 50, y: Math.floor(n / 50) };
}

export function packCoord(c: Coord): number {
	return packXY(c.x, c.y);
}

export function unpackCoord(n: number): Coord {
	return unpackXY(n);
}

/**
 * Pack 24-char hex object id → 6 UTF-16 code units (International packId).
 * ~75% smaller when stringified as a lone value / concatenated list.
 */
export function packId(id: string): string {
	let packed = '';
	for (let i = 0; i < 24; i += 4) {
		packed += String.fromCharCode(parseInt(id.substr(i, 4), 16));
	}
	return packed;
}

export function unpackId(packed: string): string {
	let id = '';
	for (let i = 0; i < packed.length; i++) {
		const c = packed.charCodeAt(i);
		id += (c >>> 8).toString(16).padStart(2, '0');
		id += (c & 0xff).toString(16).padStart(2, '0');
	}
	return id;
}

/** Concat packed ids without JSON array overhead. */
export function packIdList(ids: string[]): string {
	let s = '';
	for (let i = 0; i < ids.length; i++) s += packId(ids[i]!);
	return s;
}

export function unpackIdList(packed: string): string[] {
	const ids: string[] = [];
	for (let i = 0; i < packed.length; i += 6) {
		ids.push(unpackId(packed.substr(i, 6)));
	}
	return ids;
}
