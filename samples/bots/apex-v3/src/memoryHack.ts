/**
 * Memhack: keep one Memory object across ticks (ZeSwarm / International).
 * Avoids full JSON.parse every tick; engine serializes via RawMemory._parsed.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let cached: Memory | undefined;

export function runMemoryHack(): void {
	if (cached === undefined) {
		// Force first parse
		void Memory;
		const raw = RawMemory as RawMemory & { _parsed?: Memory };
		cached = raw._parsed || Memory;
	}
	const g = global as typeof globalThis & { Memory?: Memory };
	delete g.Memory;
	g.Memory = cached;
	(RawMemory as RawMemory & { _parsed?: Memory })._parsed = cached;
}
