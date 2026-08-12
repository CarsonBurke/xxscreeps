/**
 * Memory bags are numeric-key maps (CreepMem / RoomApexMem / EmpireMem).
 * No string property names on creep/room memory — char cost + TI style.
 *
 * Top-level short roots (MemRoot.empire = 'e', etc.) must only be accessed
 * via MemRoot enum in application code — never as bare "s" / "e" literals.
 */

interface Memory {
	/**
	 * Legacy long root still used by war/empire plan bags during transition.
	 * Prefer MemRoot.empire once fully migrated.
	 */
	empire?: Record<string | number, unknown>;
	intel?: Record<string, unknown>;
	rooms?: { [roomName: string]: RoomMemory | undefined };
	creeps?: { [creepName: string]: CreepMemory | undefined };
	flags?: { [flagName: string]: FlagMemory | undefined };
	/** Short MemRoot keys and other bags — index only; use MemRoot enum at call sites */
	[key: string]: unknown;
}

/** Only index signature — access via RoomApexMem / room bag `a` */
interface RoomMemory {
	/** Apex room bag: numeric RoomApexMem keys only */
	a?: Record<number, unknown>;
}

/** Only index signature — access via CreepMem enum */
interface CreepMemory {
	[key: number]: unknown;
}

interface FlagMemory {
	[key: number]: unknown;
}

declare const module: { exports: { loop?: () => void; [k: string]: unknown } };
declare const console: Console;
declare const global: typeof globalThis & { Memory?: Memory };
