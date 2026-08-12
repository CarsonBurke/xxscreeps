/**
 * Numeric / short keys for Memory & segment JSON.
 * Screeps bills by serialized character count — short keys win.
 * Pattern follows The International (RoomMemoryKeys, CreepMemoryKeys, stats 'eih').
 *
 * RULE: never write wire strings ('hr', 's', …) at call sites.
 * Always use these enums. Wire values exist only in the enum definitions.
 * All enums are regular (not const) so CJS require() and host tools get them.
 */

/** Top-level Memory roots we own (keep short names). */
export enum MemRoot {
	empire = 'e',
	intel = 'i',
	apexMetrics = 'm',
	apex = 'a',
	stats = 's',
}

/** Memory.e (empire) fields — numeric keys when bag is fully migrated */
export enum EmpireMem {
	version = 0,
	attacks = 1,
	claims = 2,
	forcedRemotes = 3,
	ignoreRooms = 4,
	campaigns = 5,
	economy = 6,
	plan = 7,
	ownedRooms = 8,
	rally = 9,
}

/** Memory.rooms[name].a (apex room bag) — regular enum for CJS runtime */
export enum RoomApexMem {
	remotes = 0,
	remotesAt = 1,
	fillerPads = 2,
	planTop = 3,
	planPlaced = 4,
	planTags = 5,
	planAt = 6,
	/** RoomIntent enum value — player/empire intent for this room */
	intent = 7,
	intentTick = 8,
}

/**
 * What we want to do with a room (empire + flag-driven).
 * Stored as number on room memory — never parsed from flag name strings.
 *
 * Regular enum (not const) so CJS `require` modules (war.js) get a runtime object.
 */
export enum RoomIntent {
	none = 0,
	attack = 1,
	defend = 2,
	claim = 3,
	remote = 4,
	ignore = 5,
	rally = 6,
}

/** Higher wins when multiple colored flags share a room. */
export const ROOM_INTENT_PRIORITY: Record<RoomIntent, number> = {
	[RoomIntent.none]: 0,
	[RoomIntent.rally]: 1,
	[RoomIntent.ignore]: 2,
	[RoomIntent.remote]: 3,
	[RoomIntent.claim]: 4,
	[RoomIntent.defend]: 5,
	[RoomIntent.attack]: 6,
};

/**
 * Flag primary color → RoomIntent.
 * Place a flag in the target room; color is the intent. No name parsing.
 *
 * Color constants inlined (1–10 Screeps values) so this module loads in Node
 * host tools without Game globals — still matches COLOR_RED … COLOR_WHITE.
 *
 * | Color    | Intent |
 * |----------|--------|
 * | RED      | attack |
 * | PURPLE   | defend |
 * | BLUE     | claim  |
 * | CYAN     | remote |
 * | WHITE    | ignore |
 * | YELLOW   | rally  |
 */
const COLOR = {
	RED: 1,
	PURPLE: 2,
	BLUE: 3,
	CYAN: 4,
	GREEN: 5,
	YELLOW: 6,
	ORANGE: 7,
	BROWN: 8,
	GREY: 9,
	WHITE: 10,
} as const;

export const FLAG_COLOR_INTENT: Partial<Record<number, RoomIntent>> = {
	[COLOR.RED]: RoomIntent.attack,
	[COLOR.PURPLE]: RoomIntent.defend,
	[COLOR.BLUE]: RoomIntent.claim,
	[COLOR.CYAN]: RoomIntent.remote,
	[COLOR.WHITE]: RoomIntent.ignore,
	[COLOR.YELLOW]: RoomIntent.rally,
};

/**
 * Creep.memory keys (numeric only).
 * Regular enum — CJS require() modules need runtime object (not const enum).
 */
export enum CreepMem {
	role = 0,
	home = 1,
	sourceId = 2,
	targetRoom = 3,
	remote = 4,
	seat = 5,
	working = 6,
	pioneer = 7,
	squad = 8,
	defendRoom = 9,
	queue = 10,
	arrived = 11,
	marching = 12,
	campaignId = 13,
	/** Defender idle counter */
	safeTicks = 14,
}

/**
 * Creep role — stored at CreepMem.role as a number, never a string.
 * Regular enum so CJS require() modules get a runtime object.
 */
export enum Role {
	bootstrap = 0,
	harvester = 1,
	remoteHarvester = 2,
	hauler = 3,
	remoteHauler = 4,
	filler = 5,
	upgrader = 6,
	builder = 7,
	reserver = 8,
	claimer = 9,
	scout = 10,
	defender = 11,
	attacker = 12,
	ranged = 13,
	healer = 14,
	dismantler = 15,
}

/** For logs / spawn names only — never stored in Memory */
export const ROLE_NAME: Record<Role, string> = {
	[Role.bootstrap]: 'bootstrap',
	[Role.harvester]: 'harvester',
	[Role.remoteHarvester]: 'remoteHarvester',
	[Role.hauler]: 'hauler',
	[Role.remoteHauler]: 'remoteHauler',
	[Role.filler]: 'filler',
	[Role.upgrader]: 'upgrader',
	[Role.builder]: 'builder',
	[Role.reserver]: 'reserver',
	[Role.claimer]: 'claimer',
	[Role.scout]: 'scout',
	[Role.defender]: 'defender',
	[Role.attacker]: 'attacker',
	[Role.ranged]: 'ranged',
	[Role.healer]: 'healer',
	[Role.dismantler]: 'dismantler',
};

/** Parse legacy string or number → Role (spawn merge / one-tick migration) */
export function parseRole(v: unknown): Role | undefined {
	if (typeof v === 'number' && ROLE_NAME[v as Role] != null) return v as Role;
	if (typeof v === 'string') {
		for (const [num, name] of Object.entries(ROLE_NAME)) {
			if (name === v) return Number(num) as Role;
		}
	}
	return undefined;
}

/**
 * Segment metrics sample fields (2-char max where possible).
 * Enum member name = long name for TB/JSONL expand; value = wire key.
 */
export enum MetricKey {
	t = 't',
	rclMax = 'r',
	colonies = 'n',
	creeps = 'c',
	controlPoints = 'p',
	harvested = 'h',
	build = 'b',
	upgrade = 'u',
	creepEnergy = 'e',
	storedEnergy = 'se',
	sites = 'si',
	progress = 'pr',
	gcl = 'g',
	cpu = 'k',
	bucket = 'q',
	/** Harvest energy per tick (windowed over WRITE_EVERY) — total */
	harvestRate = 'hr',
	/** Control points gained per tick (windowed) */
	controlRate = 'cr',
	/** Upgrade energy spent per tick (windowed) */
	upgradeRate = 'ur',
	/** Build energy spent per tick (windowed) */
	buildRate = 'br',
	/** Spawn-time-bound max harvest e/t (ceiling estimate) */
	maxHarvestEt = 'mh',
	/** Physics max e/t of all candidate sources */
	maxHarvestPhysics = 'mp',
	/** Energy harvested in claimed (owned) rooms — total */
	harvestedClaimed = 'hc',
	/** Energy harvested in remote (non-owned) rooms — total */
	harvestedRemote = 'ho',
	/** Claimed-room harvest e/t (windowed) */
	claimedHarvestRate = 'hcr',
	/** Remote-room harvest e/t (windowed) */
	remoteHarvestRate = 'hor',
}

/** Segment envelope */
export enum SegmentKey {
	version = 'v',
	seq = 's',
	written = 'w',
	sample = 'd',
}

export const METRICS_SEGMENT = 87;
export const PROTOCOL_VERSION = 1;
export const WRITE_EVERY = 5;

/**
 * Expand dense wire sample → long keys (enum member names).
 * Host tools and TB use long names; segment stores wire values only.
 */
export function expandMetricSample(
	dense: Partial<Record<MetricKey, number>> | Record<string, number>,
): Record<string, number> {
	const bag = dense as Record<string, number | undefined>;
	const out: Record<string, number> = {};
	for (const longName of Object.keys(MetricKey) as (keyof typeof MetricKey)[]) {
		const wire = MetricKey[longName];
		if (typeof wire !== 'string') continue;
		const v = bag[wire] ?? bag[longName];
		if (v != null && typeof v === 'number') out[longName] = v;
	}
	return out;
}

/** wire short → long name (for host decoders). Built from MetricKey only. */
export function metricWireToLong(): Record<string, string> {
	const out: Record<string, string> = {};
	for (const longName of Object.keys(MetricKey) as (keyof typeof MetricKey)[]) {
		const wire = MetricKey[longName];
		if (typeof wire === 'string') out[wire] = longName;
	}
	return out;
}

/** Segment envelope wire → long (from SegmentKey only). */
export function segmentWireToLong(): Record<string, string> {
	const out: Record<string, string> = {};
	for (const longName of Object.keys(SegmentKey) as (keyof typeof SegmentKey)[]) {
		const wire = SegmentKey[longName];
		if (typeof wire === 'string') out[wire] = longName;
	}
	return out;
}
