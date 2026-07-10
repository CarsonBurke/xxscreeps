/**
 * Lean empire metrics → RawMemory segment 87 (no host deps).
 * Protocol: samples/bots/PROTOCOL.md
 *
 * Performance:
 *  - Counters accumulate every tick (event log walk only).
 *  - Segment JSON is written every WRITE_EVERY ticks (default 5).
 *  - Payload is a single sample (no ring) to keep stringify cheap.
 */
const SEGMENT_ID = 87;
const PROTOCOL_VERSION = 1;
/** How often to serialize + write the segment. Counters still tick every tick. */
const WRITE_EVERY = 5;

function state() {
	Memory.apexMetrics ||= {
		controlPoints: 0,
		harvested: 0,
		build: 0,
		upgrade: 0,
		creepEnergy: 0,
		seenCreeps: Object.create(null),
		seq: 0,
		segmentActive: false,
	};
	return Memory.apexMetrics;
}

function bodyCost(creep) {
	let sum = 0;
	const body = creep.body;
	for (let i = 0; i < body.length; i++) {
		sum += BODYPART_COST[body[i].type] || 0;
	}
	return sum;
}

/** Event-log counters — O(events), typically tiny. */
function ingestEventLogs() {
	const s = state();
	for (const name in Game.rooms) {
		const log = Game.rooms[name].getEventLog();
		if (!log || !log.length) continue;
		for (let i = 0; i < log.length; i++) {
			const ev = log[i];
			const d = ev.data;
			const amount = d ? d.amount : ev.amount;
			const spent = d ? d.energySpent : ev.energySpent;
			switch (ev.event) {
				case EVENT_HARVEST:
					if (amount) s.harvested += amount;
					break;
				case EVENT_BUILD:
					if (spent) s.build += spent;
					break;
				case EVENT_UPGRADE_CONTROLLER:
					if (spent) s.upgrade += spent;
					if (amount) s.controlPoints += amount;
					break;
				default:
					break;
			}
		}
	}
}

/** New creeps → body energy spent. */
function ingestSpawns() {
	const s = state();
	const seen = s.seenCreeps;
	for (const name in Game.creeps) {
		if (seen[name] === undefined) {
			const cost = bodyCost(Game.creeps[name]);
			s.creepEnergy += cost;
			seen[name] = 1;
		}
	}
	// Prune dead names infrequently (not every tick).
	if (Game.time % 50 === 0) {
		for (const name in seen) {
			if (!Game.creeps[name]) delete seen[name];
		}
	}
}

/**
 * Snapshot of the few gauges we care about.
 * Avoid room.find where possible.
 */
function sample() {
	const s = state();
	let rclMax = 0;
	let colonies = 0;
	let progress = 0;
	let storedEnergy = 0;
	let sites = 0;

	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		const ctrl = room.controller;
		if (ctrl && ctrl.my) {
			colonies++;
			const level = ctrl.level || 0;
			if (level > rclMax) rclMax = level;
			progress += ctrl.progress || 0;
			if (room.storage) {
				const st = room.storage.store;
				storedEnergy += st.getUsedCapacity
					? (st.getUsedCapacity(RESOURCE_ENERGY) || 0)
					: (st[RESOURCE_ENERGY] || 0);
			}
		}
		// Construction backlog — cheap enough; skip if you need more CPU later.
		sites += room.find(FIND_MY_CONSTRUCTION_SITES).length;
	}

	let creeps = 0;
	for (const _ in Game.creeps) creeps++;

	const cpu = Game.cpu.getUsed();
	const gcl = Game.gcl;

	return {
		t: Game.time,
		rclMax,
		colonies,
		creeps,
		controlPoints: s.controlPoints,
		harvested: s.harvested,
		build: s.build,
		upgrade: s.upgrade,
		creepEnergy: s.creepEnergy,
		storedEnergy,
		sites,
		progress,
		gcl: gcl ? gcl.level : 1,
		cpu,
		bucket: Game.cpu.bucket,
	};
}

function flushSegment(sample) {
	const s = state();
	s.seq++;
	if (typeof RawMemory === 'undefined') return;

	if (!s.segmentActive) {
		try {
			RawMemory.setActiveSegments([ SEGMENT_ID ]);
			s.segmentActive = true;
		} catch {
			/* empty */
		}
	} else {
		// Re-assert occasionally in case something else called setActiveSegments.
		if (Game.time % 100 === 0) {
			try {
				RawMemory.setActiveSegments([ SEGMENT_ID ]);
			} catch {
				/* empty */
			}
		}
	}

	// Single sample only — watcher/JSONL owns history.
	RawMemory.segments[SEGMENT_ID] = JSON.stringify({
		v: PROTOCOL_VERSION,
		seq: s.seq,
		written: Game.time,
		sample,
	});
}

/**
 * Call once per tick at end of loop.
 * Heavy work (JSON + segment write) only every WRITE_EVERY ticks.
 */
function tick() {
	ingestEventLogs();
	ingestSpawns();

	// Always refresh a tiny Memory.stats for console / debugging (no stringify).
	const s = state();
	let creeps = 0;
	for (const _ in Game.creeps) creeps++;
	Memory.stats = {
		t: Game.time,
		rclMax: Memory.stats && Memory.stats.rclMax,
		creeps,
		cpu: Game.cpu.getUsed(),
		bucket: Game.cpu.bucket,
		harvested: s.harvested,
		controlPoints: s.controlPoints,
	};

	if (Game.time % WRITE_EVERY !== 0) return null;

	const snap = sample();
	// Patch rcl into the lightweight stats after full sample.
	Memory.stats.rclMax = snap.rclMax;
	Memory.stats.colonies = snap.colonies;
	Memory.stats.storedEnergy = snap.storedEnergy;
	flushSegment(snap);
	return snap;
}

module.exports = {
	SEGMENT_ID,
	PROTOCOL_VERSION,
	WRITE_EVERY,
	tick,
	state,
};
