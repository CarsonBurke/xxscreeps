/**
 * Lean empire metrics → RawMemory segment (short keys for char cost).
 * Includes windowed rates: harvest e/t (total + claimed/remote split),
 * control points/t, upgrade e/t, build e/t.
 */
import {
	MemRoot,
	MetricKey,
	METRICS_SEGMENT,
	PROTOCOL_VERSION,
	SegmentKey,
	WRITE_EVERY,
	expandMetricSample,
} from './memoryKeys';
import { bodyCost } from './util';

interface MetricsState {
	controlPoints: number;
	harvested: number;
	/** Harvest in owned (controller.my) rooms */
	harvestedClaimed: number;
	/** Harvest in non-owned rooms (remotes) */
	harvestedRemote: number;
	build: number;
	upgrade: number;
	creepEnergy: number;
	seenCreeps: Record<string, 1>;
	seq: number;
	segmentActive: boolean;
	/** Previous flush snapshot for rates */
	prevAt: number;
	prevHarvested: number;
	prevHarvestedClaimed: number;
	prevHarvestedRemote: number;
	prevControlPoints: number;
	prevUpgrade: number;
	prevBuild: number;
}

function state(): MetricsState {
	const M = Memory as Memory & { [MemRoot.apexMetrics]?: MetricsState };
	if (!M[MemRoot.apexMetrics]) {
		M[MemRoot.apexMetrics] = {
			controlPoints: 0,
			harvested: 0,
			harvestedClaimed: 0,
			harvestedRemote: 0,
			build: 0,
			upgrade: 0,
			creepEnergy: 0,
			seenCreeps: Object.create(null) as Record<string, 1>,
			seq: 0,
			segmentActive: false,
			prevAt: 0,
			prevHarvested: 0,
			prevHarvestedClaimed: 0,
			prevHarvestedRemote: 0,
			prevControlPoints: 0,
			prevUpgrade: 0,
			prevBuild: 0,
		};
	}
	const s = M[MemRoot.apexMetrics]!;
	// Migrate older state bags missing split counters
	if (s.harvestedClaimed == null) s.harvestedClaimed = 0;
	if (s.harvestedRemote == null) s.harvestedRemote = 0;
	if (s.prevHarvestedClaimed == null) s.prevHarvestedClaimed = 0;
	if (s.prevHarvestedRemote == null) s.prevHarvestedRemote = 0;
	return s;
}

function isClaimedRoom(room: Room): boolean {
	return !!(room.controller && room.controller.my);
}

function ingestEventLogs(): void {
	const s = state();
	for (const name in Game.rooms) {
		const room = Game.rooms[name]!;
		const log = room.getEventLog();
		if (!log || !log.length) continue;
		const claimed = isClaimedRoom(room);
		for (let i = 0; i < log.length; i++) {
			const ev = log[i] as {
				event: number;
				data?: { amount?: number; energySpent?: number };
				amount?: number;
				energySpent?: number;
			};
			const d = ev.data;
			const amount = d ? d.amount : ev.amount;
			const spent = d ? d.energySpent : ev.energySpent;
			switch (ev.event) {
				case EVENT_HARVEST:
					if (amount) {
						s.harvested += amount;
						if (claimed) s.harvestedClaimed += amount;
						else s.harvestedRemote += amount;
					}
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

function ingestSpawns(): void {
	const s = state();
	const seen = s.seenCreeps;
	for (const name in Game.creeps) {
		if (seen[name] === undefined) {
			const creep = Game.creeps[name]!;
			const parts = creep.body.map(p => p.type);
			s.creepEnergy += bodyCost(parts);
			seen[name] = 1;
		}
	}
	if (Game.time % 50 === 0) {
		for (const name in seen) {
			if (!Game.creeps[name]) delete seen[name];
		}
	}
}

/**
 * GCL as continuous level: integer level + fraction of progress to next.
 * e.g. level 1 with 25% progress → 1.25
 */
function gclLevelFloat(): number {
	if (!Game.gcl) return 1;
	const level = Game.gcl.level || 1;
	const total = Game.gcl.progressTotal || 0;
	const prog = Game.gcl.progress || 0;
	if (!(total > 0)) return level;
	const frac = Math.min(1, Math.max(0, prog / total));
	return level + frac;
}

/** Dense sample — short MetricKey fields only. */
export type MetricSample = {
	[MetricKey.t]: number;
	[MetricKey.rclMax]: number;
	[MetricKey.colonies]: number;
	[MetricKey.creeps]: number;
	[MetricKey.controlPoints]: number;
	[MetricKey.harvested]: number;
	[MetricKey.build]: number;
	[MetricKey.upgrade]: number;
	[MetricKey.creepEnergy]: number;
	[MetricKey.storedEnergy]: number;
	[MetricKey.sites]: number;
	[MetricKey.progress]: number;
	/** Continuous GCL: level + progress/progressTotal */
	[MetricKey.gcl]: number;
	[MetricKey.cpu]: number;
	[MetricKey.bucket]: number;
	[MetricKey.harvestRate]: number;
	[MetricKey.controlRate]: number;
	[MetricKey.upgradeRate]: number;
	[MetricKey.buildRate]: number;
	[MetricKey.maxHarvestEt]: number;
	[MetricKey.maxHarvestPhysics]: number;
	[MetricKey.harvestedClaimed]: number;
	[MetricKey.harvestedRemote]: number;
	[MetricKey.claimedHarvestRate]: number;
	[MetricKey.remoteHarvestRate]: number;
};

function windowRates(s: MetricsState): {
	harvestRate: number;
	claimedHarvestRate: number;
	remoteHarvestRate: number;
	controlRate: number;
	upgradeRate: number;
	buildRate: number;
} {
	// Before first flush, no baseline — report 0 rather than lifetime/tick.
	if (!s.prevAt) {
		return {
			harvestRate: 0,
			claimedHarvestRate: 0,
			remoteHarvestRate: 0,
			controlRate: 0,
			upgradeRate: 0,
			buildRate: 0,
		};
	}
	const dt = Math.max(1, Game.time - s.prevAt);
	return {
		harvestRate: (s.harvested - s.prevHarvested) / dt,
		claimedHarvestRate: (s.harvestedClaimed - s.prevHarvestedClaimed) / dt,
		remoteHarvestRate: (s.harvestedRemote - s.prevHarvestedRemote) / dt,
		controlRate: (s.controlPoints - s.prevControlPoints) / dt,
		upgradeRate: (s.upgrade - s.prevUpgrade) / dt,
		buildRate: (s.build - s.prevBuild) / dt,
	};
}

function sample(): MetricSample {
	const s = state();
	const rates = windowRates(s);
	let rclMax = 0;
	let colonies = 0;
	let progress = 0;
	let storedEnergy = 0;
	let sites = 0;

	for (const name in Game.rooms) {
		const room = Game.rooms[name]!;
		const ctrl = room.controller;
		if (ctrl && ctrl.my) {
			colonies++;
			const level = ctrl.level || 0;
			if (level > rclMax) rclMax = level;
			progress += ctrl.progress || 0;
			if (room.storage) {
				storedEnergy += room.storage.store.getUsedCapacity(RESOURCE_ENERGY) || 0;
			}
		}
		sites += room.find(FIND_MY_CONSTRUCTION_SITES).length;
	}

	let creeps = 0;
	for (const _ in Game.creeps) creeps++;

	// Ceiling from harvestBudget (updated by spawn each tick)
	let maxH = 0;
	let maxP = 0;
	const emp = Memory.empire as {
		spawnBudget?: Record<string, { maxEtSpawnBound?: number; maxEtPhysics?: number }>;
		harvestCeiling?: Record<string, { maxEtSpawnBound?: number; maxEtPhysics?: number }>;
	} | undefined;
	const ceilMap = (emp && (emp.harvestCeiling || emp.spawnBudget)) || {};
	for (const rn in ceilMap) {
		const c = ceilMap[rn]!;
		if (c.maxEtSpawnBound != null && c.maxEtSpawnBound > maxH) maxH = c.maxEtSpawnBound;
		if (c.maxEtPhysics != null && c.maxEtPhysics > maxP) maxP = c.maxEtPhysics;
	}

	return {
		[MetricKey.t]: Game.time,
		[MetricKey.rclMax]: rclMax,
		[MetricKey.colonies]: colonies,
		[MetricKey.creeps]: creeps,
		[MetricKey.controlPoints]: s.controlPoints,
		[MetricKey.harvested]: s.harvested,
		[MetricKey.build]: s.build,
		[MetricKey.upgrade]: s.upgrade,
		[MetricKey.creepEnergy]: s.creepEnergy,
		[MetricKey.storedEnergy]: storedEnergy,
		[MetricKey.sites]: sites,
		[MetricKey.progress]: progress,
		[MetricKey.gcl]: gclLevelFloat(),
		[MetricKey.cpu]: Game.cpu.getUsed(),
		[MetricKey.bucket]: Game.cpu.bucket,
		[MetricKey.harvestRate]: rates.harvestRate,
		[MetricKey.controlRate]: rates.controlRate,
		[MetricKey.upgradeRate]: rates.upgradeRate,
		[MetricKey.buildRate]: rates.buildRate,
		[MetricKey.maxHarvestEt]: maxH,
		[MetricKey.maxHarvestPhysics]: maxP,
		[MetricKey.harvestedClaimed]: s.harvestedClaimed,
		[MetricKey.harvestedRemote]: s.harvestedRemote,
		[MetricKey.claimedHarvestRate]: rates.claimedHarvestRate,
		[MetricKey.remoteHarvestRate]: rates.remoteHarvestRate,
	};
}

function flushSegment(snap: MetricSample): void {
	const s = state();
	s.seq++;
	// Advance rate baseline after computing this sample
	s.prevAt = Game.time;
	s.prevHarvested = s.harvested;
	s.prevHarvestedClaimed = s.harvestedClaimed;
	s.prevHarvestedRemote = s.harvestedRemote;
	s.prevControlPoints = s.controlPoints;
	s.prevUpgrade = s.upgrade;
	s.prevBuild = s.build;

	if (typeof RawMemory === 'undefined') return;

	if (!s.segmentActive || Game.time % 100 === 0) {
		try {
			RawMemory.setActiveSegments([METRICS_SEGMENT]);
			s.segmentActive = true;
		} catch {
			/* empty */
		}
	}

	const payload = {
		[SegmentKey.version]: PROTOCOL_VERSION,
		[SegmentKey.seq]: s.seq,
		[SegmentKey.written]: Game.time,
		[SegmentKey.sample]: snap,
	};
	RawMemory.segments[METRICS_SEGMENT] = JSON.stringify(payload);
}

/** Expand dense sample to long keys for TensorBoard JSONL / harness. */
export function expandSample(d: MetricSample): Record<string, number> {
	return expandMetricSample(d as unknown as Record<string, number>);
}

export function tick(): MetricSample | null {
	ingestEventLogs();
	ingestSpawns();

	let creeps = 0;
	for (const _ in Game.creeps) creeps++;
	const s = state();
	const rates = windowRates(s);
	// Lightweight stats bag — MetricKey enums only (never raw wire strings)
	const stats: Partial<Record<MetricKey, number>> = {
		[MetricKey.t]: Game.time,
		[MetricKey.creeps]: creeps,
		[MetricKey.cpu]: Game.cpu.getUsed(),
		[MetricKey.bucket]: Game.cpu.bucket,
		[MetricKey.harvested]: s.harvested,
		[MetricKey.harvestedClaimed]: s.harvestedClaimed,
		[MetricKey.harvestedRemote]: s.harvestedRemote,
		[MetricKey.controlPoints]: s.controlPoints,
		[MetricKey.harvestRate]: rates.harvestRate,
		[MetricKey.claimedHarvestRate]: rates.claimedHarvestRate,
		[MetricKey.remoteHarvestRate]: rates.remoteHarvestRate,
		[MetricKey.controlRate]: rates.controlRate,
		[MetricKey.upgradeRate]: rates.upgradeRate,
	};
	(Memory as Memory & { [key: string]: unknown })[MemRoot.stats] = stats;

	if (Game.time % WRITE_EVERY !== 0) return null;
	const snap = sample();
	flushSegment(snap);
	return snap;
}

export { METRICS_SEGMENT, MetricKey, SegmentKey, MemRoot, expandMetricSample };
