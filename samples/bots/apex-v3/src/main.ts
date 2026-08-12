/**
 * Apex v4 — phase-aware RCL push + enum memory.
 * Install from dist/:
 *   npx xxscreeps manage bot add apex-v3 samples/bots/apex-v3/dist --spawn W5N5
 *
 * Memory: MetricKey in segment 87; CreepMem / RoomApexMem / Role numeric enums.
 */
import config = require('./config');
import * as intel from './intel';
import * as roomManager from './room';
import * as roles from './roles';
import * as metrics from './metrics';
import { runMemoryHack } from './memoryHack';
import { getRole } from './creepMem';
import { beginTick, endTickMovement } from './traffic';
import { MemRoot, MetricKey } from './memoryKeys';

function cleanCreepMemory(): void {
	if (!Memory.creeps) return;
	for (const name in Memory.creeps) {
		if (!Game.creeps[name]) delete Memory.creeps[name];
	}
}

function initEmpire(): void {
	Memory.empire ||= {
		attacks: [],
		claims: [],
		defends: [],
		forcedRemotes: {},
		ignoreRooms: {},
		campaigns: {},
		economy: {},
		version: 4,
	} as Memory['empire'];
	(Memory.empire as { version: number }).version = 4;
	Memory.intel ||= {};
}

function runCreeps(): void {
	for (const name in Game.creeps) {
		const creep = Game.creeps[name]!;
		if (creep.spawning) continue;
		// getRole() migrates legacy string roles → Role enum once
		getRole(creep);
		roles.run(creep);
	}
}

function cpuBanner(): void {
	if (Game.time % (config.consoleSummaryInterval || 50) !== 0) return;
	const used = Game.cpu.getUsed();
	const creeps = Object.keys(Game.creeps).length;
	let colonies = 0;
	for (const n in Game.rooms) {
		const r = Game.rooms[n]!;
		if (r.controller && r.controller.my) colonies++;
	}
	const plan = Memory.empire && (Memory.empire as { plan?: {
		freeGcl?: number;
		expansions?: { room: string }[];
		leadColony?: string;
	} }).plan;
	const freeGcl = plan ? plan.freeGcl : '?';
	const expand = plan && plan.expansions ? plan.expansions.map(e => e.room).join(',') : '';
	const lead = plan && plan.leadColony ? plan.leadColony : '?';
	const sb = Memory.empire && (Memory.empire as {
		spawnBudget?: Record<string, {
			maxEtPhysics?: number;
			maxEtSpawnBound?: number;
			maxEtLocalBound?: number;
			maxEtRemoteBound?: number;
			realizedEt?: number;
			gapEt?: number;
			surplusEt?: number;
			harvestEfficiency?: number;
			spawnUtil?: number;
			spawnsToTakeAll?: number;
			targetUpWork?: number;
			upWork?: number;
			sources?: { haveWork?: number; needWork?: number; bodyW?: number }[];
		}>;
	}).spawnBudget;
	const leadSb = lead && lead !== '?' && sb ? sb[lead] : null;
	let mine = '';
	if (leadSb) {
		const w = (leadSb.sources || []).map(s =>
			`${s.haveWork || 0}/${s.needWork || 5}${s.bodyW != null ? `@${s.bodyW}` : ''}`).join(',');
		// got / spawnBoundCeiling / physics — gap is how far below the bar
		mine = ` mineW=${w}` +
			` H=${leadSb.realizedEt}/${leadSb.maxEtSpawnBound}/${leadSb.maxEtPhysics}` +
			` (L${leadSb.maxEtLocalBound}+R${leadSb.maxEtRemoteBound})` +
			` gap=${leadSb.gapEt}` +
			` eff=${((leadSb.harvestEfficiency || 0) * 100).toFixed(0)}%` +
			` needSpawns≈${leadSb.spawnsToTakeAll}` +
			` surp=${leadSb.surplusEt} upW=${leadSb.upWork}/${leadSb.targetUpWork}` +
			` spawnUtil=${((leadSb.spawnUtil || 0) * 100).toFixed(0)}%`;
	}
	const stats = (Memory as Memory & { [key: string]: unknown })[MemRoot.stats] as
		| Partial<Record<MetricKey, number>>
		| undefined;
	const fmt = (k: MetricKey) =>
		stats && stats[k] != null ? Number(stats[k]).toFixed(1) : '?';
	const hr = fmt(MetricKey.harvestRate);
	const hcr = fmt(MetricKey.claimedHarvestRate);
	const hor = fmt(MetricKey.remoteHarvestRate);
	const cr = fmt(MetricKey.controlRate);
	console.log(
		`Apex v4 t=${Game.time} cpu=${used.toFixed(1)}/${Game.cpu.limit} bucket=${Game.cpu.bucket}` +
		` creeps=${creeps} colonies=${colonies} freeGCL=${freeGcl} lead=${lead}` +
		` e/t=${hr} (claimed=${hcr} remote=${hor}) cp/t=${cr}` +
		mine +
		(expand ? ` expand→${expand}` : ''),
	);
}

export function loop(): void {
	try {
		runMemoryHack();
		beginTick(); // clear traffic intents
		initEmpire();
		cleanCreepMemory();
		intel.tick();
		roomManager.runAll(); // spawn / planner / defense
		runCreeps(); // roles register move intents
		endTickMovement(); // relay + traffic resolve → creep.move
		metrics.tick();
		cpuBanner();
	} catch (err) {
		console.log('Apex v4 FATAL', err && (err as Error).stack || err);
	}
}

// Screeps entry: module.exports.loop
module.exports.loop = loop;
