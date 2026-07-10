/**
 * Event-log based empire stats + periodic console summary.
 * Tracks: energyHarvested, upgradePoints, buildEnergy, spawnEnergy.
 */
const config = require('config');

function ensure() {
	Memory.stats ||= {
		tick: 0,
		energyHarvested: 0,
		upgradePoints: 0,
		buildEnergy: 0,
		spawnEnergy: 0,
		// Rolling window since last summary.
		window: {
			energyHarvested: 0,
			upgradePoints: 0,
			buildEnergy: 0,
			spawnEnergy: 0,
			start: Game.time,
		},
	};
	return Memory.stats;
}

function add(field, amount) {
	if (!amount || amount <= 0) return;
	const s = ensure();
	s[field] = (s[field] || 0) + amount;
	s.window[field] = (s.window[field] || 0) + amount;
}

function onHarvest(amount) {
	add('energyHarvested', amount);
}

function onUpgrade(amount) {
	add('upgradePoints', amount);
}

function onBuild(amount) {
	add('buildEnergy', amount);
}

function onSpawn(amount) {
	add('spawnEnergy', amount);
}

/**
 * Scan creeps' last-tick actions via memory markers and Game notifications.
 * Roles call stats hooks directly; this also samples structure deltas occasionally.
 */
function tick() {
	const s = ensure();
	s.tick = Game.time;

	// Count creeps / rooms for summary.
	s.creeps = Object.keys(Game.creeps).length;
	s.colonies = 0;
	s.rclSum = 0;
	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		if (room.controller && room.controller.my) {
			s.colonies++;
			s.rclSum += room.controller.level;
		}
	}

	// Sample spawn energy spent this tick (spawning creeps remainingTime).
	for (const name in Game.spawns) {
		const spawn = Game.spawns[name];
		if (spawn.spawning && spawn.spawning.remainingTime === spawn.spawning.needTime - 1) {
			// Just started — estimate cost from creep memory if available later.
		}
	}

	const interval = config.consoleSummaryInterval || config.statsInterval || 50;
	if (Game.time % interval === 0) {
		printSummary();
	}
}

function printSummary() {
	const s = ensure();
	const w = s.window || {};
	const elapsed = Math.max(1, Game.time - (w.start || Game.time));
	const used = Game.cpu.getUsed();
	console.log(
		`Apex v2 t=${Game.time} cpu=${used.toFixed(1)}/${Game.cpu.limit} bucket=${Game.cpu.bucket}` +
		` creeps=${s.creeps || 0} colonies=${s.colonies || 0}` +
		` | Δ${elapsed}t harvest=${w.energyHarvested || 0}` +
		` upgrade=${w.upgradePoints || 0}` +
		` build=${w.buildEnergy || 0}` +
		` spawn=${w.spawnEnergy || 0}`,
	);
	// Reset window.
	s.window = {
		energyHarvested: 0,
		upgradePoints: 0,
		buildEnergy: 0,
		spawnEnergy: 0,
		start: Game.time,
	};
}

/**
 * Estimate harvest energy from WORK parts when a miner successfully harvests.
 * Call after harvest() returns OK.
 */
function noteHarvest(creep, result) {
	if (result !== OK) return;
	const work = creep.getActiveBodyparts(WORK);
	// Up to 2 energy per WORK per tick on sources.
	onHarvest(work * 2);
}

function noteUpgrade(creep, result) {
	if (result !== OK) return;
	const work = creep.getActiveBodyparts(WORK);
	onUpgrade(work);
}

function noteBuild(creep, result) {
	if (result !== OK) return;
	const work = creep.getActiveBodyparts(WORK);
	// Each WORK spends up to 5 energy building.
	const spent = Math.min(work * 5, require('util').energyOf(creep.store));
	onBuild(spent);
}

function noteSpawnCost(cost) {
	onSpawn(cost);
}

module.exports = {
	ensure,
	tick,
	onHarvest,
	onUpgrade,
	onBuild,
	onSpawn,
	noteHarvest,
	noteUpgrade,
	noteBuild,
	noteSpawnCost,
	printSummary,
};
