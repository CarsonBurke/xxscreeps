/**
 * Apex v3 — delegation economy + war/econ projections.
 */
module.exports = {
	sign: 'Apex v3 · delegated empire',
	version: 3,
	visuals: true,

	// --- Bootstrap (only until specialization is affordable) ---
	bootstrapRcl: 2,
	emergencyEnergy: 300,
	maxBootstrapWorkers: 4,
	specializeCapacity: 550,

	// --- Harvesters (static at sources) ---
	sourceWorkParts: 5,
	// 1 CARRY to drip into container; drop if none.
	harvesterCarry: 1,
	containerRcl: 2,

	// --- Haulers ---
	// Carry ≈ 2 * pathLen * e/t energy → parts; formula uses energy/tick.
	sourceEnergyPerTick: 10,
	haulerCarryPerPathTile: 2, // energy of carry per path tile (× e/t elsewhere)
	maxHaulerCarryParts: 25,
	minHaulerCarryParts: 2,

	// --- Fillers (spawn + extensions) ---
	// Stand near spawn; top up spawn/extensions from nearby piles/containers.
	fillerRange: 3,
	maxFillers: 2,

	// --- Upgraders (controller parking) ---
	upgradeSafeTicks: 8000,
	upgradePushStorage: 40_000,
	maxUpgraders: 4,
	upgraderParkRange: 3,

	// --- Builders ---
	maxBuilders: 3,
	repairThreshold: 0.7,

	// --- Remotes ---
	// Remotes can start as soon as we can specialize a bit; empire heuristics scale count.
	remoteMinRcl: 2,
	maxRemotesPerColony: 6,
	reserveRefresh: 2000,
	remoteThreatCooldown: 500,

	// --- Expansion (GCL-driven, not RCL-gated) ---
	// Claim as soon as free GCL exists. Weak colonies can still lead if they're all we have.
	expandMinRcl: 1,
	expandMinFreeGcl: 1,
	// Soft preference only — empire.js still expands with empty storage.
	expandMinStorage: 0,
	expandPreferEstablished: true,

	// --- Combat (war.js may override budgets) ---
	defendThreatParts: 1,
	safeModeOnBreach: true,
	towerFocusHealers: true,
	towerRampartAware: true,
	defenderRecycleSafeTicks: 200,
	attackSquad: { attackers: 2, healers: 2, ranged: 1 },
	squadRallyBeforeMarch: true,
	squadRallyRange: 3,
	// War abandon defaults (war.js reads these; aliases maxDeaths/noProgressTicks OK)
	war: {
		energyBudget: 50_000,
		maxDeaths: 8,
		noProgressTicks: 1500,
		cooldownTicks: 1000,
		winCooldownTicks: 200,
		clearSafeTicks: 30,
	},

	// --- Spawning ---
	spawnPriority: [
		'bootstrap',
		'defender',
		'harvester',
		'filler',
		'hauler',
		'remoteHarvester',
		'remoteHauler',
		'reserver',
		'upgrader',
		'builder',
		'scout',
		'claimer',
		'attacker',
		'ranged',
		'healer',
		'dismantler',
	],
	spawnEnergyHysteresis: 0.92,

	// --- Metrics / intel ---
	scoutInterval: 200,
	intelStale: 1000,
	consoleSummaryInterval: 50,

	// --- Logistics ---
	linkBuffer: 400,
	terminalEnergyBalance: 50_000,
	terminalSendMin: 10_000,

	// --- CPU ---
	lowBucketThreshold: 2000,
};
