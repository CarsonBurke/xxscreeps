/**
 * Apex v2 empire-wide tunables.
 * All distances are in tiles; all energy values are units.
 */
module.exports = {
	// --- Identity ---
	sign: 'Apex v2 · xxscreeps sample empire',
	version: 2,
	visuals: true,

	// --- Bootstrap ---
	// Below this RCL (and low capacity), prefer multi-role workers.
	bootstrapRcl: 3,
	// Emergency: if no haulers and spawn energy low, force a tiny bootstrap creep.
	emergencyEnergy: 300,
	// Cap concurrent bootstrap workers so we do not starve specialization.
	maxBootstrapWorkers: 6,
	// Once capacity ≥ this, stop spawning bootstrap and use specialized roles.
	specializeCapacity: 550,

	// --- Mining ---
	// Source regen is 300 ticks / 3000 energy → 10 e/t. 5 WORK harvests 10 e/t.
	sourceWorkParts: 5,
	// Keep one free adjacent spot for a hauler when possible.
	reserveHaulerSpot: true,
	// Build a container near each source once RCL allows.
	containerRcl: 2,
	// Prefer links over containers at controller/source when RCL ≥ 5.
	linkRcl: 5,
	// Drop energy only when no container/link exists yet.
	dropMineUntilContainer: true,

	// --- Hauling ---
	// Target hauler capacity ≈ 2 * pathLength * sourceRate so pipelines stay full.
	haulerCarryPerPathTile: 2,
	maxHaulerCarryParts: 25,
	minHaulerCarryParts: 2,
	// Source rate for demand model (energy/tick at full mine).
	sourceEnergyPerTick: 10,
	// Prefer fuller containers when assigning haulers.
	haulerFullnessBias: true,

	// --- Upgrading ---
	upgradeSafeTicks: 8000,
	upgradePushStorage: 50_000,
	maxUpgraders: 4,

	// --- Building / repair ---
	maxBuilders: 3,
	repairThreshold: 0.75,
	wallHitsTarget: {
		3: 5_000,
		4: 20_000,
		5: 50_000,
		6: 200_000,
		7: 1_000_000,
		8: 10_000_000,
	},
	wallGrindMinStorage: 100_000,

	// --- Remotes (FSM) ---
	remoteMinRcl: 3,
	maxRemotesPerColony: 4,
	reserveRefresh: 2000,
	// Abandon remote if hostiles seen within this window without clearance.
	remoteThreatCooldown: 500,
	// Sustained threat ticks before abandon.
	remoteAbandonThreatTicks: 50,
	// Remote FSM phase names (documented; used as memory values).
	// scout → reserve → container → mine → haul
	remotePhases: [ 'scout', 'reserve', 'container', 'mine', 'haul' ],

	// --- Expansion ---
	expandMinRcl: 7,
	expandMinFreeGcl: 1,
	expandMinStorage: 80_000,
	// Prefer rooms with this many sources for claim/remote scoring.
	preferredSourceCount: 2,

	// --- Combat / defense ---
	towerFocusHealers: true,
	// Prefer hostiles standing on / next to ramparts last (harder targets).
	towerRampartAware: true,
	defendThreatParts: 1,
	// Recycle idle defenders after this many safe ticks.
	defenderRecycleSafeTicks: 200,
	safeModeOnBreach: true,
	attackSquad: { attackers: 2, healers: 2, ranged: 1 },
	// Squad must gather at rally before marching.
	squadRallyBeforeMarch: true,
	squadRallyRange: 3,

	// --- Spawning ---
	// Wait until available energy ≥ this fraction of full-body cost (hysteresis).
	spawnEnergyHysteresis: 0.92,
	// Ticks of headroom: replace when TTL < body cost ticks + travel buffer.
	replaceTtlBuffer: 30,
	// Fixed pre-spawn threshold fallback when ETA cannot be computed.
	replaceMinTtl: 80,
	spawnPriority: [
		'bootstrap',
		'defender',
		'miner',
		'hauler',
		'remoteMiner',
		'remoteHauler',
		'reserver',
		'upgrader',
		'builder',
		'repairer',
		'mineralMiner',
		'scout',
		'claimer',
		'attacker',
		'ranged',
		'healer',
		'dismantler',
	],

	// --- Traffic / movement ---
	// Rebuild room CostMatrix every N ticks.
	costMatrixRefresh: 25,
	// Default PathFinder plain/swamp costs; roads preferred via matrix.
	plainCost: 2,
	swampCost: 10,
	roadCost: 1,
	// reusePath by role (ticks).
	pathReuseByRole: {
		miner: 40,
		remoteMiner: 40,
		hauler: 12,
		remoteHauler: 15,
		upgrader: 25,
		builder: 15,
		repairer: 20,
		bootstrap: 12,
		scout: 30,
		reserver: 40,
		claimer: 40,
		defender: 5,
		attacker: 5,
		ranged: 5,
		healer: 3,
		dismantler: 8,
		mineralMiner: 30,
		default: 15,
	},
	// Park haulers one tile off the miner seat so they do not block harvest.
	haulerParkRange: 1,

	// --- Construction ---
	// Max construction sites per colony room.
	siteBudget: 18,
	// Max new sites of any type created in one planner pass.
	sitesPerPass: 6,
	// Max concurrent road sites; prefer heat-based placement.
	maxRoadSites: 8,
	// Path heat threshold before placing a road tile (visits).
	roadHeatThreshold: 8,
	// Decay path heat every N ticks.
	roadHeatDecayInterval: 200,
	// Complete existing sites of a type before placing more of that type.
	finishSitesBeforeMore: true,

	// --- Intel ---
	scoutInterval: 200,
	intelStale: 1000,

	// --- Logistics ---
	linkBuffer: 400,
	// Prefer link transfer over hauler when both ends exist.
	linkFirst: true,
	terminalEnergyBalance: 50_000,
	terminalSendMin: 10_000,

	// --- CPU / robustness ---
	// Skip remotes, scouts, and expansion when bucket below this.
	lowBucketThreshold: 2000,
	// Skip non-critical pathing visuals when bucket is low.
	criticalBucketThreshold: 500,

	// --- Observability ---
	statsInterval: 50,
	consoleSummaryInterval: 50,
};
