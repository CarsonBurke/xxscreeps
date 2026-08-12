/**
 * Apex tunables — only real constraints and soft bars.
 * No RCL ladders or arbitrary “max N of role” caps.
 * Physics / game limits (CONTROLLER_STRUCTURES, MAX_CREEP_SIZE, spawn duty) are the throttle.
 */
const config = {
	sign: 'Apex v4 · phase-aware empire',
	version: 4,
	visuals: true,

	/**
	 * Full mine WORK per normal source: ceil(10 e/t / HARVEST_POWER 2) = 5.
	 * Bodies may be smaller and stacked (multi-harvester).
	 */
	sourceWorkParts: 5,
	harvesterCarry: 1,

	/**
	 * Soft hauler packing hint (still clamped by MAX_CREEP_SIZE in body builder).
	 * Not a “max haulers” count.
	 */
	haulerCarryPerPathTile: 2,
	minHaulerCarryParts: 2,

	fillerRange: 3,

	/** Controller ticks-to-downgrade: push upgrade when below this (real risk). */
	upgradeSafeTicks: 8000,
	upgraderParkRange: 3,

	repairThreshold: 0.7,

	/**
	 * Planner: place whatever CONTROLLER_STRUCTURES allows.
	 * Site budget is how many new sites per pass (CPU), not a permanent room cap.
	 * Game limit is 100 construction sites.
	 */
	plannerSiteBudget: 6,
	plannerCooldown: 10,
	/** Soft ceiling on open sites (game max 100). Leave headroom for remotes. */
	maxOpenSites: 40,

	/**
	 * Remotes = economic foundation whenever a spawn exists.
	 * No RCL gate, no room-count cap — spawn duty + threat are the limits.
	 */
	/** Soft bar for spawn pushBoost / TB (not a hard stop on remotes). */
	targetRemoteEt: 40,
	reserveRefresh: 2000,
	/** Skip remotes with recent hostiles (real: don't feed invaders). */
	remoteThreatCooldown: 500,

	/**
	 * Source Keeper rooms: only remote if we can field a melee killer that wins.
	 * SK ≈ 5000 hits, ~10A+10RA (~400 dps). Solo pure A/M needs ~19A19M ≈ 2470e
	 * to kill before dying. Below this capacity, skip SK entirely.
	 */
	skKillerMinEnergy: 2500,
	skKillerMinAttack: 19,

	/** Expansion: free GCL is the scarce resource — no min RCL/storage ladder. */
	expandPreferEstablished: true,

	defendThreatParts: 1,
	safeModeOnBreach: true,
	attackSquad: { attackers: 2, healers: 2, ranged: 1 },
	war: {
		energyBudget: 50_000,
		maxDeaths: 8,
		noProgressTicks: 800,
		cooldownTicks: 2000,
	},

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
	intelStale: 1000,
	consoleSummaryInterval: 50,
	linkBuffer: 400,
	terminalEnergyBalance: 50_000,
	terminalSendMin: 10_000,
	lowBucketThreshold: 2000,
};

export = config;
