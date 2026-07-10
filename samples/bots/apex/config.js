/**
 * Empire-wide tunables. Adjust these before re-uploading code.
 * All distances are in tiles; all energy values are units.
 */
module.exports = {
	// --- Identity ---
	sign: 'Apex · xxscreeps sample empire',
	visuals: true,

	// --- Bootstrap ---
	// Below this RCL, prefer fat multi-role workers over specialized bodies.
	bootstrapRcl: 3,
	// Emergency: if no haulers and spawn energy low, force a tiny bootstrap creep.
	emergencyEnergy: 300,

	// --- Mining ---
	// Source regen is 300 ticks / 3000 energy → 10 e/t. 5 WORK harvests 10 e/t.
	sourceWorkParts: 5,
	// Keep one free adjacent spot for a hauler when possible; otherwise pack miners.
	reserveHaulerSpot: true,
	// Build a container near each source once RCL allows roads/containers.
	containerRcl: 2,
	// Prefer links over containers at the controller/source when RCL ≥ 5.
	linkRcl: 5,

	// --- Hauling ---
	// Target hauler capacity ≈ 2 * pathLength * sourceRate so pipelines stay full.
	haulerCarryPerPathTile: 2,
	maxHaulerCarryParts: 25,
	minHaulerCarryParts: 2,

	// --- Upgrading ---
	// Keep controller away from downgrade; spam upgrade when storage is fat.
	upgradeSafeTicks: 8000,
	upgradePushStorage: 50_000,
	// Max concurrent dedicated upgraders per colony (storage-fed).
	maxUpgraders: 4,

	// --- Building / repair ---
	maxBuilders: 3,
	// Towers/roads/containers repaired below this hits ratio by repairers.
	repairThreshold: 0.75,
	// Walls/ramparts: repair up to this hits while energy is plentiful.
	wallHitsTarget: {
		3: 5_000,
		4: 20_000,
		5: 50_000,
		6: 200_000,
		7: 1_000_000,
		8: 10_000_000,
	},
	// Never drain storage below this for wall grinding.
	wallGrindMinStorage: 100_000,

	// --- Remotes ---
	// Auto-claim adjacent rooms as mining outposts when colony RCL ≥ this.
	remoteMinRcl: 3,
	// Max simultaneous remote rooms per colony.
	maxRemotesPerColony: 4,
	// Reserve when reservation ticks fall below this.
	reserveRefresh: 2000,
	// Abandon a remote if hostiles seen this often without clearance.
	remoteThreatCooldown: 500,

	// --- Expansion (claim new colonies) ---
	expandMinRcl: 7,
	// Need this much free GCL (gcl.level - owned rooms) to claim.
	expandMinFreeGcl: 1,
	// Storage energy floor before sending a claimer.
	expandMinStorage: 80_000,

	// --- Combat / defense ---
	// Hostiles with ATTACK/RANGED/HEAL count as military threat.
	towerFocusHealers: true,
	// Spawn defenders when hostile military body parts ≥ this in a colony.
	defendThreatParts: 1,
	// Safe mode if hostiles inside and tower energy is low or walls breached.
	safeModeOnBreach: true,
	// Squad size for attack flags.
	attackSquad: { attackers: 2, healers: 2, ranged: 1 },

	// --- Spawning ---
	// Names are role + short id; max lifetime bookkeeping ticks.
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

	// --- Intel ---
	scoutInterval: 200,
	intelStale: 1000,

	// --- Logistics ---
	// Link: source links push to storage/controller links.
	linkBuffer: 400,
	// Terminal: keep this energy at home; excess may be sent to poorer colonies.
	terminalEnergyBalance: 50_000,
	terminalSendMin: 10_000,
};
