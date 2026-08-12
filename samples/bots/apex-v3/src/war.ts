// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * Apex v3 war / campaign strategy.
 *
 * Room orders are RoomIntent enums on Memory.rooms[name].a (synced by intel from
 * flag *colors*, never flag names). Example:
 *   if (getRoomIntent(room) === RoomIntent.attack) open attack campaign
 *
 * Campaign types: remote | claim | attack | defend
 *
 * Public API:
 *   war.tick()                  — after intel.tick() (intents already synced)
 *   war.spawnRequests(homeRoom)
 */
const config = require('./config');
const combat = require('./combat');
const {
	hostileThreat,
	ownedRooms,
	lowBucket,
} = require('./util');
const {
	RoomIntent,
	getRoomIntent,
	getIntentSnapshot,
} = require('./intents');
const {
	getRole, getCampaignId, memInit, Role, CreepMem,
} = require('./creepMem');
const { getRemotes } = require('./roomMem');

// ---------------------------------------------------------------------------
// Tunables (override via config.war if present)
// ---------------------------------------------------------------------------

const WAR_CFG = {
	// Energy spent on military for a campaign before abandon-without-progress.
	energyBudget: 50_000,
	// Deaths without a clear before abandon.
	deathLimit: 8,
	// Ticks with no progress event → abandon.
	stallTicks: 1_500,
	// After abandon / win, do not re-open same room+type for this long.
	cooldownTicks: 1_000,
	// Brief cool after win so we don't thrash.
	winCooldownTicks: 200,
	// Room is "clear" after this many consecutive safe visible ticks.
	clearSafeTicks: 30,
	// Open a remote-security campaign when threat ≥ this (military parts).
	remoteThreatOpen: 1,
	// Also open remote campaign if invader core present.
	remoteOpenOnCore: true,
	// Abandon if enemy controller level > our home RCL + this delta.
	maxEnemyRclDelta: 1,
	// Abandon if enemy towers ≥ this and our home RCL is below minRclForTowers.
	enemyTowerAbandon: 2,
	minHomeRclForTowers: 6,
	// Max concurrent active campaigns per home.
	maxActivePerHome: 3,
	// Squad compositions by campaign type.
	squads: {
		remote: { attackers: 1, healers: 1, ranged: 0, dismantlers: 0, defenders: 0 },
		claim: { attackers: 1, healers: 1, ranged: 0, dismantlers: 0, defenders: 1 },
		attack: { attackers: 2, healers: 2, ranged: 1, dismantlers: 0, defenders: 0 },
		// defend flag: station defenders
		defend: { attackers: 0, healers: 0, ranged: 0, dismantlers: 0, defenders: 2 },
	},
	// Spawn priorities (lower = more urgent in apex queues that sort ascending).
	priority: {
		defender: 5,
		attacker: 40,
		healer: 41,
		ranged: 42,
		dismantler: 43,
	},
	// Max campaign history retained.
	maxHistory: 40,
};

/**
 * Merge WAR_CFG with config.war.
 * Accepts aliases used by apex-v3 config: maxDeaths, noProgressTicks.
 */
function cfg() {
	const over = (config && config.war) || {};
	const mapped = Object.assign({}, over);
	if (over.maxDeaths != null && over.deathLimit == null) mapped.deathLimit = over.maxDeaths;
	if (over.noProgressTicks != null && over.stallTicks == null) mapped.stallTicks = over.noProgressTicks;
	return Object.assign({}, WAR_CFG, mapped, {
		squads: Object.assign({}, WAR_CFG.squads, over.squads || {}),
		priority: Object.assign({}, WAR_CFG.priority, over.priority || {}),
	});
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

/**
 * Campaigns are stored as Memory.empire.campaigns = { [id]: campaign }
 * so intel / empire can Object.values(...) them.
 */
function ensureEmpire() {
	Memory.empire ||= {};
	// Migrate array → object if an older war.js wrote an array.
	if (Array.isArray(Memory.empire.campaigns)) {
		const obj = {};
		for (const camp of Memory.empire.campaigns) {
			if (camp && camp.id) obj[camp.id] = camp;
		}
		Memory.empire.campaigns = obj;
	}
	if (!Memory.empire.campaigns || typeof Memory.empire.campaigns !== 'object') {
		Memory.empire.campaigns = {};
	}
	Memory.empire.attacks ||= [];
	Memory.empire.claims ||= [];
	Memory.empire.defends ||= [];
	Memory.empire.forcedRemotes ||= {};
	Memory.empire.ignoreRooms ||= {};
	return Memory.empire;
}

function campaignList() {
	return Object.values(ensureEmpire().campaigns);
}

function nextCampaignId(room, type) {
	Memory.empire._campaignSeq = (Memory.empire._campaignSeq || 0) + 1;
	return `${type}_${room}_${Memory.empire._campaignSeq}`;
}

function activeCampaigns(home) {
	return campaignList().filter(c =>
		c.status === 'active' && (!home || c.home === home));
}

function findCampaign(pred) {
	return campaignList().find(pred);
}

function inCooldown(room, type, home) {
	const c = cfg();
	const now = Game.time;
	return campaignList().some(camp => {
		if (camp.room !== room) return false;
		if (type && camp.type !== type) return false;
		if (home && camp.home !== home) return false;
		if (camp.status === 'active') return false;
		const until = camp.cooldownUntil ||
			((camp.ended || camp.lastProgress || 0) + c.cooldownTicks);
		return now < until;
	});
}

/** Per-home energy budget: min(config, empire.plan militaryBudget) when known. */
function budgetFor(home, baseBudget) {
	const plan = Memory.empire && Memory.empire.plan;
	const by = plan && plan.byColony && plan.byColony[home];
	if (by && by.militaryBudget != null && by.militaryBudget > 0) {
		// Allow at least half of static budget so tiny colonies can still clear invaders.
		return Math.max(baseBudget * 0.5, Math.min(baseBudget, by.militaryBudget));
	}
	return baseBudget;
}

// ---------------------------------------------------------------------------
// Intel / room observation
// ---------------------------------------------------------------------------

function roomIntel(roomName) {
	return (Memory.intel && Memory.intel[roomName]) || null;
}

function observeRoom(roomName) {
	const room = Game.rooms[roomName];
	const intel = roomIntel(roomName);
	const out = {
		visible: !!room,
		threat: 0,
		hostileCount: 0,
		core: false,
		towers: 0,
		enemyRcl: 0,
		myController: false,
		hasSpawn: false,
		reservationTicks: 0,
		owner: null,
	};

	if (room) {
		const hostiles = room.find(FIND_HOSTILE_CREEPS);
		out.hostileCount = hostiles.length;
		out.threat = hostileThreat(room);
		const coreType = typeof STRUCTURE_INVADER_CORE !== 'undefined'
			? STRUCTURE_INVADER_CORE
			: 'invaderCore';
		out.core = room.find(FIND_HOSTILE_STRUCTURES, {
			filter: s => s.structureType === coreType,
		}).length > 0;
		out.towers = room.find(FIND_HOSTILE_STRUCTURES, {
			filter: s => s.structureType === STRUCTURE_TOWER,
		}).length;
		if (room.controller) {
			out.myController = !!room.controller.my;
			out.enemyRcl = (!room.controller.my && room.controller.owner)
				? (room.controller.level || 0)
				: 0;
			out.owner = room.controller.owner && room.controller.owner.username;
			if (room.controller.reservation) {
				out.reservationTicks = room.controller.reservation.ticksToEnd || 0;
			}
		}
		out.hasSpawn = room.find(FIND_MY_SPAWNS).length > 0;
	} else if (intel) {
		out.threat = intel.threat || 0;
		out.hostileCount = out.threat > 0 ? 1 : 0;
		if (intel.controller) {
			out.myController = !!intel.controller.my;
			out.enemyRcl = (!intel.controller.my && intel.controller.owner)
				? (intel.controller.level || 0)
				: 0;
			out.owner = intel.controller.owner || intel.owner;
			out.reservationTicks = (intel.controller.reservation && intel.controller.reservation.ticks) || 0;
		}
		// Tower count from last visible snapshot if stored.
		out.towers = intel.hostileTowers || 0;
		out.core = !!intel.invaderCore;
		out.hasSpawn = !!intel.mySpawn;
	}
	return out;
}

function homeRcl(homeName) {
	const room = Game.rooms[homeName];
	if (room && room.controller) return room.controller.level || 0;
	return 0;
}

// ---------------------------------------------------------------------------
// Campaign open / progress / abandon
// ---------------------------------------------------------------------------

function openCampaign({ type, room, home, goal, squad }) {
	const c = cfg();
	if (inCooldown(room, type, home)) return null;
	if (activeCampaigns(home).length >= c.maxActivePerHome) return null;

	// Don't duplicate active campaign for same room+type+home.
	const existing = findCampaign(camp =>
		camp.status === 'active' &&
		camp.room === room &&
		camp.type === type &&
		camp.home === home);
	if (existing) return existing;

	const camp = {
		id: nextCampaignId(room, type),
		type,
		room,
		home,
		started: Game.time,
		lastProgress: Game.time,
		spentEnergy: 0,
		deaths: 0,
		kills: 0,
		safeTicks: 0,
		goal: goal || defaultGoal(type),
		status: 'active',
		/** Player/empire RoomIntent that opened this; null = auto (e.g. remote security) */
		intent: type === 'remote' ? null : typeToIntent(type),
		squad: squad || c.squads[type] || c.squads.attack,
		assigned: {},
		lastHostileCount: null,
		lastReservation: null,
		abandonReason: null,
		ended: null,
		cooldownUntil: null,
	};
	ensureEmpire().campaigns[camp.id] = camp;
	console.log(`Apex v3 WAR open ${camp.id} ${type}→${room} from ${home} goal=${camp.goal}`);
	return camp;
}

/** Player-order campaign type → RoomIntent. Auto remotes use null, not this. */
function typeToIntent(type) {
	if (type === 'attack') return RoomIntent.attack;
	if (type === 'defend') return RoomIntent.defend;
	if (type === 'claim') return RoomIntent.claim;
	return RoomIntent.none;
}

/** True while player/empire still wants this order on the room. Auto campaigns always active. */
function intentStillActive(camp) {
	if (camp.intent == null || camp.intent === RoomIntent.none) return true;
	return getRoomIntent(camp.room) === camp.intent;
}

function defaultGoal(type) {
	if (type === 'remote') return 'clear_hostiles';
	if (type === 'claim') return 'claim';
	if (type === 'defend') return 'hold';
	return 'clear_hostiles';
}

function markProgress(camp, why) {
	camp.lastProgress = Game.time;
	camp._lastProgressWhy = why;
}

function endCampaign(camp, status, reason) {
	const c = cfg();
	camp.status = status;
	camp.ended = Game.time;
	camp.abandonReason = reason || null;
	const cool = status === 'won' ? c.winCooldownTicks : c.cooldownTicks;
	camp.cooldownUntil = Game.time + cool;
	// Clear assigned tracking.
	camp.assigned = {};
	console.log(
		`Apex v3 WAR ${status} ${camp.id} ${camp.type}→${camp.room}` +
		(reason ? ` (${reason})` : '') +
		` spent=${camp.spentEnergy} deaths=${camp.deaths}`,
	);
}

function tooStrong(camp, obs) {
	const c = cfg();
	const rcl = homeRcl(camp.home);
	if (obs.enemyRcl > 0 && obs.enemyRcl > rcl + c.maxEnemyRclDelta) {
		return `enemy_rcl_${obs.enemyRcl}_vs_home_${rcl}`;
	}
	if (obs.towers >= c.enemyTowerAbandon && rcl < c.minHomeRclForTowers) {
		return `enemy_towers_${obs.towers}_home_rcl_${rcl}`;
	}
	return null;
}

/**
 * Update accounting + progress + win/abandon for one active campaign.
 */
function tickCampaign(camp) {
	const c = cfg();
	const obs = observeRoom(camp.room);

	// --- Military assignment accounting (spentEnergy + deaths) ---
	const living = {};
	for (const name in Game.creeps) {
		const creep = Game.creeps[name];
		if (getCampaignId(creep) !== camp.id) continue;
		if (!combat.MILITARY_ROLES.includes(getRole(creep))) continue;
		const cost = combat.creepBodyCost(creep);
		living[name] = cost;
		if (camp.assigned[name] == null) {
			// First seen → count spawn cost toward campaign budget.
			camp.spentEnergy = (camp.spentEnergy || 0) + cost;
			camp.assigned[name] = cost;
		}
	}
	// Deaths: previously assigned, no longer living.
	for (const name of Object.keys(camp.assigned || {})) {
		if (!living[name] && !Game.creeps[name]) {
			camp.deaths = (camp.deaths || 0) + 1;
			// Body already counted at spawn; death itself is the progress failure signal.
			delete camp.assigned[name];
		}
	}
	// Drop stale names that somehow remain.
	camp.assigned = living;

	// --- Progress indicators ---
	if (obs.visible) {
		const prevH = camp.lastHostileCount;
		if (prevH != null && obs.hostileCount < prevH) {
			camp.kills = (camp.kills || 0) + (prevH - obs.hostileCount);
			markProgress(camp, 'hostile_kill');
		}
		if (camp.lastHostileCount == null && obs.hostileCount > 0) {
			// First contact still counts as "engaged".
			markProgress(camp, 'contact');
		}
		camp.lastHostileCount = obs.hostileCount;

		if (obs.hostileCount === 0 && !obs.core) {
			camp.safeTicks = (camp.safeTicks || 0) + 1;
			if (camp.safeTicks === c.clearSafeTicks) {
				markProgress(camp, 'room_clear');
			}
		} else {
			camp.safeTicks = 0;
		}

		// Reservation gain (remote / claim support).
		if (camp.lastReservation != null && obs.reservationTicks > camp.lastReservation + 5) {
			markProgress(camp, 'reservation');
		}
		camp.lastReservation = obs.reservationTicks;

		// Claim success.
		if (camp.type === 'claim' && obs.myController) {
			markProgress(camp, 'claimed');
		}
		if (camp.type === 'claim' && obs.hasSpawn) {
			markProgress(camp, 'spawn_up');
		}

		// Core destroyed: was true in intel / prior, now false.
		if (camp._hadCore && !obs.core) {
			markProgress(camp, 'core_cleared');
		}
		camp._hadCore = obs.core;
	}

	// --- Win conditions (RoomIntent status, not flag name presence) ---
	if (camp.type === 'remote' || camp.type === 'attack' || camp.type === 'defend') {
		const orderLifted = !intentStillActive(camp);
		if (camp.type === 'attack' && orderLifted && (camp.safeTicks || 0) >= c.clearSafeTicks) {
			endCampaign(camp, 'won', 'intent_cleared');
			return;
		}
		if ((camp.goal === 'clear_hostiles' || camp.goal === 'hold') &&
			(camp.safeTicks || 0) >= c.clearSafeTicks &&
			obs.hostileCount === 0 && !obs.core) {
			// Defend holds while RoomIntent.defend remains.
			if (camp.type === 'defend' && intentStillActive(camp)) {
				// stay active
			} else if (camp.type !== 'defend') {
				endCampaign(camp, 'won', 'cleared');
				return;
			}
		}
		if (camp.type === 'defend' && orderLifted) {
			endCampaign(camp, 'won', 'defend_intent_cleared');
			return;
		}
	}

	if (camp.type === 'claim') {
		if (obs.myController && obs.hasSpawn) {
			endCampaign(camp, 'won', 'spawn_established');
			return;
		}
		// Intent lifted and we never got the room — stall/abandon paths handle it.
	}

	// --- Abandon: enemy too strong ---
	const strong = tooStrong(camp, obs);
	if (strong) {
		endCampaign(camp, 'abandoned', strong);
		return;
	}

	// --- Abandon: deaths without clearing ---
	if ((camp.deaths || 0) > c.deathLimit && (camp.safeTicks || 0) < c.clearSafeTicks) {
		endCampaign(camp, 'abandoned', `deaths_${camp.deaths}`);
		return;
	}

	// --- Abandon: energy budget with no recent progress ---
	const stalled = Game.time - (camp.lastProgress || camp.started) > c.stallTicks;
	const budget = budgetFor(camp.home, c.energyBudget);
	if ((camp.spentEnergy || 0) > budget && stalled) {
		endCampaign(camp, 'abandoned', `budget_${camp.spentEnergy}`);
		return;
	}

	// --- Abandon: pure stall ---
	if (stalled) {
		endCampaign(camp, 'abandoned', `stall_${c.stallTicks}`);
		return;
	}
}

// ---------------------------------------------------------------------------
// Campaign discovery (RoomIntent enums — never flag-name strings)
// ---------------------------------------------------------------------------

function nearestHome(roomName) {
	const owned = ownedRooms();
	let best = null;
	let bestDist = Infinity;
	for (const r of owned) {
		const d = Game.map.getRoomLinearDistance(r.name, roomName);
		if (d < bestDist) {
			bestDist = d;
			best = r.name;
		}
	}
	return best;
}

/**
 * Open campaigns from room intent status:
 *   getRoomIntent(room) === RoomIntent.attack | defend | claim
 * Snapshot is filled by intel.syncIntentsFromFlags (color → enum).
 */
function openFromIntents() {
	const snap = getIntentSnapshot();
	if (!snap) return;
	const c = cfg();

	for (const room of snap.attacks) {
		const home = nearestHome(room);
		if (!home) continue;
		openCampaign({
			type: 'attack',
			room,
			home,
			goal: 'clear_hostiles',
			squad: c.squads.attack,
		});
	}

	for (const room of snap.claims) {
		// Already established colony — don't thrash claim campaigns.
		const owned = Game.rooms[room];
		if (owned && owned.controller && owned.controller.my &&
			owned.find(FIND_MY_SPAWNS).length > 0) {
			continue;
		}
		const home = nearestHome(room);
		if (!home) continue;
		openCampaign({
			type: 'claim',
			room,
			home,
			goal: 'claim',
			squad: c.squads.claim,
		});
	}

	for (const room of snap.defends) {
		const home = nearestHome(room);
		if (!home) continue;
		openCampaign({
			type: 'defend',
			room,
			home,
			goal: 'hold',
			squad: c.squads.defend,
		});
	}
}

/**
 * Open remote-security campaigns for remotes that show hostiles / cores.
 * Reads colony remotes from RoomApexMem bag (getRemotes).
 */
function openRemoteSecurity() {
	const c = cfg();
	if (lowBucket()) return;

	for (const homeRoom of ownedRooms()) {
		const home = homeRoom.name;
		const remotes = new Set(getRemotes(home));
		// Forced remotes adjacent-ish to this home.
		for (const r of Object.keys(Memory.empire.forcedRemotes || {})) {
			if (Game.map.getRoomLinearDistance(home, r) <= 2) remotes.add(r);
		}

		for (const remoteName of remotes) {
			if (Memory.empire.ignoreRooms && Memory.empire.ignoreRooms[remoteName]) continue;

			const obs = observeRoom(remoteName);
			const need =
				obs.threat >= c.remoteThreatOpen ||
				(c.remoteOpenOnCore && obs.core);

			if (!need) continue;

			openCampaign({
				type: 'remote',
				room: remoteName,
				home,
				goal: 'clear_hostiles',
				squad: c.squads.remote,
			});
		}
	}
}

/**
 * Support claim/expand rooms we already own but lack a spawn (pioneer phase).
 */
function openClaimSupport() {
	const c = cfg();
	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		if (!room.controller || !room.controller.my) continue;
		const spawns = room.find(FIND_MY_SPAWNS);
		if (spawns.length) continue;
		// Owned, no spawn — military screen while pioneers build.
		const home = nearestHome(name) || name;
		// Prefer a different home if this room has no spawn energy base.
		const donor = ownedRooms().find(r =>
			r.name !== name && r.find(FIND_MY_SPAWNS).length > 0);
		openCampaign({
			type: 'claim',
			room: name,
			home: donor ? donor.name : home,
			goal: 'claim',
			squad: c.squads.claim,
		});
	}
}

function pruneCampaigns() {
	const empire = ensureEmpire();
	const c = cfg();
	const now = Game.time;
	const list = campaignList();

	for (const camp of list) {
		if (camp.status === 'active') continue;
		const until = camp.cooldownUntil || (camp.ended || 0) + c.cooldownTicks;
		if (now < until) {
			if (camp.status !== 'cooldown') camp.status = 'cooldown';
			continue;
		}
		// Past cooldown → drop.
		delete empire.campaigns[camp.id];
	}

	const remaining = campaignList();
	if (remaining.length > c.maxHistory) {
		// Prefer dropping oldest non-active.
		remaining.sort((a, b) => {
			const aa = a.status === 'active' ? 1 : 0;
			const bb = b.status === 'active' ? 1 : 0;
			if (aa !== bb) return bb - aa;
			return (b.started || 0) - (a.started || 0);
		});
		for (let i = c.maxHistory; i < remaining.length; i++) {
			delete empire.campaigns[remaining[i].id];
		}
	}
}

// ---------------------------------------------------------------------------
// Spawn requests
// ---------------------------------------------------------------------------

function roleKey(role) {
	// Role enum → squad field name in WAR_CFG.squads
	const map = {
		[Role.attacker]: 'attackers',
		[Role.healer]: 'healers',
		[Role.ranged]: 'ranged',
		[Role.dismantler]: 'dismantlers',
		[Role.defender]: 'defenders',
	};
	return map[role];
}

/**
 * @param {Room|string} homeRoom
 * @returns {{ role: Role, priority: number, memory: CreepMemory, context: object }[]}
 */
function spawnRequests(homeRoom) {
	const home = typeof homeRoom === 'string' ? homeRoom : homeRoom.name;
	const c = cfg();
	const reqs = [];
	// Real gate: need a spawn + active campaign. No RCL ladder.
	const homeRoomObj = Game.rooms[home];
	if (!homeRoomObj || !homeRoomObj.find(FIND_MY_SPAWNS).length) return reqs;

	const camps = activeCampaigns(home);

	for (const camp of camps) {
		const counts = combat.squadCounts(camp.room, camp.id);
		const want = camp.squad || c.squads[camp.type] || c.squads.attack;
		const squadId = camp.id;

		const order = [
			Role.defender, Role.attacker, Role.healer, Role.ranged, Role.dismantler,
		];
		const prioName = {
			[Role.defender]: 'defender',
			[Role.attacker]: 'attacker',
			[Role.healer]: 'healer',
			[Role.ranged]: 'ranged',
			[Role.dismantler]: 'dismantler',
		};
		for (const role of order) {
			const key = roleKey(role);
			const need = (want && want[key]) || 0;
			if (need <= 0) continue;
			const have = counts[role] || 0;
			if (have >= need) continue;
			const pn = prioName[role];
			const fields = {
				[CreepMem.role]: role,
				[CreepMem.home]: home,
				[CreepMem.targetRoom]: camp.room,
				[CreepMem.squad]: squadId,
				[CreepMem.campaignId]: camp.id,
				[CreepMem.marching]: false,
			};
			if (camp.type === 'defend') fields[CreepMem.defendRoom] = camp.room;
			reqs.push({
				role,
				priority: c.priority[pn] != null ? c.priority[pn] : 40,
				memory: memInit(fields),
				context: { campaignId: camp.id, campaignType: camp.type },
			});
		}
	}

	// Local home defense is handled by spawn — war only requests campaign military.
	return reqs;
}

// ---------------------------------------------------------------------------
// Main tick
// ---------------------------------------------------------------------------

/**
 * @param {Room[]} [_colonies] optional list from room.js (unused; we re-scan owned)
 */
function tick(_colonies) {
	ensureEmpire();
	// Intents already synced in intel.tick() → RoomIntent on room memory.
	openFromIntents();
	openRemoteSecurity();
	openClaimSupport();

	for (const camp of campaignList()) {
		if (camp.status === 'active') tickCampaign(camp);
	}

	pruneCampaigns();
}

/**
 * Optional: run towers for all owned rooms (if main does not use defense.js).
 */
function tickDefense() {
	for (const room of ownedRooms()) {
		combat.runDefense(room);
	}
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
	tick,
	tickDefense,
	spawnRequests,
	openCampaign,
	activeCampaigns,
	observeRoom,
	cfg,
	WAR_CFG,
	// Re-export combat runners for roles.js convenience.
	runners: {
		defender: combat.runDefender,
		attacker: combat.runAttacker,
		ranged: combat.runRanged,
		healer: combat.runHealer,
		dismantler: combat.runDismantler,
	},
	runDefender: combat.runDefender,
	runAttacker: combat.runAttacker,
	runRanged: combat.runRanged,
	runHealer: combat.runHealer,
	runDismantler: combat.runDismantler,
	runDefense: combat.runDefense,
	runTowers: combat.runTowers,
};

export {};
