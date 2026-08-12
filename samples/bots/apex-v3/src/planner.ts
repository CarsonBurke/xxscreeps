// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * General construction planner — utility scoring, not a fixed structure ladder.
 *
 * Idea:
 *   1. Enumerate *candidate jobs* (places we could put something useful now).
 *   2. Score each job with multi-factor heuristics from current world state
 *      (bottlenecks, leverage, prerequisites-as-soft-penalties, saturation).
 *   3. Place the highest-scoring jobs until the site budget is spent.
 *
 * There is no hard-coded "always containers then roads then storage".
 * If spawn energy capacity is the binding constraint, extensions win.
 * If miners are dropping on bare ground, source containers win.
 * If paths are long and RCL allows, roads compete on logistics value.
 * Storage only scores well once logistics/spawn bandwidth exist to fill it.
 *
 * Remotes use the same scorer with a smaller candidate set.
 */
const config = require('./config');
const { getFillerPads, setFillerPads, setPlanDebug, getPlanAt } = require('./roomMem');

function maxOpenSites() {
	// Game hard limit is 100; leave headroom. Not an RCL ladder.
	const soft = config.maxOpenSites != null ? config.maxOpenSites : 40;
	return Math.min(100, Math.max(1, soft));
}

// ---------------------------------------------------------------------------
// Terrain / placement primitives
// ---------------------------------------------------------------------------

function maxOf(type, rcl) {
	const table = CONTROLLER_STRUCTURES[type];
	return table ? (table[rcl] || 0) : 0;
}

function countBuilt(room, type) {
	return room.find(FIND_STRUCTURES, { filter: s => s.structureType === type }).length;
}

function countSites(room, type) {
	return room.find(FIND_MY_CONSTRUCTION_SITES, { filter: s => s.structureType === type }).length;
}

function countTotal(room, type) {
	return countBuilt(room, type) + countSites(room, type);
}

function slotsLeft(room, type) {
	return Math.max(0, maxOf(type, room.controller.level) - countTotal(room, type));
}

function trySite(room, x, y, type) {
	if (x < 1 || x > 48 || y < 1 || y > 48) return false;
	if (room.getTerrain().get(x, y) === TERRAIN_MASK_WALL) return false;
	const pos = new RoomPosition(x, y, room.name);
	for (const s of pos.lookFor(LOOK_STRUCTURES)) {
		if (s.structureType === STRUCTURE_ROAD && type !== STRUCTURE_ROAD) continue;
		if (s.structureType === STRUCTURE_RAMPART) continue;
		return false;
	}
	if (pos.lookFor(LOOK_CONSTRUCTION_SITES).length) return false;

	// Soft reservation: filler pads stay clear of bulky buildings
	const pads = getFillerPads(room.name);
	if (pads.length && type !== STRUCTURE_ROAD && type !== STRUCTURE_CONTAINER) {
		for (const p of pads) {
			if (p.x === x && p.y === y) return false;
		}
	}
	return room.createConstructionSite(x, y, type) === OK;
}

function openNear(pos, range, opts = {}) {
	const room = Game.rooms[pos.roomName];
	if (!room) return [];
	const terrain = room.getTerrain();
	const pads = opts.avoidPads || [];
	const out = [];
	for (let dx = -range; dx <= range; dx++) {
		for (let dy = -range; dy <= range; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = pos.x + dx;
			const y = pos.y + dy;
			if (x < 1 || x > 48 || y < 1 || y > 48) continue;
			if (terrain.get(x, y) === TERRAIN_MASK_WALL) continue;
			if (pads.some(p => p.x === x && p.y === y)) continue;
			const p = new RoomPosition(x, y, pos.roomName);
			const blocked = p.lookFor(LOOK_STRUCTURES).some(s =>
				s.structureType !== STRUCTURE_ROAD &&
				s.structureType !== STRUCTURE_RAMPART &&
				!(opts.allowContainer && s.structureType === STRUCTURE_CONTAINER));
			if (blocked) continue;
			if (p.lookFor(LOOK_CONSTRUCTION_SITES).length) continue;
			out.push({ pos: p, range: Math.max(Math.abs(dx), Math.abs(dy)), manhattan: Math.abs(dx) + Math.abs(dy) });
		}
	}
	out.sort((a, b) => a.manhattan - b.manhattan);
	return out;
}

function ensureFillerPads(room, spawn) {
	const existing = getFillerPads(room.name);
	if (existing.length) return existing;
	const pads = [];
	const terrain = room.getTerrain();
	const cands = [];
	for (let dx = -1; dx <= 1; dx++) {
		for (let dy = -1; dy <= 1; dy++) {
			if (dx === 0 && dy === 0) continue;
			const x = spawn.pos.x + dx;
			const y = spawn.pos.y + dy;
			if (terrain.get(x, y) === TERRAIN_MASK_WALL) continue;
			cands.push({ x, y, d: Math.abs(dx) + Math.abs(dy) });
		}
	}
	cands.sort((a, b) => a.d - b.d);
	for (const c of cands) {
		if (pads.length >= 2) break;
		const pos = new RoomPosition(c.x, c.y, room.name);
		if (pos.lookFor(LOOK_STRUCTURES).some(s =>
			s.structureType !== STRUCTURE_ROAD && s.structureType !== STRUCTURE_RAMPART)) continue;
		pads.push({ x: c.x, y: c.y });
	}
	setFillerPads(room.name, pads);
	return pads;
}

// ---------------------------------------------------------------------------
// World features used by the scorer (bottleneck signals)
// ---------------------------------------------------------------------------

function analyze(room, spawn) {
	const rcl = room.controller.level;
	let spawnCap = 0;
	let spawnEnergy = 0;
	for (const s of room.find(FIND_MY_STRUCTURES)) {
		if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
			spawnCap += s.store.getCapacity(RESOURCE_ENERGY) || 0;
			spawnEnergy += s.store[RESOURCE_ENERGY] || 0;
		}
	}
	const sources = room.find(FIND_SOURCES);
	let sourcesWithoutContainer = 0;
	for (const source of sources) {
		const has = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		}).length + source.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		}).length;
		if (!has) sourcesWithoutContainer++;
	}
	const ctrl = room.controller;
	const ctrlHasEnergySpot = ctrl.pos.findInRange(FIND_STRUCTURES, 3, {
		filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_LINK,
	}).length + ctrl.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 3, {
		filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_LINK,
	}).length;

	const spawnHasStaging = spawn.pos.findInRange(FIND_STRUCTURES, 2, {
		filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_STORAGE,
	}).length + spawn.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 2, {
		filter: s => s.structureType === STRUCTURE_CONTAINER || s.structureType === STRUCTURE_STORAGE,
	}).length;

	const towers = countBuilt(room, STRUCTURE_TOWER);
	const openSites = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	const storage = room.storage;
	const threat = room.find(FIND_HOSTILE_CREEPS).length;

	// Binding constraints (0..1 urgency)
	const extensionFill = maxOf(STRUCTURE_EXTENSION, rcl) === 0
		? 1
		: countTotal(room, STRUCTURE_EXTENSION) / Math.max(1, maxOf(STRUCTURE_EXTENSION, rcl));
	const spawnCapUrgency = 1 - Math.min(1, spawnCap / Math.max(300, 300 + maxOf(STRUCTURE_EXTENSION, rcl) * 50));
	const logisticsUrgency = sourcesWithoutContainer / Math.max(1, sources.length);
	const upgradeLogisticsUrgency = ctrlHasEnergySpot ? 0 : 0.8;
	const defenseUrgency = threat > 0 ? 1 : (towers < maxOf(STRUCTURE_TOWER, rcl) ? 0.4 : 0);
	const storageUrgency = storage ? 0 : (spawnCap >= 550 ? 0.7 : 0.2);
	const siteBacklog = Math.min(1, openSites / 15);

	return {
		rcl,
		spawnCap,
		spawnEnergy,
		sources,
		sourcesWithoutContainer,
		ctrlHasEnergySpot,
		spawnHasStaging,
		towers,
		openSites,
		storage,
		threat,
		extensionFill,
		spawnCapUrgency,
		logisticsUrgency,
		upgradeLogisticsUrgency,
		defenseUrgency,
		storageUrgency,
		siteBacklog,
	};
}

// ---------------------------------------------------------------------------
// Candidate generation — anything we *could* place; scoring decides order
// ---------------------------------------------------------------------------

/**
 * @returns {Array<{ type: string, x: number, y: number, tag: string, meta?: object }>}
 */
function candidates(room, spawn, remotes, ctx) {
	const jobs = [];
	const rcl = ctx.rcl;
	const pads = ensureFillerPads(room, spawn);
	// Containers/roads have no RCL unlock — only CONTROLLER_STRUCTURES / energy.
	// Prefer extensions while spawn cap < 550 (real: unlock 5W bodies).
	const extensionRush = slotsLeft(room, STRUCTURE_EXTENSION) > 0 &&
		countTotal(room, STRUCTURE_EXTENSION) < maxOf(STRUCTURE_EXTENSION, rcl) &&
		ctx.spawnCap < 550;
	const allowContainers = !extensionRush;
	const allowRoads = !extensionRush && ctx.spawnCap >= 400;

	const push = (type, x, y, tag, meta) => {
		if (slotsLeft(room, type) <= 0 && type !== STRUCTURE_ROAD && type !== STRUCTURE_RAMPART &&
			type !== STRUCTURE_CONTAINER) {
			// container has global cap 5 — still check
			if (type === STRUCTURE_CONTAINER && slotsLeft(room, STRUCTURE_CONTAINER) <= 0) return;
			if (type !== STRUCTURE_CONTAINER) return;
		}
		if (type === STRUCTURE_CONTAINER && slotsLeft(room, STRUCTURE_CONTAINER) <= 0) return;
		jobs.push({ type, x, y, tag, meta: meta || {} });
	};

	// --- Extensions first (RCL2 capacity is the binding constraint) ---
	if (slotsLeft(room, STRUCTURE_EXTENSION) > 0) {
		let n = 0;
		for (let r = 1; r <= 7 && n < 12; r++) {
			for (let dx = -r; dx <= r && n < 12; dx++) {
				for (let dy = -r; dy <= r && n < 12; dy++) {
					if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
					const x = spawn.pos.x + dx;
					const y = spawn.pos.y + dy;
					if ((x + y) % 2 === (spawn.pos.x + spawn.pos.y) % 2) continue;
					if (pads.some(p => p.x === x && p.y === y)) continue;
					push(STRUCTURE_EXTENSION, x, y, 'extension', { r });
					n++;
				}
			}
		}
	}

	// During extension rush, skip containers/roads so all build energy grows spawn bandwidth.
	if (extensionRush) return jobs;

	// --- Logistics nodes (containers / links near features) ---
	if (allowContainers) {
		for (const source of ctx.sources) {
			const has = source.pos.findInRange(FIND_STRUCTURES, 1, {
				filter: s => s.structureType === STRUCTURE_CONTAINER,
			}).length;
			const site = source.pos.findInRange(FIND_MY_CONSTRUCTION_SITES, 1, {
				filter: s => s.structureType === STRUCTURE_CONTAINER,
			}).length;
			if (!has && !site) {
				const spot = openNear(source.pos, 1)[0];
				if (spot) push(STRUCTURE_CONTAINER, spot.pos.x, spot.pos.y, 'container:source', { feature: 'source', id: source.id });
			}
			// Link candidates near sources when unlocked
			if (maxOf(STRUCTURE_LINK, rcl) > 0) {
				const hasL = source.pos.findInRange(FIND_MY_STRUCTURES, 2, {
					filter: s => s.structureType === STRUCTURE_LINK,
				}).length;
				if (!hasL && slotsLeft(room, STRUCTURE_LINK) > 0) {
					const spot = openNear(source.pos, 2, { avoidPads: pads })[0];
					if (spot) push(STRUCTURE_LINK, spot.pos.x, spot.pos.y, 'link:source', { feature: 'source' });
				}
			}
		}

		if (!ctx.ctrlHasEnergySpot && room.controller) {
			const spot = openNear(room.controller.pos, 2)[0];
			if (spot) push(STRUCTURE_CONTAINER, spot.pos.x, spot.pos.y, 'container:controller', { feature: 'controller' });
			if (maxOf(STRUCTURE_LINK, rcl) > 0 && slotsLeft(room, STRUCTURE_LINK) > 0) {
				const spotL = openNear(room.controller.pos, 2, { avoidPads: pads })[0];
				if (spotL) push(STRUCTURE_LINK, spotL.pos.x, spotL.pos.y, 'link:controller', { feature: 'controller' });
			}
		}

		if (!ctx.spawnHasStaging && !ctx.storage) {
			const spot = openNear(spawn.pos, 2, { avoidPads: pads })[0];
			if (spot) push(STRUCTURE_CONTAINER, spot.pos.x, spot.pos.y, 'container:spawn', { feature: 'spawn' });
		}
	}

	if (!allowRoads) return jobs;

	// Filler pad roads
	for (const p of pads) {
		const pos = new RoomPosition(p.x, p.y, room.name);
		if (!pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_ROAD)) {
			push(STRUCTURE_ROAD, p.x, p.y, 'road:fillerPad', { feature: 'filler' });
		}
	}

	// --- Roads along paths to important goals ---
	const goals = [
		...ctx.sources.map(s => ({ pos: s.pos, tag: 'source', weight: 1 })),
		room.controller && { pos: room.controller.pos, tag: 'controller', weight: 1 },
		ctx.storage && { pos: ctx.storage.pos, tag: 'storage', weight: 0.8 },
	].filter(Boolean);

	for (const rn of remotes || []) {
		const exit = room.findExitTo(rn);
		if (exit === ERR_NO_PATH || exit === ERR_INVALID_ARGS) continue;
		const exitPos = spawn.pos.findClosestByPath(exit);
		if (exitPos) goals.push({ pos: exitPos, tag: 'remoteExit', weight: 0.6, remote: rn });
	}

	for (const g of goals) {
		const path = spawn.pos.findPathTo(g.pos, { ignoreCreeps: true, maxOps: 3000 });
		let i = 0;
		for (const step of path) {
			if (i++ > 40) break;
			if (g.pos.roomName === room.name && step.x === g.pos.x && step.y === g.pos.y) continue;
			const pos = new RoomPosition(step.x, step.y, room.name);
			if (pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_ROAD)) continue;
			push(STRUCTURE_ROAD, step.x, step.y, `road:${g.tag}`, { goal: g.tag, weight: g.weight, index: i });
		}
	}

	// Base lattice roads (between extension slots)
	for (let r = 1; r <= 4; r++) {
		for (let dx = -r; dx <= r; dx++) {
			for (let dy = -r; dy <= r; dy++) {
				if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
				const x = spawn.pos.x + dx;
				const y = spawn.pos.y + dy;
				if ((x + y) % 2 !== (spawn.pos.x + spawn.pos.y) % 2) continue;
				push(STRUCTURE_ROAD, x, y, 'road:lattice', { r });
			}
		}
	}

	// --- Combat / logistics / industry structures near spawn ---
	const nearSpawnTypes = [
		STRUCTURE_TOWER,
		STRUCTURE_STORAGE,
		STRUCTURE_TERMINAL,
		STRUCTURE_SPAWN,
		STRUCTURE_OBSERVER,
		STRUCTURE_POWER_SPAWN,
		STRUCTURE_NUKER,
		STRUCTURE_FACTORY,
	];
	const offsets = [
		[ 0, -2 ], [ 0, 2 ], [ -2, 0 ], [ 2, 0 ],
		[ -2, -2 ], [ 2, 2 ], [ -2, 2 ], [ 2, -2 ],
		[ 0, -3 ], [ 0, 3 ], [ -3, 0 ], [ 3, 0 ],
		[ 1, -2 ], [ -1, -2 ], [ 2, 1 ], [ -2, -1 ],
	];
	for (const type of nearSpawnTypes) {
		if (slotsLeft(room, type) <= 0) continue;
		for (const [ dx, dy ] of offsets) {
			push(type, spawn.pos.x + dx, spawn.pos.y + dy, type, { near: 'spawn' });
		}
	}

	// Extractor on mineral
	if (slotsLeft(room, STRUCTURE_EXTRACTOR) > 0) {
		const mineral = room.find(FIND_MINERALS)[0];
		if (mineral) push(STRUCTURE_EXTRACTOR, mineral.pos.x, mineral.pos.y, 'extractor', {});
	}

	// Labs cluster
	if (slotsLeft(room, STRUCTURE_LAB) > 0) {
		for (let dx = -2; dx <= 2; dx++) {
			for (let dy = 2; dy <= 4; dy++) {
				push(STRUCTURE_LAB, spawn.pos.x + dx, spawn.pos.y + dy, 'lab', {});
			}
		}
	}

	// Ramparts on core
	if (maxOf(STRUCTURE_RAMPART, rcl) > 0) {
		const cores = [ spawn ];
		if (room.storage) cores.push(room.storage);
		if (room.terminal) cores.push(room.terminal);
		for (const t of room.find(FIND_MY_STRUCTURES, { filter: s => s.structureType === STRUCTURE_TOWER })) {
			cores.push(t);
		}
		for (const core of cores) {
			for (let dx = -1; dx <= 1; dx++) {
				for (let dy = -1; dy <= 1; dy++) {
					const x = core.pos.x + dx;
					const y = core.pos.y + dy;
					const pos = new RoomPosition(x, y, room.name);
					if (pos.lookFor(LOOK_STRUCTURES).some(s => s.structureType === STRUCTURE_RAMPART)) continue;
					push(STRUCTURE_RAMPART, x, y, 'rampart', { core: core.structureType || 'spawn' });
				}
			}
		}
	}

	// Storage/spawn link
	if (slotsLeft(room, STRUCTURE_LINK) > 0 && (ctx.storage || spawn)) {
		const anchor = ctx.storage || spawn;
		const spot = openNear(anchor.pos, 2, { avoidPads: pads })[0];
		if (spot) push(STRUCTURE_LINK, spot.pos.x, spot.pos.y, 'link:storage', { feature: 'storage' });
	}

	return jobs;
}

// ---------------------------------------------------------------------------
// Utility scoring — general factors, not a type ladder
// ---------------------------------------------------------------------------

/**
 * Score a job. Higher = more valuable to place *right now*.
 *
 * Factors (all soft — anything can win if the world state says so):
 *  - bottleneckRelief: does this fix the binding constraint?
 *  - leverage: does this unlock / multiply other systems?
 *  - readiness: soft penalty if supporting context is weak (not a hard gate)
 *  - saturation: prefer filling early slots of a type; diminishing returns
 *  - backlog: global site pressure reduces appetite for low-urgency jobs
 *  - costBias: expensive structures need stronger justification early
 */
function scoreJob(job, room, ctx) {
	const rcl = ctx.rcl;
	const type = job.type;
	let score = 0;

	// --- Bottleneck relief ---
	if (type === STRUCTURE_CONTAINER) {
		if (job.meta.feature === 'source') {
			score += 100 * ctx.logisticsUrgency + 40;
		} else if (job.meta.feature === 'controller') {
			score += 80 * ctx.upgradeLogisticsUrgency + 20;
		} else if (job.meta.feature === 'spawn') {
			// Staging for fillers — high when no storage yet and spawn is active
			score += ctx.storage ? 5 : (30 + 40 * ctx.spawnCapUrgency);
		} else {
			score += 15;
		}
	}

	if (type === STRUCTURE_EXTENSION) {
		score += 90 * ctx.spawnCapUrgency + 25 * (1 - ctx.extensionFill);
		// Early RCL: extensions are the main power spike
		if (rcl <= 3) score += 20;
	}

	if (type === STRUCTURE_ROAD) {
		const w = job.meta.weight != null ? job.meta.weight : 0.5;
		// Roads matter more when logistics exist to use them and paths are worked
		const logisticsReady = 1 - ctx.logisticsUrgency; // containers mostly done
		score += 15 * w;
		score += 35 * logisticsReady * w;
		if (job.tag === 'road:fillerPad') score += 40; // cheap, high local value
		if (job.tag === 'road:lattice') score += 8 * logisticsReady;
		if (job.tag === 'road:remoteExit') score += 12 * logisticsReady;
		// Until something to haul, roads are low value
		score -= 25 * ctx.logisticsUrgency;
	}

	if (type === STRUCTURE_TOWER) {
		score += 70 * ctx.defenseUrgency + 25;
		if (ctx.threat > 0) score += 80;
		if (ctx.towers === 0 && rcl >= 3) score += 40;
	}

	if (type === STRUCTURE_STORAGE) {
		score += 60 * ctx.storageUrgency;
		// Soft readiness: weak if spawn cap tiny or no source containers
		score -= 40 * ctx.logisticsUrgency;
		score -= 20 * ctx.spawnCapUrgency;
		if (ctx.spawnCap >= 1000) score += 25;
	}

	if (type === STRUCTURE_LINK) {
		const hasStorage = !!ctx.storage;
		score += hasStorage ? 45 : 10;
		score -= 30 * ctx.logisticsUrgency;
		if (job.meta.feature === 'controller' && hasStorage) score += 15;
		if (job.meta.feature === 'source' && hasStorage) score += 20;
	}

	if (type === STRUCTURE_TERMINAL) {
		score += ctx.storage ? 35 : 5;
		score -= 25 * ctx.logisticsUrgency;
	}

	if (type === STRUCTURE_EXTRACTOR) {
		score += ctx.storage ? 30 : 8;
	}

	if (type === STRUCTURE_LAB) {
		score += ctx.storage ? 20 : 0;
		score -= 15 * ctx.spawnCapUrgency;
	}

	if (type === STRUCTURE_SPAWN) {
		// Extra spawns: when energy capacity and workload justify parallel production
		score += ctx.spawnCap >= 1000 ? 40 : 10;
		score += (countBuilt(room, STRUCTURE_SPAWN) === 1 && rcl >= 7) ? 30 : 0;
	}

	if (type === STRUCTURE_FACTORY || type === STRUCTURE_POWER_SPAWN ||
		type === STRUCTURE_NUKER || type === STRUCTURE_OBSERVER) {
		score += ctx.storage && rcl >= 8 ? 15 : 2;
	}

	if (type === STRUCTURE_RAMPART) {
		score += 20 * ctx.defenseUrgency + (ctx.towers > 0 ? 15 : 0);
		score -= 10 * ctx.spawnCapUrgency;
	}

	// --- Saturation (diminishing returns for this type) ---
	const built = countTotal(room, type);
	const cap = maxOf(type, rcl) || (type === STRUCTURE_ROAD ? 200 : 1);
	const sat = cap > 0 ? built / cap : 1;
	score *= 1 - 0.55 * Math.min(1, sat);

	// --- Global backlog: if many sites open, only high scores survive ---
	score -= 50 * ctx.siteBacklog;

	// --- Already have a site of this exact cell? candidates filter handles it ---

	// Prefer closer-to-spawn utility structures slightly (build time / defense)
	if (job.meta.near === 'spawn' || job.meta.feature === 'spawn') score += 5;

	return score;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const MIN_SCORE = 12;

function run(room, remotes) {
	if (!room.controller || !room.controller.my) return;
	const spawn = room.find(FIND_MY_SPAWNS)[0];
	if (!spawn) return;

	const rcl = room.controller.level || 1;
	const planCooldown = config.plannerCooldown || 10;
	const planAt = getPlanAt(room.name);
	if (planAt && Game.time - planAt < planCooldown) return;

	const siteCap = maxOpenSites();
	const open = room.find(FIND_MY_CONSTRUCTION_SITES).length;
	if (open >= siteCap) {
		setPlanDebug(room.name, `site_cap:${open}`, 0, []);
		return;
	}

	const ctx = analyze(room, spawn);
	ensureFillerPads(room, spawn);

	const jobs = candidates(room, spawn, remotes || [], ctx);
	// Dedupe by type+xy
	const seen = new Set();
	const unique = [];
	for (const j of jobs) {
		const k = `${j.type}:${j.x},${j.y}`;
		if (seen.has(k)) continue;
		seen.add(k);
		unique.push(j);
	}

	const scored = unique.map(j => ({ job: j, score: scoreJob(j, room, ctx) }));
	scored.sort((a, b) => b.score - a.score);

	const siteBudget = Math.min(
		config.plannerSiteBudget || 4,
		Math.max(0, siteCap - open),
	);
	let placed = 0;
	const placedTags = [];
	for (const { job, score } of scored) {
		if (placed >= siteBudget) break;
		if (score < MIN_SCORE) break;
		if (trySite(room, job.x, job.y, job.type)) {
			placed++;
			placedTags.push(`${job.tag}:${score.toFixed(0)}`);
		}
	}

	const top = scored[0] ? `${scored[0].job.tag}@${scored[0].score.toFixed(1)}` : 'none';
	setPlanDebug(room.name, top, placed, placedTags.slice(0, 5));
}

function runRemote(room) {
	if (!room) return;
	// Use highest owned RCL + spawn capacity as gate — don't steal builders
	// from home extension rush with remote containers/roads.
	let homeRcl = 0;
	let homeCap = 0;
	for (const n in Game.rooms) {
		const r = Game.rooms[n];
		if (!r.controller || !r.controller.my) continue;
		if (r.controller.level > homeRcl) homeRcl = r.controller.level;
		for (const s of r.find(FIND_MY_STRUCTURES)) {
			if (s.structureType === STRUCTURE_SPAWN || s.structureType === STRUCTURE_EXTENSION) {
				homeCap += s.store.getCapacity(RESOURCE_ENERGY) || 0;
			}
		}
	}
	// Remote infra only after home can field 5W (cap≥550) — real: don't steal builders
	// from extension rush. No RCL magic number.
	// Wait until home can field 5W miners (capacity ≥ 550) before remote infra
	if (homeCap < 550) return;
	// Cooldown + site budget (shared with home pressure)
	const key = `_remotePlan_${room.name}`;
	const last = Memory.empire && Memory.empire[key];
	if (last && Game.time - last < (config.plannerCooldown || 10)) return;
	if (Memory.empire) Memory.empire[key] = Game.time;

	const homeSites = (() => {
		let n = 0;
		for (const rn in Game.rooms) {
			const r = Game.rooms[rn];
			if (r.controller && r.controller.my) n += r.find(FIND_MY_CONSTRUCTION_SITES).length;
		}
		return n;
	})();
	if (homeSites >= maxSitesForRcl(homeRcl)) return;

	const jobs = [];
	const sources = room.find(FIND_SOURCES);
	for (const source of sources) {
		const has = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		}).length;
		const site = source.pos.findInRange(FIND_CONSTRUCTION_SITES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		}).length;
		if (!has && !site) {
			const spot = openNear(source.pos, 1)[0];
			if (spot) {
				jobs.push({
					type: STRUCTURE_CONTAINER,
					x: spot.pos.x,
					y: spot.pos.y,
					score: 100,
					tag: 'remote-container',
				});
			}
		}
	}
	// Roads once home can afford logistics (cap≥400); not an RCL number
	const needContainer = jobs.some(j => j.type === STRUCTURE_CONTAINER);
	if (!needContainer && homeCap >= 400) {
		const center = new RoomPosition(25, 25, room.name);
		for (const source of sources) {
			const path = source.pos.findPathTo(center, { ignoreCreeps: true, maxOps: 2000 });
			let i = 0;
			for (const step of path.slice(0, 8)) {
				jobs.push({
					type: STRUCTURE_ROAD,
					x: step.x,
					y: step.y,
					score: 40 - i,
					tag: 'remote-road',
				});
				i++;
			}
		}
	}
	jobs.sort((a, b) => b.score - a.score);
	let placed = 0;
	const budget = Math.min(2, (config.plannerSiteBudget || 4));
	for (const j of jobs) {
		if (placed >= budget) break;
		if (trySite(room, j.x, j.y, j.type)) placed++;
	}
}

module.exports = {
	run,
	runRemote,
	ensureFillerPads,
	analyze,
	candidates,
	scoreJob,
};

export { run, runRemote, ensureFillerPads, analyze, candidates, scoreJob };
