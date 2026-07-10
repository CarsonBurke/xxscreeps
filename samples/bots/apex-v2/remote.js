/**
 * Remote room FSM per colony:
 *   scout → reserve → container → mine → haul
 * Abandon on sustained threat; re-scout after cooldown.
 */
const config = require('config');
const intel = require('intel');
const { lowBucket, hostileThreat } = require('util');

const PHASES = [ 'scout', 'reserve', 'container', 'mine', 'haul', 'abandoned' ];

function ensureState(room) {
	room.memory.apex ||= {};
	room.memory.apex.remoteState ||= {};
	return room.memory.apex.remoteState;
}

function getPhase(colonyRoom, remoteName) {
	const st = ensureState(colonyRoom);
	return (st[remoteName] && st[remoteName].phase) || 'scout';
}

/**
 * Advance FSM for each active remote of a colony.
 */
function tickRemote(colonyRoom, remoteName) {
	const st = ensureState(colonyRoom);
	const entry = st[remoteName] ||= {
		phase: 'scout',
		threatTicks: 0,
		since: Game.time,
		abandonedAt: 0,
	};

	// Recovery from abandoned after cooldown.
	if (entry.phase === 'abandoned') {
		const cool = config.remoteThreatCooldown || 500;
		if (Game.time - (entry.abandonedAt || 0) > cool) {
			entry.phase = 'scout';
			entry.threatTicks = 0;
			entry.since = Game.time;
		}
		return entry;
	}

	const info = intel.get(remoteName);
	const visible = Game.rooms[remoteName];

	// Threat tracking.
	let threat = 0;
	if (visible) {
		threat = hostileThreat(visible);
		// Also record via intel.
	} else if (info) {
		threat = info.threat || 0;
		// Only count recent threat.
		if (Game.time - (info.lastSeen || 0) > config.remoteThreatCooldown) threat = 0;
	}

	if (threat > 0) {
		entry.threatTicks = (entry.threatTicks || 0) + 1;
	} else {
		entry.threatTicks = Math.max(0, (entry.threatTicks || 0) - 1);
	}

	const abandonAfter = config.remoteAbandonThreatTicks || 50;
	if (entry.threatTicks >= abandonAfter) {
		entry.phase = 'abandoned';
		entry.abandonedAt = Game.time;
		return entry;
	}

	// Phase transitions.
	switch (entry.phase) {
		case 'scout': {
			// Need visibility / intel on sources.
			if (info && info.lastSeen && info.sourceCount > 0 &&
				Game.time - info.lastSeen < config.intelStale) {
				entry.phase = 'reserve';
				entry.since = Game.time;
			} else if (visible) {
				entry.phase = 'reserve';
				entry.since = Game.time;
			}
			break;
		}
		case 'reserve': {
			const ctrl = info && info.controller;
			const ticks = ctrl && ctrl.reservation ? ctrl.reservation.ticks : 0;
			const reservedOk = ctrl && ctrl.reservation && ticks > 200;
			// Also proceed if we have a reserver en route for a while, or room has no controller (rare).
			if (reservedOk || (visible && visible.controller && visible.controller.reservation &&
				visible.controller.reservation.ticksToEnd > 200)) {
				entry.phase = 'container';
				entry.since = Game.time;
			} else if (!ctrl && info && info.lastSeen) {
				// No controller data — skip to container if sources known.
				if (info.sourceCount > 0) {
					entry.phase = 'container';
					entry.since = Game.time;
				}
			}
			// Timeout: if stuck in reserve > 800 ticks with sources, still place containers.
			if (Game.time - entry.since > 800 && info && info.sourceCount > 0) {
				entry.phase = 'container';
				entry.since = Game.time;
			}
			break;
		}
		case 'container': {
			// Check containers at sources when visible.
			if (visible) {
				const sources = visible.find(FIND_SOURCES);
				let ready = 0;
				for (const s of sources) {
					const c = s.pos.findInRange(FIND_STRUCTURES, 1, {
						filter: x => x.structureType === STRUCTURE_CONTAINER,
					});
					const sites = s.pos.findInRange(FIND_CONSTRUCTION_SITES, 1, {
						filter: x => x.structureType === STRUCTURE_CONTAINER,
					});
					if (c.length) ready++;
					else if (!sites.length) {
						// Construction module places sites; wait.
					}
				}
				if (sources.length && ready >= sources.length) {
					entry.phase = 'mine';
					entry.since = Game.time;
				} else if (Game.time - entry.since > 1500 && ready > 0) {
					// Partial containers: start mining anyway.
					entry.phase = 'mine';
					entry.since = Game.time;
				} else if (Game.time - entry.since > 2500) {
					// Timeout: mine with drop mining.
					entry.phase = 'mine';
					entry.since = Game.time;
				}
			} else if (Game.time - entry.since > 2000) {
				entry.phase = 'mine';
				entry.since = Game.time;
			}
			break;
		}
		case 'mine': {
			// Once miner has been assigned long enough and containers hold energy, enter haul.
			if (visible) {
				const containers = visible.find(FIND_STRUCTURES, {
					filter: s => s.structureType === STRUCTURE_CONTAINER,
				});
				const hasEnergy = containers.some(c =>
					c.store && c.store.getUsedCapacity(RESOURCE_ENERGY) > 100);
				const miners = [];
				for (const n in Game.creeps) {
					const c = Game.creeps[n];
					if (c.memory && c.memory.role === 'remoteMiner' &&
						(c.memory.remote === remoteName || c.memory.targetRoom === remoteName)) {
						miners.push(c);
					}
				}
				if (miners.length > 0 && (hasEnergy || Game.time - entry.since > 100)) {
					entry.phase = 'haul';
					entry.since = Game.time;
				}
			} else if (Game.time - entry.since > 300) {
				entry.phase = 'haul';
				entry.since = Game.time;
			}
			break;
		}
		case 'haul': {
			// Steady state. Optionally drop back to mine if no haulers and containers overflow.
			break;
		}
		default:
			entry.phase = 'scout';
	}

	return entry;
}

/**
 * Update remote list + FSM for a colony. Returns active (non-abandoned) remotes.
 */
function updateRemotes(room) {
	room.memory.apex ||= {};
	const m = room.memory.apex;
	m.remotes ||= [];

	if (lowBucket()) {
		// Do not expand remotes when bucket is low; still tick FSM for abandon.
		for (const name of m.remotes) tickRemote(room, name);
		return m.remotes.filter(n => getPhase(room, n) !== 'abandoned');
	}

	if (!m.remotesAt || Game.time - m.remotesAt >= 100) {
		const picked = intel.pickRemotes(room);
		const next = [];
		for (const name of m.remotes) {
			if (picked.includes(name) && getPhase(room, name) !== 'abandoned') {
				next.push(name);
			}
		}
		for (const name of picked) {
			if (!next.includes(name) && next.length < config.maxRemotesPerColony) {
				next.push(name);
			}
		}
		m.remotes = next;
		m.remotesAt = Game.time;
	}

	for (const name of m.remotes) {
		tickRemote(room, name);
	}

	// Drop abandoned from active work list after processing (keep state for cooldown).
	return m.remotes.filter(n => getPhase(room, n) !== 'abandoned');
}

function phaseOf(colonyRoom, remoteName) {
	return getPhase(colonyRoom, remoteName);
}

module.exports = {
	updateRemotes,
	tickRemote,
	phaseOf,
	getPhase,
	PHASES,
};
