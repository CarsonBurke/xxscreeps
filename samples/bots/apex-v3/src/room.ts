// @ts-nocheck — ported from JS; tighten types incrementally
/* eslint-disable */
/**
 * Per-colony loop: remotes, planner, defense, spawn; empire/war/economy hooks.
 */
const config = require('./config');
const intel = require('./intel');
const planner = require('./planner');
const defense = require('./defense');
const spawn = require('./spawn');
const empire = require('./empire');
const { energyOf, lowBucket } = require('./util');
const { getRemotes, setRemotes, getRemotesAt, getPlanTop } = require('./roomMem');
const { RoomIntent, getRoomIntent, setRoomIntent } = require('./intents');

let war = null;
let economy = null;
try { war = require('./war'); } catch { /* */ }
try { economy = require('./economy'); } catch { /* */ }

/**
 * Reassert empire remotes every tick (flag sync clears non-flag intents each tick).
 */
function assertRemoteIntents(remotes) {
	for (const r of remotes || []) {
		const cur = getRoomIntent(r);
		// Don't stomp player attack/claim/ignore/defend
		if (
			cur === RoomIntent.attack ||
			cur === RoomIntent.claim ||
			cur === RoomIntent.ignore ||
			cur === RoomIntent.defend
		) continue;
		setRoomIntent(r, RoomIntent.remote);
	}
}

function updateRemotes(room) {
	const name = room.name;
	const at = getRemotesAt(name);
	const cp = empire.planFor(name);
	const plan = Memory.empire && Memory.empire.plan;
	const planTick = plan && plan.tick != null ? plan.tick : 0;
	// Invalidate cache when empire plan is newer, or after short TTL (≤ plan interval)
	const cacheOk = at && Game.time - at < 10 && planTick <= at;
	if (cacheOk) {
		const cached = getRemotes(name);
		// If plan now has remotes and cache is empty, force refresh
		if (cached.length || !(cp && cp.remotes && cp.remotes.length)) {
			assertRemoteIntents(cached);
			return cached;
		}
	}

	let picked;
	if (lowBucket()) {
		picked = getRemotes(name);
	} else if (cp && Array.isArray(cp.remotes) && cp.remotes.length) {
		picked = cp.remotes;
	} else {
		// All viable remotes — spawn EV / duty is the throttle, not a room count
		picked = intel.pickRemotes(room);
	}
	setRemotes(name, picked);
	assertRemoteIntents(picked);
	return picked;
}

function runColony(room) {
	if (!room.controller || !room.controller.my) return;
	const remotes = updateRemotes(room);
	defense.run(room);
	planner.run(room, remotes);
	for (const rn of remotes) {
		if (Game.rooms[rn]) planner.runRemote(Game.rooms[rn]);
	}
	spawn.run(room, remotes);

	if (config.visuals && !lowBucket()) {
		const vis = room.visual;
		const storageE = room.storage ? energyOf(room.storage.store) : 0;
		const cp = empire.planFor(room.name);
		const top = getPlanTop(room.name);
		const mode = cp ? cp.mode : '?';
		vis.text(
			`v3 RCL${room.controller.level} ${mode} rem:${remotes.join(',') || '—'} build:${top || '—'} E:${storageE}`,
			1, 1, { align: 'left', font: 0.5, opacity: 0.7 },
		);
	}
}

function runAll() {
	try { empire.tick(); } catch (e) { console.log('empire', e); }

	const colonies = [];
	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		if (room.controller && room.controller.my) {
			colonies.push(room);
			runColony(room);
		}
	}
	if (economy && economy.tick) {
		try { economy.tick(colonies); } catch (e) { console.log('economy', e); }
	}
	if (war && war.tick) {
		try { war.tick(colonies); } catch (e) { console.log('war', e); }
	}
}

module.exports = { runAll, runColony, updateRemotes };
export { runAll, runColony, updateRemotes };

export {};
