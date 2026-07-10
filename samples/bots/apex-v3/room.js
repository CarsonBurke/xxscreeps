/**
 * Per-colony loop: remotes, planner, defense, spawn; empire/war/economy hooks.
 */
const config = require('config');
const intel = require('intel');
const planner = require('planner');
const defense = require('defense');
const spawn = require('spawn');
const empire = require('empire');
const { energyOf, lowBucket } = require('util');

let war = null;
let economy = null;
try { war = require('war'); } catch { /* */ }
try { economy = require('economy'); } catch { /* */ }

function updateRemotes(room) {
	const m = room.memory.apex ||= {};
	if (m.remotesAt && Game.time - m.remotesAt < 50) return m.remotes || [];

	const cp = empire.planFor(room.name);
	let picked;
	if (lowBucket()) {
		picked = m.remotes || [];
	} else if (cp && cp.remotes) {
		picked = cp.remotes;
	} else {
		const maxR = (cp && cp.maxRemotes != null) ? cp.maxRemotes : config.maxRemotesPerColony;
		picked = intel.pickRemotes(room).slice(0, maxR);
	}
	m.remotes = picked;
	m.remotesAt = Game.time;
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
		const top = room.memory.apex && room.memory.apex.planTop;
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
