/**
 * Per-colony orchestration: remotes, construction, logistics, defense, spawn.
 */
const config = require('config');
const intel = require('intel');
const construction = require('construction');
const logistics = require('logistics');
const defense = require('defense');
const spawn = require('spawn');
const { energyOf } = require('util');

function ensureRoomMemory(room) {
	room.memory.apex ||= {};
	const m = room.memory.apex;
	m.remotes ||= [];
	return m;
}

/**
 * Refresh remote list periodically; keep stable assignments between refreshes.
 */
function updateRemotes(room) {
	const m = ensureRoomMemory(room);
	if (m.remotesAt && Game.time - m.remotesAt < 100) return m.remotes;

	const picked = intel.pickRemotes(room);
	// Merge with existing that still score ok; drop threatened.
	const next = [];
	for (const name of m.remotes) {
		if (picked.includes(name)) next.push(name);
		else {
			const info = intel.get(name);
			if (info && info.threat > 0) continue;
			// Keep briefly if just stale visibility.
			if (picked.length < config.maxRemotesPerColony && next.length < config.maxRemotesPerColony) {
				// only keep if was intentional
			}
		}
	}
	for (const name of picked) {
		if (!next.includes(name) && next.length < config.maxRemotesPerColony) next.push(name);
	}

	m.remotes = next;
	m.remotesAt = Game.time;
	return m.remotes;
}

function visualize(room, remotes) {
	if (!config.visuals) return;
	const vis = room.visual;
	const rcl = room.controller ? room.controller.level : 0;
	const storageE = room.storage ? energyOf(room.storage.store) : 0;
	vis.text(
		`Apex RCL${rcl}  remotes:${remotes.join(',') || '—'}  store:${storageE}`,
		1, 1,
		{ align: 'left', font: 0.6, opacity: 0.7 },
	);
	for (const name of remotes) {
		const info = intel.get(name);
		const threat = info ? info.threat : '?';
		vis.text(`→ ${name} thr=${threat}`, 1, 2 + remotes.indexOf(name) * 0.7, {
			align: 'left', font: 0.5, opacity: 0.6, color: '#8cf',
		});
	}
}

function runColony(room) {
	if (!room.controller || !room.controller.my) return;

	const remotes = updateRemotes(room);

	defense.run(room);
	logistics.run(room);
	construction.run(room);

	// Remote construction when visible.
	for (const rn of remotes) {
		if (Game.rooms[rn]) construction.runRemote(Game.rooms[rn]);
	}

	spawn.run(room, remotes);
	visualize(room, remotes);

	// Sign controller occasionally.
	if (config.sign && room.controller && Game.time % 500 === 0) {
		// Signed by reserver/claimer; skip if no creep nearby.
	}
}

function runAll() {
	for (const name in Game.rooms) {
		const room = Game.rooms[name];
		if (room.controller && room.controller.my) {
			runColony(room);
		}
	}
	logistics.runEmpire();
}

module.exports = { runAll, runColony, updateRemotes };
