/**
 * Per-colony orchestration: remotes FSM, construction, logistics, defense, spawn.
 */
const config = require('config');
const remote = require('remote');
const construction = require('construction');
const logistics = require('logistics');
const defense = require('defense');
const spawn = require('spawn');
const traffic = require('traffic');
const intel = require('intel');
const { energyOf, lowBucket } = require('util');

function visualize(room, remotes) {
	if (!config.visuals) return;
	const vis = room.visual;
	const rcl = room.controller ? room.controller.level : 0;
	const storageE = room.storage ? energyOf(room.storage.store) : 0;
	vis.text(
		`Apex v2 RCL${rcl}  remotes:${remotes.join(',') || '—'}  store:${storageE}`,
		1, 1,
		{ align: 'left', font: 0.6, opacity: 0.7 },
	);
	for (const name of remotes) {
		const info = intel.get(name);
		const threat = info ? info.threat : '?';
		const phase = remote.phaseOf(room, name);
		vis.text(`→ ${name} [${phase}] thr=${threat}`, 1, 2 + remotes.indexOf(name) * 0.7, {
			align: 'left', font: 0.5, opacity: 0.6, color: '#8cf',
		});
	}
}

function runColony(room) {
	if (!room.controller || !room.controller.my) return;

	const remotes = remote.updateRemotes(room);

	defense.run(room);
	logistics.run(room);
	construction.run(room);
	traffic.decayHeat(room);

	// Remote construction when visible and not abandoned / not low bucket.
	if (!lowBucket()) {
		for (const rn of remotes) {
			const phase = remote.phaseOf(room, rn);
			if (phase === 'abandoned' || phase === 'scout') continue;
			if (Game.rooms[rn]) construction.runRemote(Game.rooms[rn], phase);
		}
	}

	spawn.run(room, remotes);
	visualize(room, remotes);
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

module.exports = { runAll, runColony };
