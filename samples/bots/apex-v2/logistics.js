/**
 * Structure logistics: link-first network + terminal balancing.
 * Market order book is stubbed in xxscreeps — only terminal.send.
 */
const config = require('config');
const { energyOf, freeCapacity, ownedRooms } = require('util');

function runLinks(room) {
	const links = room.find(FIND_MY_STRUCTURES, {
		filter: s => s.structureType === STRUCTURE_LINK,
	});
	if (links.length < 2) return;

	const storage = room.storage;
	const ctrl = room.controller;

	const controllerLinks = links.filter(l => ctrl && l.pos.inRangeTo(ctrl, 2));
	const storageLinks = links.filter(l => storage && l.pos.inRangeTo(storage, 2));
	const sourceLinks = links.filter(l =>
		!controllerLinks.includes(l) && !storageLinks.includes(l));

	const sinks = [ ...controllerLinks, ...storageLinks ];
	if (!sinks.length) return;

	// Source links always push first (link-first logistics).
	for (const src of sourceLinks) {
		if (src.cooldown > 0) continue;
		if (energyOf(src.store) < config.linkBuffer) continue;
		// Prefer controller link if hungry, else storage link.
		let dest = controllerLinks.find(l => freeCapacity(l.store, RESOURCE_ENERGY) > 100);
		if (!dest) dest = storageLinks.find(l => freeCapacity(l.store, RESOURCE_ENERGY) > 100);
		if (!dest) dest = sinks.find(l => freeCapacity(l.store, RESOURCE_ENERGY) > 50 && l.id !== src.id);
		if (!dest) continue;
		const amount = Math.min(energyOf(src.store), freeCapacity(dest.store, RESOURCE_ENERGY));
		if (amount > 0) src.transferEnergy(dest, amount);
	}

	// Storage link → controller link top-up.
	for (const src of storageLinks) {
		if (src.cooldown > 0) continue;
		const dest = controllerLinks.find(l =>
			energyOf(l.store) < 400 && freeCapacity(l.store, RESOURCE_ENERGY) > 100);
		if (!dest) continue;
		if (energyOf(src.store) < 200) continue;
		src.transferEnergy(dest);
	}
}

/**
 * Suggest which source needs a hauler most (by container fullness).
 * Used by hauler role for dynamic reassignment when idle.
 */
function neediestSource(room) {
	const sources = room.find(FIND_SOURCES);
	let best = null;
	let bestScore = -1;
	for (const source of sources) {
		const container = source.pos.findInRange(FIND_STRUCTURES, 1, {
			filter: s => s.structureType === STRUCTURE_CONTAINER,
		})[0];
		let score = 0;
		if (container) {
			const cap = container.store.getCapacity(RESOURCE_ENERGY) || 2000;
			score = energyOf(container.store) / cap;
		} else {
			const drop = source.pos.findInRange(FIND_DROPPED_RESOURCES, 1, {
				filter: r => r.resourceType === RESOURCE_ENERGY,
			})[0];
			if (drop) score = Math.min(1, drop.amount / 1000);
		}
		// Count haulers already assigned.
		let haulers = 0;
		for (const n in Game.creeps) {
			const c = Game.creeps[n];
			if (c.memory && (c.memory.role === 'hauler' || c.memory.role === 'remoteHauler') &&
				c.memory.sourceId === source.id) haulers++;
		}
		score -= haulers * 0.35;
		if (score > bestScore) {
			bestScore = score;
			best = source;
		}
	}
	return bestScore > 0.15 ? best : null;
}

function runTerminals() {
	const colonies = ownedRooms().filter(r => r.terminal && r.storage);
	if (colonies.length < 2) return;
	if (Game.time % 20 !== 0) return;

	const rich = [];
	const poor = [];
	for (const room of colonies) {
		const e = energyOf(room.terminal.store) + energyOf(room.storage.store);
		const entry = { room, e, terminal: room.terminal };
		if (e > config.terminalEnergyBalance + config.terminalSendMin) rich.push(entry);
		if (e < config.terminalEnergyBalance - config.terminalSendMin) poor.push(entry);
	}
	if (!rich.length || !poor.length) return;

	rich.sort((a, b) => b.e - a.e);
	poor.sort((a, b) => a.e - b.e);

	const from = rich[0];
	const to = poor[0];
	if (from.terminal.cooldown > 0) return;
	const amount = Math.min(
		config.terminalSendMin,
		energyOf(from.terminal.store),
		freeCapacity(to.terminal.store, RESOURCE_ENERGY),
	);
	if (amount < 1000) return;
	const cost = Game.market.calcTransactionCost(amount, from.room.name, to.room.name);
	if (energyOf(from.terminal.store) < amount + cost) return;
	from.terminal.send(RESOURCE_ENERGY, amount, to.room.name, 'apex-v2-balance');
}

function run(room) {
	runLinks(room);
}

function runEmpire() {
	runTerminals();
}

module.exports = { run, runEmpire, runLinks, neediestSource };
