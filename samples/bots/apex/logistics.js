/**
 * Structure logistics: links + terminal balancing between colonies.
 * Market order book is stubbed in xxscreeps — we only use terminal.send.
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
		const dest = controllerLinks.find(l => energyOf(l.store) < 400 && freeCapacity(l.store, RESOURCE_ENERGY) > 100);
		if (!dest) continue;
		if (energyOf(src.store) < 200) continue;
		src.transferEnergy(dest);
	}
}

function runTerminals() {
	// Balance energy among colonies that both have terminals.
	const colonies = ownedRooms().filter(r => r.terminal && r.storage);
	if (colonies.length < 2) return;

	// Only act every 20 ticks (terminal cooldown is 10).
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
	from.terminal.send(RESOURCE_ENERGY, amount, to.room.name, 'apex-balance');
}

function run(room) {
	runLinks(room);
}

function runEmpire() {
	runTerminals();
}

module.exports = { run, runEmpire, runLinks };
