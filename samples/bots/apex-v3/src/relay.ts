// @ts-nocheck
/**
 * Hauler relay — pass energy mid-route (TI-style) to raise pipeline throughput.
 * Full homebound meets empty outbound on same trunk → transfer + optional swap intent.
 */
import { getRole, getWorking, Role } from './creepMem';

// Local store helpers — do not import util (util → creepMove → trafficCore cycle risk)
function energyOf(store: { getUsedCapacity?: (r?: string) => number; energy?: number } | null): number {
	if (!store) return 0;
	if (typeof store.getUsedCapacity === 'function') {
		const v = store.getUsedCapacity(RESOURCE_ENERGY as unknown as string);
		if (v != null) return v;
	}
	return (store as { energy?: number }).energy || 0;
}
function freeCap(store: { getFreeCapacity?: (r?: string) => number } | null): number {
	if (!store || typeof store.getFreeCapacity !== 'function') return 0;
	let v = store.getFreeCapacity(RESOURCE_ENERGY as unknown as string);
	if (v == null || Number.isNaN(v)) v = store.getFreeCapacity();
	return v || 0;
}

const lastRelay = new Map<string, { partner: string; tick: number }>();

function isHauler(c: Creep): boolean {
	const r = getRole(c);
	return r === Role.hauler || r === Role.remoteHauler;
}

/**
 * Run after roles, before traffic resolve.
 */
export function runRelay(room: Room): void {
	const haulers = room.find(FIND_MY_CREEPS).filter(isHauler);
	if (haulers.length < 2) return;

	const used = new Set<string>();

	for (let i = 0; i < haulers.length; i++) {
		const a = haulers[i]!;
		if (used.has(a.name)) continue;
		// working=true ⇒ delivering home; false ⇒ fetching — require opposite direction
		const aHome = !!getWorking(a);
		const aEnergy = energyOf(a.store);
		if (aEnergy < 10 && aHome) continue; // empty but "home" flag — skip
		if (aEnergy > 10 && !aHome) {
			// has cargo but still outbound — only relay if mostly full
		}

		const near = a.pos.findInRange(FIND_MY_CREEPS, 1).filter(c =>
			c.name !== a.name && isHauler(c) && !used.has(c.name),
		);
		for (const b of near) {
			const bHome = !!getWorking(b);
			// Opposite logistics direction only
			if (aHome === bHome) continue;

			const full = aHome ? a : b;   // homebound (should carry energy)
			const empty = aHome ? b : a;  // outbound (should be empty-ish)
			if (energyOf(full.store) < 10) continue;
			if (energyOf(empty.store) > empty.store.getCapacity()! * 0.5) continue;

			// Don't re-relay same pair every tick
			const prev = lastRelay.get(full.name);
			if (prev && prev.partner === empty.name && Game.time - prev.tick < 2) continue;

			// Need free capacity on empty
			if (freeCap(empty.store) < 10) continue;

			const amt = Math.min(energyOf(full.store), freeCap(empty.store));
			if (amt < 10) continue;

			const res = full.transfer(empty, RESOURCE_ENERGY, amt);
			if (res === OK) {
				used.add(full.name);
				used.add(empty.name);
				lastRelay.set(full.name, { partner: empty.name, tick: Game.time });
				lastRelay.set(empty.name, { partner: full.name, tick: Game.time });
				// Working flags: full may still have energy; empty now working home
				break;
			}
		}
	}
}

/** All owned rooms */
export function runRelayAll(): void {
	for (const name in Game.rooms) {
		const room = Game.rooms[name]!;
		if (!room.controller || !room.controller.my) {
			// Still relay in remote rooms with our creeps
			const mine = room.find(FIND_MY_CREEPS);
			if (!mine.length) continue;
		}
		runRelay(room);
	}
}
