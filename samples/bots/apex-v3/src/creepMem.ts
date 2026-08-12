/**
 * Typed accessors for creep memory — CreepMem / Role enums only.
 * No string property names on CreepMemory.
 */
import { CreepMem, Role, ROLE_NAME, parseRole } from './memoryKeys';

export type CreepBag = CreepMemory;

export function cm(creep: Creep): CreepBag {
	return creep.memory;
}

export function getRole(creep: Creep): Role {
	const m = cm(creep) as CreepMemory & { role?: unknown };
	const v = m[CreepMem.role];
	if (typeof v === 'number' && ROLE_NAME[v as Role] != null) return v as Role;
	// One-tick migration: legacy string key `role` or bad value on enum key
	const parsed = parseRole(v) ?? parseRole(m.role);
	if (parsed != null) {
		m[CreepMem.role] = parsed;
		delete m.role;
		return parsed;
	}
	return Role.bootstrap;
}

export function setRole(creep: Creep, role: Role): void {
	cm(creep)[CreepMem.role] = role;
}

export function roleName(role: Role): string {
	return ROLE_NAME[role] || String(role);
}

export function getHome(creep: Creep): string | undefined {
	return cm(creep)[CreepMem.home] as string | undefined;
}

export function setHome(creep: Creep, home: string): void {
	cm(creep)[CreepMem.home] = home;
}

export function getSourceId(creep: Creep): Id<Source> | undefined {
	return cm(creep)[CreepMem.sourceId] as Id<Source> | undefined;
}

export function setSourceId(creep: Creep, id: Id<Source> | undefined): void {
	if (id == null) delete cm(creep)[CreepMem.sourceId];
	else cm(creep)[CreepMem.sourceId] = id;
}

export function getTargetRoom(creep: Creep): string | undefined {
	return cm(creep)[CreepMem.targetRoom] as string | undefined;
}

export function setTargetRoom(creep: Creep, room: string | undefined): void {
	if (room == null) delete cm(creep)[CreepMem.targetRoom];
	else cm(creep)[CreepMem.targetRoom] = room;
}

export function getRemote(creep: Creep): string | undefined {
	return cm(creep)[CreepMem.remote] as string | undefined;
}

export function setRemote(creep: Creep, room: string | undefined): void {
	if (room == null) delete cm(creep)[CreepMem.remote];
	else cm(creep)[CreepMem.remote] = room;
}

export function getSeat(creep: Creep): Id<Structure> | undefined {
	return cm(creep)[CreepMem.seat] as Id<Structure> | undefined;
}

export function setSeat(creep: Creep, id: Id<Structure> | undefined): void {
	if (id == null) delete cm(creep)[CreepMem.seat];
	else cm(creep)[CreepMem.seat] = id;
}

export function getWorking(creep: Creep): boolean {
	return !!cm(creep)[CreepMem.working];
}

export function setWorking(creep: Creep, v: boolean): void {
	cm(creep)[CreepMem.working] = v;
}

export function getPioneer(creep: Creep): boolean {
	return !!cm(creep)[CreepMem.pioneer];
}

export function setPioneer(creep: Creep, v: boolean): void {
	cm(creep)[CreepMem.pioneer] = v;
}

export function getSquad(creep: Creep): string | undefined {
	return cm(creep)[CreepMem.squad] as string | undefined;
}

export function setSquad(creep: Creep, id: string | undefined): void {
	if (id == null) delete cm(creep)[CreepMem.squad];
	else cm(creep)[CreepMem.squad] = id;
}

export function getQueue(creep: Creep): string[] | undefined {
	return cm(creep)[CreepMem.queue] as string[] | undefined;
}

export function setQueue(creep: Creep, q: string[] | undefined): void {
	if (q == null) delete cm(creep)[CreepMem.queue];
	else cm(creep)[CreepMem.queue] = q;
}

export function getArrived(creep: Creep): number | undefined {
	return cm(creep)[CreepMem.arrived] as number | undefined;
}

export function setArrived(creep: Creep, t: number | undefined): void {
	if (t == null) delete cm(creep)[CreepMem.arrived];
	else cm(creep)[CreepMem.arrived] = t;
}

export function getMarching(creep: Creep): boolean {
	return !!cm(creep)[CreepMem.marching];
}

export function setMarching(creep: Creep, v: boolean): void {
	cm(creep)[CreepMem.marching] = v;
}

export function getDefendRoom(creep: Creep): string | undefined {
	return cm(creep)[CreepMem.defendRoom] as string | undefined;
}

export function setDefendRoom(creep: Creep, room: string | undefined): void {
	if (room == null) delete cm(creep)[CreepMem.defendRoom];
	else cm(creep)[CreepMem.defendRoom] = room;
}

export function getCampaignId(creep: Creep): string | undefined {
	return cm(creep)[CreepMem.campaignId] as string | undefined;
}

export function setCampaignId(creep: Creep, id: string | undefined): void {
	if (id == null) delete cm(creep)[CreepMem.campaignId];
	else cm(creep)[CreepMem.campaignId] = id;
}

export function getSafeTicks(creep: Creep): number {
	return (cm(creep)[CreepMem.safeTicks] as number) || 0;
}

export function setSafeTicks(creep: Creep, n: number): void {
	cm(creep)[CreepMem.safeTicks] = n;
}

/** Memory object for spawnCreep({ memory }) — numeric CreepMem keys only */
export function memInit(fields: Partial<Record<CreepMem, unknown>>): CreepMemory {
	const m: CreepMemory = {};
	for (const key of Object.keys(fields)) {
		const k = Number(key) as CreepMem;
		if (Number.isNaN(k)) continue;
		const v = fields[k];
		if (v !== undefined) m[k] = v;
	}
	return m;
}

export { CreepMem, Role, ROLE_NAME, parseRole };
