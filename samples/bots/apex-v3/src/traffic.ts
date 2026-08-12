/**
 * Traffic stack barrel (flat module for sandbox require).
 */
export {
	resetTraffic,
	runTraffic,
	markImmovable,
	registerMove,
	registerDirection,
} from './trafficCore';
export { goTo, goDo, moveToRoom, clearCreepPath, gcCreepPaths } from './creepMove';
export { runRelayAll, runRelay } from './relay';
export {
	bumpStructureEpoch,
	pathCacheStats,
	clearPathCache,
	getCachedPath,
	findPathIgnoreCreeps,
} from './pathCache';
export { movePriority, MovePrio } from './movePriorities';

import { resetTraffic, runTraffic } from './trafficCore';
import { runRelayAll } from './relay';
import { gcCreepPaths } from './creepMove';

export function beginTick(): void {
	resetTraffic();
	if (Game.time % 50 === 0) gcCreepPaths();
}

export function endTickMovement(): void {
	runRelayAll();
	runTraffic();
}
