export interface WakeableWorker {
	currentTime: number;
	finalizedTime: number;
	idle: boolean;
	pendingTime: number | undefined;
}

/**
 * Request work from a worker. Returns true when the caller must start its worker loop. A request
 * received by a running worker is latched so the worker checks its queue again before going idle.
 */
export function requestWorkerWake(worker: WakeableWorker, time: number) {
	if (time <= worker.finalizedTime || time < worker.currentTime) {
		return false;
	}
	if (worker.idle) {
		worker.idle = false;
		return true;
	}
	worker.pendingTime = Math.max(worker.pendingTime ?? -Infinity, time);
	return false;
}

/** Run each requested tick until no wake request arrived during the preceding queue drain. */
export async function runWorkerUntilIdle(
	worker: WakeableWorker, initialTime: number, run: (time: number) => Promise<void>,
) {
	let time = initialTime;
	const getPendingTime = () => worker.pendingTime;
	while (true) {
		worker.currentTime = time;
		worker.pendingTime = undefined;
		await run(time);
		const pendingTime = getPendingTime();
		if (pendingTime === undefined) {
			worker.idle = true;
			return;
		}
		time = pendingTime;
	}
}
