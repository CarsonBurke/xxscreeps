import { assert, describe, test } from 'xxscreeps/test/index.js';
import { requestWorkerWake, runWorkerUntilIdle } from './wakeable-worker.js';

function deferred() {
	let resolve!: () => void;
	const promise = new Promise<void>(done => {
		resolve = done;
	});
	return { promise, resolve };
}

describe('Wakeable worker', () => {
	test('rechecks its queue when a wake arrives during a drain', async () => {
		const worker = {
			currentTime: -Infinity, finalizedTime: -Infinity, idle: true,
			pendingTime: undefined as number | undefined,
		};
		const firstDrain = deferred();
		const drains: number[] = [];

		assert.strictEqual(requestWorkerWake(worker, 10), true);
		const running = runWorkerUntilIdle(worker, 10, async time => {
			drains.push(time);
			if (drains.length === 1) {
				await firstDrain.promise;
			}
		});

		assert.strictEqual(requestWorkerWake(worker, 10), false);
		firstDrain.resolve();
		await running;

		assert.deepStrictEqual(drains, [ 10, 10 ]);
		assert.strictEqual(worker.idle, true);
	});

	test('switches to a newer tick requested while the old queue drain unwinds', async () => {
		const worker = {
			currentTime: -Infinity, finalizedTime: -Infinity, idle: true,
			pendingTime: undefined as number | undefined,
		};
		const firstDrain = deferred();
		const drains: number[] = [];

		assert.strictEqual(requestWorkerWake(worker, 10), true);
		const running = runWorkerUntilIdle(worker, 10, async time => {
			drains.push(time);
			if (drains.length === 1) {
				await firstDrain.promise;
			}
		});

		assert.strictEqual(requestWorkerWake(worker, 11), false);
		firstDrain.resolve();
		await running;

		assert.deepStrictEqual(drains, [ 10, 11 ]);
		assert.strictEqual(worker.idle, true);
	});

	test('keeps the newest tick when several wakes arrive during one drain', () => {
		const worker = {
			currentTime: 11, finalizedTime: 10, idle: false,
			pendingTime: undefined as number | undefined,
		};

		assert.strictEqual(requestWorkerWake(worker, 12), false);
		assert.strictEqual(requestWorkerWake(worker, 11), false);
		assert.strictEqual(requestWorkerWake(worker, 13), false);

		assert.strictEqual(worker.pendingTime, 13);
	});

	test('ignores a wake older than the tick currently being drained', async () => {
		const worker = {
			currentTime: -Infinity, finalizedTime: -Infinity, idle: true,
			pendingTime: undefined as number | undefined,
		};
		const firstDrain = deferred();
		const drains: number[] = [];

		assert.strictEqual(requestWorkerWake(worker, 11), true);
		const running = runWorkerUntilIdle(worker, 11, async time => {
			drains.push(time);
			await firstDrain.promise;
		});

		assert.strictEqual(requestWorkerWake(worker, 10), false);
		firstDrain.resolve();
		await running;

		assert.deepStrictEqual(drains, [ 11 ]);
		assert.strictEqual(worker.idle, true);
	});

	test('does not restart an idle worker for an older tick', () => {
		const worker = {
			currentTime: 11, finalizedTime: 10, idle: true,
			pendingTime: undefined as number | undefined,
		};

		assert.strictEqual(requestWorkerWake(worker, 10), false);
		assert.strictEqual(worker.idle, true);
	});

	test('does not restart an idle worker for its finalized tick', () => {
		const worker = {
			currentTime: 11, finalizedTime: 11, idle: true,
			pendingTime: undefined as number | undefined,
		};

		assert.strictEqual(requestWorkerWake(worker, 11), false);
		assert.strictEqual(worker.idle, true);
	});
});
