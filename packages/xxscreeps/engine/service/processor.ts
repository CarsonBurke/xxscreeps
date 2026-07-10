import type { ProcessorRequest } from 'xxscreeps/engine/processor/worker.js';
import type { Effect } from 'xxscreeps/utility/types.js';
import { config } from 'xxscreeps/config/index.js';
import { consumeSet, consumeSortedSet, consumeSortedSetMembers } from 'xxscreeps/engine/db/async.js';
import { Database, Shard } from 'xxscreeps/engine/db/index.js';
import { getProcessorChannel, processRoomsSetKey } from 'xxscreeps/engine/processor/model.js';
import { Fn } from 'xxscreeps/functional/fn.js';
import * as Async from 'xxscreeps/utility/async.js';
import { negotiateResponderClient } from 'xxscreeps/utility/responder.js';
import { clamp } from 'xxscreeps/utility/utility.js';
import { handleInterruptSignal } from './signal.js';
import { requestWorkerWake, runWorkerUntilIdle } from './wakeable-worker.js';
import { checkIsEntry, getServiceChannel } from './index.js';

const isEntry = checkIsEntry();
const log = isEntry || config.processor.log
	? (message: string) => process.stderr.write(message)
	: () => {};

// Interrupt handler. Standalone-only self-halt: under the launcher, main owns shutdown via the
// processor channel, and breaking iterators here races its next-tick `process` publish.
let halt: Effect | undefined;
let halted = false as boolean;
let processing = false;
using signal = handleInterruptSignal(() => {
	halted = true;
	if (isEntry && !processing) {
		halt?.();
	}
});

// Connect to main & storage
await using db = await Database.connect();
await using shard = await Shard.connect(db, config.shards[0]!.name);
await using disposable = new AsyncDisposableStack();
const worldBlob = await shard.data.req('terrain', { blob: true });
const processorSubscription = disposable.adopt(
	await getProcessorChannel(shard).subscribe(),
	subscription => subscription.disconnect());

// Sync with main
await async function() {
	await using disposable = new AsyncDisposableStack();
	const channel = disposable.adopt(
		await getServiceChannel(shard).subscribe(),
		subscription => subscription.disconnect());
	const messages = channel.iterable();
	await channel.publish({ type: 'processorConnected' });
	for await (const message of Async.breakable(messages, breaker => halt = breaker)) {
		if (message.type === 'shutdown' || halted) {
			break;
		} else if (message.type === 'mainConnected') {
			return;
		}
	}
	throw new Error('Processor initialization failure');
}();

// Create processor workers
type RoomWorker = typeof workers extends (infer Type)[] ? Type : never;
const userCount = Number(await db.data.sCard('users')) - 3; // minus Invader, Source Keeper, Screeps
const singleThreaded = config.launcher?.singleThreaded;
const processorCount = clamp(1, config.processor.concurrency, singleThreaded ? 1 : Math.ceil(userCount / 2));
const workers = await Fn.pipe(
	Fn.range(processorCount),
	$$ => Fn.mapAwait($$, async () => {
		const client = disposable.adopt(
			await negotiateResponderClient<ProcessorRequest, unknown>('xxscreeps/engine/processor/worker.js', singleThreaded),
			client => {
				client.close();
				return client.wait();
			});
		return {
			...client,
			affinity: [] as string[],
			checkAffinity: true,
			currentTime: -Infinity,
			finalizedTime: -Infinity,
			idle: true,
			pendingTime: undefined as number | undefined,
			processed: [] as string[],
		};
	}));
const affinityByRoom = new Map<string, RoomWorker>();

// Pulls rooms from process queue for a given worker. Prioritizes affinity rooms, but will return
// other rooms if needed.
async function *consumeRoomsQueue(worker: RoomWorker, time: number): AsyncIterable<string> {
	const queueKey = processRoomsSetKey(time);
	loop: while (true) {

		// Yield affinity rooms first
		while (worker.checkAffinity) {
			const affinityIterator = consumeSortedSetMembers(shard.scratch, queueKey, worker.affinity, 0, 0);
			// eslint-disable-next-line @typescript-eslint/require-await, require-yield
			const endOfAffinity = async function*() {
				worker.checkAffinity = false;
			}();
			const iterators = Fn.concatAsync([ affinityIterator, endOfAffinity ]);
			for await (const roomName of Fn.lookAhead(iterators, 1)) {
				yield roomName;
			}
		}

		// Yield non-affinity rooms until there's no more, or it's time to check affinity again
		for await (const roomName of consumeSortedSet(shard.scratch, queueKey, 0, 0)) {
			yield roomName;
			// nb: eslint ignore is automatically removed
			if (worker.checkAffinity as unknown) {
				continue loop;
			}
		}
		break;
	}
}

function startWorker(worker: RoomWorker, time: number) {
	Async.mustNotReject(async () => {
		await runWorkerUntilIdle(worker, time, async requestedTime => {
			// Continue processing until the queue is empty. Empty queue may not mean processing is
			// done, it also may mean we're waiting on runner intents. A notification received while
			// this drain is unwinding forces another drain before sleeping. The requested time is
			// carried with the wake because finalization can advance the tick during that window.
			for await (const roomName of consumeRoomsQueue(worker, requestedTime)) {
				worker.processed.push(roomName);
				await worker.responder({ type: 'process', roomName, time: requestedTime });
				log(`${roomName}, `);
			}
		});
	});
}

// Initialize workers and rooms
await Fn.mapAwait(workers, async worker => {
	await worker.responder({ type: 'world', worldBlob });
	for await (const roomName of consumeSet(shard.scratch, 'initializeRooms')) {
		await worker.responder({ type: 'initialize', roomName });
		if (halted) {
			break;
		}
	}
});
const processorMessages = processorSubscription.iterable();
await getServiceChannel(shard).publish({ type: 'processorInitialized' });

// Process messages
loop: for await (const message of Async.breakable(processorMessages, breaker => halt = breaker)) {
	switch (message.type) {
		case 'shutdown':
			break loop;

		case 'process': {
			const { time, roomNames } = message;
			processing = true;

			// Update checkAffinity flag on workers
			let activations = function() {
				if (roomNames) {
					for (const roomName of roomNames) {
						const worker = affinityByRoom.get(roomName);
						if (worker) {
							worker.checkAffinity = true;
						}
					}
					return roomNames.length;
				} else {
					return Infinity;
				}
			}();

			// Prefer idle workers, then latch enough busy workers to cover notifications that arrived
			// while every worker was transitioning to idle.
			const busyWorkers = workers.filter(worker => !worker.idle);
			for (const worker of workers) {
				if (worker.idle && requestWorkerWake(worker, time)) {
					startWorker(worker, time);
					if (--activations <= 0) {
						break;
					}
				}
			}
			if (activations > 0) {
				for (const worker of busyWorkers) {
					requestWorkerWake(worker, time);
					if (--activations <= 0) {
						break;
					}
				}
			}
			break;
		}

		// Second processing phase. This waits until all player code and first phase processing has
		// run.
		case 'finalize': {
			const { time } = message;
			log(`finalized tick ${time}\n`);
			// Run finalization in worker
			await Promise.all(Fn.map(workers, async worker => {
				if (worker.processed.length > 0) {
					await worker.responder({ type: 'finalize', time });
				}
			}));
			for (const worker of workers) {
				worker.finalizedTime = Math.max(worker.finalizedTime, time);
			}
			processing = false;
			if (halted) {
				// We check for interrupts at the end of tick
				break loop;
			}
			// Reset affinity for each worker
			affinityByRoom.clear();
			for (const worker of workers) {
				worker.affinity = worker.processed;
				worker.processed = [];
				for (const roomName of worker.affinity) {
					affinityByRoom.set(roomName, worker);
				}
			}
			break;
		}
	}
}
