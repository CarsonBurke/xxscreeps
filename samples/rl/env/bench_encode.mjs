import { performance } from 'node:perf_hooks';
import { encodeObservationFromRooms } from './encode.mjs';
import { makeEncodeFixture } from './encode_fixture.mjs';

const iterations = Number(process.env.RL_ENCODE_BENCH_ITERS || 2000);
const warmup = Number(process.env.RL_ENCODE_BENCH_WARMUP || 200);
const { Game, roomNames } = makeEncodeFixture();

function run(count) {
	let checksum = 0;
	const start = performance.now();
	for (let i = 0; i < count; i++) {
		Game.time++;
		const observation = encodeObservationFromRooms(Game, '100', roomNames);
		const patches = observation._raw.patches;
		checksum = (checksum + patches[0] + patches[patches.length - 1]
			+ observation._raw.actorMask[0] + observation._raw.targetMask[0]) >>> 0;
	}
	return { elapsedMs: performance.now() - start, checksum };
}

run(warmup);
const result = run(iterations);
const memory = process.memoryUsage();
console.log(JSON.stringify({
	iterations,
	rooms: roomNames.length,
	elapsedMs: Number(result.elapsedMs.toFixed(3)),
	usPerObservation: Number((result.elapsedMs * 1000 / iterations).toFixed(3)),
	observationsPerSecond: Number((iterations * 1000 / result.elapsedMs).toFixed(1)),
	checksum: result.checksum,
	heapUsedMiB: Number((memory.heapUsed / 1048576).toFixed(2)),
	arrayBuffersMiB: Number((memory.arrayBuffers / 1048576).toFixed(2)),
}));
