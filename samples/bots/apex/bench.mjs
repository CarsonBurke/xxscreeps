/**
 * Benchmark Apex for N ticks inside the xxscreeps test shard simulator.
 *
 * Usage (from repo root, Node ≥ 22 with SuppressedError):
 *   mise exec node@24 -- node --import xxscreeps/loader \
 *     samples/bots/apex/bench.mjs [ticks=20000] [botDir=.] [flags]
 *
 * Flags:
 *   --no-tb / --no-tensorboard   skip TensorBoard event generation
 *   --tb                         force TensorBoard generation on (default)
 *   --tensorboard                start TensorBoard UI + open browser (implies TB on)
 *   --headful / --client         serve BM on the Screeps client (HTTP :21025).
 *                                Same process as the sim — open the map while it runs.
 *
 * Env (optional):
 *   APEX_BENCH_TB=0              same as --no-tb
 *   APEX_BENCH_TENSORBOARD=1     same as --tensorboard
 *   APEX_BENCH_TB_PORT=6006      TensorBoard port
 *   APEX_BENCH_HEADFUL=1         same as --headful
 *   APEX_BENCH_PASSWORD=…        login password for Player 1 (default: apexbench)
 *   APEX_BENCH_TICK_MS=N         headful tick delay ms (default 100; 0 = max speed)
 *   APEX_BENCH_NO_OPEN=1         do not auto-open the browser
 *
 * Headful example (Screeps map of this BM):
 *   mise exec node@24 -- node --import xxscreeps/loader \
 *     samples/bots/apex/bench.mjs 20000 . --headful
 *   → http://127.0.0.1:21025/  login Player 1 / apexbench → W7N3
 *   View-only: don't place/click intents or upload code (mutates the BM).
 *   Numbers from headful runs are not pure BM scores (use headless for that).
 *
 * Live world (not the isolated BM):
 *   mise exec node@24 -- npx xxscreeps start
 *
 * Writes (each BM is a new stamp — never overwrites prior runs):
 *   samples/bots/apex/runs/<stamp>/metrics.jsonl
 *   samples/bots/apex/runs/<stamp>/tb/
 *   samples/bots/apex/runs/tb/<stamp> → multi-run index
 *
 * Note: flags are stripped here before loading the test harness (argparse
 * only allows --test-redis on the simulator import path).
 */

const OUR_FLAGS = new Set([
	'--no-tb',
	'--no-tensorboard',
	'--tb',
	'--tensorboard',
	'--headful',
	'--client',
]);

const raw = process.argv.slice(2);
const has = f => raw.includes(f);
const noTb = has('--no-tb') || has('--no-tensorboard') ||
	process.env.APEX_BENCH_TB === '0' ||
	process.env.APEX_BENCH_TB === 'false';
const forceTb = has('--tb');
// --tensorboard = open UI (and force event generation)
const openTensorBoard = has('--tensorboard') ||
	process.env.APEX_BENCH_TENSORBOARD === '1' ||
	process.env.APEX_BENCH_TENSORBOARD === 'true';
const headful = has('--headful') || has('--client') ||
	process.env.APEX_BENCH_HEADFUL === '1' ||
	process.env.APEX_BENCH_HEADFUL === 'true';

// Generation default ON unless disabled; --tensorboard forces it on
process.env.APEX_BENCH_TB = (forceTb || openTensorBoard || !noTb) ? '1' : '0';
process.env.APEX_BENCH_TENSORBOARD = openTensorBoard ? '1' : '0';
process.env.APEX_BENCH_HEADFUL = headful ? '1' : '0';

// Remove our flags so xxscreeps/test argparse does not reject them
process.argv = process.argv.filter(a => !OUR_FLAGS.has(a));

await import('./bench-impl.mjs');
