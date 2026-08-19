import type ivm from 'isolated-vm';
import type { InitializationPayload, TickPayload } from 'xxscreeps/engine/runner/index.js';
import type { CPU } from 'xxscreeps/game/game.js';
import * as Runtime from 'xxscreeps/driver/runtime/index.js';
import { hooks } from 'xxscreeps/game/index.js';

export { tick } from 'xxscreeps/driver/runtime/index.js';

export let isolate: ivm.Isolate;

declare module 'xxscreeps/game/game.js' {
	interface CPU {
		/**
		 * Use this method to get heap statistics for your virtual machine. The return value is almost
		 * identical to the Node.js function `v8.getHeapStatistics()`. This function returns one
		 * additional property: `externally_allocated_size` which is the total amount of currently
		 * allocated memory which is not included in the v8 heap but counts against this isolate's memory
		 * limit. `ArrayBuffer` instances over a certain size are externally allocated and will be counted
		 * here.
		 */
		getHeapStatistics: () => ivm.HeapStatistics;

		/**
		 * Reset your runtime environment and wipe all data in heap memory.
		 */
		halt: () => never;
	}
}

class IsolatedCPU implements CPU {
	bucket;
	limit;
	tickLimit;
	readonly #startTime;
	readonly #deterministic;
	// A virtual clock advances a fixed step per call, so code that measures its
	// own cost sees the same sequence on every replay. Monotonic by construction,
	// so a loop waiting for the budget to be consumed still terminates.
	#virtual = 0;
	readonly #heapLimit = 256 * 1024 * 1024;

	constructor(data: TickPayload) {
		this.bucket = data.cpu.bucket;
		this.limit = data.cpu.limit;
		this.tickLimit = data.cpu.tickLimit;
		this.#startTime = isolate.wallTime;
		this.#deterministic = data.cpu.deterministic ?? false;
	}

	// Heap occupancy is a function of garbage-collection timing, and player code
	// divides by these fields, so a reproducible run reports a fixed plausible
	// heap rather than the real one. Zeros are wrong here: a ratio of zeros is
	// NaN and silently changes which branch player code takes.
	getHeapStatistics = () => this.#deterministic
		? {
			total_heap_size: 16 * 1024 * 1024,
			total_heap_size_executable: 1024 * 1024,
			total_physical_size: 16 * 1024 * 1024,
			total_available_size: (this.#heapLimit) - 16 * 1024 * 1024,
			used_heap_size: 8 * 1024 * 1024,
			heap_size_limit: this.#heapLimit,
			malloced_memory: 1024 * 1024,
			peak_malloced_memory: 2 * 1024 * 1024,
			does_zap_garbage: 0,
			number_of_native_contexts: 1,
			number_of_detached_contexts: 0,
			externally_allocated_size: 0,
		} as ivm.HeapStatistics
		: isolate.getHeapStatisticsSync();

	getUsed = () => this.#deterministic
		? (this.#virtual += 0.05)
		: Number(isolate.wallTime - this.#startTime) / 1e6;

	halt = () => {
		isolate.dispose();
		return undefined as never;
	};
}

hooks.register('gameInitializer', (game, data) => {
	game.cpu = new IsolatedCPU(data!);
});

export function initialize(
	isolate_: ivm.Isolate,
	context: ivm.Context,
	data: InitializationPayload,
) {
	isolate = isolate_;
	// Evaluation for plain JS scripts
	const evaluate: Runtime.Evaluate = (source, filename): unknown => {
		const script = isolate_.compileScriptSync(source, { filename });
		return script.runSync(context, { reference: true }).deref();
	};

	// Compilation for ES Modules
	type WithFilename = ivm.Module & Record<'filename', string>;
	const compiler: Runtime.Compiler<WithFilename> = {
		compile(source, filename) {
			const module = isolate_.compileModuleSync(source, { filename }) as WithFilename;
			module.filename = filename;
			return module;
		},
		evaluate(module, linker): unknown {
			module.instantiateSync(context, (specifier, referrer) =>
				linker(specifier, (referrer as WithFilename).filename));
			module.evaluateSync();
			return module.namespace.deref();
		},
	};

	Runtime.initialize(compiler, evaluate, data);
}
