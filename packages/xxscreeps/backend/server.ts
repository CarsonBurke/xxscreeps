import { handleInterruptSignal } from 'xxscreeps/engine/service/signal.js';
import { initializeGameEnvironment } from 'xxscreeps/game/index.js';
import { BackendContext } from './context.js';
import { listenBackend } from './listen.js';
import 'xxscreeps:mods/backend';
import 'xxscreeps:mods/game';
import 'xxscreeps:mods/processor';

initializeGameEnvironment();

// Initialize services
await using backendContext = await BackendContext.connect();
const handle = await listenBackend(backendContext);

// Interrupt handler
const halt = Promise.withResolvers<void>();
using _signal = handleInterruptSignal(halt.resolve);
await halt.promise;

// Start graceful exit
await handle.stop();
