import type { BackendContext } from './context.js';
import type { Context, State } from './index.js';
import * as http from 'node:http';
import Koa from 'koa';
import bodyParser from 'koa-bodyparser';
import ConditionalGet from 'koa-conditional-get';
import Router from 'koa-router';
import { config } from 'xxscreeps/config/index.js';
import { authentication } from './auth/index.js';
import { installEndpointHandlers } from './endpoints/index.js';
import { setupGracefulShutdown } from './graceful.js';
import { installSocketHandlers, installUpgradeHandlers } from './socket.js';
import { hooks } from './symbols.js';

export type BackendListenHandle = {
	url: string;
	port: number;
	host: string | undefined;
	httpServer: http.Server;
	stop: () => Promise<void>;
};

/**
 * Start the HTTP + SockJS backend against an existing `BackendContext`.
 * Used by `server.ts` and in-process tools (e.g. headful bench) that already hold the shard open.
 */
export async function listenBackend(backendContext: BackendContext): Promise<BackendListenHandle> {
	hooks.makeIterated('backendReady')(backendContext.db, backendContext.shard);

	const koa = new Koa<State, Context>();
	const router = new Router<State, Context>();
	const httpServer = http.createServer(koa.callback());
	const unlistenServer = setupGracefulShutdown(httpServer);
	installUpgradeHandlers(koa, httpServer);
	const socketHandler = installSocketHandlers(koa, backendContext);

	koa.use(ConditionalGet());
	koa.use(async (context, next) => {
		try {
			await next();
		} catch (err) {
			console.error(`Unhandled error. Endpoint: ${context.url}\n`, err);
			context.status = 500;
			context.body = '';
		}
	});
	koa.use((context, next) => {
		context.backend = backendContext;
		context.db = backendContext.db;
		context.shard = backendContext.shard;
		return next();
	});
	koa.use(bodyParser({
		jsonLimit: '8mb',
	}));
	koa.use(authentication());
	hooks.makeIterated('middleware')(koa, router);
	koa.use(router.routes());
	koa.use(router.allowedMethods());
	installEndpointHandlers(koa, router);

	// Read configuration — same bind parsing as the standalone backend service
	const bind = config.backend.bind;
	const parts = String(bind).split(':');
	const hostPart = parts[0] === '*' || parts[0] === '' ? undefined : parts[0];
	const port = Number(parts[1] ?? 21025);

try {
		await new Promise<void>((resolve, reject) => {
			httpServer.once('error', reject);
			const onListen = () => {
				httpServer.off('error', reject);
				console.log('🌎 Listening');
				resolve();
			};
			if (hostPart === undefined) {
				httpServer.listen(port, onListen);
			} else {
				httpServer.listen(port, hostPart, onListen);
			}
		});
	} catch (err) {
		httpServer.close();
		throw err;
	}

	const urlHost = hostPart && hostPart !== '0.0.0.0' ? hostPart : '127.0.0.1';
	return {
		url: `http://${urlHost}:${port}/`,
		port,
		host: hostPart,
		httpServer,
		async stop() {
			await unlistenServer();
			await socketHandler.flush();
		},
	};
}
