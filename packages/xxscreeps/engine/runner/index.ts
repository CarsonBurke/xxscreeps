import type { MessageFor } from 'xxscreeps/engine/db/channel.js';
import type { CodeBlobs } from 'xxscreeps/engine/db/user/code-schema.js';
import type { RoomIntentPayload } from 'xxscreeps/engine/processor/index.js';
import type { RunnerIntent, getRunnerUserChannel } from 'xxscreeps/engine/runner/model.js';

export { hooks } from './symbols.js';

export interface InitializationPayload {
	userId: string;
	codeBlob: CodeBlobs | undefined;
	shardName: string;
	terrainBlob: Readonly<Uint8Array>;
	/**
	 * Seed for a deterministic `Math.random` inside the player sandbox. A sandbox
	 * is a separate realm, so a host-side override of `Math.random` never reaches
	 * player code and an unseeded sandbox replays a different game from the same
	 * world. Left undefined the sandbox keeps the platform generator.
	 */
	randomSeed?: number;
}

export interface TickPayload {
	cpu: {
		bucket: number;
		limit: number;
		tickLimit: number;
		/** Report a virtual `getUsed` instead of wall-clock elapsed time. */
		deterministic?: boolean;
	};
	roomBlobs: Readonly<Uint8Array>[];
	time: number;
	backendIntents?: RunnerIntent[];
	eval: Extract<MessageFor<typeof getRunnerUserChannel>, { type: 'eval' }>['payload'][];
	usernames?: Record<string, string>;
	// User ids a connector wants resolved to usernames this tick (e.g. market transaction parties not
	// visible in any of the player's rooms). The runner merges these into the unseen-user resolution.
	userIds?: string[];
}

export interface TickResult {
	error?: true;
	console: string | undefined;
	evalAck?: {
		id: string;
		result: {
			error: boolean;
			value: string;
		};
	}[];
	intentPayloads: Record<string, RoomIntentPayload>;
	usage: TickUsageResult;
}

export interface TickUsageResult {
	cpu?: number;
}
