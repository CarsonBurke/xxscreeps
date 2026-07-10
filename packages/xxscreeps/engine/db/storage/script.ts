import type { LocalKeyValResponder } from './local/keyval.js';
import type { Value } from './provider.js';
import { createHash } from 'node:crypto';

/**
 * Represents a script which can be sent to the keyval storage engine and run locally on that
 * service. This is a plain JavaScript function in the local provider, or a lua script on Redis.
 */
export class KeyvalScript<Result extends Value | Value[] | null = any, Keys extends string[] = [], Argv extends Value[] = []> {
	readonly [provider: string]: string;
	readonly local: string;
	readonly localId: string;
	constructor(basicImpl: (keyval: LocalKeyValResponder, keys: Keys, argv: Argv) => Result, extra: Record<string, string> = {}) {
		this.local = basicImpl.toString();
		this.localId = createHash('sha256').update(this.local).digest('base64url').slice(0, 16);
		Object.assign(this, extra);
	}
}
