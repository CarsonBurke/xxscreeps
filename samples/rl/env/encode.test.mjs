import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { encodeObservationFromRooms, SCHEMA } from './encode.mjs';
import { makeEncodeFixture } from './encode_fixture.mjs';

const EXPECTED_TENSOR_SHA256 = '94577a949c1cb14575aa90e7231ba91b1807bc1b5ebfb7db1608ef88ce616f5b';

function tensorDigest(observation) {
	const hash = createHash('sha256');
	for (const value of Object.values(observation._raw)) {
		hash.update(Buffer.from(value.buffer, value.byteOffset, value.byteLength));
	}
	return hash.digest('hex');
}

function patchOffset(room, x, y, feature) {
	const patchX = Math.floor(x / SCHEMA.patchSize);
	const patchY = Math.floor(y / SCHEMA.patchSize);
	const patch = patchY * SCHEMA.patchesPerSide + patchX;
	const cell = (y % SCHEMA.patchSize) * SCHEMA.patchSize + (x % SCHEMA.patchSize);
	return ((room * SCHEMA.patchesPerRoom + patch) * SCHEMA.patchSize * SCHEMA.patchSize
		+ cell) * SCHEMA.tileFeat + feature;
}

test('current-schema binary tensors remain byte-exact after packed painting', () => {
	const { Game, roomNames } = makeEncodeFixture();
	const observation = encodeObservationFromRooms(Game, '100', roomNames);

	assert.equal(observation.encoding, 'bin');
	assert.deepEqual(observation.roomNames, roomNames);
	assert.deepEqual(observation.shapes.patches, [ 4, 100, 700 ]);
	assert.equal(observation._raw.patches.byteLength, 280000);
	assert.equal(tensorDigest(observation), EXPECTED_TENSOR_SHA256);
});

test('pooled snapshots reset dynamic paint and preserve the adjacent result', () => {
	const { Game, roomNames } = makeEncodeFixture({ roomCount: 1, creepsPerRoom: 1 });
	const creep = Object.values(Game.creeps)[0];
	const ownOffset = patchOffset(0, creep.pos.x, creep.pos.y, 20);
	const hostileOffset = patchOffset(0, creep.pos.x, creep.pos.y, 24);
	const first = encodeObservationFromRooms(Game, '100', roomNames);
	const firstBytes = Buffer.from(first._raw.patches);

	assert.equal(first._raw.patches[ownOffset], 255);
	assert.equal(first._raw.patches[hostileOffset], 0);
	creep.my = false;
	const second = encodeObservationFromRooms(Game, '100', roomNames);
	assert.deepEqual(Buffer.from(first._raw.patches), firstBytes);
	assert.notEqual(first._raw.patches.buffer, second._raw.patches.buffer);
	assert.equal(second._raw.patches[ownOffset], 0);
	assert.equal(second._raw.patches[hostileOffset], 255);

	creep.my = true;
	const third = encodeObservationFromRooms(Game, '100', roomNames);
	assert.equal(first._raw.patches.buffer, third._raw.patches.buffer);
	assert.equal(third._raw.patches[ownOffset], 255);
	assert.equal(third._raw.patches[hostileOffset], 0);
});

test('zero visible rooms emit the required empty one-room patch payload', () => {
	const { Game } = makeEncodeFixture({ roomCount: 0, creepsPerRoom: 0 });
	const observation = encodeObservationFromRooms(Game, '100', []);

	assert.equal(observation.roomsUsed, 1);
	assert.deepEqual(observation.shapes.patches, [ 1, 100, 700 ]);
	assert.ok(observation._raw.patches.every(value => value === 0));
	assert.ok(observation._raw.roomMask.every(value => value === 0));
});
