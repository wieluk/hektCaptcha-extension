'use strict';

import { Time } from '../src/utils';

// ---------------------------------------------------------------------------
// Time.time()
// ---------------------------------------------------------------------------
describe('Time.time()', () => {
  it('returns a number', () => {
    expect(typeof Time.time()).toBe('number');
  });

  it('returns the current timestamp (within 100 ms of Date.now)', () => {
    const before = Date.now();
    const result = Time.time();
    const after = Date.now();
    expect(result).toBeGreaterThanOrEqual(before);
    expect(result).toBeLessThanOrEqual(after);
  });

  it('falls back to new Date().getTime() when Date.now is unavailable', () => {
    const original = Date.now;
    Date.now = undefined;
    const result = Time.time();
    Date.now = original;
    expect(typeof result).toBe('number');
    expect(result).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Time.sleep()
// ---------------------------------------------------------------------------
describe('Time.sleep()', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it('returns a Promise', () => {
    const p = Time.sleep(100);
    expect(p).toBeInstanceOf(Promise);
    jest.runAllTimers();
  });

  it('resolves after the specified number of milliseconds', async () => {
    const promise = Time.sleep(500);
    jest.advanceTimersByTime(500);
    await expect(promise).resolves.toBeUndefined();
  });

  it('uses a default delay of 1000 ms', async () => {
    const promise = Time.sleep();
    jest.advanceTimersByTime(1000);
    await expect(promise).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Time.random_sleep()
// ---------------------------------------------------------------------------
describe('Time.random_sleep()', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it('resolves after a duration within [min, max)', async () => {
    // Force Math.random() to return 0.5 so duration = floor(0.5*(max-min)+min)
    const randomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.5);
    const promise = Time.random_sleep(200, 400);
    // duration = floor(0.5 * (400-200) + 200) = floor(300) = 300
    jest.advanceTimersByTime(300);
    await expect(promise).resolves.toBeUndefined();
    randomSpy.mockRestore();
  });

  it('always sleeps at least min ms when Math.random returns 0', async () => {
    const randomSpy = jest.spyOn(Math, 'random').mockReturnValue(0);
    const promise = Time.random_sleep(100, 500);
    // duration = floor(0 * (500-100) + 100) = 100
    jest.advanceTimersByTime(100);
    await expect(promise).resolves.toBeUndefined();
    randomSpy.mockRestore();
  });
});
