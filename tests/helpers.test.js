'use strict';

import {
  cosineSimilarity,
  softmax,
  scaleBoxes,
  overflowBoxes,
  preprocessImageData,
} from '../src/helpers';

// ---------------------------------------------------------------------------
// cosineSimilarity()
// ---------------------------------------------------------------------------
describe('cosineSimilarity()', () => {
  it('returns 1 for identical vectors', () => {
    const v = [1, 2, 3];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
  });

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [-1, 0, 0])).toBeCloseTo(-1, 5);
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
  });

  it('handles single-element vectors', () => {
    expect(cosineSimilarity([5], [5])).toBeCloseTo(1, 5);
    expect(cosineSimilarity([3], [-3])).toBeCloseTo(-1, 5);
  });

  it('returns a value in [-1, 1] for arbitrary vectors', () => {
    const a = [0.1, 0.5, -0.3, 0.9];
    const b = [-0.4, 0.2, 0.8, 0.1];
    const result = cosineSimilarity(a, b);
    expect(result).toBeGreaterThanOrEqual(-1);
    expect(result).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// softmax()
// ---------------------------------------------------------------------------
describe('softmax()', () => {
  it('outputs values that sum to 1', () => {
    const result = softmax([1, 2, 3]);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it('all output values are in (0, 1)', () => {
    const result = softmax([-1, 0, 1, 2]);
    result.forEach((v) => {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    });
  });

  it('assigns higher probability to the larger logit', () => {
    const result = softmax([0, 10]);
    expect(result[1]).toBeGreaterThan(result[0]);
  });

  it('handles a single-element input', () => {
    const result = softmax([42]);
    expect(result).toHaveLength(1);
    expect(result[0]).toBeCloseTo(1, 5);
  });

  it('handles Float32Array input', () => {
    const input = new Float32Array([1.0, 2.0, 3.0]);
    const result = softmax(input);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it('handles empty input', () => {
    expect(softmax([])).toEqual([]);
  });

  it('is numerically stable with large logits (constant offset invariance)', () => {
    const base = [1, 2, 3];
    const shifted = [101, 102, 103];
    const r1 = softmax(base);
    const r2 = softmax(shifted);
    r1.forEach((v, i) => expect(v).toBeCloseTo(r2[i], 5));
  });
});

// ---------------------------------------------------------------------------
// scaleBoxes()
// ---------------------------------------------------------------------------
describe('scaleBoxes()', () => {
  it('returns the same coordinates when image and scaled dims are equal', () => {
    const result = scaleBoxes([10, 20, 30, 40], [100, 100], [100, 100]);
    expect(result).toEqual([10, 20, 30, 40]);
  });

  it('scales boxes correctly for a 2x upscale with no padding', () => {
    // imageDims=[50,50], scaledDims=[100,100] → gain=2, wPad=0, hPad=0
    const result = scaleBoxes([20, 30, 40, 60], [50, 50], [100, 100]);
    expect(result[0]).toBeCloseTo(10, 5); // (20 - 0) / 2
    expect(result[1]).toBeCloseTo(15, 5); // (30 - 0) / 2
    expect(result[2]).toBeCloseTo(20, 5); // 40 / 2 (width scaled by gain only)
    expect(result[3]).toBeCloseTo(30, 5); // 60 / 2 (height scaled by gain only)
  });

  it('accounts for letterbox padding correctly with [x, y, w, h] semantics', () => {
    // imageDims=[200,100], scaledDims=[100,100]
    // gain = min(100/200, 100/100) = 0.5
    // wPad = (100 - 0.5*200)/2 = 0, hPad = (100 - 0.5*100)/2 = 25
    // x and y account for padding and gain; w and h are only scaled by gain.
    const result = scaleBoxes([50, 50, 150, 75], [200, 100], [100, 100]);
    expect(result[0]).toBeCloseTo(100, 5); // (50 - 0) / 0.5
    expect(result[1]).toBeCloseTo(50, 5); // (50 - 25) / 0.5
    expect(result[2]).toBeCloseTo(300, 5); // 150 / 0.5 (width scaled by gain only)
    expect(result[3]).toBeCloseTo(150, 5); // 75 / 0.5 (height scaled by gain only)
  });

  it('returns an empty array for empty input', () => {
    expect(scaleBoxes([], [100, 100], [100, 100])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// overflowBoxes()
// ---------------------------------------------------------------------------
describe('overflowBoxes()', () => {
  it('does not modify a box fully inside the canvas', () => {
    expect(overflowBoxes([10, 20, 30, 40], 100)).toEqual([10, 20, 30, 40]);
  });

  it('clamps negative x and y to 0', () => {
    const result = overflowBoxes([-5, -10, 20, 30], 100);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0);
  });

  it('reduces width when x+w exceeds maxSize', () => {
    // x=80, w=30 → x+w=110 > 100, new w = 100-80 = 20
    const result = overflowBoxes([80, 10, 30, 20], 100);
    expect(result[2]).toBe(20);
  });

  it('reduces height when y+h exceeds maxSize', () => {
    // y=90, h=20 → y+h=110 > 100, new h = 100-90 = 10
    const result = overflowBoxes([10, 90, 20, 20], 100);
    expect(result[3]).toBe(10);
  });

  it('handles a box that starts at the canvas boundary', () => {
    const result = overflowBoxes([100, 100, 0, 0], 100);
    expect(result).toEqual([100, 100, 0, 0]);
  });

  it('does not mutate the input array', () => {
    const original = [80, 10, 30, 20];
    const copy = [...original];
    overflowBoxes(original, 100);
    expect(original).toEqual(copy);
  });
});

// ---------------------------------------------------------------------------
// preprocessImageData()
// ---------------------------------------------------------------------------
describe('preprocessImageData()', () => {
  // Helper: build a 1-pixel RGBA buffer
  function pixelBuffer(r, g, b, a = 255) {
    return new Uint8Array([r, g, b, a]);
  }

  it('returns a Float32Array', () => {
    const result = preprocessImageData(pixelBuffer(128, 64, 32));
    expect(result).toBeInstanceOf(Float32Array);
  });

  it('produces 3 values per pixel (alpha channel stripped)', () => {
    // 1 pixel → 3 floats (R, G, B)
    const result = preprocessImageData(pixelBuffer(10, 20, 30));
    expect(result).toHaveLength(3);
  });

  it('lays out data in channel-first order [R…, G…, B…]', () => {
    // 2 pixels: [255,0,0,255] and [0,255,0,255]
    const buf = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]);
    const result = preprocessImageData(buf, false); // skip normalization
    // Expected: [R0, R1, G0, G1, B0, B1] = [1, 0, 0, 1, 0, 0]
    expect(result[0]).toBeCloseTo(1, 5); // R of pixel 0
    expect(result[1]).toBeCloseTo(0, 5); // R of pixel 1
    expect(result[2]).toBeCloseTo(0, 5); // G of pixel 0
    expect(result[3]).toBeCloseTo(1, 5); // G of pixel 1
    expect(result[4]).toBeCloseTo(0, 5); // B of pixel 0
    expect(result[5]).toBeCloseTo(0, 5); // B of pixel 1
  });

  it('normalizes values to [0, 1] before ImageNet standardisation', () => {
    // With normalize=false, a pixel value of 255 should become 1.0
    const result = preprocessImageData(pixelBuffer(255, 255, 255), false);
    expect(result[0]).toBeCloseTo(1.0, 5);
    expect(result[1]).toBeCloseTo(1.0, 5);
    expect(result[2]).toBeCloseTo(1.0, 5);
  });

  it('applies ImageNet mean/std normalisation when normalize=true', () => {
    // For a 255 (=1.0) red pixel: (1.0 - 0.485) / 0.229 ≈ 2.2489
    const result = preprocessImageData(pixelBuffer(255, 0, 0));
    expect(result[0]).toBeCloseTo((1.0 - 0.485) / 0.229, 3); // R channel
    expect(result[1]).toBeCloseTo((0.0 - 0.456) / 0.224, 3); // G channel
    expect(result[2]).toBeCloseTo((0.0 - 0.406) / 0.225, 3); // B channel
  });

  it('normalize defaults to true', () => {
    const withTrue = preprocessImageData(pixelBuffer(100, 150, 200), true);
    const withDefault = preprocessImageData(pixelBuffer(100, 150, 200));
    expect(Array.from(withTrue)).toEqual(Array.from(withDefault));
  });

  it('handles an empty buffer', () => {
    const result = preprocessImageData(new Uint8Array([]));
    expect(result).toHaveLength(0);
  });

  it('applies ImageNet normalisation correctly for multi-pixel input', () => {
    // Three pixels with distinct colors to exercise all channel slices:
    // P0 = (255,   0,   0)
    // P1 = (  0, 255,   0)
    // P2 = (  0,   0, 255)
    const buf = new Uint8Array([
      255, 0,   0,   255, // P0
      0,   255, 0,   255, // P1
      0,   0,   255, 255, // P2
    ]);

    const result = preprocessImageData(buf, true);
    const numPixels = 3;

    const imagenetMean = [0.485, 0.456, 0.406];
    const imagenetStd = [0.229, 0.224, 0.225];

    const reds   = [255, 0, 0];
    const greens = [0, 255, 0];
    const blues  = [0, 0, 255];

    for (let i = 0; i < numPixels; i++) {
      const rIndex = i;
      const gIndex = numPixels + i;
      const bIndex = 2 * numPixels + i;

      const expectedR = (reds[i]   / 255 - imagenetMean[0]) / imagenetStd[0];
      const expectedG = (greens[i] / 255 - imagenetMean[1]) / imagenetStd[1];
      const expectedB = (blues[i]  / 255 - imagenetMean[2]) / imagenetStd[2];

      expect(result[rIndex]).toBeCloseTo(expectedR, 3);
      expect(result[gIndex]).toBeCloseTo(expectedG, 3);
      expect(result[bIndex]).toBeCloseTo(expectedB, 3);
    }
  });
});
