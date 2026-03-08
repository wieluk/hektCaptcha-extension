'use strict';

/**
 * Computes the cosine similarity between two numeric vectors.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
export function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((acc, val, i) => acc + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((acc, val) => acc + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((acc, val) => acc + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Applies the softmax function to an array of logit values.
 * Uses the numerically stable max-subtraction trick to prevent overflow.
 * @param {number[]|Float32Array} x
 * @returns {number[]}
 */
export function softmax(x) {
  const arr = Array.from(x);

  if (arr.length === 0) {
    return [];
  }

  // Numerically stable softmax: shift by the maximum logit
  const maxLogit = Math.max(...arr);
  const shiftedExp = arr.map((v) => Math.exp(v - maxLogit));
  const sumShiftedExp = shiftedExp.reduce((a, b) => a + b, 0);

  // If everything underflowed or sum is not finite, fall back to uniform
  if (sumShiftedExp === 0 || !Number.isFinite(sumShiftedExp)) {
    const uniformProb = 1 / arr.length;
    return new Array(arr.length).fill(uniformProb);
  }

  return shiftedExp.map((v) => v / sumShiftedExp);
}

/**
 * Scales bounding-box coordinates back from the letterboxed/padded image
 * space to the original image space.
 * Expects boxes in [x, y, w, h] format: x and y are coordinates (offset by
 * letterbox padding and gain), while w and h are sizes (scaled by gain only).
 * @param {number[]} boxes  [x, y, w, h]
 * @param {number[]} imageDims  [width, height] of the original image
 * @param {number[]} scaledDims [width, height] of the scaled image
 * @returns {number[]}
 */
export function scaleBoxes(boxes, imageDims, scaledDims) {
  const gain = Math.min(
    scaledDims[0] / imageDims[0],
    scaledDims[1] / imageDims[1]
  );
  const wPad = (scaledDims[0] - gain * imageDims[0]) / 2;
  const hPad = (scaledDims[1] - gain * imageDims[1]) / 2;
  return boxes.map((box, i) => {
    const pad = i % 2 === 0 ? wPad : hPad;
    return i < 2 ? (box - pad) / gain : box / gain;
  });
}

/**
 * Clamps a bounding box so it does not exceed the given square canvas size.
 * Returns a new array; the original box is not mutated.
 * @param {number[]} box  [x, y, w, h]
 * @param {number} maxSize
 * @returns {number[]}
 */
export function overflowBoxes(box, maxSize) {
  const x = box[0] >= 0 ? box[0] : 0;
  const y = box[1] >= 0 ? box[1] : 0;
  const w = x + box[2] <= maxSize ? box[2] : maxSize - x;
  const h = y + box[3] <= maxSize ? box[3] : maxSize - y;
  return [x, y, w, h];
}

/**
 * Converts raw RGBA image buffer data into a normalised Float32Array in
 * channel-first (CHW) order suitable for ONNX model input.
 *
 * The output layout is [R…, G…, B…] (alpha channel is ignored).
 * When `normalize` is true the values are standardised with
 * mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] (ImageNet norms).
 *
 * @param {Uint8Array|number[]} imageBufferData  Raw RGBA pixel data
 * @param {boolean} [normalize=true]
 * @returns {Float32Array}
 */
export function preprocessImageData(imageBufferData, normalize = true) {
  const [redArray, greenArray, blueArray] = [[], [], []];

  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
  }

  const transposedData = redArray.concat(greenArray, blueArray);
  const float32Data = new Float32Array(transposedData.map((x) => x / 255.0));

  if (normalize) {
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const pixelsPerChannel = float32Data.length / 3;
    for (let i = 0; i < float32Data.length; i++) {
      const channelIndex = Math.floor(i / pixelsPerChannel);
      float32Data[i] = (float32Data[i] - mean[channelIndex]) / std[channelIndex];
    }
  }

  return float32Data;
}
