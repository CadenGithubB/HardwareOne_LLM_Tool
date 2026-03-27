/**
 * worker.js — quantization runs off the main thread
 * Receives tensors, returns INT8 quantized data + scales
 */

self.onmessage = function ({ data }) {
  if (data.type === 'quantize') {
    const { name, tensor, groupSize, isNorm } = data;

    if (isNorm) {
      // Norm weights stay as F32 — they're tiny and precision matters
      self.postMessage(
        { type: 'done', name, mode: 'f32', data: tensor },
        [tensor.buffer]
      );
      return;
    }

    const result = quantizeInt8(tensor, groupSize);
    self.postMessage(
      { type: 'done', name, mode: 'int8', scales: result.scales, quant: result.quant },
      [result.scales.buffer, result.quant.buffer]
    );
  }
};

function quantizeInt8(f32, groupSize) {
  const len    = f32.length;
  const nGroups = Math.ceil(len / groupSize);
  const scales = new Float32Array(nGroups);
  const quant  = new Int8Array(len);

  for (let g = 0; g < nGroups; g++) {
    const start = g * groupSize;
    const end   = Math.min(start + groupSize, len);

    let absMax = 0;
    for (let i = start; i < end; i++) {
      const v = Math.abs(f32[i]);
      if (v > absMax) absMax = v;
    }

    const scale = absMax > 0 ? absMax / 127.0 : 1.0;
    scales[g]   = scale;
    const inv   = 1.0 / scale;

    for (let i = start; i < end; i++) {
      let q = Math.round(f32[i] * inv);
      if (q >  127) q =  127;
      if (q < -128) q = -128;
      quant[i] = q;
    }
  }

  return { scales, quant };
}
