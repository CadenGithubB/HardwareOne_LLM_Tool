/**
 * safetensors.js — parse HuggingFace .safetensors files
 *
 * Format:
 *   [8 bytes: header_length uint64-LE]
 *   [header_length bytes: UTF-8 JSON]
 *   [tensor data bytes, concatenated]
 *
 * Each tensor entry: { dtype, shape, data_offsets: [start, end] }
 */

export async function parseSafetensors(file) {
  const prefix    = await file.slice(0, 8).arrayBuffer();
  const headerLen = Number(new DataView(prefix).getBigUint64(0, true));
  const headerBuf = await file.slice(8, 8 + headerLen).arrayBuffer();
  const header    = JSON.parse(new TextDecoder().decode(headerBuf));
  const dataStart = 8 + headerLen;

  const names = Object.keys(header).filter(k => k !== '__metadata__');

  return {
    names,
    header,

    async getTensor(name) {
      const meta = header[name];
      if (!meta) return null;
      const [start, end] = meta.data_offsets;
      const buf = await file.slice(dataStart + start, dataStart + end).arrayBuffer();
      return convertToF32(buf, meta.dtype, meta.shape);
    },

    getMeta(name) {
      return header[name] ?? null;
    },

    /** Returns the number of elements for a tensor (product of shape dims) */
    getNumel(name) {
      const meta = header[name];
      if (!meta) return 0;
      return meta.shape.reduce((a, b) => a * b, 1);
    }
  };
}

function convertToF32(buf, dtype, shape) {
  const numel = shape.reduce((a, b) => a * b, 1);

  switch (dtype) {
    case 'F32':
      return new Float32Array(buf);

    case 'F16': {
      const u16 = new Uint16Array(buf);
      const f32 = new Float32Array(numel);
      for (let i = 0; i < numel; i++) f32[i] = f16ToFloat(u16[i]);
      return f32;
    }

    case 'BF16': {
      // BF16 is just the high 16 bits of F32 — shift up 16 bits
      const u16 = new Uint16Array(buf);
      const f32 = new Float32Array(numel);
      const u32 = new Uint32Array(f32.buffer);
      for (let i = 0; i < numel; i++) u32[i] = u16[i] << 16;
      return f32;
    }

    case 'I8':
      return float32From(new Int8Array(buf));

    case 'U8':
      return float32From(new Uint8Array(buf));

    default:
      console.warn(`Unsupported dtype: ${dtype}, treating as F32`);
      return new Float32Array(buf);
  }
}

function float32From(typedArr) {
  const f32 = new Float32Array(typedArr.length);
  for (let i = 0; i < typedArr.length; i++) f32[i] = typedArr[i];
  return f32;
}

function f16ToFloat(h) {
  const sign = (h >>> 15) & 1;
  const exp  = (h >>> 10) & 0x1f;
  const frac =  h         & 0x3ff;

  if (exp === 0) {
    const val = Math.pow(2, -14) * (frac / 1024);
    return sign ? -val : val;
  }
  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }
  const val = Math.pow(2, exp - 15) * (1 + frac / 1024);
  return sign ? -val : val;
}
