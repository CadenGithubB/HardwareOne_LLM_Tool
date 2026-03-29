/**
 * tokenizer.js — parse HuggingFace tokenizer.json and encode to binary
 *
 * Output binary block layout:
 *   uint32: vocab_size
 *   uint32: n_merges
 *   For each vocab entry (id=0..vocab_size-1):
 *     uint8:  byte_length of string
 *     bytes:  UTF-8 string
 *   For each merge rule:
 *     uint16: left_token_id
 *     uint16: right_token_id
 *     uint32: merged_token_id  (the result token id)
 */

export function parseTokenizerJson(json) {
  const model = json.model;
  if (!model) throw new Error('tokenizer.json has no .model field');

  // Build id→string vocab array
  const vocabMap = model.vocab ?? {};
  const addedTokens = json.added_tokens ?? [];

  // Merge added_tokens into vocab map
  for (const t of addedTokens) {
    vocabMap[t.content] = t.id;
  }

  const maxId     = Math.max(...Object.values(vocabMap));
  const vocabSize = maxId + 1;
  const vocab     = new Array(vocabSize).fill('');

  for (const [str, id] of Object.entries(vocabMap)) {
    vocab[id] = str;
  }

  // BPE merges: array of "tokenA tokenB" strings
  const mergeStrings = model.merges ?? [];
  const merges = [];

  for (const mergeStr of mergeStrings) {
    const spaceIdx = mergeStr.indexOf(' ');
    if (spaceIdx < 0) continue;
    const left  = mergeStr.slice(0, spaceIdx);
    const right = mergeStr.slice(spaceIdx + 1);
    const leftId  = vocabMap[left];
    const rightId = vocabMap[right];
    const mergedId = vocabMap[left + right];
    if (leftId !== undefined && rightId !== undefined && mergedId !== undefined) {
      merges.push({ leftId, rightId, mergedId });
    }
  }

  return { vocab, vocabSize, merges };
}

export function encodeTokenizerBinary(tokenizer) {
  const { vocab, vocabSize, merges } = tokenizer;
  const enc = new TextEncoder();
  const parts = [];

  // Header: vocab_size, n_merges
  const header = new Uint32Array([vocabSize, merges.length]);
  parts.push(header.buffer);

  // Vocab strings
  for (let id = 0; id < vocabSize; id++) {
    const str     = unescape_hf_token(vocab[id] ?? '');
    const encoded = enc.encode(str);
    const len     = Math.min(encoded.length, 255);
    const lenBuf  = new Uint8Array([len]);
    parts.push(lenBuf.buffer);
    parts.push(encoded.slice(0, len).buffer);
  }

  // Merge rules
  for (const { leftId, rightId, mergedId } of merges) {
    const entry = new Uint32Array([leftId, rightId, mergedId]);
    parts.push(entry.buffer);
  }

  // Concatenate all parts
  const totalSize = parts.reduce((n, b) => n + b.byteLength, 0);
  const out = new Uint8Array(totalSize);
  let offset = 0;
  for (const buf of parts) {
    out.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  return out.buffer;
}

/** HuggingFace BPE uses Ġ for spaces and Ċ for newline — convert back */
function unescape_hf_token(s) {
  return s
    .replace(/Ġ/g, ' ')
    .replace(/Ċ/g, '\n')
    .replace(/ĉ/g, '\t');
}

export function tokenizerStats(tokenizer) {
  return {
    vocabSize: tokenizer.vocabSize,
    mergeCount: tokenizer.merges.length,
    sampleTokens: tokenizer.vocab.slice(0, 10).map((s, i) => `[${i}]="${s}"`).join(', ')
  };
}
