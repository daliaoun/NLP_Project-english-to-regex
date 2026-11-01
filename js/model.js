/*
 * Inference for the trained encoder-decoder, in plain JavaScript (no libraries).
 * Mirrors src/models.py: LSTM encoder, linear+tanh bridge, LSTM decoder, optional additive attention,
 * greedy decoding. The weights are the ones trained with PyTorch, exported by tools/export_site_data.py.
 */

export const PAD = 0, UNK = 1, SOS = 2, EOS = 3;
const MAX_SRC_LEN = 40;   // same truncation as training
const MAX_TGT_LEN = 60;

// ------------------------------------------------------------------ text preprocessing
// Same normalisation and tokenisation as src/data.py (lowercase, quotes removed, words + punctuation).
export function normalizeSrc(text) {
  return text.toLowerCase().replace(/['"]/g, " ").replace(/\s+/g, " ").trim();
}
const SRC_TOKEN = /[a-z0-9_]+(?:-[a-z0-9_]+)*|[^\sa-z0-9_]/g;
export function tokenizeSrc(text) {
  return normalizeSrc(text).match(SRC_TOKEN) ?? [];
}
// Regex side: \b, the four placeholder words and every other character are tokens.
const TGT_TOKEN = /\\b|dog|truck|ring|lake|[\s\S]/g;
export function tokenizeTgt(rx) {
  return rx.match(TGT_TOKEN) ?? [];
}

// ---------------------------------------------------------------------------- math helpers
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** out[r] = sum_k W[r*cols + k] * x[k]  (+ optional bias), W row-major [rows, cols] */
function matvec(W, rows, cols, x, out, bias = null) {
  for (let r = 0; r < rows; r++) {
    let s = bias ? bias[r] : 0;
    const base = r * cols;
    for (let k = 0; k < cols; k++) s += W[base + k] * x[k];
    out[r] = s;
  }
  return out;
}

export class Seq2SeqModel {
  constructor(entry, buffer) {
    this.label = entry.label;
    this.cfg = entry.config;
    this.srcVocab = entry.src_vocab;
    this.tgtVocab = entry.tgt_vocab;
    this.srcIndex = new Map(this.srcVocab.map((t, i) => [t, i]));
    this.nParams = entry.n_params;
    this.W = {};
    for (const t of entry.tensors) this.W[t.name] = new Float32Array(buffer, t.offset * 4, t.length);
    this.H = this.cfg.hid_dim;
    this.E = this.cfg.emb_dim;
    this.attention = !!this.cfg.attention;
  }

  /** token ids of a description, as used at training time */
  encodeSource(text) {
    const toks = tokenizeSrc(text).slice(0, MAX_SRC_LEN);
    const ids = toks.map((t) => this.srcIndex.get(t) ?? UNK);
    return { tokens: toks, ids: ids.length ? ids : [UNK] };
  }

  /** one LSTM step; returns new [h, c]. Gate order is PyTorch's: input, forget, cell, output. */
  lstmStep(prefix, x, inSize, h, c) {
    const { H } = this;
    const Wih = this.W[`${prefix}.weight_ih_l0`], Whh = this.W[`${prefix}.weight_hh_l0`];
    const bih = this.W[`${prefix}.bias_ih_l0`], bhh = this.W[`${prefix}.bias_hh_l0`];
    const gates = new Float32Array(4 * H);
    for (let r = 0; r < 4 * H; r++) {
      let s = bih[r] + bhh[r];
      let base = r * inSize;
      for (let k = 0; k < inSize; k++) s += Wih[base + k] * x[k];
      base = r * H;
      for (let k = 0; k < H; k++) s += Whh[base + k] * h[k];
      gates[r] = s;
    }
    const h2 = new Float32Array(H), c2 = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const i = sigmoid(gates[j]), f = sigmoid(gates[H + j]);
      const g = Math.tanh(gates[2 * H + j]), o = sigmoid(gates[3 * H + j]);
      c2[j] = f * c[j] + i * g;
      h2[j] = o * Math.tanh(c2[j]);
    }
    return [h2, c2];
  }

  embed(table, id, size) {
    return this.W[table].subarray(id * size, (id + 1) * size);
  }

  encode(ids) {
    const { H, E } = this;
    let h = new Float32Array(H), c = new Float32Array(H);
    const encOut = [];
    for (const id of ids) {
      [h, c] = this.lstmStep("encoder.rnn", this.embed("encoder.embedding.weight", id, E), E, h, c);
      encOut.push(h);
    }
    // bridge: tanh(Linear(final state)), separately for h and c
    const h0 = matvec(this.W["bridge_h.weight"], H, H, h, new Float32Array(H), this.W["bridge_h.bias"]).map(Math.tanh);
    const c0 = matvec(this.W["bridge_c.weight"], H, H, c, new Float32Array(H), this.W["bridge_c.bias"]).map(Math.tanh);
    let encProj = null;
    if (this.attention) {
      const Wh = this.W["decoder.attention.W_h.weight"];
      encProj = encOut.map((e) => matvec(Wh, H, H, e, new Float32Array(H)));
    }
    return { encOut, encProj, h0, c0 };
  }

  /** additive attention: softmax_i( v . tanh(W_h h_i + W_s s) ), context = sum_i alpha_i h_i */
  attend(h, encOut, encProj) {
    const { H } = this;
    const sProj = matvec(this.W["decoder.attention.W_s.weight"], H, H, h, new Float32Array(H),
      this.W["decoder.attention.W_s.bias"]);
    const v = this.W["decoder.attention.v.weight"];
    const S = encOut.length;
    const scores = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      let acc = 0;
      const p = encProj[s];
      for (let j = 0; j < H; j++) acc += v[j] * Math.tanh(p[j] + sProj[j]);
      scores[s] = acc;
    }
    let max = -Infinity;
    for (let s = 0; s < S; s++) if (scores[s] > max) max = scores[s];
    let z = 0;
    const alpha = new Float64Array(S);
    for (let s = 0; s < S; s++) { alpha[s] = Math.exp(scores[s] - max); z += alpha[s]; }
    const ctx = new Float32Array(H);
    for (let s = 0; s < S; s++) {
      alpha[s] /= z;
      const e = encOut[s];
      for (let j = 0; j < H; j++) ctx[j] += alpha[s] * e[j];
    }
    return { ctx, alpha };
  }

  /** Greedy decoding. Returns { tokens, text, attention: [[alpha_over_input] per generated token] } */
  greedy(ids, maxLen = MAX_TGT_LEN) {
    const { H, E } = this;
    const { encOut, encProj, h0, c0 } = this.encode(ids);
    let h = h0, c = c0, prev = SOS;
    const tokens = [];
    const attention = [];
    const outW = this.W["decoder.out.weight"], outB = this.W["decoder.out.bias"];
    const V = this.tgtVocab.length;
    for (let t = 0; t < maxLen; t++) {
      const emb = this.embed("decoder.embedding.weight", prev, E);
      let rnnIn = emb, inSize = E, ctx = null;
      if (this.attention) {
        const a = this.attend(h, encOut, encProj);
        ctx = a.ctx;
        attention.push(Array.from(a.alpha));
        rnnIn = new Float32Array(E + H);
        rnnIn.set(emb, 0);
        rnnIn.set(ctx, E);
        inSize = E + H;
      }
      [h, c] = this.lstmStep("decoder.rnn", rnnIn, inSize, h, c);
      let feats = h;
      if (this.attention) {           // output layer sees [hidden, context, previous-token embedding]
        feats = new Float32Array(H + H + E);
        feats.set(h, 0);
        feats.set(ctx, H);
        feats.set(emb, 2 * H);
      }
      const logits = matvec(outW, V, feats.length, feats, new Float32Array(V), outB);
      let best = 0;
      for (let k = 1; k < V; k++) if (logits[k] > logits[best]) best = k;
      if (best === EOS) break;
      prev = best;
      if (best !== PAD && best !== SOS) tokens.push(this.tgtVocab[best]);
    }
    return { tokens, text: tokens.join(""), attention };
  }

  /** description -> regex */
  translate(text) {
    const { tokens, ids } = this.encodeSource(text);
    const out = this.greedy(ids);
    return { ...out, inputTokens: tokens, inputIds: ids, unknown: tokens.filter((t, i) => ids[i] === UNK) };
  }
}

export async function loadModel(name, base = "data/") {
  const manifest = await (await fetch(`${base}models.json`)).json();
  const entry = manifest[name];
  const res = await fetch(`${base}${entry.file}`);
  if (!res.ok) throw new Error(`could not load ${entry.file}`);
  return new Seq2SeqModel(entry, await res.arrayBuffer());
}

export function modelFromFiles(entry, buffer) {
  return new Seq2SeqModel(entry, buffer);
}
