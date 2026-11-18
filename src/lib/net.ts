// A decoder-only Transformer with sparse weights.
//
//   x_t = E[token_t] + PE(t)                          (sinusoidal position code in a few dims)
//   for each layer:
//     x += Σ_heads Wo · Σ_s softmax_s(q_t·k_s·scale − slope·(t−s)) v_s     (causal, ALiBi)
//     x[dst..] = x[src..] / ‖x[src..]‖   for each norm of the layer (after attention)
//     x += W2 · ReLU(W1 · x + b1)                                           (optional MLP)
//   logits = Wu · x + bu
//
// The only normalization is the block L2 norm above. All matrices are stored sparse; the
// forward pass is exact float arithmetic (keys and scores in float64), so a weight that was
// constructed to select one position does so up to e^-margin.
//
// Functions take named parameters, except the sparse kernels (Csr.addMul, Csr.addRow,
// Acc.addRow, Acc.add): they run millions of times per token, and an argument object per
// call would dominate the forward pass.

export type Sparse = { rows: number; cols: number; r: Int32Array; c: Int32Array; v: Float32Array };

export type Head = {
  name: string;
  dk: number;
  dv: number;
  scale: number;
  slope: number;
  Wq: Sparse; // d × dk
  Wk: Sparse; // d × dk
  Wv: Sparse; // d × dv
  Wo: Sparse; // dv × d
};

export type Mlp = { n: number; W1: Sparse; b1: Float32Array; W2: Sparse };

// L2 normalization of a block of the residual into another block (like RMSNorm, without gain)
export type Norm = { src: number; dst: number; n: number };

export type Layer = { heads: Head[]; mlp: Mlp | null; norms?: Norm[] };

export type Net = {
  d: number;
  vocab: number;
  pe: { offset: number; freqs: number[] };
  E: Sparse; // vocab × d
  layers: Layer[];
  Wu: Sparse; // d × vocab
  bu: Float32Array;
  meta: Record<string, unknown>;
};

// ---------- sparse matrix in row-major CSR (rows = input dimension) ----------

export class Csr {
  readonly rows: number;
  readonly cols: number;
  readonly ptr: Int32Array;
  readonly col: Int32Array;
  readonly val: Float32Array;

  constructor({ matrix: m }: { matrix: Sparse }) {
    this.rows = m.rows;
    this.cols = m.cols;
    const ptr = new Int32Array(m.rows + 1);
    for (let i = 0; i < m.r.length; i++) ptr[m.r[i] + 1]++;
    for (let i = 0; i < m.rows; i++) ptr[i + 1] += ptr[i];
    const fill = ptr.slice(0, m.rows);
    const col = new Int32Array(m.r.length);
    const val = new Float32Array(m.r.length);
    for (let i = 0; i < m.r.length; i++) {
      const at = fill[m.r[i]]++;
      col[at] = m.c[i];
      val[at] = m.v[i];
    }
    this.ptr = ptr;
    this.col = col;
    this.val = val;
  }

  // out[cols] += x[rows] · M, iterating only the nonzero entries of x (hot path: positional)
  addMul(x: Float32Array | Float64Array, out: Float32Array | Float64Array, nz?: Int32Array, nnz?: number): void {
    const { ptr, col, val } = this;
    if (nz && nnz !== undefined) {
      for (let k = 0; k < nnz; k++) {
        const i = nz[k];
        const xi = x[i];
        if (xi === 0) continue;
        for (let j = ptr[i]; j < ptr[i + 1]; j++) out[col[j]] += xi * val[j];
      }
      return;
    }
    for (let i = 0; i < this.rows; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      for (let j = ptr[i]; j < ptr[i + 1]; j++) out[col[j]] += xi * val[j];
    }
  }

  // single row (one-hot input) (hot path: positional)
  addRow(i: number, scale: number, out: Float32Array | Float64Array): void {
    for (let j = this.ptr[i]; j < this.ptr[i + 1]; j++) out[this.col[j]] += scale * this.val[j];
  }
}

// A dense scratch buffer that remembers which entries were touched, so that the result of
// a sparse product can be read and cleared in time proportional to its nonzeros.
export class Acc {
  readonly val: Float64Array;
  readonly idx: Int32Array;
  readonly mark: Uint8Array;
  n = 0;
  constructor({ size }: { size: number }) {
    this.val = new Float64Array(size);
    this.idx = new Int32Array(size);
    this.mark = new Uint8Array(size);
  }
  // hot path: positional
  addRow(m: Csr, i: number, scale: number): void {
    const { ptr, col, val } = m;
    for (let j = ptr[i]; j < ptr[i + 1]; j++) {
      const c = col[j];
      if (!this.mark[c]) {
        this.mark[c] = 1;
        this.idx[this.n++] = c;
      }
      this.val[c] += scale * val[j];
    }
  }
  // hot path: positional
  add(c: number, v: number): void {
    if (!this.mark[c]) {
      this.mark[c] = 1;
      this.idx[this.n++] = c;
    }
    this.val[c] += v;
  }
  clear(): void {
    for (let k = 0; k < this.n; k++) {
      const c = this.idx[k];
      this.val[c] = 0;
      this.mark[c] = 0;
    }
    this.n = 0;
  }
}

// ---------- runtime ----------

type HeadRt = {
  h: Head;
  q: Csr;
  k: Csr;
  v: Csr;
  o: Csr;
  keys: Float64Array[];
  skeys: { i: Int32Array; v: Float64Array }[];
  vals: { i: Int32Array; v: Float32Array }[];
  acc: Acc;
};
type LayerRt = {
  heads: HeadRt[];
  W1: Csr | null;
  W2: Csr | null;
  b1: Float32Array | null;
  n: number;
  acc: Acc | null;
  norms: Norm[];
};
const EMPTY = new Float64Array(0);
const csr = ({ matrix }: { matrix: Sparse }) => new Csr({ matrix });

export class Runner {
  readonly net: Net;
  private E: Csr;
  private Wu: Csr;
  private layers: LayerRt[];
  private xr: Acc; // the residual stream, with the list of the entries it touched
  private dacc: Acc; // the sum of the attention heads' outputs of a layer
  private logits: Float32Array;
  private oacc: Acc;
  t = 0;

  constructor({ net }: { net: Net }) {
    this.net = net;
    this.E = csr({ matrix: net.E });
    this.Wu = csr({ matrix: net.Wu });
    this.layers = net.layers.map((L) => {
      if (L.mlp)
        for (const b of L.mlp.b1) if (b > 0) throw new Error("positive MLP bias is not supported by the sparse runner");
      return {
        heads: L.heads.map((h) => ({
          h,
          q: csr({ matrix: h.Wq }),
          k: csr({ matrix: h.Wk }),
          v: csr({ matrix: h.Wv }),
          o: csr({ matrix: h.Wo }),
          keys: [],
          skeys: [],
          vals: [],
          acc: new Acc({ size: h.dv }),
        })),
        W1: L.mlp ? csr({ matrix: L.mlp.W1 }) : null,
        W2: L.mlp ? csr({ matrix: L.mlp.W2 }) : null,
        b1: L.mlp ? L.mlp.b1 : null,
        n: L.mlp ? L.mlp.n : 0,
        acc: L.mlp ? new Acc({ size: L.mlp.n }) : null,
        norms: L.norms ?? [],
      };
    });
    this.xr = new Acc({ size: net.d });
    this.dacc = new Acc({ size: net.d });
    this.logits = new Float32Array(net.vocab);
    this.oacc = new Acc({ size: net.d });
  }

  reset(): void {
    this.t = 0;
    for (const L of this.layers)
      for (const h of L.heads) {
        h.keys = [];
        h.skeys = [];
        h.vals = [];
      }
  }

  // Feed one token; returns the logits for the next token (a reused buffer).
  step({ token }: { token: number }): Float32Array {
    const { net, xr, dacc } = this;
    xr.clear();
    xr.addRow(this.E, token, 1);
    const t = this.t;
    net.pe.freqs.forEach((f, i) => {
      xr.add(net.pe.offset + 2 * i, Math.cos(f * t));
      xr.add(net.pe.offset + 2 * i + 1, Math.sin(f * t));
    });
    const x = xr.val;
    for (const L of this.layers) {
      // the entries of x touched so far (some may be zero, which is harmless)
      let nz = xr.idx;
      let nnz = xr.n;
      for (const H of L.heads) {
        const { h, acc } = H;
        const q = new Float64Array(h.dk);
        const k = new Float64Array(h.dk);
        H.q.addMul(x, q, nz, nnz);
        H.k.addMul(x, k, nz, nnz);
        for (let a = 0; a < nnz; a++) if (x[nz[a]] !== 0) acc.addRow(H.v, nz[a], x[nz[a]]);
        let vn = 0;
        for (let a = 0; a < acc.n; a++) if (acc.val[acc.idx[a]] !== 0) vn++;
        const vi = new Int32Array(vn);
        const vv = new Float32Array(vn);
        for (let a = 0, j = 0; a < acc.n; a++) {
          const c = acc.idx[a];
          if (acc.val[c] !== 0) {
            vi[j] = c;
            vv[j++] = acc.val[c];
          }
        }
        acc.clear();
        if (h.dk > 64) {
          // a wide key (e.g. the name match) is kept sparse
          let kn = 0;
          for (let j = 0; j < h.dk; j++) if (k[j] !== 0) kn++;
          const ki = new Int32Array(kn);
          const kv = new Float64Array(kn);
          for (let j = 0, m = 0; j < h.dk; j++)
            if (k[j] !== 0) {
              ki[m] = j;
              kv[m++] = k[j];
            }
          H.skeys.push({ i: ki, v: kv });
        }
        H.keys.push(h.dk > 64 ? EMPTY : k);
        H.vals.push({ i: vi, v: vv });
        // scores
        const T = H.keys.length;
        const sc = new Float64Array(T);
        let mx = -Infinity;
        // q is mostly zeros (e.g. the 650-dim name match): sum over its nonzeros only
        let qn = 0;
        for (let j = 0; j < h.dk; j++) if (q[j] !== 0) qn++;
        const qi = new Int32Array(qn);
        for (let j = 0, m = 0; j < h.dk; j++) if (q[j] !== 0) qi[m++] = j;
        for (let s = 0; s < T; s++) {
          let a = 0;
          if (h.dk > 64) {
            const { i: ki, v: kv } = H.skeys[s];
            for (let m = 0; m < ki.length; m++) a += q[ki[m]] * kv[m];
          } else {
            const ks = H.keys[s];
            for (let m = 0; m < qn; m++) a += q[qi[m]] * ks[qi[m]];
          }
          a = a * h.scale - h.slope * (t - s);
          sc[s] = a;
          if (a > mx) mx = a;
        }
        let z = 0;
        for (let s = 0; s < T; s++) {
          sc[s] = Math.exp(sc[s] - mx);
          z += sc[s];
        }
        // weighted sum of the values, straight through Wo
        const oa = this.oacc;
        for (let s = 0; s < T; s++) {
          const w = sc[s] / z;
          if (w < 1e-12) continue;
          const { i: ii, v: vvv } = H.vals[s];
          for (let j = 0; j < ii.length; j++) oa.addRow(H.o, ii[j], w * vvv[j]);
        }
        for (let a = 0; a < oa.n; a++) dacc.add(oa.idx[a], oa.val[oa.idx[a]]);
        oa.clear();
      }
      for (let a = 0; a < dacc.n; a++) xr.add(dacc.idx[a], dacc.val[dacc.idx[a]]);
      dacc.clear();
      for (const nm of L.norms) {
        let ss = 0;
        const n0 = xr.n;
        for (let a = 0; a < n0; a++) {
          const i = xr.idx[a];
          if (i >= nm.src && i < nm.src + nm.n) ss += x[i] * x[i];
        }
        if (ss > 0) {
          const inv = 1 / Math.sqrt(ss);
          for (let a = 0; a < n0; a++) {
            const i = xr.idx[a];
            if (i >= nm.src && i < nm.src + nm.n && x[i] !== 0) xr.add(nm.dst + (i - nm.src), x[i] * inv);
          }
        }
      }
      if (L.W1 && L.W2 && L.b1 && L.acc) {
        // every bias is ≤ 0, so only neurons that receive input can fire
        nz = xr.idx;
        nnz = xr.n;
        const acc = L.acc;
        for (let a = 0; a < nnz; a++) if (x[nz[a]] !== 0) acc.addRow(L.W1, nz[a], x[nz[a]]);
        for (let a = 0; a < acc.n; a++) {
          const i = acc.idx[a];
          const pre = acc.val[i] + L.b1[i];
          if (pre > 0) xr.addRow(L.W2, i, pre);
        }
        acc.clear();
      }
    }
    const logits = this.logits;
    logits.set(net.bu);
    for (let a = 0; a < xr.n; a++) {
      const i = xr.idx[a];
      if (x[i] !== 0) this.Wu.addRow(i, x[i], logits);
    }
    this.t++;
    return logits;
  }

  // residual of the last step (for inspection)
  get residual(): Float64Array {
    return this.xr.val;
  }
}

export function argmax({ values }: { values: Float32Array }): number {
  let b = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[b]) b = i;
  return b;
}

// ---------- serialization: <name>.json (config) + <name>.bin (arrays) ----------

type Ref = { off: number; len: number; type: "i32" | "f32" };
type SparseRef = { rows: number; cols: number; r: Ref; c: Ref; v: Ref };
type HeadRef = Omit<Head, "Wq" | "Wk" | "Wv" | "Wo"> & { Wq: SparseRef; Wk: SparseRef; Wv: SparseRef; Wo: SparseRef };
type Config = {
  format: string;
  d: number;
  vocab: number;
  pe: Net["pe"];
  E: SparseRef;
  layers: { heads: HeadRef[]; mlp: { n: number; W1: SparseRef; b1: Ref; W2: SparseRef } | null; norms: Norm[] }[];
  Wu: SparseRef;
  bu: Ref;
  meta: Record<string, unknown>;
};

export async function saveNet({ net, path }: { net: Net; path: string }): Promise<void> {
  const chunks: Uint8Array[] = [];
  let off = 0;
  const put = ({ array: a }: { array: Int32Array | Float32Array }): Ref => {
    const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    chunks.push(bytes.slice());
    const ref: Ref = { off, len: a.length, type: a instanceof Int32Array ? "i32" : "f32" };
    off += a.byteLength;
    return ref;
  };
  const sp = ({ matrix: m }: { matrix: Sparse }): SparseRef => ({
    rows: m.rows,
    cols: m.cols,
    r: put({ array: m.r }),
    c: put({ array: m.c }),
    v: put({ array: m.v }),
  });
  const cfg: Config = {
    format: "sparse-transformer-v1",
    d: net.d,
    vocab: net.vocab,
    pe: net.pe,
    E: sp({ matrix: net.E }),
    layers: net.layers.map((L) => ({
      heads: L.heads.map((h) => ({
        name: h.name,
        dk: h.dk,
        dv: h.dv,
        scale: h.scale,
        slope: h.slope,
        Wq: sp({ matrix: h.Wq }),
        Wk: sp({ matrix: h.Wk }),
        Wv: sp({ matrix: h.Wv }),
        Wo: sp({ matrix: h.Wo }),
      })),
      mlp: L.mlp
        ? { n: L.mlp.n, W1: sp({ matrix: L.mlp.W1 }), b1: put({ array: L.mlp.b1 }), W2: sp({ matrix: L.mlp.W2 }) }
        : null,
      norms: L.norms ?? [],
    })),
    Wu: sp({ matrix: net.Wu }),
    bu: put({ array: net.bu }),
    meta: net.meta,
  };
  await Bun.write(`${path}.bin`, new Blob(chunks as BlobPart[]));
  await Bun.write(`${path}.json`, JSON.stringify(cfg));
}

export async function loadNet({ path }: { path: string }): Promise<Net> {
  const cfg = (await Bun.file(`${path}.json`).json()) as Config;
  const buf = await Bun.file(`${path}.bin`).arrayBuffer();
  const get = ({ ref: r }: { ref: Ref }) =>
    r.type === "i32" ? new Int32Array(buf, r.off, r.len) : new Float32Array(buf, r.off, r.len);
  const sp = ({ matrix: m }: { matrix: SparseRef }): Sparse => ({
    rows: m.rows,
    cols: m.cols,
    r: get({ ref: m.r }) as Int32Array,
    c: get({ ref: m.c }) as Int32Array,
    v: get({ ref: m.v }) as Float32Array,
  });
  return {
    d: cfg.d,
    vocab: cfg.vocab,
    pe: cfg.pe,
    E: sp({ matrix: cfg.E }),
    layers: cfg.layers.map((L) => ({
      heads: L.heads.map((h) => ({
        ...h,
        Wq: sp({ matrix: h.Wq }),
        Wk: sp({ matrix: h.Wk }),
        Wv: sp({ matrix: h.Wv }),
        Wo: sp({ matrix: h.Wo }),
      })),
      mlp: L.mlp
        ? {
            n: L.mlp.n,
            W1: sp({ matrix: L.mlp.W1 }),
            b1: get({ ref: L.mlp.b1 }) as Float32Array,
            W2: sp({ matrix: L.mlp.W2 }),
          }
        : null,
      norms: L.norms ?? [],
    })),
    Wu: sp({ matrix: cfg.Wu }),
    bu: get({ ref: cfg.bu }) as Float32Array,
    meta: cfg.meta,
  };
}

// ---------- builder helper: a growable COO matrix ----------

export class Coo {
  readonly rows: number;
  readonly cols: number;
  r: number[] = [];
  c: number[] = [];
  v: number[] = [];

  constructor({ rows, cols }: { rows: number; cols: number }) {
    this.rows = rows;
    this.cols = cols;
  }

  add({ row, col, value }: { row: number; col: number; value: number }): void {
    if (value === 0) return;
    if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
      throw new Error(`Coo.add out of range ${row},${col} in ${this.rows}x${this.cols}`);
    }
    this.r.push(row);
    this.c.push(col);
    this.v.push(value);
  }

  done(): Sparse {
    return {
      rows: this.rows,
      cols: this.cols,
      r: Int32Array.from(this.r),
      c: Int32Array.from(this.c),
      v: Float32Array.from(this.v),
    };
  }
}
