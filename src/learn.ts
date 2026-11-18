// learn.ts — build the weights of a Transformer that reproduces pybot, without gradients.
//
// Every weight is set by counting and by closed-form construction (no gradient descent):
//   - vocabulary: the pieces of the harvested conversations, single characters, the names of
//     pybot's lexicon (COMMON_NAMES), the words of pybot's knowledge keys, typo forms
//   - reply classes: the distinct pybot replies of the training turns (with the user's name
//     replaced by a slot) and the reply of every knowledge entry
//   - selectors: one per distinct training (prompt, reply) — nearest neighbour by the cosine
//     of IDF-weighted bags of words — and one per knowledge entry rule. Each selector has a
//     hidden ID token <s:i> that the network emits right after <bot>; it carries its class.
//   - layer 1 attention: previous m tokens (m = 1..P), latest <user> position, user side,
//     latest ID token's class
//   - layer 1 MLP: recital neurons (class + shortest unambiguous context → next token or a
//     name slot) and name-introduction detectors ("i'm <Name>", "my name is <Name>", ...)
//   - layer 2 attention: bag of words of the current user turn; the latest introduced name;
//     then the bag is L2-normalized (UBAG); MLP: which words are present (PRES)
//   - layer 3 attention: the latest introduced name that differs from the latest one;
//     MLP: fill a name slot, phrases (AND of words), SAMEP
//   - layers 4-5 MLP: pybot's knowledge entries as rules (OR of synonyms, AND of key groups,
//     no avoid word, at least one non-frame word)
//   - unembedding: at <bot>, logit(<s:i>) = 100·cos(bag, selector_i) for a neighbour, or
//     100·TAU + 0.5·rank for a firing rule (rank as pybot's `_rank`), plus name terms
//
// usage: bun run src/learn.ts [--data a.jsonl,b.jsonl] [--train-groups ec,name,dolly1,dolly2]
//                         [--out weights/model] [--tau 0.8] [--prev 16] [--no-kb] [--no-typos]
//                         [--dev] [--sim] [--sim-only] [--errors] [--rule-errors]

import { flag, option } from "@/util/cli.ts";
import { type Conv, isDev, isTest, readConvs } from "@/lib/data.ts";
import { lexeme, stem } from "@/lib/lex.ts";
import { Coo, type Head, type Net, saveNet } from "@/lib/net.ts";
import { BOS, BOT, EOT, SPECIALS, Tokenizer, USER, charTokens, pieces } from "@/lib/tok.ts";

// every harvest file (script/download.sh); --train-groups decides which ones are trained on
const DEFAULT_DATA = ["ec_all", "names", "dolly1", "dolly2", "kb_probes", "stb", "pert", "namex", "namey"].map(
  (n) => `data/harvest/${n}.jsonl`,
);
const DATA = option({ name: "--data", fallback: "" })
  ? option({ name: "--data", fallback: "" }).split(",")
  : (await Promise.all(DEFAULT_DATA.map(async (f) => ((await Bun.file(f).exists()) ? f : "")))).filter(Boolean);

// ---------- options ----------
// prettier-ignore
const OUT          = option({ name: "--out",          fallback: "weights/model" }),
      TRAIN_GROUPS = option({ name: "--train-groups", fallback: "ec,name,dolly1,dolly2" }), // trained on; the rest is evaluation only
      DEV          = flag({ name: "--dev" }),                                            // hold out the dev split (for tuning with --sim)
      SEL          = option({ name: "--sel",          fallback: "knn" }),                // centroid | knn
      P            = Number(option({ name: "--prev",       fallback: "16" })),           // how many tokens back the recital can look
      STEM         = !flag({ name: "--no-stem" }),                                       // bag of word stems rather than word forms
      IDFPOW       = Number(option({ name: "--idfpow",     fallback: "1" })),            // bag weight of a word = idf^IDFPOW
      BIGRAM       = Number(option({ name: "--bigram",     fallback: "0" })),            // weight of word-pair features in the bag (0 = off)
      FIRSTW       = Number(option({ name: "--firstw",     fallback: "0.1" })),          // weight of the first-word features
      FIRSTN       = Number(option({ name: "--firstn",     fallback: "3" })),            // how many first words are features
      MIX          = Number(option({ name: "--mix",        fallback: "0" })),            // also add class centroids as selectors, scaled by MIX (0 = off)
      POOLMIX      = Number(option({ name: "--poolmix",    fallback: "0" })),            // also add pool centroids as selectors (0 = off)
      NAMES_FILE   = option({ name: "--names",        fallback: "data/kb/names.json" }), // pybot's lexicon names
      EXTRA_NAMES  = flag({ name: "--no-extra-names" }) ? "" : option({ name: "--extra-names", fallback: "data/raw/first-names.txt,data/raw/names_all.txt" }),
      KB_FILE      = flag({ name: "--no-kb" })    ? "" : option({ name: "--kb",    fallback: "data/kb/entries.jsonl" }), // pybot's knowledge entries
      POOLS_FILE   = flag({ name: "--no-pools" }) ? "" : option({ name: "--pools", fallback: "data/kb/pools.json" }),    // pybot's reply pools
      TAU          = Number(option({ name: "--tau",        fallback: "0.8" })),          // a firing rule beats a neighbour whose cosine is below TAU
      RANK_W       = Number(option({ name: "--rank-w",     fallback: "0.5" })),          // logit per rank point of a firing rule
      CALW         = Number(option({ name: "--calw",       fallback: "5" })),            // weight of the rule calibration
      NORULE_PEN   = Number(option({ name: "--norule",     fallback: "10" })),           // a neighbour with an entry's reply loses this while no rule fires
      POOL_BONUS   = Number(option({ name: "--pool-bonus", fallback: "12" })),           // added to a selector whose reply is a _choose pool option
      TWO_POS      = !flag({ name: "--no-two-pos" }),                                    // an entry must cover two positions of the line
      SKIP_GUARDED = flag({ name: "--skip-guarded" }),                                   // leave out entries pybot also gates by a guard function
      PREVW        = Number(option({ name: "--prevw",      fallback: "0" })),            // sim only: weight of the previous user turn's bag
      RECENT_ON    = flag({ name: "--recent" }),                                         // sim only: _choose's "not said in the last 5 turns"
      UW           = Number(option({ name: "--oracle-understood", fallback: "0" })); // sim only: pybot's own reading as features

// ---------- constants of the construction ----------
// prettier-ignore
const F        = 16,    // number of position frequencies
      BIG      = 1e6,   // "attend to marked positions" score (beats ALiBi over any context length)
      SLOPE    = 20,    // ALiBi slope of the "latest marked" heads (recency)
      G        = 1000,  // gate between ID logits and word logits
      K        = 10,    // logit given by a recital neuron
      S        = 100,   // similarity → logit
      LAMBDA   = 0.02,  // weight of the has-name feature in the selection
      EOT_BASE = 0.5,   // the reply ends if no neuron proposes a token
      SLOT_PEN = 50,    // a reply with a name slot is ruled out while no name is known
      MAXK     = 6,     // pools of 2..MAXK options get the seed circuit
      G2       = 1000,  // the logit taken from a pool option that choice() does not take
      IMAX     = 30,    // turn indices the seed circuit knows
      HMAX     = 120,   // seeds are clipped to 1000·HMAX + 999
      NMAX     = 400,   // tokens in a turn the seed circuit can count
      MAXLEN   = 8192,  // context length the position code is checked for
      LAMBDA_P = 1e4,   // BAG[l] ≥ 1e-4 whenever l is present (a turn of fewer than ~1000 tokens)
      MBIG     = 1e6; // gate size in the character-sum neurons
const FREQS = Array.from({ length: F }, (_, i) => 1.0 * (1 / 3000) ** (i / (F - 1)));

const t0 = performance.now();
const log = (...a: unknown[]) => console.log(`[${((performance.now() - t0) / 1000).toFixed(1)}s]`, ...a);

// ---------- data ----------
const convs: Conv[] = (await Promise.all(DATA.map((path) => readConvs({ path })))).flat();
// by id, so the weights do not depend on the order of the lines in the harvest files
convs.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
const inTrainGroups = ({ id }: { id: string }) => !TRAIN_GROUPS || TRAIN_GROUPS.split(",").includes(id.split("-")[0]);
const train = convs.filter(
  (c) => !isTest({ id: c.id }) && !(DEV && isDev({ id: c.id })) && inTrainGroups({ id: c.id }),
);
log(`conversations: ${convs.length}, train ${train.length}`);

// ---------- base vocabulary (the selector ID tokens are appended at the end) ----------
const cap = ({ s }: { s: string }) => s.charAt(0).toUpperCase() + s.slice(1);
const nameList: string[] = (await Bun.file(NAMES_FILE).json()) as string[];
const PYBOT_NN = nameList.length; // names [0, PYBOT_NN) are pybot's lexicon; the rest are extra
const allText: string[] = [];
for (const c of train) for (const t of c.turns) allText.push(t.user, t.bot);
const pieceSet = new Set<string>();
for (const s of allText) for (const p of pieces({ text: s })) pieceSet.add(p);
for (const n of nameList)
  for (const f of [n, cap({ s: n })]) {
    pieceSet.add(f);
    pieceSet.add(` ${f}`);
  }
// pybot's knowledge entries: the words of their keys and the text of their replies
type Entry = {
  id: string;
  domain: string;
  guard?: boolean;
  key: string[][];
  avoid: string[];
  ask: string[];
  bias: number;
  reply: string | null;
};
const entries: Entry[] = KB_FILE
  ? (await Bun.file(KB_FILE).text())
      .split("\n")
      .filter(Boolean)
      .map((l) => JSON.parse(l) as Entry)
      .filter((e) => e.reply)
  : [];
for (const e of entries) {
  for (const a of [...e.key.flat(), ...e.avoid, ...e.ask])
    for (const p of pieces({ text: a })) {
      const w = p.trim();
      for (const f of [w, cap({ s: w })]) {
        pieceSet.add(f);
        pieceSet.add(` ${f}`);
      }
    }
  allText.push(e.reply as string);
  for (const p of pieces({ text: e.reply as string })) pieceSet.add(p);
}
// pybot's reply pools (bot/compose/fallback_data.py): a fallback reply is one option of a pool,
// chosen by random.Random(turn·7919 + Σ ord(c) of the lowercased line).choice(pool)
const poolList: string[][] = POOLS_FILE
  ? Object.values((await Bun.file(POOLS_FILE).json()) as Record<string, string[]>).filter(
      (o) => o.length >= 2 && o.length <= MAXK,
    )
  : [];
const poolOfText = new Map<string, { p: number; j: number }>();
poolList.forEach((opts, p) => {
  opts.forEach((o, j) => {
    if (!poolOfText.has(o)) poolOfText.set(o, { p, j });
  });
});
for (const opts of poolList)
  for (const o of opts) {
    allText.push(o);
    for (const p of pieces({ text: o })) pieceSet.add(p);
  }
// the options of the pick() calls recorded in the training harvest (py/harvest.py)
type Pick = { options: string[]; seed_text: string; turns: number; out: string };
const picksOf = ({ turn: t }: { turn: unknown }) =>
  ((t as { picks?: Pick[] }).picks ?? []).filter((p) => p.options.length >= 2 && p.options.length <= MAXK);
if (POOLS_FILE)
  for (const c of train)
    for (const t of c.turns)
      for (const p of picksOf({ turn: t }))
        for (const o of p.options) {
          allText.push(o);
          for (const q of pieces({ text: o })) pieceSet.add(q);
        }
// extra first names (a public list): only the capitalized forms, and never a word the texts or
// the knowledge keys use in lowercase ("Happy", "Faith", "Will" stay words)
for (const NF of EXTRA_NAMES ? EXTRA_NAMES.split(",") : []) {
  if (!(await Bun.file(NF).exists())) continue;
  const low = new Map<string, number>();
  for (const t of allText)
    for (const p of pieces({ text: t })) {
      const w = p.trim();
      if (/^[a-z]/u.test(w)) low.set(w, (low.get(w) ?? 0) + 1);
    }
  const keyWords = new Set(
    entries.flatMap((e) => [...e.key.flat(), ...e.avoid, ...e.ask]).flatMap((a) => a.toLowerCase().split(/\W+/u)),
  );
  const have = new Set(nameList);
  for (const line of (await Bun.file(NF).text()).split("\n")) {
    const n = line
      .trim()
      .replace(/^\uFEFF/u, "")
      .toLowerCase();
    if (!/^[a-z]{2,}$/u.test(n) || have.has(n) || (low.get(n) ?? 0) >= 1 || keyWords.has(n)) continue;
    // a word the texts already use capitalized ("Paris", "London") is not taken as a name
    if (pieceSet.has(cap({ s: n })) || pieceSet.has(` ${cap({ s: n })}`)) continue;
    have.add(n);
    nameList.push(n);
    pieceSet.add(cap({ s: n }));
    pieceSet.add(` ${cap({ s: n })}`);
  }
}
const chars = charTokens({ texts: allText });
const charSet = new Set(chars);
// typo forms (off with --no-typos): two adjacent letters swapped (not the first letter) in a word of 5+
// letters that the user turns or the knowledge keys use; the form is a token whose lexeme is
// the word's, so the bag and the rules see the word. A form that is a real word is skipped.
const typoOf = new Map<string, string>();
if (!flag({ name: "--no-typos" })) {
  const cnt = new Map<string, number>();
  for (const c of train)
    for (const t of c.turns)
      for (const p of pieces({ text: t.user })) {
        const w = p.trim().toLowerCase();
        if (/^[a-z]{5,}$/u.test(w)) cnt.set(w, (cnt.get(w) ?? 0) + 1);
      }
  for (const e of entries)
    for (const a of e.key.flat())
      for (const p of pieces({ text: a })) {
        const w = p.trim().toLowerCase();
        if (/^[a-z]{5,}$/u.test(w)) cnt.set(w, (cnt.get(w) ?? 0) + 2);
      }
  const real = new Set([...pieceSet].map((p) => p.trim().toLowerCase()));
  for (const [w, n] of cnt) {
    if (n < 2) continue;
    for (let i = 1; i < w.length - 1; i++) {
      if (w[i] === w[i + 1]) continue;
      const v = w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2);
      if (real.has(v) || typoOf.has(v)) continue;
      typoOf.set(v, w);
    }
  }
  for (const v of typoOf.keys()) {
    pieceSet.add(v);
    pieceSet.add(` ${v}`);
  }
}
const words = [...pieceSet].filter((p) => !charSet.has(p)).sort();
const baseTokens = [...SPECIALS, "<id>", ...chars, ...words];
const NB = baseTokens.length;
const IDPH = SPECIALS.length; // the placeholder every ID token presents as its current token
const tok = new Tokenizer({ tokens: baseTokens });
const NN = nameList.length;
const nameOf = new Int32Array(NB).fill(-1); // name index of a token that spells a name
nameList.forEach((n, i) => {
  const forms = i < PYBOT_NN ? [n, cap({ s: n }), ` ${n}`, ` ${cap({ s: n })}`] : [cap({ s: n }), ` ${cap({ s: n })}`];
  for (const f of forms) nameOf[tok.index.get(f) as number] = i;
});
const nameTokSp = nameList.map((n) => tok.index.get(` ${cap({ s: n })}`) as number);
const nameTokNs = nameList.map((n) => tok.index.get(cap({ s: n })) as number);

// lexemes (what the bag of words counts). A capitalized name is the one lexeme "<name>",
// unless the name is also an ordinary word ("will", "may", "bill": lowercase in the texts).
const lowerCount = new Map<string, number>();
for (const s of allText)
  for (const p of pieces({ text: s })) {
    const w = p.trim();
    if (/^[a-z]/u.test(w)) lowerCount.set(w, (lowerCount.get(w) ?? 0) + 1);
  }
const isNameLex = ({ v }: { v: number }) =>
  nameOf[v] >= 0 && /^\s?[A-Z]/u.test(baseTokens[v]) && (lowerCount.get(nameList[nameOf[v]]) ?? 0) < 3;
const lexIndex = new Map<string, number>();
const lexOfBase = new Int32Array(NB).fill(-1);
for (let v = 0; v < NB; v++) {
  const w = lexeme({ token: baseTokens[v] });
  const w0 = w && typoOf.has(w) ? (typoOf.get(w) as string) : w;
  const l = isNameLex({ v }) ? "<name>" : w0 && STEM ? stem({ word: w0 }) : w0;
  if (!l) continue;
  if (!lexIndex.has(l)) lexIndex.set(l, lexIndex.size);
  lexOfBase[v] = lexIndex.get(l) as number;
}
const L = lexIndex.size;
const lexStr = ({ v }: { v: number }) => (v >= 0 && v < NB ? (lexeme({ token: baseTokens[v] }) ?? "") : "");

// ---------- name memory: the rule the network implements, run in JS for the labels ----------
// A user token that spells a name is an introduction when the tokens before it are one of:
const INTRO = [["i'm"], ["im"], ["i", "am"], ["name", "is"], ["name's"], ["call", "me"]];
function introducedIn({ text }: { text: string }): number[] {
  const ids = [USER, ...tok.encode({ text })];
  const found: number[] = [];
  for (let i = 1; i < ids.length; i++) {
    if (nameOf[ids[i]] < 0) continue;
    for (const p of INTRO) {
      if (i - p.length < 1) continue;
      if (p.every((w, k) => lexStr({ v: ids[i - p.length + k] }) === w)) {
        found.push(nameOf[ids[i]]);
        break;
      }
    }
  }
  return found;
}
type Mem = { latest: number; pname: number };
function memAfter({ history }: { history: number[] }): Mem {
  if (!history.length) return { latest: -1, pname: -1 };
  const latest = history[history.length - 1];
  let pname = latest;
  for (let i = history.length - 1; i >= 0; i--)
    if (history[i] !== latest) {
      pname = history[i];
      break;
    }
  return { latest, pname };
}
// slot codes in a templated reply
const SLOT_NAME_SP = -1;
const SLOT_NAME_NS = -2;
const SLOT_PNAME_SP = -3;
const SLOT_PNAME_NS = -4;
function template({ reply, m }: { reply: string; m: Mem }): number[] {
  return tok.encode({ text: reply }).map((v) => {
    const n = nameOf[v];
    if (n < 0 || m.latest < 0) return v;
    const sp = baseTokens[v].startsWith(" ");
    const capd = /^\s?[A-Z]/u.test(baseTokens[v]);
    if (!capd) return v;
    if (n === m.latest) return sp ? SLOT_NAME_SP : SLOT_NAME_NS;
    if (n === m.pname) return sp ? SLOT_PNAME_SP : SLOT_PNAME_NS;
    return v;
  });
}

// every turn with its memory state
type Labeled = {
  conv: string;
  ti: number;
  user: string;
  prev: string;
  bot: string;
  module: string | null;
  mem: Mem;
  key: string;
  toks: number[];
};
function label({ set }: { set: Conv[] }): Labeled[] {
  const out: Labeled[] = [];
  for (const c of set) {
    const hist: number[] = [];
    let prev = "";
    c.turns.forEach((t, ti) => {
      hist.push(...introducedIn({ text: t.user }));
      const mem = memAfter({ history: hist });
      const toks = template({ reply: t.bot, m: mem });
      out.push({ conv: c.id, ti, user: t.user, prev, bot: t.bot, module: t.module, mem, key: toks.join(","), toks });
      prev = t.user;
    });
  }
  return out;
}
const trainTurns = label({ set: train });

// pick() pools (bot/compose/__init__.py `pick`): the recorded calls whose choice is the whole
// reply. Without a seed text the seed is the turn index (type B); with the lowercased line as
// seed text it is turn·7919 + Σ ord (type A, the same seed as _choose). Options are templated
// with the memory of that turn, so "Nice to meet you, Sam" is a pool option with a name slot.
type PickPool = { toks: number[][]; type: "A" | "B" };
const pickPools: PickPool[] = [];
{
  const seen = new Set<string>();
  for (const c of train) {
    const hist: number[] = [];
    for (const t of c.turns) {
      hist.push(...introducedIn({ text: t.user }));
      const mem = memAfter({ history: hist });
      for (const p of picksOf({ turn: t })) {
        if (p.out.split(/\s+/u).filter(Boolean).join(" ") !== t.bot) continue;
        const type = !p.seed_text ? "B" : p.seed_text === t.user.toLowerCase() ? "A" : null;
        if (!type) continue;
        const toks = p.options.map((o) => template({ reply: o.split(/\s+/u).filter(Boolean).join(" "), m: mem }));
        const key = `${type}:${toks.map((x) => x.join(",")).join("|")}`;
        if (seen.has(key)) continue;
        seen.add(key);
        pickPools.push({ toks, type });
      }
    }
  }
}

// reply classes = distinct templated replies (most frequent first)
const classCount = new Map<string, { n: number; toks: number[] }>();
for (const t of trainTurns) {
  const e = classCount.get(t.key);
  if (e) e.n++;
  else classCount.set(t.key, { n: 1, toks: t.toks });
}
// every entry reply is a class too (it may never have been seen in a training turn)
for (const e of entries) {
  const toks = tok.encode({ text: e.reply as string });
  const k = toks.join(",");
  if (!classCount.has(k)) classCount.set(k, { n: 0, toks });
}
// every pool option is a class too (pybot may choose an option no training turn showed)
for (const opts of poolList)
  for (const o of opts) {
    const toks = tok.encode({ text: o });
    const k = toks.join(",");
    if (!classCount.has(k)) classCount.set(k, { n: 0, toks });
  }
for (const pp of pickPools)
  for (const toks of pp.toks) {
    const k = toks.join(",");
    if (!classCount.has(k)) classCount.set(k, { n: 0, toks });
  }
const classList = [...classCount.entries()].sort((a, b) => b[1].n - a[1].n || (a[0] < b[0] ? -1 : 1));
const classOf = new Map(classList.map(([k], i) => [k, i]));
const classToks = classList.map(([, e]) => e.toks);
const C = classList.length;
const renderClass = ({ toks }: { toks: number[] }) =>
  toks
    .map((v) =>
      v === SLOT_NAME_SP
        ? " {name}"
        : v === SLOT_NAME_NS
          ? "{name}"
          : v === SLOT_PNAME_SP
            ? " {pname}"
            : v === SLOT_PNAME_NS
              ? "{pname}"
              : baseTokens[v],
    )
    .join("");
log(
  `reply classes: ${C} (train turns ${trainTurns.length}; with a name slot ${classToks.filter((t) => t.some((v) => v < 0)).length})`,
);

// ---------- IDF and bags ----------
const df = new Float64Array(L);
const encCache = new Map<string, number[]>();
const encUser = ({ s }: { s: string }) => {
  let e = encCache.get(s);
  if (!e) {
    e = tok.encode({ text: s });
    encCache.set(s, e);
  }
  return e;
};
for (const t of trainTurns) {
  const seen = new Set<number>();
  for (const v of encUser({ s: t.user })) if (lexOfBase[v] >= 0) seen.add(lexOfBase[v]);
  for (const l of seen) df[l]++;
}
const nDocs = trainTurns.length;
const idf = Float64Array.from(df, (d) => Math.log((nDocs + 1) / (d + 1)) + 1);
const baseIdf = new Float64Array(NB).fill(1);
for (let v = 0; v < NB; v++) if (lexOfBase[v] >= 0) baseIdf[v] = idf[lexOfBase[v]] ** IDFPOW;

// word pairs (lexeme of the previous token, lexeme of this token) seen in ≥ 2 training turns
const bigramIndex = new Map<string, number>();
if (BIGRAM > 0) {
  const cnt = new Map<string, number>();
  for (const t of trainTurns) {
    const e = encUser({ s: t.user });
    const seen = new Set<string>();
    for (let i = 1; i < e.length; i++)
      if (lexOfBase[e[i - 1]] >= 0 && lexOfBase[e[i]] >= 0) seen.add(`${lexOfBase[e[i - 1]]}|${lexOfBase[e[i]]}`);
    for (const k of seen) cnt.set(k, (cnt.get(k) ?? 0) + 1);
  }
  for (const [k, n] of cnt) if (n >= 2) bigramIndex.set(k, bigramIndex.size);
}
const NBI = bigramIndex.size;

type Bag = Map<number, number>;
// bag of a user turn, as the network computes it: Σ idf·e_lex / Σ idf (the <user> marker
// takes part in the softmax with weight 1 and no lexeme)
function bagOf({ text }: { text: string }): Bag {
  const m: Bag = new Map();
  let z = 1;
  const e = encUser({ s: text });
  e.forEach((v, i) => {
    z += baseIdf[v];
    if (lexOfBase[v] >= 0) m.set(lexOfBase[v], (m.get(lexOfBase[v]) ?? 0) + baseIdf[v]);
    if (NBI && i > 0) {
      const b = bigramIndex.get(`${lexOfBase[e[i - 1]]}|${lexOfBase[v]}`);
      if (b !== undefined) m.set(2 * L + b, (m.get(2 * L + b) ?? 0) + BIGRAM * baseIdf[v]);
    }
  });
  for (const [k, w] of m) m.set(k, w / z);
  return m;
}
function unit({ b }: { b: Bag }): Bag {
  let n = 0;
  for (const w of b.values()) n += w * w;
  n = Math.sqrt(n) || 1;
  const o: Bag = new Map();
  for (const [l, w] of b) o.set(l, w / n);
  return o;
}

// the selection feature: the bag of the turn, plus (sim experiment) the previous turn's bag
// the lexemes of the first FIRSTN tokens of the turn as features of their own (the form of the
// question: "what do you think", "can you recommend", "is it"), weight FIRSTW
const FOFF = () => 2 * L + NBI;
// upper-bound experiment (--oracle-understood w): pybot's own reading of the line as features
const understood: Record<string, string[]> = UW > 0 ? await Bun.file("data/kb/understood.json").json() : {};
const ufeatIdx = new Map<string, number>();
const UOFF = () => 2 * L + NBI + FIRSTN * L;
function featOf({ user, prev }: { user: string; prev: string }): Bag {
  const b = bagOf({ text: user });
  if (UW > 0)
    for (const f of understood[user] ?? []) {
      if (!ufeatIdx.has(f)) ufeatIdx.set(f, ufeatIdx.size);
      b.set(UOFF() + (ufeatIdx.get(f) as number), UW);
    }
  if (FIRSTW > 0) {
    const e = encUser({ s: user });
    for (let i = 0; i < Math.min(FIRSTN, e.length); i++)
      if (lexOfBase[e[i]] >= 0) b.set(FOFF() + i * L + lexOfBase[e[i]], FIRSTW);
  }
  if (PREVW > 0 && prev) for (const [l, w] of bagOf({ text: prev })) b.set(L + l, PREVW * w);
  return b;
}

// ---------- selectors ----------
type Selector = { bag: Bag; cls: number; n: number; hasName: boolean; rule?: number };
const selectors: Selector[] = [];
if (SEL === "centroid") {
  const acc = new Map<string, { bag: Bag; cls: number; n: number; hasName: boolean }>();
  for (const t of trainTurns) {
    const k = classOf.get(t.key) as number;
    const h = t.mem.latest >= 0;
    const id = `${k}|${h}`;
    const e = acc.get(id) ?? { bag: new Map(), cls: k, n: 0, hasName: h };
    for (const [l, w] of unit({ b: bagOf({ text: t.user }) })) e.bag.set(l, (e.bag.get(l) ?? 0) + w);
    e.n++;
    acc.set(id, e);
  }
  for (const e of acc.values()) selectors.push({ ...e, bag: unit({ b: e.bag }) });
} else {
  const seen = new Map<string, Selector>();
  for (const t of trainTurns) {
    const k = classOf.get(t.key) as number;
    const h = t.mem.latest >= 0;
    const id = `${k}|${h}|${t.user}`;
    const s = seen.get(id);
    if (s) s.n++;
    else {
      const sel = { bag: unit({ b: featOf({ user: t.user, prev: t.prev }) }), cls: k, n: 1, hasName: h };
      seen.set(id, sel);
      selectors.push(sel);
    }
  }
}
if (MIX > 0 && SEL !== "centroid") {
  const acc = new Map<string, Selector>();
  for (const t of trainTurns) {
    const k = classOf.get(t.key) as number;
    const h = t.mem.latest >= 0;
    const id = `${k}|${h}`;
    const e = acc.get(id) ?? { bag: new Map(), cls: k, n: 0, hasName: h };
    for (const [l, w] of unit({ b: bagOf({ text: t.user }) })) e.bag.set(l, (e.bag.get(l) ?? 0) + w);
    e.n++;
    acc.set(id, e);
  }
  for (const e of acc.values())
    if (e.n > 1) {
      const u = unit({ b: e.bag });
      for (const [l, w] of u) u.set(l, w * MIX);
      selectors.push({ ...e, bag: u });
    }
}
// ---------- pybot's knowledge rules ----------
// An entry answers when every key group has one of its alternatives in the turn, no avoid
// alternative is there, and (if it has ask words) one ask word is there. An alternative of
// several words counts as present when all its words are in the turn (order is ignored).
// Words are compared as lexemes (lowercase stems), like the bag.
const altList: number[][] = [];
const altIdx = new Map<string, number>();
const singleLetter = new Set([...lexIndex].filter(([w]) => /^[a-z]$/u.test(w)).map(([, l]) => l));
function altOf({ a }: { a: string }): number {
  const ls = [
    ...new Set(
      tok
        .encode({ text: a })
        .map((v) => lexOfBase[v])
        .filter((l) => l >= 0),
    ),
  ].sort((x, y) => x - y);
  if (!ls.length) return -1;
  // an alternative of single letters only ("i e", "a p r", "e.g.") would match the letters of
  // any spelled-out unknown word ("FrieslandCampina"): such spellings are left out
  if (ls.length > 1 && ls.every((l) => singleLetter.has(l))) return -1;
  const k = ls.join(",");
  let i = altIdx.get(k);
  if (i === undefined) {
    i = altList.length;
    altList.push(ls);
    altIdx.set(k, i);
  }
  return i;
}
const groupList: number[][] = [];
const groupIdx = new Map<string, number>();
function groupOf({ alts }: { alts: string[] }): number {
  const ids = [...new Set(alts.map((a) => altOf({ a })).filter((i) => i >= 0))].sort((x, y) => x - y);
  if (!ids.length) return -1;
  const k = ids.join(",");
  let i = groupIdx.get(k);
  if (i === undefined) {
    i = groupList.length;
    groupList.push(ids);
    groupIdx.set(k, i);
  }
  return i;
}
// pybot ranks the entries that match (bot/knowledge/lookup.py `_rank`) by: for each key part,
// 3 if a phrase matched it and 2 if a word did; +1 for each matched key word that only this
// entry uses; + the entry's bias. We use the same sum over lexemes.
const pgroupList: number[][] = []; // the phrase alternatives of a group
const pgroupIdx = new Map<string, number>();
// how many entries of the same domain use a key lexeme (pybot counts rarity per domain)
const entriesOfLex = new Map<string, number>();
for (const e of entries) {
  const ls = new Set<number>();
  for (const a of e.key.flat()) for (const v of tok.encode({ text: a })) if (lexOfBase[v] >= 0) ls.add(lexOfBase[v]);
  for (const l of ls) entriesOfLex.set(`${e.domain}|${l}`, (entriesOfLex.get(`${e.domain}|${l}`) ?? 0) + 1);
}
// pybot gives rank 1 to an entry matched only by asking words ("what", "should", "how", "it"...)
// (`says_what` in `_rank`); here such a match does not fire: the rule needs one matched
// alternative with a word outside that frame.
const frameWords: string[] = KB_FILE ? ((await Bun.file("data/kb/frame_words.json").json()) as string[]) : [];
const frameLex = new Set(
  frameWords.map((w) => lexIndex.get(STEM ? stem({ word: w }) : w)).filter((l) => l !== undefined) as number[],
);
type Rule = {
  id: string;
  cls: number;
  groups: number[];
  pgroups: number[];
  rare: number[];
  avoid: number[];
  bias: number;
  content: number[];
};
const rules: Rule[] = [];
for (const e of entries) {
  if (SKIP_GUARDED && e.guard) continue;
  const gs = e.key.map((alts) => groupOf({ alts }));
  if (e.ask.length) gs.push(groupOf({ alts: e.ask }));
  if (!gs.length || gs.some((g) => g < 0)) continue;
  let groups = [...new Set(gs)];
  // pybot scores 0 an entry that covers fewer than two positions of the line (`_two_positions`):
  // with one key part, only a phrase alternative can match
  if (groups.length === 1 && TWO_POS) {
    const ph = groupList[groups[0]].filter((a) => altList[a].length > 1);
    if (!ph.length) continue;
    const k = ph.join(",");
    let gi = groupIdx.get(k);
    if (gi === undefined) {
      gi = groupList.length;
      groupList.push(ph);
      groupIdx.set(k, gi);
    }
    groups = [gi];
  }
  const pgroups: number[] = [];
  for (const g of groups) {
    const ph = groupList[g].filter((a) => altList[a].length > 1);
    if (!ph.length) continue;
    const k = ph.join(",");
    let i = pgroupIdx.get(k);
    if (i === undefined) {
      i = pgroupList.length;
      pgroupList.push(ph);
      pgroupIdx.set(k, i);
    }
    pgroups.push(i);
  }
  const rare = [...new Set(groups.flatMap((g) => groupList[g].flatMap((a) => altList[a])))].filter(
    (l) => entriesOfLex.get(`${e.domain}|${l}`) === 1,
  );
  const avoid = [...new Set(e.avoid.map((a) => altOf({ a })).filter((i) => i >= 0))];
  const cls = classOf.get(tok.encode({ text: e.reply as string }).join(",")) as number;
  const alts = [...new Set(groups.flatMap((g) => groupList[g]))];
  const content = alts.filter((a) => altList[a].some((l) => !frameLex.has(l)));
  rules.push({
    id: e.id,
    cls,
    groups,
    pgroups,
    rare,
    avoid,
    bias: e.bias,
    content: content.length === alts.length ? [] : content,
  });
}
rules.forEach((r, i) => selectors.push({ bag: new Map(), cls: r.cls, n: 0, hasName: false, rule: i }));
const RULE_ON = S * TAU; // the logit of a firing rule, before its rank
// the largest rank any rule can reach; RULE_OFF keeps a rule that does not fire below 0
// (below every neighbour) whatever is present
const MAX_RANK = Math.max(0, ...rules.map((r) => 3 * r.groups.length + r.rare.length + Math.max(0, r.bias)));
const RULE_OFF = -Math.max(100, RANK_W * MAX_RANK + 50);
// which rules fire: walk only from the lexemes that are present
const altsOfLex = new Map<number, number[]>();
altList.forEach((ls, a) => {
  for (const l of ls) {
    const x = altsOfLex.get(l) ?? [];
    x.push(a);
    altsOfLex.set(l, x);
  }
});
const groupsOfAlt: number[][] = altList.map(() => []);
groupList.forEach((as, g) => {
  for (const a of as) groupsOfAlt[a].push(g);
});
const rulesOfGroup: number[][] = groupList.map(() => []);
rules.forEach((r, i) => {
  for (const g of r.groups) rulesOfGroup[g].push(i);
});
let lastPresent = new Set<number>();
let lastAltP = new Set<number>();
function firing({ text }: { text: string }): Uint8Array {
  const out = new Uint8Array(rules.length);
  const cnt = new Map<number, number>();
  for (const l of bagOf({ text }).keys())
    if (l < L) for (const a of altsOfLex.get(l) ?? []) cnt.set(a, (cnt.get(a) ?? 0) + 1);
  const altP = new Set<number>();
  for (const [a, n] of cnt) if (n === altList[a].length) altP.add(a);
  const grpP = new Set<number>();
  for (const a of altP) for (const g of groupsOfAlt[a]) grpP.add(g);
  const rc = new Map<number, number>();
  for (const g of grpP) for (const r of rulesOfGroup[g]) rc.set(r, (rc.get(r) ?? 0) + 1);
  for (const [r, n] of rc) {
    const ru = rules[r];
    if (
      n === ru.groups.length &&
      !ru.avoid.some((a) => altP.has(a)) &&
      (!ru.content.length || ru.content.some((a) => altP.has(a)))
    )
      out[r] = 1;
  }
  lastPresent = new Set([...bagOf({ text }).keys()].filter((l) => l < L));
  lastAltP = altP;
  return out;
}
// the rank of a rule from what is present (meaningful when it fires)
function rankOf({ r }: { r: Rule }): number {
  let k = r.bias;
  for (const g of r.groups) if (groupList[g].some((a) => lastAltP.has(a))) k += 2;
  for (const pg of r.pgroups) if (pgroupList[pg].some((a) => lastAltP.has(a))) k += 1;
  for (const l of r.rare) if (lastPresent.has(l)) k += 1;
  return k;
}

// a selector whose reply is an option of a pool becomes one selector per option of that pool;
// the seed circuit gives −G2 to every option but the one pybot's choice() takes
type PoolSel = { p: number; j: number };
// all pools as lists of classes: _choose pools (type A) and pick() pools (type A or B)
type UPool = { cls: number[]; type: "A" | "B" };
const U: UPool[] = [
  ...poolList.map((opts) => ({
    cls: opts.map((o) => classOf.get(tok.encode({ text: o }).join(",")) as number),
    type: "A" as const,
  })),
  ...pickPools.map((pp) => ({ cls: pp.toks.map((t) => classOf.get(t.join(",")) as number), type: pp.type })),
];
const poolOfClass = new Map<number, PoolSel>();
U.forEach((u, p) => {
  u.cls.forEach((c, j) => {
    if (!poolOfClass.has(c)) poolOfClass.set(c, { p, j });
  });
});
void poolOfText;
// pool centroids (--poolmix w): one selector per pool, its bag the normalized mean of the bags of
// the training prompts whose reply is an option of that pool, scaled by w
if (POOLMIX > 0) {
  const acc = new Map<number, Bag>();
  const cnt = new Map<number, number>();
  for (const t of trainTurns) {
    const k = classOf.get(t.key) as number;
    const ps = poolOfClass.get(k);
    if (!ps) continue;
    const b = acc.get(ps.p) ?? new Map();
    for (const [l, w] of unit({ b: bagOf({ text: t.user }) })) b.set(l, (b.get(l) ?? 0) + w);
    acc.set(ps.p, b);
    cnt.set(ps.p, (cnt.get(ps.p) ?? 0) + 1);
  }
  for (const [p, b] of acc) {
    if ((cnt.get(p) ?? 0) < 3) continue;
    const u = unit({ b });
    for (const [l, w] of u) u.set(l, w * POOLMIX);
    selectors.push({ bag: u, cls: U[p].cls[0], n: cnt.get(p) ?? 0, hasName: false });
  }
}
{
  const expanded: (Selector & { pv?: PoolSel })[] = [];
  for (const sel of selectors) {
    const ps = sel.rule === undefined ? poolOfClass.get(sel.cls) : undefined;
    if (!ps) {
      expanded.push(sel);
      continue;
    }
    U[ps.p].cls.forEach((c, j) => {
      expanded.push({ ...sel, cls: c, pv: { p: ps.p, j } });
    });
  }
  selectors.length = 0;
  selectors.push(...expanded);
}
const poolOptClass = U.map((u) => u.cls);
// the character-code sum of a token (lowercased), as the seed circuit adds it up
const csOf = ({ v }: { v: number }) =>
  v < SPECIALS.length || v === IDPH
    ? 0
    : [...baseTokens[v].toLowerCase()].reduce((a, ch) => a + (ch.codePointAt(0) as number), 0);
const seedTable = new Uint8Array(await Bun.file("data/kb/seed_table.bin").arrayBuffer());
function seedOf({ user, ti }: { user: string; ti: number }): number {
  const e = encUser({ s: user });
  const S = e.length <= NMAX ? e.reduce((a, v) => a + csOf({ v }), 0) : 0; // the circuit counts up to NMAX tokens
  const seed = S + (ti <= IMAX ? 7919 * ti : 0);
  return Math.min(seed, 1000 * HMAX + 999);
}
const choiceOf = ({ seed, k }: { seed: number; k: number }) => seedTable[seed * 5 + k - 2];

// calibration of the rules on the training turns (counting): when a rule is the best-ranked
// firing rule, how often did pybot answer with its entry? A rule pybot usually passes over
// (a guard, a context, a reading this approximation does not have) gets CALW·log((ok+½)/(n+1)).
// a neighbour whose reply is a knowledge entry's reply loses NORULE_PEN when no rule fires:
// pybot answers from its knowledge base only when an entry matches
const entryClass = new Set(rules.map((r) => r.cls));
const calib = new Float64Array(rules.length);
if (CALW > 0) {
  const ok = new Float64Array(rules.length);
  const n = new Float64Array(rules.length);
  for (const t of trainTurns) {
    const f = firing({ text: t.user });
    let best = -1;
    let bestK = -Infinity;
    for (let r = 0; r < rules.length; r++)
      if (f[r]) {
        const k = rankOf({ r: rules[r] });
        if (k > bestK) {
          bestK = k;
          best = r;
        }
      }
    if (best < 0) continue;
    n[best]++;
    if (classOf.get(t.key) === rules[best].cls) ok[best]++;
  }
  for (let r = 0; r < rules.length; r++) if (n[r] > 0) calib[r] = CALW * Math.log((ok[r] + 0.5) / (n[r] + 1));
  log(`rule calibration: ${n.filter((x) => x > 0).length} rules seen on training turns`);
}

const NS = selectors.length;
log(
  `lexemes: ${L}, selectors (${SEL}): ${NS} (of which knowledge rules ${rules.length}; alternatives ${altList.length}, groups ${groupList.length})`,
);

// ---------- simulation of the selection (same arithmetic as the network's ID logits) ----------
const prior = Float64Array.from(selectors, (s) => 1e-3 * Math.log(1 + s.n));
const slotted = classToks.map((t) => t.some((v) => v < 0));
// a reply with the previous-name slot ("Tom, not Maria") needs two different names
const pslotted = classToks.map((t) => t.some((v) => v === SLOT_PNAME_SP || v === SLOT_PNAME_NS));
const inv: [number, number][][] = Array.from({ length: 2 * L + NBI + FIRSTN * L + 2000 }, () => []);
selectors.forEach((s, i) => {
  for (const [l, w] of s.bag) inv[l].push([i, w]);
});
// the option of pool p that pybot's _choose takes: choice() over the options not said in the
// last 5 turns (all options when every one was)
const RECENT_HITS = { n: 0 };
const POOL_ERR = new Map<string, number>();
function chosenOption({
  p,
  seed,
  recent,
  ti = 0,
}: {
  p: number;
  seed: number;
  recent: Set<number>;
  ti?: number;
}): number {
  const opts = U[p].cls;
  if (U[p].type === "B") return ti <= IMAX ? choiceOf({ seed: ti, k: opts.length }) : -1;
  let fresh = opts.map((_, j) => j).filter((j) => !(RECENT_ON && recent.has(poolOptClass[p][j])));
  if (!fresh.length) fresh = opts.map((_, j) => j);
  if (fresh.length < opts.length) RECENT_HITS.n++;
  return fresh[fresh.length === 1 ? 0 : choiceOf({ seed, k: fresh.length })];
}
function select({
  text,
  hasName,
  prev = "",
  samePname = false,
  seed = 0,
  recent = new Set(),
  ti = 0,
}: {
  text: string;
  hasName: boolean;
  prev?: string;
  samePname?: boolean;
  seed?: number;
  recent?: Set<number>;
  ti?: number;
}): number {
  const sc = new Float64Array(NS);
  for (const [l, w] of unit({ b: featOf({ user: text, prev }) })) for (const [i, pw] of inv[l]) sc[i] += S * w * pw;
  const fire = rules.length ? firing({ text }) : null;
  const anyFire = fire ? fire.some((x) => x === 1) : false;
  const hq = hasName ? 1 : -1;
  let best = -1;
  let bestV = -Infinity;
  for (let i = 0; i < NS; i++) {
    let v = sc[i] + prior[i] + S * LAMBDA * hq * (selectors[i].hasName ? 1 : -1);
    const ri = selectors[i].rule;
    // a rule that does not fire scores ≤ RULE_OFF + RANK_W·(max rank) < 0 and never wins, so
    // its rank is not worked out here
    if (ri !== undefined && fire) v += fire[ri] ? RULE_ON + RANK_W * rankOf({ r: rules[ri] }) + calib[ri] : RULE_OFF;
    if (!hasName && slotted[selectors[i].cls]) v -= SLOT_PEN;
    if (hasName && samePname && pslotted[selectors[i].cls]) v -= SLOT_PEN;
    if (NORULE_PEN && fire && ri === undefined && !anyFire && entryClass.has(selectors[i].cls)) v -= NORULE_PEN;
    const pv = (selectors[i] as Selector & { pv?: PoolSel }).pv;
    if (pv && pv.p < poolList.length) v += POOL_BONUS; // _choose (fallback) pools only
    if (pv && pv.j !== chosenOption({ p: pv.p, seed, recent, ti })) v -= G2;
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return selectors[best].cls;
}
if (flag({ name: "--sim" })) {
  const sets: [string, Conv[]][] = [["train", train]];
  if (DEV) sets.push(["dev", convs.filter((c) => isDev({ id: c.id }) && !isTest({ id: c.id }))]);
  else sets.push(["test", convs.filter((c) => isTest({ id: c.id }))]);
  for (const [name, set] of sets) {
    const groups = new Map<string, { n: number; reach: number; ok: number }>();
    // the classes this network said in earlier turns of the conversation (its own history,
    // as in eval.ts: the recency filter looks at what the network said, not at pybot)
    const said = new Map<string, number[]>();
    for (const t of label({ set })) {
      const hist = said.get(t.conv) ?? [];
      const recent = new Set(hist.slice(Math.max(0, t.ti - 5), t.ti));
      const got = select({
        text: t.user,
        hasName: t.mem.latest >= 0,
        prev: t.prev,
        samePname: t.mem.pname === t.mem.latest,
        seed: seedOf({ user: t.user, ti: t.ti }),
        recent,
        ti: t.ti,
      });
      hist.push(got);
      said.set(t.conv, hist);
      const kk = classOf.get(t.key);
      const gnames = ["all", t.conv.split("-")[0]];
      if (kk !== undefined && poolOfClass.has(kk)) {
        gnames.push("pool-reply");
        if (poolOfClass.has(got) && poolOfClass.get(got)?.p === poolOfClass.get(kk)?.p)
          gnames.push("pool-reply-same-pool");
      }
      if (flag({ name: "--pool-errors" }) && kk !== undefined && poolOfClass.has(kk) && got !== kk) {
        const f = firing({ text: t.user });
        const any = f.some((x) => x === 1);
        const kind = poolOfClass.has(got)
          ? poolOfClass.get(got)?.p === poolOfClass.get(kk)?.p
            ? "same pool, other option"
            : "other pool"
          : entryClass.has(got)
            ? "entry reply"
            : "other reply";
        POOL_ERR.set(`${kind} | rule fires: ${any}`, (POOL_ERR.get(`${kind} | rule fires: ${any}`) ?? 0) + 1);
      }
      for (const gname of gnames) {
        const g = groups.get(gname) ?? { n: 0, reach: 0, ok: 0 };
        g.n++;
        const k = classOf.get(t.key);
        if (k !== undefined) {
          g.reach++;
          if (got === k) g.ok++;
        }
        groups.set(gname, g);
      }
    }
    if (flag({ name: "--rule-errors" })) {
      const rulesOfClass = new Map<number, number[]>();
      rules.forEach((r, i) => rulesOfClass.set(r.cls, [...(rulesOfClass.get(r.cls) ?? []), i]));
      const st = new Map<string, number>();
      const bump = ({ key: k }: { key: string }) => st.set(k, (st.get(k) ?? 0) + 1);
      for (const t of label({ set })) {
        const k = classOf.get(t.key);
        const rs = k === undefined ? undefined : rulesOfClass.get(k);
        if (!rs) continue;
        bump({ key: "turns whose reply is an entry" });
        const f = firing({ text: t.user });
        const nf = f.reduce((a, b) => a + b, 0);
        const own = rs.some((r) => f[r]);
        const got = select({
          text: t.user,
          hasName: t.mem.latest >= 0,
          prev: t.prev,
          samePname: t.mem.pname === t.mem.latest,
          seed: seedOf({ user: t.user, ti: t.ti }),
          recent: new Set(),
          ti: t.ti,
        });
        if (got === k) bump({ key: own ? "  right (own rule fires)" : "  right (by a neighbour)" });
        else if (own) {
          bump({
            key:
              nf > 1
                ? "  wrong: own rule fires but another wins"
                : "  wrong: own rule fires alone but a neighbour wins",
          });
          if (nf > 1 && flag({ name: "--rule-examples" }) && Math.random() < 0.15) {
            const ownR = rs.find((r) => f[r]) as number;
            const others = [...f.keys()]
              .filter((r) => f[r] && !rs.includes(r))
              .map((r) => `${rules[r].id}:${rankOf({ r: rules[r] })}`);
            log(`    ex: "${t.user}" own=${rules[ownR].id}:${rankOf({ r: rules[ownR] })} others=${others.join(" ")}`);
          }
        } else bump({ key: nf ? "  wrong: own rule silent, another rule fires" : "  wrong: no rule fires" });
      }
      for (const [k, n] of st) log(`  ${name}: ${n}  ${k}`);
    }
    if (flag({ name: "--errors" })) {
      const modOf = new Map<number, string>();
      for (const t of trainTurns) modOf.set(classOf.get(t.key) as number, t.module ?? "none");
      const pairs = new Map<string, number>();
      for (const t of label({ set })) {
        const k = classOf.get(t.key);
        if (k === undefined) continue;
        const got = select({
          text: t.user,
          hasName: t.mem.latest >= 0,
          prev: t.prev,
          samePname: t.mem.pname === t.mem.latest,
          seed: seedOf({ user: t.user, ti: t.ti }),
          recent: new Set(),
          ti: t.ti,
        });
        if (got === k) continue;
        const key = `${t.module} -> ${modOf.get(got)}`;
        if (
          flag({ name: "--error-examples" }) &&
          t.module === "greeting" &&
          modOf.get(got) === undefined &&
          Math.random() < 0.2
        ) {
          const f = firing({ text: t.user });
          const fr = [...f.keys()].filter((r) => f[r]).map((r) => `${rules[r].id}:${rankOf({ r: rules[r] })}`);
          log(`    fb-ex: "${t.user}" → fired ${fr.join(" ")}`);
        }
        pairs.set(key, (pairs.get(key) ?? 0) + 1);
      }
      for (const [k, n] of [...pairs].sort((a, b) => b[1] - a[1]).slice(0, 12)) log(`  error ${name}: ${n}  ${k}`);
    }
    for (const [k, n] of [...POOL_ERR].sort((a, b) => b[1] - a[1])) log(`  pool-reply error ${name}: ${n}  ${k}`);
    POOL_ERR.clear();
    log(`  recency filter changed the candidates ${RECENT_HITS.n} times (summed over selectors)`);
    for (const [g, v] of groups)
      log(
        `sim ${name}/${g}: turns ${v.n}, reachable ${(v.reach / v.n).toFixed(3)}, selected right ${(v.ok / v.n).toFixed(3)} (of reachable ${(v.ok / Math.max(1, v.reach)).toFixed(3)})`,
      );
  }
  if (flag({ name: "--sim-only" })) process.exit(0);
}

if (PREVW > 0) throw new Error("--prevw is a simulation-only experiment");
if (RECENT_ON) throw new Error("--recent is a simulation-only experiment");
if (UW > 0) throw new Error("--oracle-understood is a simulation-only experiment (pybot's own reading of the line)");

// ---------- full vocabulary ----------
const tokens = [...baseTokens, ...selectors.map((_, i) => `<s:${i}>`)];
const V = tokens.length;
const ID0 = NB; // token id of <s:0>
log(`vocab: ${V} (chars ${chars.length}, words ${words.length}, ids ${NS})`);

// tokens the PREV blocks copy: those that occur in a reply (the recital's contexts), the
// specials, and the words of the name-introduction patterns (and every word, with bigrams)
const introWords = new Set(INTRO.flat());
const inR = new Uint8Array(NB);
for (const t of classToks) for (const v of t) if (v >= 0) inR[v] = 1;
for (let v = 0; v <= IDPH; v++) inR[v] = 1;
for (let v = 0; v < NB; v++) if (introWords.has(lexStr({ v })) || (NBI && lexOfBase[v] >= 0)) inR[v] = 1;
const rIdx = new Int32Array(NB).fill(-1);
let NR = 0;
for (let v = 0; v < NB; v++) if (inR[v]) rIdx[v] = NR++;
log(`tokens copied by the PREV heads: ${NR} of ${NB}`);

// ---------- residual layout ----------
let d = 0;
const block = ({ n }: { n: number }) => {
  const o = d;
  d += n;
  return o;
};
const ANCHOR = block({ n: 1 });
const IS_USER = block({ n: 1 });
const IS_BOT = block({ n: 1 });
const IS_ID = block({ n: 1 });
const PE = block({ n: 2 * F });
const TOK = block({ n: NB }); // current token (ID tokens present as <id>)
const PREV = [TOK, ...Array.from({ length: P }, () => block({ n: NR }))]; // PREV[m] = token m back (R tokens)
// the residual dim that is 1 when token v is m tokens back (m = 0: the current token)
const prevKey = ({ m, v }: { m: number; v: number }) => {
  if (m === 0) return TOK + v;
  if (rIdx[v] < 0) throw new Error(`token ${v} is not copied by the PREV heads`);
  return PREV[m] + rIdx[v];
};
const UPE = block({ n: 2 * F }); // position code of the latest <user>
const USIDE = block({ n: 1 }); // 1 on the user side of the turn
const IDC = block({ n: C }); // class of an ID token (embedding)
const IDR = block({ n: C }); // class of the latest ID token (copied)
const NBAG = L + NBI + (FIRSTW > 0 ? FIRSTN * L : 0);
const BAG = block({ n: NBAG }); // bag of words (word pairs, first words) of the current user turn
const BIGR = block({ n: NBI }); // this token completes word pair b (layer-1 MLP)
const UBAG = block({ n: NBAG }); // BAG / ‖BAG‖ (layer-2 norm): the selection compares directions
const NAMEID = block({ n: NN }); // which name a token spells (embedding)
const NAMEFLAG = block({ n: 1 }); // this token is an introduced name
const NAMEV = block({ n: NN }); // the latest introduced name
const HASNAME = block({ n: 1 }); // some name was introduced
const PNAMEV = block({ n: NN }); // the latest introduced name that differs from NAMEV
const SAMEP = block({ n: 1 }); // PNAMEV is the same name as NAMEV (no different earlier name)
const GATE = block({ n: 4 }); // name slot requested here: name/pname × with/without space
const OUTB = block({ n: V }); // logits
// the knowledge rules
const presLex = [...new Set(altList.flat())];
const presIdx = new Map(presLex.map((l, i) => [l, i]));
const phrases = altList.map((ls, a) => (ls.length > 1 ? a : -1)).filter((a) => a >= 0);
const phrIdx = new Map(phrases.map((a, i) => [a, i]));
const avoiders = rules.map((r, i) => (r.avoid.length ? i : -1)).filter((i) => i >= 0);
const avdIdx = new Map(avoiders.map((r, i) => [r, i]));
const PRES = block({ n: presLex.length }); // lexeme l is in the current user turn (0/1)
const PHR = block({ n: phrases.length }); // all words of a phrase are there (0/1)
const GRP = block({ n: groupList.length }); // some alternative of a key group is there (0/1)
const AVD = block({ n: avoiders.length }); // some avoid alternative of a rule is there (0/1)
const PGRP = block({ n: pgroupList.length }); // some phrase alternative of a key group is there (0/1)
const contentRules = rules.map((r, i) => (r.content.length ? i : -1)).filter((i) => i >= 0);
const cntIdx = new Map(contentRules.map((r, i) => [r, i]));
const CNT = block({ n: contentRules.length }); // some non-frame alternative of the rule is there (0/1)
const ENT = block({ n: rules.length }); // the rule fires (0/1)
// the seed circuit (pybot's choice of a pool option)
const IS_BOS = block({ n: 1 });
const CSV = block({ n: 1 }); // character-code sum of the current token (embedding)
const TX = block({ n: 1 }); // 1 / (1 + number of <user> so far)          (layer-1 head "turns")
const AV = block({ n: 1 }); // Σ CSV over the current user turn / (n + 1)  (layer-2 head "csum")
const BV = block({ n: 1 }); // 1 / (n + 1), n = tokens in the turn          (layer-2 head "csum")
const NOH = block({ n: NMAX + 1 }); // one-hot of n                          (layer-2 MLP)
const IOH = block({ n: IMAX + 1 }); // one-hot of the turn index            (layer-2 MLP)
const SV = block({ n: 1 }); // Σ ord of the lowercased line                 (layer-3 MLP)
const HOH = block({ n: HMAX + 1 }); // one-hot of ⌊seed / 1000⌋              (layer-4 MLP)
const LOV = block({ n: 1 }); // seed mod 1000                               (layer-4 MLP)
const LOH = block({ n: 1000 }); // one-hot of seed mod 1000                  (layer-5 MLP)
const IDXB = Array.from({ length: MAXK + 1 }, (_, k) => (k >= 2 ? block({ n: k }) : -1)); // choice() index for seed = turn index (layer-3 MLP)
const ANYRULE = block({ n: 1 }); // some knowledge rule fires (layer-6 MLP)
const IDXK = Array.from({ length: MAXK + 1 }, (_, k) => (k >= 2 ? block({ n: k }) : -1)); // choice() index, per pool size (layer-6 MLP)
const altDim = ({ a }: { a: number }) =>
  altList[a].length > 1 ? PHR + (phrIdx.get(a) as number) : PRES + (presIdx.get(altList[a][0]) as number);

log(`d_model: ${d}`);

// ---------- embedding ----------
const E = new Coo({ rows: V, cols: d });
for (let v = 0; v < V; v++) {
  E.add({ row: v, col: ANCHOR, value: 1 });
  E.add({ row: v, col: TOK + (v < ID0 ? v : IDPH), value: 1 });
  if (v < NB && nameOf[v] >= 0) E.add({ row: v, col: NAMEID + nameOf[v], value: 1 });
}
E.add({ row: USER, col: IS_USER, value: 1 });
E.add({ row: BOT, col: IS_BOT, value: 1 });
E.add({ row: BOS, col: IS_BOS, value: 1 });
for (let v = 0; v < NB; v++) E.add({ row: v, col: CSV, value: csOf({ v }) });
for (let i = 0; i < NS; i++) {
  E.add({ row: ID0 + i, col: IS_ID, value: 1 });
  E.add({ row: ID0 + i, col: IDC + selectors[i].cls, value: 1 }); // an ID token carries its reply class
}

// ---------- position margins ----------
// a "m tokens back" head scores β·Σ_f cos(f(t−m−s)); the margin to the next best s is
// min_{Δ≠0} (F − Σ cos(fΔ)) over the context length.
let margin = Infinity;
for (let D = 1; D < MAXLEN; D++) {
  let s = 0;
  for (const f of FREQS) s += Math.cos(f * D);
  margin = Math.min(margin, F - s);
}
const BETA = 60 / margin; // ≥ 60 nats between the target and any other position
log(`position margin ${margin.toFixed(4)} → β = ${BETA.toFixed(1)}`);

function offsetHead({
  name,
  m,
  srcOff,
  dstOff,
  n,
}: {
  name: string;
  m: number;
  srcOff: number;
  dstOff: number;
  n: number;
}): Head {
  const Wq = new Coo({ rows: d, cols: 2 * F });
  const Wk = new Coo({ rows: d, cols: 2 * F });
  FREQS.forEach((f, i) => {
    const c = Math.cos(f * m);
    const s = Math.sin(f * m);
    // q = rotation of pe_t by −m·f  → q·pe_s = cos(f(t−m−s))
    Wq.add({ row: PE + 2 * i, col: 2 * i, value: c });
    Wq.add({ row: PE + 2 * i + 1, col: 2 * i, value: s });
    Wq.add({ row: PE + 2 * i + 1, col: 2 * i + 1, value: c });
    Wq.add({ row: PE + 2 * i, col: 2 * i + 1, value: -s });
    Wk.add({ row: PE + 2 * i, col: 2 * i, value: 1 });
    Wk.add({ row: PE + 2 * i + 1, col: 2 * i + 1, value: 1 });
  });
  const Wv = new Coo({ rows: d, cols: n });
  const Wo = new Coo({ rows: n, cols: d });
  for (let i = 0; i < n; i++) {
    Wv.add({ row: srcOff + i, col: i, value: 1 });
    Wo.add({ row: i, col: dstOff + i, value: 1 });
  }
  return { name, dk: 2 * F, dv: n, scale: BETA, slope: 0, Wq: Wq.done(), Wk: Wk.done(), Wv: Wv.done(), Wo: Wo.done() };
}

// latest position whose key flag is set (recency through ALiBi); copies src → dst
function latestHead({ name, flags, copy }: { name: string; flags: number[]; copy: [number, number][] }): Head {
  const Wq = new Coo({ rows: d, cols: 1 });
  const Wk = new Coo({ rows: d, cols: 1 });
  Wq.add({ row: ANCHOR, col: 0, value: 1 });
  for (const f of flags) Wk.add({ row: f, col: 0, value: 1 });
  const Wv = new Coo({ rows: d, cols: copy.length });
  const Wo = new Coo({ rows: copy.length, cols: d });
  copy.forEach(([from, to], i) => {
    Wv.add({ row: from, col: i, value: 1 });
    Wo.add({ row: i, col: to, value: 1 });
  });
  return {
    name,
    dk: 1,
    dv: copy.length,
    scale: BIG,
    slope: SLOPE,
    Wq: Wq.done(),
    Wk: Wk.done(),
    Wv: Wv.done(),
    Wo: Wo.done(),
  };
}
const range = ({ n, from, to }: { n: number; from: number; to: number }): [number, number][] =>
  Array.from({ length: n }, (_, i) => [from + i, to + i]);

const heads1: Head[] = [];
function prevHead({ m }: { m: number }): Head {
  const h = offsetHead({ name: `prev${m}`, m, srcOff: 0, dstOff: 0, n: 0 });
  const Wv = new Coo({ rows: d, cols: NR });
  const Wo = new Coo({ rows: NR, cols: d });
  for (let v = 0; v < NB; v++)
    if (rIdx[v] >= 0) {
      Wv.add({ row: TOK + v, col: rIdx[v], value: 1 });
      Wo.add({ row: rIdx[v], col: PREV[m] + rIdx[v], value: 1 });
    }
  return { ...h, dv: NR, Wv: Wv.done(), Wo: Wo.done() };
}
for (let m = 1; m <= P; m++) heads1.push(prevHead({ m }));
heads1.push(latestHead({ name: "lastUser", flags: [IS_USER], copy: range({ n: 2 * F, from: PE, to: UPE }) }));
heads1.push(latestHead({ name: "lastMarker", flags: [IS_USER, IS_BOT], copy: [[IS_USER, USIDE]] }));
heads1.push(latestHead({ name: "lastId", flags: [IS_ID], copy: range({ n: C, from: IDC, to: IDR }) }));
// turns: equal weight on <bos> and every <user> → the weight on <bos> is 1/(1+u)
heads1.push({ ...latestHead({ name: "turns", flags: [IS_USER, IS_BOS], copy: [[IS_BOS, TX]] }), slope: 0 });

// ---------- layer 1 MLP ----------
type Neuron = { keys: number[]; out: number; w: number };
const neurons1: Neuron[] = [];

// (a) recital. The neuron for step j of class c fires on IDR=c and the last k input tokens,
// where k is the shortest context that no other step of the same class with a different
// next token shares. A name slot in the context is a wildcard (no key).
let unresolved = 0;
const kHist = new Map<number, number>();
for (let c = 0; c < C; c++) {
  const r = [...classToks[c], EOT];
  const seq = [BOT, IDPH, ...r]; // when predicting r[j] the current input is seq[j+1]
  // does the neuron of step j with context length k also fire at step jj? (a slot in j's
  // context is a wildcard; a slot at jj is some name token, which no literal key matches)
  const fires = ({ j, k, jj }: { j: number; k: number; jj: number }) => {
    for (let m = 0; m < k; m++) {
      const a = seq[j + 1 - m];
      if (a < 0) continue;
      const i = jj + 1 - m;
      if (i < 0 || seq[i] !== a) return false;
    }
    return true;
  };
  const made = new Set<string>();
  for (let j = 0; j < r.length; j++) {
    let k = 1;
    const kmax = Math.min(P + 1, j + 2);
    for (; k <= kmax; k++) {
      let clash = false;
      for (let jj = 0; jj < r.length && !clash; jj++)
        if (jj !== j && r[jj] !== r[j] && fires({ j, k, jj })) clash = true;
      if (!clash) break;
    }
    if (k > kmax) {
      unresolved++;
      k = kmax;
    }
    kHist.set(k, (kHist.get(k) ?? 0) + 1);
    const keys = [IDR + c];
    for (let m = 0; m < k; m++) {
      const v = seq[j + 1 - m];
      if (v >= 0) keys.push(prevKey({ m, v }));
    }
    const out = r[j] >= 0 ? OUTB + r[j] : GATE + (-r[j] - 1);
    const key = `${keys.join(",")}>${out}`;
    if (made.has(key)) continue;
    made.add(key);
    // a token gets logit K; a slot gate gets exactly 1 (it is an input to the fill neurons)
    neurons1.push({ keys, out, w: r[j] >= 0 ? 2 * K : 2 });
  }
}
const nRecital = neurons1.length;

// (b) name introductions: a user-side token that spells a name, after one of the INTRO
// patterns. The pattern words are matched by every token whose lexeme is that word.
const tokensOfLex = new Map<string, number[]>();
for (let v = 0; v < NB; v++) {
  const l = lexStr({ v });
  if (!l) continue;
  const a = tokensOfLex.get(l) ?? [];
  a.push(v);
  tokensOfLex.set(l, a);
}
const nameToks: number[] = [];
for (let v = 0; v < NB; v++) if (nameOf[v] >= 0) nameToks.push(v);
for (const p of INTRO) {
  // keys: USIDE, "is a name" (any name token at TOK), and each pattern word at PREV[m]
  const keys: number[][] = [[USIDE], nameToks.map((v) => TOK + v)];
  p.forEach((w, k) => {
    const m = p.length - k;
    keys.push((tokensOfLex.get(w) ?? []).map((v) => prevKey({ m, v })));
  });
  neurons1.push({ keys: [], out: NAMEFLAG, w: 2, ...{ multi: keys } } as Neuron & { multi: number[][] });
}
// (c) word pairs for the bag
if (NBI) {
  const tokOfLexId = new Map<number, number[]>();
  for (let v = 0; v < NB; v++)
    if (lexOfBase[v] >= 0) {
      const a = tokOfLexId.get(lexOfBase[v]) ?? [];
      a.push(v);
      tokOfLexId.set(lexOfBase[v], a);
    }
  for (const [k, b] of bigramIndex) {
    const [l1, l2] = k.split("|").map(Number);
    const multi = [
      [USIDE],
      (tokOfLexId.get(l2) ?? []).map((v) => TOK + v),
      (tokOfLexId.get(l1) ?? []).map((v) => prevKey({ m: 1, v })),
    ];
    neurons1.push({ keys: [], out: BIGR + b, w: 2, multi } as Neuron & { multi: number[][] });
  }
}
log(
  `layer-1 neurons: recital ${nRecital} (context lengths ${JSON.stringify(Object.fromEntries([...kHist].sort((a, b) => a[0] - b[0])))}, unresolved ${unresolved}), name detectors ${INTRO.length}`,
);

function mlp({ neurons }: { neurons: (Neuron & { multi?: number[][] })[] }) {
  const n = neurons.length;
  const W1 = new Coo({ rows: d, cols: n });
  const b1 = new Float32Array(n);
  const W2 = new Coo({ rows: n, cols: d });
  neurons.forEach((nu, i) => {
    // a key group is an OR of one-hot dims (at most one of them is 1)
    const groups = nu.multi ?? nu.keys.map((k) => [k]);
    for (const g of groups) for (const k of g) W1.add({ row: k, col: i, value: 1 });
    b1[i] = -(groups.length - 0.5);
    W2.add({ row: i, col: nu.out, value: nu.w });
  });
  return { n, W1: W1.done(), b1, W2: W2.done() };
}

// ---------- layer 2 attention ----------
// bag: attend to the tokens of the current user turn with weight ∝ idf
const BAG_BETA = 60 / margin;
const BAG_SIDE = 80;
const bagHead = (() => {
  const dk = 2 * F + 1;
  const Wq = new Coo({ rows: d, cols: dk });
  const Wk = new Coo({ rows: d, cols: dk });
  for (let i = 0; i < 2 * F; i++) {
    Wq.add({ row: UPE + i, col: i, value: BAG_BETA });
    Wk.add({ row: UPE + i, col: i, value: 1 });
  }
  Wq.add({ row: ANCHOR, col: 2 * F, value: 1 });
  Wk.add({ row: USIDE, col: 2 * F, value: BAG_SIDE });
  for (let v = 0; v < NB; v++) Wk.add({ row: TOK + v, col: 2 * F, value: Math.log(baseIdf[v]) });
  const Wv = new Coo({ rows: d, cols: L + NBI });
  for (let v = 0; v < NB; v++) if (lexOfBase[v] >= 0) Wv.add({ row: TOK + v, col: lexOfBase[v], value: 1 });
  for (let b = 0; b < NBI; b++) Wv.add({ row: BIGR + b, col: L + b, value: BIGRAM });
  const Wo = new Coo({ rows: L + NBI, cols: d });
  for (let l = 0; l < L + NBI; l++) Wo.add({ row: l, col: BAG + l, value: 1 });
  return {
    name: "bag",
    dk,
    dv: L + NBI,
    scale: 1,
    slope: 0,
    Wq: Wq.done(),
    Wk: Wk.done(),
    Wv: Wv.done(),
    Wo: Wo.done(),
  } as Head;
})();
// csum: equal weight on the tokens of the current user turn and its <user> marker
const csumHead = (() => {
  const dk = 2 * F + 1;
  const Wq = new Coo({ rows: d, cols: dk });
  const Wk = new Coo({ rows: d, cols: dk });
  for (let i = 0; i < 2 * F; i++) {
    Wq.add({ row: UPE + i, col: i, value: BAG_BETA });
    Wk.add({ row: UPE + i, col: i, value: 1 });
  }
  Wq.add({ row: ANCHOR, col: 2 * F, value: 1 });
  Wk.add({ row: USIDE, col: 2 * F, value: BAG_SIDE });
  const Wv = new Coo({ rows: d, cols: 2 });
  Wv.add({ row: CSV, col: 0, value: 1 });
  Wv.add({ row: IS_USER, col: 1, value: 1 });
  const Wo = new Coo({ rows: 2, cols: d });
  Wo.add({ row: 0, col: AV, value: 1 });
  Wo.add({ row: 1, col: BV, value: 1 });
  return {
    name: "csum",
    dk,
    dv: 2,
    scale: 1,
    slope: 0,
    Wq: Wq.done(),
    Wk: Wk.done(),
    Wv: Wv.done(),
    Wo: Wo.done(),
  } as Head;
})();
// first words: the token m = 1..FIRSTN after the latest <user> (its position code rotated by m),
// written as its lexeme × FIRSTW into the first-word dims of the bag
const firstHeads: Head[] = [];
if (FIRSTW > 0)
  for (let m = 1; m <= FIRSTN; m++) {
    const Wq = new Coo({ rows: d, cols: 2 * F });
    const Wk = new Coo({ rows: d, cols: 2 * F });
    FREQS.forEach((f, i) => {
      const c = Math.cos(f * m);
      const sn = Math.sin(f * m);
      // q = pe of position (u + m): cos(f(u+m)) = cos fu cos fm − sin fu sin fm, ...
      Wq.add({ row: UPE + 2 * i, col: 2 * i, value: c });
      Wq.add({ row: UPE + 2 * i + 1, col: 2 * i, value: -sn });
      Wq.add({ row: UPE + 2 * i + 1, col: 2 * i + 1, value: c });
      Wq.add({ row: UPE + 2 * i, col: 2 * i + 1, value: sn });
      Wk.add({ row: PE + 2 * i, col: 2 * i, value: 1 });
      Wk.add({ row: PE + 2 * i + 1, col: 2 * i + 1, value: 1 });
    });
    const Wv = new Coo({ rows: d, cols: L });
    for (let v = 0; v < NB; v++) if (lexOfBase[v] >= 0) Wv.add({ row: TOK + v, col: lexOfBase[v], value: FIRSTW });
    const Wo = new Coo({ rows: L, cols: d });
    for (let l = 0; l < L; l++) Wo.add({ row: l, col: BAG + L + NBI + (m - 1) * L + l, value: 1 });
    firstHeads.push({
      name: `first${m}`,
      dk: 2 * F,
      dv: L,
      scale: BETA,
      slope: 0,
      Wq: Wq.done(),
      Wk: Wk.done(),
      Wv: Wv.done(),
      Wo: Wo.done(),
    });
  }
const lastName = latestHead({
  name: "lastName",
  flags: [NAMEFLAG],
  copy: [...range({ n: NN, from: NAMEID, to: NAMEV }), [NAMEFLAG, HASNAME]],
});

// ---------- layer 3 attention: the latest introduced name that is not NAMEV ----------
const pnameHead = (() => {
  const dk = 1 + NN;
  const Wq = new Coo({ rows: d, cols: dk });
  const Wk = new Coo({ rows: d, cols: dk });
  Wq.add({ row: ANCHOR, col: 0, value: 1 });
  Wk.add({ row: NAMEFLAG, col: 0, value: 1 });
  for (let n = 0; n < NN; n++) {
    Wq.add({ row: NAMEV + n, col: 1 + n, value: -0.5 });
    Wk.add({ row: NAMEID + n, col: 1 + n, value: 1 });
  }
  const Wv = new Coo({ rows: d, cols: NN });
  const Wo = new Coo({ rows: NN, cols: d });
  for (let n = 0; n < NN; n++) {
    Wv.add({ row: NAMEID + n, col: n, value: 1 });
    Wo.add({ row: n, col: PNAMEV + n, value: 1 });
  }
  return {
    name: "prevName",
    dk,
    dv: NN,
    scale: BIG,
    slope: SLOPE,
    Wq: Wq.done(),
    Wk: Wk.done(),
    Wv: Wv.done(),
    Wo: Wo.done(),
  } as Head;
})();

// ---------- layer 3 MLP: fill a name slot ----------
const neurons3: Neuron[] = [];
for (let n = 0; n < NN; n++) {
  neurons3.push({ keys: [NAMEV + n, GATE + 0], out: OUTB + nameTokSp[n], w: 2 * K });
  neurons3.push({ keys: [NAMEV + n, GATE + 1], out: OUTB + nameTokNs[n], w: 2 * K });
  neurons3.push({ keys: [PNAMEV + n, GATE + 2], out: OUTB + nameTokSp[n], w: 2 * K });
  neurons3.push({ keys: [PNAMEV + n, GATE + 3], out: OUTB + nameTokNs[n], w: 2 * K });
}

// ---------- layers 2-5 MLP: the knowledge rules ----------
type GNeuron = { inp: [number, number][]; b: number; out: [number, number][] };
function mlpGen({ neurons }: { neurons: GNeuron[] }) {
  const n = neurons.length;
  const W1 = new Coo({ rows: d, cols: n });
  const b1 = new Float32Array(n);
  const W2 = new Coo({ rows: n, cols: d });
  neurons.forEach((nu, i) => {
    for (const [k, w] of nu.inp) W1.add({ row: k, col: i, value: w });
    b1[i] = nu.b;
    for (const [k, w] of nu.out) W2.add({ row: i, col: k, value: w });
  });
  return { n, W1: W1.done(), b1, W2: W2.done() };
}
// layer 2: PRES = min(1, λ·BAG) = ReLU(λ·BAG) − ReLU(λ·BAG − 1)
const nPres: GNeuron[] = [];
presLex.forEach((l, i) => {
  nPres.push({ inp: [[BAG + l, LAMBDA_P]], b: 0, out: [[PRES + i, 1]] });
  nPres.push({ inp: [[BAG + l, LAMBDA_P]], b: -1, out: [[PRES + i, -1]] });
});
// layer 3: PHR = ReLU(Σ PRES − (n − 1))
const nPhr: GNeuron[] = phrases.map((a, i) => ({
  inp: altList[a].map((l) => [PRES + (presIdx.get(l) as number), 1] as [number, number]),
  b: -(altList[a].length - 1),
  out: [[PHR + i, 1]],
}));
// layer 4: GRP = min(1, Σ alternatives), AVD = min(1, Σ avoid alternatives)
const nGrp: GNeuron[] = [];
const orOf = ({ alts, out }: { alts: number[]; out: number }) => {
  const inp = alts.map((a) => [altDim({ a }), 1] as [number, number]);
  nGrp.push({ inp, b: 0, out: [[out, 1]] });
  nGrp.push({ inp, b: -1, out: [[out, -1]] });
};
groupList.forEach((as, g) => orOf({ alts: as, out: GRP + g }));
avoiders.forEach((ri, i) => orOf({ alts: rules[ri].avoid, out: AVD + i }));
pgroupList.forEach((as, i) => orOf({ alts: as, out: PGRP + i }));
contentRules.forEach((ri, i) => orOf({ alts: rules[ri].content, out: CNT + i }));
// layer 5: ENT = 2·ReLU(Σ GRP − 2·AVD − (G − 0.5))
const nEnt: GNeuron[] = rules.map((r, i) => {
  const inp = r.groups.map((g) => [GRP + g, 1] as [number, number]);
  if (avdIdx.has(i)) inp.push([AVD + (avdIdx.get(i) as number), -2]);
  const needs = r.groups.length + (cntIdx.has(i) ? 1 : 0);
  if (cntIdx.has(i)) inp.push([CNT + (cntIdx.get(i) as number), 1]);
  return { inp, b: -(needs - 0.5), out: [[ENT + i, 2]] };
});
log(
  `rule neurons: presence ${nPres.length}, phrases ${nPhr.length}, groups/avoid ${nGrp.length}, entries ${nEnt.length}; d_model now ${d}`,
);

function mergeMlp({ a, b }: { a: ReturnType<typeof mlpGen>; b: ReturnType<typeof mlpGen> }) {
  const shift = ({ m, byRow }: { m: { r: Int32Array; c: Int32Array; v: Float32Array }; byRow: boolean }) => ({
    r: byRow ? m.r.map((x) => x + a.n) : m.r,
    c: byRow ? m.c : m.c.map((x) => x + a.n),
    v: m.v,
  });
  const cat = ({ x, y }: { x: Int32Array | Float32Array; y: Int32Array | Float32Array }) => {
    const o = new (x.constructor as Int32ArrayConstructor)(x.length + y.length);
    o.set(x as Int32Array);
    o.set(y as Int32Array, x.length);
    return o;
  };
  const b1W1 = shift({ m: b.W1, byRow: false });
  const b1W2 = shift({ m: b.W2, byRow: true });
  return {
    n: a.n + b.n,
    W1: {
      rows: d,
      cols: a.n + b.n,
      r: cat({ x: a.W1.r, y: b1W1.r }) as Int32Array,
      c: cat({ x: a.W1.c, y: b1W1.c }) as Int32Array,
      v: Float32Array.from([...a.W1.v, ...b1W1.v]),
    },
    b1: Float32Array.from([...a.b1, ...b.b1]),
    W2: {
      rows: a.n + b.n,
      cols: d,
      r: cat({ x: a.W2.r, y: b1W2.r }) as Int32Array,
      c: cat({ x: a.W2.c, y: b1W2.c }) as Int32Array,
      v: Float32Array.from([...a.W2.v, ...b1W2.v]),
    },
  };
}

// layer 3 MLP: SAMEP = 1 when PNAMEV and NAMEV are the same name
// ---------- the seed circuit ----------
// a step of a scalar x at threshold t with slope λ: clamp(λ(x − t), 0, 1) as two ReLUs
const stepPair = ({
  inp,
  lam,
  t,
  out,
}: {
  inp: [number, number][];
  lam: number;
  t: number;
  out: [number, number][];
}): GNeuron[] => [
  { inp: inp.map(([k, w]) => [k, w * lam] as [number, number]), b: -lam * t, out },
  {
    inp: inp.map(([k, w]) => [k, w * lam] as [number, number]),
    b: -lam * t - 1,
    out: out.map(([k, w]) => [k, -w] as [number, number]),
  },
];
// one-hot of m from x = 1/(m + off): steps between consecutive values
function invOneHot({ x, off, M, dst }: { x: number; off: number; M: number; dst: number }): GNeuron[] {
  const out: GNeuron[] = [];
  for (let m = 0; m <= M; m++) {
    const c = 1 / (m + off);
    const c1 = 1 / (m + 1 + off);
    // [x ≥ (c + c1)/2] is 1 for every m' ≤ m: it adds 1 to one-hot[m], takes 1 from one-hot[m+1]
    const o: [number, number][] = [[dst + m, 1]];
    if (m < M) o.push([dst + m + 1, -1]);
    // slope 4/(c − c1): the clamp saturates (at exactly 0 or 1) a quarter-gap before c and c1
    out.push(...stepPair({ inp: [[x, 1]], lam: 4 / (c - c1), t: (c + c1) / 2, out: o }));
  }
  return out;
}
const nSteps2: GNeuron[] = [
  ...invOneHot({ x: BV, off: 1, M: NMAX, dst: NOH }),
  ...invOneHot({ x: TX, off: 2, M: IMAX, dst: IOH }),
];
// type-B pools: the seed is the turn index itself, so the choice is read off its one-hot
const nPickB: GNeuron[] = [];
for (let i = 0; i <= IMAX; i++) {
  const out: [number, number][] = [];
  for (let k = 2; k <= MAXK; k++) out.push([IDXB[k] + choiceOf({ seed: i, k }), 1]);
  nPickB.push({ inp: [[IOH + i, 2]], b: -1, out });
}
// S = (n + 1)·AV for the n whose one-hot is 1 (a gate of size MBIG)
const nSum3: GNeuron[] = Array.from({ length: NMAX + 1 }, (_, n) => ({
  inp: [
    [AV, n + 1],
    [NOH + n, MBIG],
  ] as [number, number][],
  b: -MBIG,
  out: [[SV, 1]] as [number, number][],
}));
// seed = S + 7919·i; HOH = one-hot of ⌊seed/1000⌋ (clipped at HMAX); LOV = seed − 1000·⌊seed/1000⌋
const seedIn: [number, number][] = [
  [SV, 1],
  ...Array.from({ length: IMAX + 1 }, (_, i) => [IOH + i, 7919 * i] as [number, number]),
];
const nHi4: GNeuron[] = [
  { inp: [[ANCHOR, 1]], b: 0, out: [[HOH, 1]] },
  { inp: seedIn, b: 0, out: [[LOV, 1]] },
];
// integer steps: threshold k − 0.5 and slope 4 saturate within ±0.25 of an integer
for (let h = 1; h <= HMAX; h++)
  nHi4.push(
    ...stepPair({
      inp: seedIn,
      lam: 4,
      t: 1000 * h - 0.5,
      out: [
        [HOH + h, 1],
        [HOH + h - 1, -1],
        [LOV, -1000],
      ],
    }),
  );
// LOH = one-hot of LOV (clipped at 999)
const nLo5: GNeuron[] = [{ inp: [[ANCHOR, 1]], b: 0, out: [[LOH, 1]] }];
for (let l = 1; l <= 999; l++)
  nLo5.push(
    ...stepPair({
      inp: [[LOV, 1]],
      lam: 4,
      t: l - 0.5,
      out: [
        [LOH + l, 1],
        [LOH + l - 1, -1],
      ],
    }),
  );
// layer 6: one neuron per seed = 1000·h + l, writing choice()'s index for every pool size
const nCell6: GNeuron[] = [];
for (let h = 0; h <= HMAX; h++)
  for (let l = 0; l < 1000; l++) {
    const seed = 1000 * h + l;
    const out: [number, number][] = [];
    for (let k = 2; k <= MAXK; k++) out.push([IDXK[k] + choiceOf({ seed, k }), 2]);
    nCell6.push({
      inp: [
        [HOH + h, 1],
        [LOH + l, 1],
      ],
      b: -1.5,
      out,
    });
  }
// ANYRULE = min(1, Σ ENT)
const entIn = rules.map((_, i) => [ENT + i, 1] as [number, number]);
nCell6.push({ inp: entIn, b: 0, out: [[ANYRULE, 1]] }, { inp: entIn, b: -1, out: [[ANYRULE, -1]] });
log(`seed circuit neurons: ${nSteps2.length + nSum3.length + nHi4.length + nLo5.length + nCell6.length}`);

const nSame: GNeuron[] = Array.from({ length: NN }, (_, n) => ({
  inp: [
    [PNAMEV + n, 1],
    [NAMEV + n, 1],
  ] as [number, number][],
  b: -1.5,
  out: [[SAMEP, 2]] as [number, number][],
}));

// ---------- unembedding ----------
const Wu = new Coo({ rows: d, cols: V });
const bu = new Float32Array(V);
for (let v = 0; v < V; v++) Wu.add({ row: OUTB + v, col: v, value: 1 });
bu[EOT] = EOT_BASE;
for (let i = 0; i < NS; i++) {
  const v = ID0 + i;
  Wu.add({ row: IS_BOT, col: v, value: G });
  // feature dims of the simulation → residual dims: [0, L) words, [2L, 2L+NBI) pairs, then first words
  for (const [l, w] of selectors[i].bag)
    Wu.add({
      row: UBAG + (l < L ? l : l < 2 * L + NBI ? L + (l - 2 * L) : L + NBI + (l - 2 * L - NBI)),
      col: v,
      value: S * w,
    });
  // has-name agreement: S·λ·(±1)·(2·HASNAME − 1)
  const hs = selectors[i].hasName ? 1 : -1;
  Wu.add({ row: HASNAME, col: v, value: 2 * S * LAMBDA * hs });
  Wu.add({ row: ANCHOR, col: v, value: -S * LAMBDA * hs });
  if (slotted[selectors[i].cls]) {
    Wu.add({ row: HASNAME, col: v, value: SLOT_PEN });
    Wu.add({ row: ANCHOR, col: v, value: -SLOT_PEN });
  }
  if (pslotted[selectors[i].cls]) Wu.add({ row: SAMEP, col: v, value: -SLOT_PEN });
  // base so that an ID always beats the word logits at <bot>; tiny prior for ties
  bu[v] = -G + 1 + prior[i];
  if (NORULE_PEN && selectors[i].rule === undefined && entryClass.has(selectors[i].cls)) {
    // −NORULE_PEN while no knowledge rule fires
    bu[v] -= NORULE_PEN;
    Wu.add({ row: ANYRULE, col: v, value: NORULE_PEN });
  }
  const pv = (selectors[i] as Selector & { pv?: PoolSel }).pv;
  if (pv && pv.p < poolList.length) bu[v] += POOL_BONUS; // _choose (fallback) pools only
  if (pv) {
    // −G2 unless choice() takes this option (exactly one IDX of the pool's size is 1)
    bu[v] -= G2;
    Wu.add({ row: (U[pv.p].type === "A" ? IDXK : IDXB)[U[pv.p].cls.length] + pv.j, col: v, value: G2 });
  }
  const ri = selectors[i].rule;
  if (ri !== undefined) {
    // RULE_OFF while the rule does not fire, RULE_ON when it does, plus RANK_W · rank
    const r = rules[ri];
    Wu.add({ row: ENT + ri, col: v, value: RULE_ON - RULE_OFF + calib[ri] }); // calib ≤ 0 only applies when it fires
    bu[v] += RULE_OFF + RANK_W * r.bias;
    for (const g of r.groups) Wu.add({ row: GRP + g, col: v, value: 2 * RANK_W });
    for (const pg of r.pgroups) Wu.add({ row: PGRP + pg, col: v, value: RANK_W });
    for (const l of r.rare) Wu.add({ row: PRES + (presIdx.get(l) as number), col: v, value: RANK_W });
  }
}

const net: Net = {
  d,
  vocab: V,
  pe: { offset: PE, freqs: FREQS },
  E: E.done(),
  layers: [
    { heads: heads1, mlp: mlp({ neurons: neurons1 }) },
    {
      heads: [bagHead, lastName, csumHead, ...firstHeads],
      mlp: mergeMlp({ a: mlpGen({ neurons: nPres }), b: mlpGen({ neurons: nSteps2 }) }),
      norms: [{ src: BAG, dst: UBAG, n: NBAG }],
    },
    {
      heads: [pnameHead],
      mlp: mergeMlp({
        a: mergeMlp({
          a: mergeMlp({ a: mlp({ neurons: neurons3 }), b: mlpGen({ neurons: nPhr }) }),
          b: mlpGen({ neurons: nSame }),
        }),
        b: mergeMlp({ a: mlpGen({ neurons: nSum3 }), b: mlpGen({ neurons: nPickB }) }),
      }),
    },
    { heads: [], mlp: mergeMlp({ a: mlpGen({ neurons: nGrp }), b: mlpGen({ neurons: nHi4 }) }) },
    { heads: [], mlp: mergeMlp({ a: mlpGen({ neurons: nEnt }), b: mlpGen({ neurons: nLo5 }) }) },
    { heads: [], mlp: mlpGen({ neurons: nCell6 }) },
  ],
  Wu: Wu.done(),
  bu,
  meta: {
    // where the blocks of the residual are (for inspection)
    layout: {
      IS_BOT,
      TX,
      AV,
      BV,
      SV,
      LOV,
      HASNAME,
      SAMEP,
      NOH,
      NOH_n: NMAX + 1,
      IOH,
      IOH_n: IMAX + 1,
      HOH,
      HOH_n: HMAX + 1,
      LOH,
      LOH_n: 1000,
      IDX3: IDXK[3],
      IDX3_n: 3,
      IDR,
      IDR_n: C,
      OUTB,
      OUTB_n: V,
    },
    tokens,
    ID0,
    classes: C,
    selector: SEL,
    idClass: selectors.map((x) => x.cls),
    replies: classToks.map((toks) => renderClass({ toks })),
    data: DATA,
    builtAt: new Date().toISOString(),
  },
};
await saveNet({ net, path: OUT });
const size = Bun.file(`${OUT}.bin`).size;
log(`saved ${OUT}.{json,bin} (${(size / 1e6).toFixed(1)} MB)`);
