// eval.ts — how faithfully does the network reproduce pybot?
//
// Each conversation is replayed with the same user turns; the network answers freely
// (its own replies stay in its context). Per turn we compare with pybot's reply:
//   exact   : identical text
//   f1      : word-level F1 (lowercased words)
// (how many test replies are reachable at all is reported by `learn.ts --sim`)
//
// usage: bun run src/eval.ts [--weights weights/model] [--data a.jsonl,b.jsonl]
//                        [--split train|test|all] [--limit N] [--out results/x.json] [--show N]

import { Chat } from "@/lib/chat.ts";
import { option } from "@/util/cli.ts";
import { type Conv, isTest, readConvs } from "@/lib/data.ts";
import { loadNet } from "@/lib/net.ts";

// prettier-ignore
const weights = option({ name: "--weights", fallback: "weights/model" }),
      data    = option({ name: "--data",    fallback: "data/harvest/ec_all.jsonl" }).split(","),
      split   = option({ name: "--split",   fallback: "test" }),
      limit   = Number(option({ name: "--limit", fallback: "0" })),
      show    = Number(option({ name: "--show",  fallback: "0" })),
      outPath = option({ name: "--out",     fallback: "" });

const net = await loadNet({ path: weights });

let convs: Conv[] = (await Promise.all(data.map((path) => readConvs({ path })))).flat();
if (split !== "all") convs = convs.filter((c) => (split === "test") === isTest({ id: c.id }));
convs.sort((a, b) => (a.id < b.id ? -1 : 1));
if (limit > 0) convs = convs.slice(0, limit);

function words({ text }: { text: string }): string[] {
  return text.toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

// word-level F1 between the network's reply and pybot's
function f1({ got, want }: { got: string; want: string }): number {
  const wa = words({ text: got });
  const wb = words({ text: want });
  if (!wa.length && !wb.length) return 1;
  const cnt = new Map<string, number>();
  for (const w of wb) cnt.set(w, (cnt.get(w) ?? 0) + 1);
  let common = 0;
  for (const w of wa) {
    const n = cnt.get(w) ?? 0;
    if (n > 0) {
      common++;
      cnt.set(w, n - 1);
    }
  }
  if (!common) return 0;
  const p = common / wa.length;
  const r = common / wb.length;
  return (2 * p * r) / (p + r);
}

type Tally = { n: number; exact: number; f1: number };
const tally = (): Tally => ({ n: 0, exact: 0, f1: 0 });
const add = ({ to, ok, score }: { to: Tally; ok: boolean; score: number }) => {
  to.n++;
  if (ok) to.exact++;
  to.f1 += score;
};

const chat = new Chat({ net });
const all = tally();
const first = tally();
const byModule = new Map<string, Tally>();
const byGroup = new Map<string, Tally>();
const samples: unknown[] = [];
const t0 = performance.now();
for (const c of convs) {
  chat.reset();
  c.turns.forEach((t, i) => {
    const r = chat.reply({ user: t.user });
    const ok = r.text === t.bot;
    const score = f1({ got: r.text, want: t.bot });
    const group = c.id.split("-")[0];
    const module = t.module ?? "none";
    if (!byGroup.has(group)) byGroup.set(group, tally());
    if (!byModule.has(module)) byModule.set(module, tally());
    add({ to: all, ok, score });
    if (i === 0) add({ to: first, ok, score });
    add({ to: byGroup.get(group) as Tally, ok, score });
    add({ to: byModule.get(module) as Tally, ok, score });
    if (samples.length < show && !ok) {
      samples.push({ id: c.id, user: t.user, pybot: t.bot, net: r.text, cls: r.hidden.join(" ") });
    }
  });
}
const secs = (performance.now() - t0) / 1000;
const res = {
  weights,
  split,
  conversations: convs.length,
  turns: all.n,
  exact: all.exact / all.n,
  f1: all.f1 / all.n,
  firstTurnExact: first.n ? first.exact / first.n : 0,
  secondsPerTurn: secs / all.n,
  groups: Object.fromEntries(
    [...byGroup.entries()].map(([k, v]) => [
      k,
      { turns: v.n, exact: +(v.exact / v.n).toFixed(4), f1: +(v.f1 / v.n).toFixed(4) },
    ]),
  ),
  modules: Object.fromEntries(
    [...byModule.entries()]
      .sort((a, b) => b[1].n - a[1].n)
      .slice(0, 15)
      .map(([k, v]) => [k, `${v.exact}/${v.n}`]),
  ),
};
console.log(JSON.stringify(res, null, 2));
for (const s of samples) console.log(JSON.stringify(s));
if (outPath) await Bun.write(outPath, JSON.stringify({ ...res, samples }, null, 2));
