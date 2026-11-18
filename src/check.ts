// check.ts — recital check: force each reply ID and verify the network spells the reply.
//
// usage: bun run src/check.ts [--weights weights/model] [--limit N]

import { option } from "@/util/cli.ts";
import { Runner, argmax, loadNet } from "@/lib/net.ts";
import { BOS, BOT, EOT, Tokenizer, USER } from "@/lib/tok.ts";

const net = await loadNet({ path: option({ name: "--weights", fallback: "weights/model" }) });
const tok = new Tokenizer({ tokens: net.meta.tokens as string[] });

// prettier-ignore
const replies = net.meta.replies as string[],
      ID0     = net.meta.ID0     as number,
      idClass = net.meta.idClass as number[],
      limit   = Number(option({ name: "--limit", fallback: "0" })) || replies.length;

// the first ID token of every class
const selOf = new Map<number, number>();
idClass.forEach((c, i) => {
  if (!selOf.has(c)) selOf.set(c, i);
});

// the class's text as the network should say it after "my name is Sam"
function expected({ cls }: { cls: number }): string {
  return replies[cls].replaceAll("{name}", "Sam").replaceAll("{pname}", "Sam");
}

// force the class's ID token and decode greedily until <eot>
function recite({ runner, cls }: { runner: Runner; cls: number }): string {
  runner.reset();
  for (const token of [BOS, USER, ...tok.encode({ text: "my name is Sam" }), BOT]) runner.step({ token });
  let logits = runner.step({ token: ID0 + (selOf.get(cls) as number) });
  const out: number[] = [];
  for (let i = 0; i < 1000; i++) {
    const v = argmax({ values: logits });
    if (v === EOT) break;
    out.push(v);
    logits = runner.step({ token: v });
  }
  return tok.decode({ ids: out });
}

const runner = new Runner({ net });
let ok = 0;
let noSel = 0; // classes no ID token leads to (unused pool options): they are never said
const bad: string[] = [];
const t0 = performance.now();
for (let cls = 0; cls < limit; cls++) {
  if (!selOf.has(cls)) {
    noSel++;
    continue;
  }
  const text = recite({ runner, cls });
  if (text === expected({ cls })) ok++;
  else if (bad.length < 5) bad.push(`#${cls}\n  want: ${replies[cls].slice(0, 200)}\n  got:  ${text.slice(0, 200)}`);
}
const secs = ((performance.now() - t0) / 1000).toFixed(1);
console.log(`recital exact: ${ok}/${limit - noSel} (classes without an ID token: ${noSel}) (${secs}s)`);
for (const b of bad) console.log(b);
