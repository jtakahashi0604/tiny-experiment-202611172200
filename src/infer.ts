// infer.ts — chat with the constructed network.
//
// usage: bun run src/infer.ts [--weights weights/model] [--show-id] [-p "one message"]
//   /reset starts a new conversation, /exit quits.

import { Chat } from "@/lib/chat.ts";
import { flag, option } from "@/util/cli.ts";
import { loadNet } from "@/lib/net.ts";

// prettier-ignore
const showId  = flag({ name: "--show-id" }),
      weights = option({ name: "--weights", fallback: "weights/model" }),
      one     = option({ name: "-p",        fallback: "" });

const net = await loadNet({ path: weights });
const chat = new Chat({ net });

function say({ text }: { text: string }): void {
  const t0 = performance.now();
  const r = chat.reply({ user: text });
  const extra = showId ? `  [${r.hidden.join(" ")} ${((performance.now() - t0) / 1000).toFixed(2)}s]` : "";
  console.log(`bot: ${r.text}${extra}`);
}

if (one) {
  say({ text: one });
  process.exit(0);
}

console.log("A Transformer whose weights were constructed from pybot. /reset, /exit");
process.stdout.write("user: ");
for await (const line of console) {
  const text = line.trim();
  if (text === "/exit") break;
  if (text === "/reset") {
    chat.reset();
    console.log("(new conversation)");
  } else if (text) {
    say({ text });
  }
  process.stdout.write("user: ");
}
