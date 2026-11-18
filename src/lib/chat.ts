// A conversation with the network: feeds turns and decodes replies greedily.

import { type Net, Runner, argmax } from "@/lib/net.ts";
import { BOS, BOT, EOT, Tokenizer, USER, isHidden } from "@/lib/tok.ts";

export type Reply = { text: string; ids: number[]; hidden: string[] };

export class Chat {
  readonly runner: Runner;
  readonly tok: Tokenizer;
  readonly maxReply: number;

  constructor({ net, maxReply = 600 }: { net: Net; maxReply?: number }) {
    this.runner = new Runner({ net });
    this.tok = new Tokenizer({ tokens: net.meta.tokens as string[] });
    this.maxReply = maxReply;
    this.reset();
  }

  reset(): void {
    this.runner.reset();
    this.runner.step({ token: BOS });
  }

  reply({ user }: { user: string }): Reply {
    const r = this.runner;
    r.step({ token: USER });
    for (const token of this.tok.encode({ text: user })) r.step({ token });
    let logits = r.step({ token: BOT });
    const ids: number[] = [];
    const hidden: string[] = [];
    for (let i = 0; i < this.maxReply; i++) {
      const v = argmax({ values: logits });
      logits = r.step({ token: v });
      if (v === EOT) break;
      const token = this.tok.tokens[v];
      if (isHidden({ token })) hidden.push(token);
      else ids.push(v);
    }
    return { text: this.tok.decode({ ids }), ids, hidden };
  }
}
