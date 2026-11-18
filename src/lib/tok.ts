// Word-level tokenizer with character fallback.
//
// Text is split into "pieces": an optional single leading space followed by a word
// (letters with inner apostrophes), a number, or one other character. A piece that is in
// the vocabulary is one token; otherwise it is spelled with single-character tokens
// (the first keeps the leading space). pybot collapses whitespace, so a single optional
// leading space is enough to make detokenization exact.

export const SPECIALS = ["<user>", "<bot>", "<eot>", "<unk>", "<bos>"] as const;

// prettier-ignore
export const USER = 0, // a user turn starts
             BOT  = 1, // the bot's turn starts
             EOT  = 2, // the end of a reply
             UNK  = 3, // a character outside the vocabulary
             BOS  = 4; // the first token of every conversation

const PIECE = /\s?(?:[A-Za-z]+(?:['’][A-Za-z]+)*|[0-9]+|[^\sA-Za-z0-9])/gu;

export function normalizeText({ text }: { text: string }): string {
  return text.split(/\s+/u).filter(Boolean).join(" ");
}

export function pieces({ text }: { text: string }): string[] {
  return normalizeText({ text }).match(PIECE) ?? [];
}

// a special or ID token ("<user>", "<s:12>") is not shown
export function isHidden({ token }: { token: string }): boolean {
  return token.length > 2 && token.startsWith("<") && token.endsWith(">");
}

export class Tokenizer {
  readonly tokens: string[];
  readonly index = new Map<string, number>();

  constructor({ tokens }: { tokens: string[] }) {
    this.tokens = tokens;
    tokens.forEach((t, i) => this.index.set(t, i));
  }

  get size(): number {
    return this.tokens.length;
  }

  encode({ text }: { text: string }): number[] {
    const out: number[] = [];
    for (const piece of pieces({ text })) {
      const id = this.index.get(piece);
      if (id !== undefined) {
        out.push(id);
        continue;
      }
      for (const c of spell({ piece })) out.push(this.index.get(c) ?? UNK);
    }
    return out;
  }

  decode({ ids }: { ids: number[] }): string {
    return ids
      .map((i) => this.tokens[i])
      .filter((token) => !isHidden({ token }))
      .join("");
  }
}

// A piece spelled as single characters; a leading space stays on the first character.
export function spell({ piece }: { piece: string }): string[] {
  const chars = Array.from(piece);
  if (chars[0] === " " && chars.length > 1) return [` ${chars[1]}`, ...chars.slice(2)];
  return chars;
}

// Every single character (with and without a leading space) that appears in the corpus is
// added to the vocabulary so that any text can be spelled.
export function charTokens({ texts }: { texts: Iterable<string> }): string[] {
  const set = new Set<string>();
  for (const text of texts) {
    for (const c of Array.from(normalizeText({ text }))) {
      if (c === " ") continue;
      set.add(c);
      set.add(` ${c}`);
    }
  }
  return [...set].sort();
}
