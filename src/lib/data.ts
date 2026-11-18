// Harvested pybot conversations and the train/test split.

export type Turn = { user: string; bot: string; module: string | null };
export type Conv = { id: string; turns: Turn[] };

export async function readConvs({ path }: { path: string }): Promise<Conv[]> {
  const text = await Bun.file(path).text();
  return text
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Conv);
}

// FNV-1a, for a split that does not depend on file order
export function hash({ text }: { text: string }): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

// evaluation only: the official everyday-conversations test set, the name conversations with
// held-out names, SmolTalkBench, the perturbed test conversations
const TEST_ONLY = ["ec-test-", "name-test-", "stb-", "pert-", "namex-", "namey-"];

// test = TEST_ONLY and 10% (by id hash) of everything else (everyday train, knowledge probes,
// Dolly). dev = another 10% of everything else; it is training data unless --dev is given.
export function isTest({ id }: { id: string }): boolean {
  if (TEST_ONLY.some((p) => id.startsWith(p))) return true;
  if (id.startsWith("name-train-")) return false;
  return hash({ text: id }) % 10 === 0;
}

export function isDev({ id }: { id: string }): boolean {
  if (TEST_ONLY.some((p) => id.startsWith(p)) || id.startsWith("name-")) return false;
  return hash({ text: id }) % 10 === 1;
}
