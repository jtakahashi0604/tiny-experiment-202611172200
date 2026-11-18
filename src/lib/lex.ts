// Lexeme of a token: what the bag of words counts. Case and the leading space are ignored,
// curly apostrophes are straightened. Specials and ID tokens have no lexeme.

import { isHidden } from "@/lib/tok.ts";

export function lexeme({ token }: { token: string }): string | null {
  if (isHidden({ token })) return null;
  const t = token.trim().toLowerCase().replace(/’/gu, "'");
  return t || null;
}

// A crude suffix stripper, so that "sleeping", "sleeps" and "sleep" count as one word.
export function stem({ word }: { word: string }): string {
  if (!/^[a-z]+(?:'[a-z]+)?$/u.test(word)) return word;
  let s = word.replace(/'s$/u, "");
  if (s.length > 4 && s.endsWith("ies")) s = `${s.slice(0, -3)}y`;
  else if (s.length > 4 && /(?:ches|shes|sses|xes)$/u.test(s)) s = s.slice(0, -2);
  else if (s.length > 3 && s.endsWith("s") && !/(?:ss|us|is)$/u.test(s)) s = s.slice(0, -1);
  if (s.length > 5 && s.endsWith("ing")) s = s.slice(0, -3);
  else if (s.length > 4 && s.endsWith("ed")) s = s.slice(0, -2);
  if (s.length > 4 && /([b-df-hj-np-tv-z])\1$/u.test(s) && !/(?:ll|ss|zz)$/u.test(s)) s = s.slice(0, -1);
  return s;
}
