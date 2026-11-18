"""Upper-bound check: pick the entry whose bare rule fires (most key groups, then bias)."""
import json, re, sys, collections
sys.path.insert(0, "src--py")
ents = [json.loads(l) for l in open("data/kb/entries.jsonl")]
W = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def toks(s): return W.findall(s.lower().replace("’", "'"))
def has(ws, phrase, wset):
    p = toks(phrase)
    if not p: return False
    if len(p) == 1: return p[0] in wset
    n = len(p)
    return any(ws[i:i+n] == p for i in range(len(ws) - n + 1))
def best(ws):
    wset = set(ws); b = None
    for e in ents:
        if not e["reply"]: continue
        if not all(any(has(ws, a, wset) for a in g) for g in e["key"]): continue
        if any(has(ws, a, wset) for a in e["avoid"]): continue
        if e["ask"] and not any(has(ws, a, wset) for a in e["ask"]): continue
        k = (len(e["key"]), e["bias"])
        if b is None or k > b[0]: b = (k, e)
    return b[1] if b else None
st = collections.Counter()
for line in open(sys.argv[1]):
    c = json.loads(line)
    if not (c["id"].startswith("ec-test") or c["id"].startswith("ec-train")): continue
    for t in c["turns"]:
        e = best(toks(t["user"]))
        isent = any(x["reply"] == t["bot"] for x in ents if x["reply"])
        st["turns"] += 1
        if isent: st["pybot used an entry"] += 1
        if e:
            st["a rule fires"] += 1
            if e["reply"] == t["bot"]: st["  rule pick == pybot"] += 1
            elif isent: st["  rule pick is another entry"] += 1
            else: st["  pybot did not use an entry"] += 1
for k, v in st.items(): print(f"{v:6d} {k}")
