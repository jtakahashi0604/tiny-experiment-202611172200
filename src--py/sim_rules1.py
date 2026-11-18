"""How well do the bare entry rules (key groups AND, no avoid word, ask) predict pybot's pick?"""
import json, re, sys, collections
ents = [json.loads(l) for l in open("data/kb/entries.jsonl")]
convs = [json.loads(l) for l in open(sys.argv[1])]
W = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def toks(s): return W.findall(s.lower().replace("’", "'"))
def has(ws, phrase):
    p = toks(phrase)
    if not p: return False
    n = len(p)
    return any(ws[i:i+n] == p for i in range(len(ws) - n + 1))
def fires(e, ws):
    if not all(any(has(ws, a) for a in g) for g in e["key"]): return False
    if any(has(ws, a) for a in e["avoid"]): return False
    if e["ask"] and not any(has(ws, a) for a in e["ask"]): return False
    return True
byreply = collections.defaultdict(list)
for i, e in enumerate(ents):
    if e["reply"]: byreply[e["reply"]].append(i)
stat = collections.Counter()
for c in convs:
    for t in c["turns"]:
        stat["turns"] += 1
        hit = [i for i, e in enumerate(ents) if e["reply"] and e["reply"] in t["bot"]]
        ws = toks(t["user"])
        fired = [i for i, e in enumerate(ents) if fires(e, ws)]
        if hit:
            stat["reply is an entry"] += 1
            if t["bot"] in byreply: stat["  reply == entry exactly"] += 1
            if any(i in fired for i in hit): stat["  its rule fires"] += 1
            if fired and fired[0] in hit and len(fired) == 1: stat["  only its rule fires"] += 1
        else:
            if fired: stat["not an entry, some rule fires"] += 1
        stat[f"module:{t['module']}"] += 1
for k, v in stat.most_common(): print(f"{v:6d}  {k}")
