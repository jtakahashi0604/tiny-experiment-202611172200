"""For every reply that is an option of some pool, keep the pool whose seeded choice predicts the
training turns best (ignoring the recency filter). Writes the kept pools as {name: options}."""
import json, random, sys
sys.path.insert(0, "src")
pools = json.load(open(sys.argv[1]))
by = {}
for name, opts in pools.items():
    for o in opts: by.setdefault(o, []).append(name)
score = {}
n = 0
for path in sys.argv[3].split(","):
    for line in open(path):
        c = json.loads(line)
        if c["id"].startswith(("ec-test", "stb", "pert", "kb", "name-test")): continue
        for i, t in enumerate(c["turns"]):
            if t["bot"] not in by: continue
            n += 1
            seed = i * 7919 + sum(ord(ch) for ch in t["user"].lower())
            for name in by[t["bot"]]:
                ok = random.Random(seed).choice(pools[name]) == t["bot"]
                s = score.setdefault(name, [0, 0]); s[0] += ok; s[1] += 1
keep = {name: pools[name] for name, (ok, tot) in sorted(score.items()) if tot >= 2 and ok / tot >= 0.6}
# a reply in several kept pools: keep the best one only (ties by name, so the result does not
# depend on the order of the harvest lines or on Python's hash seed)
best = {}
for name in sorted(keep, key=lambda k: (-score[k][0] / score[k][1], k)):
    for o in keep[name]: best.setdefault(o, name)
final = {name: keep[name] for name in sorted(set(best.values()))}
json.dump(final, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
cov = sum(score[nm][1] for nm in final); okc = sum(score[nm][0] for nm in final)
print(f"turns with a pool reply {n}; kept {len(final)} pools covering {cov} with seeded accuracy {okc/cov:.3f}")
