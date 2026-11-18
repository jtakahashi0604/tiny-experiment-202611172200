"""pybot's own reading of each user line (bot.understanding.understand), for an upper-bound
experiment: how much would its 'asks'/'topic'/'wants'/'kind' help the selection?
Writes {user text: [features]} for every user turn of the given harvest files (fresh memory)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "pybot"))
from bot.understanding import understand  # noqa: E402
out = {}
for path in sys.argv[2].split(","):
    for line in open(path):
        for t in json.loads(line)["turns"]:
            u = t["user"]
            if u in out: continue
            try:
                r = understand(u, {})
            except Exception:  # noqa: BLE001
                r = {}
            f = [f"kind:{r.get('kind')}", f"wants:{r.get('wants')}", f"topic:{r.get('topic')}"] + [f"ask:{a}" for a in (r.get("asks") or [])]
            out[u] = f
json.dump(out, open(sys.argv[1], "w"))
print(len(out))
