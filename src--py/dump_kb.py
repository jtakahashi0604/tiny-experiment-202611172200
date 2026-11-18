"""Dump pybot's knowledge entries (id, key groups, avoid, ask, reply if a plain string)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "pybot"))
from bot import knowledge  # noqa: E402


def groups(key):
    out = []
    for g in key:
        if isinstance(g, str):
            out.append([g])
        else:
            out.append([str(x) for x in g])
    return out


n = 0
with open(sys.argv[1], "w") as f:
    for e in knowledge.entries():
        rec = {"id": e.id, "domain": getattr(e, "domain", None), "key": groups(e.key),
               "avoid": [str(a) for a in (e.avoid or ())], "ask": [str(a) for a in (e.ask or ())],
               "bias": e.bias, "guard": e.guard is not None,
               "reply": e.reply if isinstance(e.reply, str) else None}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print(n, "entries")
