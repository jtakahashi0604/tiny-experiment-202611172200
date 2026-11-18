"""Dump the reply pools of pybot's bot/compose package: {module.NAME: [options...]}.

A fallback reply is chosen from its pool by fallback_chain._choose:
random.Random(len(turns) * 7919 + sum(ord(c) for c in line.lower())).choice(fresh).
Every module-level tuple of 2..6 strings is a candidate pool; which one a reply really came
from is decided later on the training turns (learn.ts keeps the pool that predicts best)."""
import importlib
import json
import os
import pkgutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "pybot"))
import bot.compose as compose  # noqa: E402

pools = {}
for item in pkgutil.iter_modules(compose.__path__):
    try:
        mod = importlib.import_module(f"bot.compose.{item.name}")
    except Exception:  # noqa: BLE001
        continue
    for name in sorted(dir(mod)):
        v = getattr(mod, name)
        if isinstance(v, tuple) and 2 <= len(v) <= 6 and all(isinstance(x, str) and len(x) > 15 for x in v):
            opts = [" ".join(x.split()) for x in v]
            if opts not in pools.values():
                pools[f"{item.name}.{name}"] = opts
json.dump(pools, open(sys.argv[1], "w"), ensure_ascii=False, indent=0)
print(len(pools), "pools")
