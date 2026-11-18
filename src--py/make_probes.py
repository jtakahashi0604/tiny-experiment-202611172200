"""Prompts made from pybot's own knowledge base: each entry's key words put into templates.

One conversation per prompt (single turn, fresh memory). Two prompts per entry, each with a
random choice of one alternative per key group (seeded).
"""
import json
import random
import sys

TEMPLATES = [
    "{k}",
    "how do i deal with {k}?",
    "what should i know about {k}?",
    "any tips on {k}?",
    "can you help me with {k}",
    "i have a question about {k}",
    "what do you think about {k}?",
    "tell me about {k}",
]
rng = random.Random(0)
out = open(sys.argv[2], "w")
n = 0
for line in open(sys.argv[1]):
    e = json.loads(line)
    if not e["key"]:
        continue
    for v in range(2):
        words = [rng.choice(g) for g in e["key"]]
        if e["ask"]:
            words.insert(0, rng.choice(e["ask"]))
        k = " ".join(words)
        t = rng.choice(TEMPLATES)
        out.write(json.dumps({"id": f"kb-{e['id']}-{v}", "users": [t.format(k=k)]}) + "\n")
        n += 1
print(n, "probes")
