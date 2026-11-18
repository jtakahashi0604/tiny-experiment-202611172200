"""Perturbed copies of the test conversations: lowercase, no final punctuation, and a typo
(two adjacent letters swapped in one word of 5+ letters) in half of the turns."""
import json
import random
import re
import sys

rng = random.Random(2)
out = open(sys.argv[2], "w")
for line in open(sys.argv[1]):
    c = json.loads(line)
    users = []
    for u in c["users"]:
        u = u.lower().rstrip(" .!?")
        if rng.random() < 0.5:
            words = u.split(" ")
            long = [i for i, w in enumerate(words) if re.fullmatch(r"[a-z]{5,}", w)]
            if long:
                i = rng.choice(long)
                w = words[i]
                j = rng.randrange(1, len(w) - 2)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
                u = " ".join(words)
        users.append(u)
    out.write(json.dumps({"id": f"pert-{c['id']}", "users": users}) + "\n")
