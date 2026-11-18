"""Build the user-turn inputs (data/convs/*.jsonl) from the downloaded sources (data/raw/).

Each line is {"id": ..., "users": [...]}; pybot answers them in harvest.py. Every random choice is
seeded, so the files come out the same on every run.

usage: uv run python make_convs.py <data dir>
"""
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq

D = sys.argv[1]
RAW = os.path.join(D, "raw")
CONVS = os.path.join(D, "convs")
os.makedirs(CONVS, exist_ok=True)


def write(name, rows, ensure_ascii=True):
    with open(os.path.join(CONVS, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")


def read(name):
    return [json.loads(line) for line in open(os.path.join(CONVS, name))]


def fnv1a(text):
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def is_test(conv_id):  # the same split as src/lib/data.ts (for the ids used here)
    if conv_id.startswith("ec-test-"):
        return True
    return fnv1a(conv_id) % 10 == 0


# everyday-conversations (smoltalk): every user turn of every conversation
ec = {}
for split in ("train", "test"):
    rows = pq.read_table(os.path.join(RAW, f"ec_{split}.parquet")).to_pylist()
    ec[split] = [{"id": f"ec-{split}-{i}", "users": [m["content"] for m in r["messages"] if m["role"] == "user"]}
                 for i, r in enumerate(rows)]
    write(f"ec_{split}.jsonl", ec[split])
write("ec_all.jsonl", ec["test"] + ec["train"])

# SmolTalkBench: single-turn prompts (evaluation only)
stb = pq.read_table(os.path.join(RAW, "stb_gold.parquet")).to_pylist()
write("stb.jsonl", [{"id": f"stb-{i}", "users": [r["instruction"]]} for i, r in enumerate(stb)])

# Dolly: short questions without context
dolly = [json.loads(line) for line in open(os.path.join(RAW, "dolly.jsonl"))]
keep1 = [r for r in dolly if not r["context"] and len(r["instruction"]) < 140
         and r["category"] in ("open_qa", "brainstorming", "general_qa", "creative_writing")]
random.Random(0).shuffle(keep1)
dolly1 = [{"id": f"dolly1-{i}", "users": [r["instruction"]]} for i, r in enumerate(keep1[:6000])]
write("dolly1.jsonl", dolly1, ensure_ascii=False)
used = {c["users"][0] for c in dolly1}
keep2 = [r["instruction"] for r in dolly if not r["context"] and len(r["instruction"]) < 200
         and r["instruction"] not in used]
random.Random(1).shuffle(keep2)
write("dolly2.jsonl", [{"id": f"dolly2-{i}", "users": [u]} for i, u in enumerate(keep2)])

# the everyday test conversations, for make_perturbed.py
lines = open(os.path.join(CONVS, "ec_all.jsonl")).read().split("\n")
with open(os.path.join(CONVS, "ec_testonly.jsonl"), "w") as f:
    f.write("\n".join(l for l in lines if l and is_test(json.loads(l)["id"])) + "\n")

# name conversations with names outside pybot's lexicon (evaluation only)
INTRO = ["hi, i'm {n}", "Hello! My name is {n}.", "I'm {n}.", "hey, call me {n}", "hi there, I am {n}", "my name is {n}"]
ASK = ["what's my name?", "what is my name?", "do you remember my name?", "who am i?"]
RENAME = ["actually my name is {n}", "call me {n} instead"]
pyb = set(json.load(open(os.path.join(D, "kb", "names.json"))))
fill = [c["users"] for c in read("ec_train.jsonl")]

# namex: names of first-names.txt
extra = [l.strip() for l in open(os.path.join(RAW, "first-names.txt"))
         if re.fullmatch(r"[A-Za-z]{3,}", l.strip()) and l.strip().lower() not in pyb]
rng = random.Random(5)
rng.shuffle(extra)
rows = []
for i in range(120):
    n = extra[i]
    turns = [rng.choice(INTRO).format(n=n)]
    if rng.random() < 0.6:
        turns.append(rng.choice(rng.choice(fill)))
    if rng.random() < 0.3:
        turns.append(rng.choice(RENAME).format(n=extra[200 + i]))
    turns.append(rng.choice(ASK))
    rows.append({"id": f"namex-test-{i}", "users": turns})
write("namex.jsonl", rows)

# namey: names only in names_all.txt
first = {l.strip().lower() for l in open(os.path.join(RAW, "first-names.txt"))}
allx = [l.strip().lstrip("﻿") for l in open(os.path.join(RAW, "names_all.txt"))]
cand = [n for n in allx if re.fullmatch(r"[A-Za-z]{3,}", n) and n.lower() not in pyb and n.lower() not in first]
rng = random.Random(7)
rng.shuffle(cand)
pick = ["Jun", "Wei", "Yuki", "Hiroshi", "Priya", "Kenji", "Aiko", "Mohammed"] + cand[:200]
rows = []
for i in range(120):
    n = pick[i]
    turns = [rng.choice(INTRO).format(n=n)]
    if rng.random() < 0.6:
        turns.append(rng.choice(rng.choice(fill)))
    if rng.random() < 0.3:
        turns.append(rng.choice(RENAME).format(n=pick[150 + i % 50]))
    turns.append(rng.choice(ASK))
    rows.append({"id": f"namey-test-{i}", "users": turns})
write("namey.jsonl", rows)
print("convs written to", CONVS)
