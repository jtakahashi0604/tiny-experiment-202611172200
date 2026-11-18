"""Synthetic conversations about the user's name (pybot answers them; we only write the user side).

Names come from pybot's lexicon (COMMON_NAMES). Even-indexed names go to train conversations,
odd-indexed ones to test, so the test checks copying a name never seen in a training reply.
"""
import json
import random
import sys

names = [n.capitalize() for n in json.load(open(sys.argv[1]))]
fill = [json.loads(l)["users"] for l in open(sys.argv[2])]
rng = random.Random(1)
INTRO = ["hi, i'm {n}", "Hello! My name is {n}.", "I'm {n}.", "hey, call me {n}", "hi there, I am {n}",
         "my name is {n}", "Hi, I'm {n}. How are you?", "hello, i'm {n}"]
ASK = ["what's my name?", "what is my name?", "do you remember my name?", "who am i?",
       "what did i say my name was?", "what's my name"]
RENAME = ["actually my name is {n}", "sorry, I'm {n}, not {o}", "call me {n} instead"]
out = open(sys.argv[3], "w")
k = 0
for split, pool in (("train", names[0::2]), ("test", names[1::2])):
    for i in range(400 if split == "train" else 120):
        n = rng.choice(pool)
        turns = [rng.choice(INTRO).format(n=n)]
        if rng.random() < 0.7:
            turns.append(rng.choice(rng.choice(fill)))
        if rng.random() < 0.3:
            m = rng.choice([x for x in pool if x != n])
            turns.append(rng.choice(RENAME).format(n=m, o=n))
        turns.append(rng.choice(ASK))
        if rng.random() < 0.2:
            turns.insert(0, rng.choice(ASK))  # asked before telling
        out.write(json.dumps({"id": f"name-{split}-{i}", "users": turns}) + "\n")
        k += 1
print(k, "conversations")
