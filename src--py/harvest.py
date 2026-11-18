"""Run pybot (the oracle) over conversations and record every turn.

input : JSONL, one conversation per line: {"id": str, "users": [str, ...]}
output: JSONL, one conversation per line: {"id", "turns": [{"user", "bot", "module"}]}

pybot picks some wording with `random`; we call random.seed(0) before every turn so the
reply is a deterministic function of (user text, memory).

usage: uv run python harvest.py IN.jsonl OUT.jsonl [--procs 8]
"""
import json
import os
import random
import sys
import time
from multiprocessing import Pool

PYBOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "pybot")
_runtime = None


_picks = []


def _init():
    """Import pybot with compose.pick wrapped, so every pick (options, seed text, turn count,
    choice) is recorded. The wrapper is installed before the modules import `pick`."""
    global _runtime
    sys.path.insert(0, PYBOT)
    import bot.compose as compose  # noqa: PLC0415

    original = compose.pick

    def pick(options, memory, seed_text=""):
        out = original(options, memory, seed_text)
        try:
            turns = memory.get("turns") or []
            _picks.append({"options": [str(o) for o in options], "seed_text": str(seed_text),
                           "turns": len(turns), "out": str(out)})
        except Exception:  # noqa: BLE001
            pass
        return out

    compose.pick = pick
    from bot import runtime  # noqa: PLC0415

    _runtime = runtime


def run_conv(conv):
    memory = {}
    turns = []
    for text in conv["users"]:
        random.seed(0)
        _picks.clear()
        try:
            out = _runtime.turn(text, memory)
            reply = " ".join(str(out["reply"]).split()) or "Hello!"
            module = out["module"]
        except Exception:  # noqa: BLE001
            reply, module = "Hello!", None
        # keep the picks whose choice is (part of) the reply
        picks = [p for p in _picks if p["out"] and " ".join(p["out"].split()) in reply]
        turns.append({"user": text, "bot": reply, "module": module, "picks": picks})
    return {"id": conv["id"], "turns": turns}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 8
    convs = [json.loads(line) for line in open(src)]
    done = set()
    if os.path.exists(dst):
        done = {json.loads(line)["id"] for line in open(dst)}
    todo = [c for c in convs if c["id"] not in done]
    print(f"{len(convs)} conversations, {len(todo)} to do", flush=True)
    t0 = time.time()
    n_turns = 0
    with Pool(procs, initializer=_init) as pool, open(dst, "a") as f:
        for i, res in enumerate(pool.imap(run_conv, todo, chunksize=4)):  # in input order
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_turns += len(res["turns"])
            if (i + 1) % 200 == 0:
                f.flush()
                dt = time.time() - t0
                print(f"{i + 1}/{len(todo)} convs, {n_turns} turns, {n_turns / dt:.1f} turns/s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
