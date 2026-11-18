"""Export two of pybot's word lists: its first names (COMMON_NAMES) and the asking-frame words of
its knowledge matcher (ASK_FRAME, WHEN_WORDS, A_LENGTH_WORD, A_STAND_IN).

usage: uv run python export_lexicon.py <names.json> <frame_words.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "pybot"))
from bot.knowledge import lookup  # noqa: E402
from bot.understanding import lexicon  # noqa: E402

json.dump(sorted(lexicon.COMMON_NAMES), open(sys.argv[1], "w"))
words = set()
for name in ("ASK_FRAME", "WHEN_WORDS", "A_LENGTH_WORD", "A_STAND_IN"):
    words |= set(getattr(lookup, name))
json.dump(sorted(words), open(sys.argv[2], "w"))
print(len(lexicon.COMMON_NAMES), "names,", len(words), "frame words")
