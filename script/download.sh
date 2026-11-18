#!/bin/sh
# Download the sources and build data/: the user turns, pybot's knowledge base and pybot's replies.
#
# usage: sh script/download.sh [--procs N]
#
# Every step skips what already exists, so an interrupted run can be started again. Asking pybot
# (harvest.py) is the slow part: about 4 hours with 7 processes on an M1.

set -eu
cd "$(dirname "$0")/.."

PROCS=7
if [ "${1:-}" = "--procs" ]; then PROCS=$2; fi

# pybot's knowledge base lists some of its words from sets; a fixed hash seed keeps their order
export PYTHONHASHSEED=0

D=data
mkdir -p "$D/raw" "$D/kb" "$D/convs" "$D/harvest"

# ---------- sources ----------
HF=https://huggingface.co
fetch() { # url file sha256
  if [ ! -s "$2" ]; then
    echo "download $2"
    curl -fsSL -o "$2" "$1"
  fi
  if [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" != "$3" ]; then
    echo "warning: $2 differs from the version the weights were built from" >&2
  fi
}
fetch "$HF/api/datasets/HuggingFaceTB/smoltalk/parquet/everyday-conversations/train/0.parquet" "$D/raw/ec_train.parquet" 32eec6e30ae3110302a70285a272088bd4f228026935dd8cb3efab90f9efb0f0
fetch "$HF/api/datasets/HuggingFaceTB/smoltalk/parquet/everyday-conversations/test/0.parquet" "$D/raw/ec_test.parquet" b707e4eea46c39c13103ec7d72b315b98923c42af8c343cd4dceef81ad423f32
fetch "$HF/api/datasets/HuggingFaceSwarms/SmolTalkBench/parquet/gold/test/0.parquet" "$D/raw/stb_gold.parquet" 8ae9449c58ed49e6446265ed0aa26ab7da95913c8704e2e59aa70f1524181529
fetch "$HF/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl" "$D/raw/dolly.jsonl" 2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec
fetch "https://raw.githubusercontent.com/dominictarr/random-name/master/first-names.txt" "$D/raw/first-names.txt" 17d20fdf5281781bf2f2f73119e39d040f5a6da40ffa93d2cab06027df5fb37d
fetch "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/first%20names/all.txt" "$D/raw/names_all.txt" 11ded6df3688a6473e4f10398c7c05340bb08e3dc9ac4b230dc77b248f59b124

cd src--py
py() { uv run python "$@"; }

# ---------- pybot's knowledge base ----------
[ -s ../$D/kb/names.json ] || py export_lexicon.py ../$D/kb/names.json ../$D/kb/frame_words.json
[ -s ../$D/kb/entries.jsonl ] || py dump_kb.py ../$D/kb/entries.jsonl
[ -s ../$D/kb/pools_all.json ] || py dump_pools.py ../$D/kb/pools_all.json
[ -s ../$D/kb/seed_table.bin ] || py seed_table.py ../$D/kb/seed_table.bin 121000

# ---------- user turns ----------
[ -s ../$D/convs/namey.jsonl ] || py make_convs.py ../$D
[ -s ../$D/convs/names.jsonl ] || py make_names.py ../$D/kb/names.json ../$D/convs/ec_train.jsonl ../$D/convs/names.jsonl
[ -s ../$D/convs/kb_probes.jsonl ] || py make_probes.py ../$D/kb/entries.jsonl ../$D/convs/kb_probes.jsonl
[ -s ../$D/convs/pert.jsonl ] || py make_perturbed.py ../$D/convs/ec_testonly.jsonl ../$D/convs/pert.jsonl

# ---------- pybot's replies ----------
for n in ec_all names dolly1 dolly2 kb_probes stb pert namex namey; do
  py harvest.py ../$D/convs/$n.jsonl ../$D/harvest/$n.jsonl --procs "$PROCS"
done

# ---------- the pools pybot's seeded choices draw from ----------
py pool_fit.py ../$D/kb/pools_all.json ../$D/kb/pools.json \
  ../$D/harvest/ec_all.jsonl,../$D/harvest/names.jsonl,../$D/harvest/dolly1.jsonl,../$D/harvest/dolly2.jsonl

echo "data/ is ready"
