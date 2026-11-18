"""Python's random.Random(seed).choice over a pool of k options, as a table:
byte [seed * 5 + (k - 2)] = the index chosen, for seed in [0, N) and k in 2..6."""
import random
import sys

N = int(sys.argv[2])
out = bytearray(N * 5)
for seed in range(N):
    for k in range(2, 7):
        out[seed * 5 + k - 2] = random.Random(seed).choice(range(k))
open(sys.argv[1], "wb").write(bytes(out))
print(N, "seeds")
