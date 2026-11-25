#!/usr/bin/env bash
# Benchmark sweeps -> results/*.csv   (run from the project root after `make gpu`)
set -e
mkdir -p results
BIN=./bench_gpu
[ -x "$BIN" ] || { echo "build first: make gpu ARCH=sm_75  (T4=75, A100=80, RTX30=86, RTX40=89)"; exit 1; }

hdr=0
emit() {  # append rows, keep a single header
  if [ $hdr -eq 0 ]; then "$@" > "$OUT"; hdr=1; else "$@" | tail -n +2 >> "$OUT"; fi
}

# 1) scaling with corpus size N
OUT=results/sweep_N.csv; hdr=0
for N in 65536 131072 262144 524288 1048576 2097152; do emit $BIN --N $N --D 128 --Q 64 --K 10 --csv; done

# 2) effect of K
OUT=results/sweep_K.csv; hdr=0
for K in 1 5 10 16 32; do emit $BIN --N 1048576 --D 128 --Q 64 --K $K --csv; done

# 3) effect of embedding dimension D
OUT=results/sweep_D.csv; hdr=0
for D in 64 128 256 384 768; do emit $BIN --N 524288 --D $D --Q 64 --K 10 --csv; done

# 4) effect of query batch size Q
OUT=results/sweep_Q.csv; hdr=0
for Q in 1 8 32 128 512; do emit $BIN --N 524288 --D 128 --Q $Q --K 10 --csv; done

# 5) stage-1 blocks per query B (custom top-k tuning)
OUT=results/sweep_B.csv; hdr=0
for B in 1 4 16 64 256 1024; do emit $BIN --N 1048576 --D 128 --Q 16 --K 10 --B $B --csv; done

echo "done -> results/*.csv   (python3 scripts/plot.py to make figures)"
