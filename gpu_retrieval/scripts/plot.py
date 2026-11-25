#!/usr/bin/env python3
"""Plot results/sweep_*.csv produced by scripts/sweep.sh into results/*.png"""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(name):
    with open(f"results/{name}.csv") as f:
        return list(csv.DictReader(f))

def total_by_impl(rows, xkey):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["impl"]].append((float(r[xkey]), float(r["total_ms"])))
    return d

for name, xkey, xlabel in [("sweep_N", "N", "corpus size N"), ("sweep_K", "K", "k"),
                           ("sweep_D", "D", "dimension D"), ("sweep_Q", "Q", "queries Q")]:
    if not os.path.exists(f"results/{name}.csv"):
        continue
    d = total_by_impl(load(name), xkey)
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl, pts in d.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl)
    ax.set_xlabel(xlabel); ax.set_ylabel("total time (ms)"); ax.set_yscale("log")
    if xkey in ("N", "Q"): ax.set_xscale("log", base=2)
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"results/{name}.png", dpi=150); plt.close(fig)

# stage breakdown (stacked) for the largest N
if os.path.exists("results/sweep_N.csv"):
    rows = load("sweep_N")
    nmax = max(int(r["N"]) for r in rows)
    rows = [r for r in rows if int(r["N"]) == nmax and r["impl"].startswith("gpu")]
    stages = ["h2d_ms", "preprocess_ms", "score_ms", "topk_ms", "d2h_ms"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = [0.0] * len(rows)
    for s in stages:
        vals = [float(r[s]) for r in rows]
        ax.bar([r["impl"] for r in rows], vals, bottom=bottom, label=s[:-3])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("ms"); ax.set_title(f"stage breakdown, N={nmax}"); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig("results/stage_breakdown.png", dpi=150, bbox_inches="tight")

if os.path.exists("results/sweep_B.csv"):
    rows = [r for r in load("sweep_B") if r["impl"] == "gpu_custom_topk"]
    print("sweep_B (custom top-k) topk_ms by B: ", [(r["B"], r["topk_ms"]) for r in rows])

# ---- extra: speedup vs N, and a Markdown summary of every sweep -------------------------
def trio(name):
    rows = load(name)
    return [(rows[i], rows[i + 1], rows[i + 2]) for i in range(0, len(rows), 3)]

def comp(r):
    return float(r["preprocess_ms"]) + float(r["score_ms"]) + float(r["topk_ms"])

if os.path.exists("results/sweep_N.csv"):
    T = trio("sweep_N")
    xs = [int(c["N"]) for c, g, t in T]
    e2e = [float(c["total_ms"]) / float(g["total_ms"]) for c, g, t in T]
    co = [float(c["total_ms"]) / comp(g) for c, g, t in T]
    tk = [float(t["topk_ms"]) / float(g["topk_ms"]) for c, g, t in T]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, e2e, marker="o", label="end-to-end vs CPU (copies included)")
    ax.plot(xs, co, marker="o", label="GPU compute only vs CPU")
    ax.plot(xs, tk, marker="o", label="custom top-k vs Thrust sort")
    ax.set_xscale("log", base=2); ax.set_xlabel("corpus size N"); ax.set_ylabel("speedup (x)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("results/speedup_N.png", dpi=150, bbox_inches="tight"); plt.close(fig)

with open("results/summary.md", "w") as out:
    def table(title, name, xkey):
        out.write(f"### {title}\n\n| {xkey} | CPU 20 thr (ms) | GPU custom (ms) | GPU Thrust (ms) | speedup custom | custom top-k (ms) | Thrust top-k (ms) |\n|---:|---:|---:|---:|---:|---:|---:|\n")
        for c, g, t in trio(name):
            out.write(f"| {int(c[xkey]):,} | {float(c['total_ms']):.1f} | {float(g['total_ms']):.1f} | {float(t['total_ms']):.1f} | "
                      f"{float(c['total_ms'])/float(g['total_ms']):.1f}x | {float(g['topk_ms']):.2f} | {float(t['topk_ms']):.1f} |\n")
        out.write("\n")
    for title, name, xkey in [("Corpus size N (D=128, Q=64, K=10)", "sweep_N", "N"), ("K (N=1,048,576, D=128, Q=64)", "sweep_K", "K"),
                              ("Queries Q (N=524,288, D=128, K=10)", "sweep_Q", "Q"), ("Dimension D (N=524,288, Q=64, K=10)", "sweep_D", "D")]:
        if os.path.exists(f"results/{name}.csv"):
            table(title, name, xkey)
print("figures and results/summary.md written")
