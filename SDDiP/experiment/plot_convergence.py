"""
plot_convergence.py  —  绘制 SDDiP 收敛曲线

读取 results/convergence_curve/convergence_curve.csv，
输出 bound 和 sim_μ 随迭代次数的变化曲线。

Usage:
    python experiment/plot_convergence.py
    python experiment/plot_convergence.py --x time   # 横轴改为时间
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--x", choices=["iter", "time"], default="iter",
                    help="x-axis: iter (iterations) or time (seconds)")
parser.add_argument("--csv", default="results/convergence_curve/convergence_curve.csv")
args = parser.parse_args()

df = pd.read_csv(args.csv)
methods = df["method"].unique()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = plt.cm.tab10.colors
x_col = "iter" if args.x == "iter" else "elapsed_s"
x_label = "Iterations" if args.x == "iter" else "Time (s)"

# ── 左图：bound 和 sim_μ 收敛曲线 ─────────────────────────────────────────
ax = axes[0]
for i, method in enumerate(methods):
    d = df[df["method"] == method].sort_values(x_col)
    c = colors[i % len(colors)]
    ax.plot(d[x_col], d["bound"],  color=c, linestyle="-",  lw=1.8, label=f"{method} bound")
    ax.plot(d[x_col], d["sim_mu"], color=c, linestyle="--", lw=1.4, label=f"{method} sim_μ")
    # 95% CI 阴影
    ax.fill_between(d[x_col],
                    d["sim_mu"] - d["sim_ci"],
                    d["sim_mu"] + d["sim_ci"],
                    color=c, alpha=0.1)

ax.set_xlabel(x_label)
ax.set_ylabel("Objective value")
ax.set_title("Convergence of SDDP bound vs. simulated policy value (sim_μ)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# ── 右图：gap% 收敛曲线 ───────────────────────────────────────────────────
ax = axes[1]
for i, method in enumerate(methods):
    d = df[df["method"] == method].sort_values(x_col)
    c = colors[i % len(colors)]
    ax.plot(d[x_col], d["gap_pct"], color=c, lw=1.8, label=method)

ax.axhline(5,  color="gray",   linestyle="--", lw=1, label="5% threshold")
ax.axhline(10, color="silver", linestyle="--", lw=1, label="10% threshold")
ax.set_xlabel(x_label)
ax.set_ylabel("gap% = (bound − sim_μ) / |sim_μ| × 100")
ax.set_title("Convergence gap% (lower is better)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.set_ylim(bottom=0)

plt.tight_layout()
out_path = Path(args.csv).parent / f"convergence_{args.x}.png"
plt.savefig(out_path, dpi=150)
print(f"Figure saved: {out_path}")
plt.show()
