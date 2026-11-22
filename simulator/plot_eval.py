import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reads evaluation_results.csv written by evaluate.py
with open("../results/tables/evaluation_results.csv") as f:
    rows = list(csv.DictReader(f))

names = [r["policy"] for r in rows]
means = [float(r["mean_slowdown"]) for r in rows]
lo = [float(r["ci_low"]) for r in rows]
hi = [float(r["ci_high"]) for r in rows]
caps = [int(r["cap_hits"]) for r in rows]
n = rows[0]["n_seeds"]

err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

fig, ax = plt.subplots(figsize=(9, 4.4))
bars = ax.bar(names, means, yerr=err, capsize=6, color=colors[:len(names)])
for b, m, c in zip(bars, means, caps):
    label = f"{m:.2f}" + (f"\n{c}/{n} hit cap" if c else "")
    ax.text(b.get_x() + b.get_width() / 2, m + 0.25, label, ha="center", va="bottom", fontsize=9)
ax.set_ylabel("mean job slowdown (lower is better)")
ax.set_title(f"Unseen jobsets, {n} seeds, error bars = 95% CI")
ax.set_ylim(0, max(hi) * 1.25)
plt.xticks(rotation=15, fontsize=8)
plt.tight_layout()
plt.savefig("../results/figures/evaluation.png", dpi=150)
plt.savefig("../results/figures/evaluation.pdf")
print("saved evaluation.png, evaluation.pdf")
