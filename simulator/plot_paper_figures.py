"""Two-panel figures for the term paper, made from the saved results (no simulation is run).
Run from the simulator/ folder:  python plot_paper_figures.py
Reads  ../results/tables/evaluation_results.csv, load_sweep.csv, load_sweep_high.csv
       ../results/logs/history.npy, history_small.npy
Writes ../results/figures/paper_fig_eval.pdf (bars + training curve)
       ../results/figures/paper_fig_load.pdf (slowdown against load, two arrival models)"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results/"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def read(path):
    with open(R + path, newline="") as f:
        return list(csv.DictReader(f))


# ---------- figure 1: (a) evaluation bars, (b) training curve ----------
rows = read("tables/evaluation_results.csv")
short = {"Random": "Random", "SJF": "SJF", "Agent (argmax)": "Agent\nargmax",
         "Agent (sample)": "Agent\nsample", "Agent (argmax, masked)": "Agent\nargmax\nmasked",
         "Agent (sample, masked)": "Agent\nsample\nmasked"}
names = [short[r["policy"]] for r in rows]
mean = [float(r["mean_slowdown"]) for r in rows]
lo = [float(r["ci_low"]) for r in rows]
hi = [float(r["ci_high"]) for r in rows]
cap = [int(r["cap_hits"]) for r in rows]
colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
ax1.bar(range(len(rows)), mean, yerr=[[m - l for m, l in zip(mean, lo)], [h - m for m, h in zip(mean, hi)]],
        capsize=3, color=colors)
for i, (m, c) in enumerate(zip(mean, cap)):
    ax1.text(i, m + 0.35, f"{m:.2f}" + (f"\n{c}/100 cap" if c else ""), ha="center", va="bottom", fontsize=6.5)
ax1.set_xticks(range(len(rows)))
ax1.set_xticklabels(names)
ax1.set_ylabel("mean job slowdown")
ax1.set_ylim(0, max(hi) * 1.22)
ax1.set_title("(a) 100 unseen jobsets, load p = 0.7, 95% CI")

h_final = np.load(R + "logs/history.npy")
h_small = np.load(R + "logs/history_small.npy")
ax2.plot(h_small[:, 1], color="tab:orange", lw=1, label="10 jobsets x 10 episodes, 100 iterations")
ax2.plot(h_final[:, 1], color="tab:green", lw=1, label="20 jobsets x 10 episodes, 300 iterations")
ax2.set_xlabel("training iteration")
ax2.set_ylabel("mean slowdown (training jobsets)")
ax2.set_title("(b) learning curve")
ax2.legend(loc="upper right")
fig.tight_layout()
fig.savefig(R + "figures/paper_fig_eval.pdf")
plt.close(fig)

# ---------- figure 2: slowdown against load ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, path, xcol, title in [(ax1, "tables/load_sweep.csv", "load_pct", "(a) one Bernoulli arrival per step at most"),
                              (ax2, "tables/load_sweep_high.csv", "load_pct", "(b) Poisson arrivals, up to 190% load")]:
    data = read(path)
    for name, col in [("Random", "tab:gray"), ("SJF", "tab:blue"), ("Agent (sample)", "tab:green")]:
        r = [x for x in data if x["policy"] == name]
        x = [float(v[xcol]) for v in r]
        m = np.array([float(v["mean_slowdown"]) for v in r])
        l = np.array([float(v["ci_low"]) for v in r])
        h = np.array([float(v["ci_high"]) for v in r])
        ax.errorbar(x, m, yerr=[m - l, h - m], marker="o", ms=3, lw=1, capsize=2, color=col, label=name)
    ax.set_xlabel("average cluster load (%)")
    ax.set_ylabel("mean job slowdown")
    ax.set_title(title)
    ax.legend(loc="center left" if ax is ax1 else "upper left")
fig.tight_layout()
fig.savefig(R + "figures/paper_fig_load.pdf")
print("saved paper_fig_eval.pdf and paper_fig_load.pdf")
