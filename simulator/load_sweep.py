"""Slowdown at different loads (paper Figure 4 style). Load is about 1.025 * p (arrival probability).
Agent was trained only at p = 0.7. Writes results/tables/load_sweep.csv and results/figures/load_sweep.png"""
import csv
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from policy import PolicyNet
from evaluate import random_policy, sjf_policy, make_agent_policy, evaluate, mean_ci

P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = range(1000, 1100)                              # unseen, same for all policies

if __name__ == "__main__":
    torch.manual_seed(0)
    net = PolicyNet()
    net.load_state_dict(torch.load("../results/models/policy.pt"))
    net.eval()
    policies = [("Random", random_policy), ("SJF", sjf_policy),
                ("Agent (sample)", make_agent_policy(net, "sample"))]

    rows = []
    for p in P_VALUES:
        for name, pol in policies:
            sd, rw, cap = evaluate(pol, SEEDS, p_arrival=p)      # evaluate.py
            m, h = mean_ci(sd)
            rows.append([p, round(1.025 * p * 100), name, round(m, 3), round(m - h, 3), round(m + h, 3), cap])
            print(f"p={p:.1f} load~{1.025 * p * 100:3.0f}%  {name:15s} {m:7.3f} [{m - h:7.3f}, {m + h:7.3f}] cap hits {cap}", flush=True)

    with open("../results/tables/load_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "load_pct", "policy", "mean_slowdown", "ci_low", "ci_high", "cap_hits"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, _ in policies:
        r = [x for x in rows if x[2] == name]
        load = [x[1] for x in r]
        mean = [x[3] for x in r]
        ax.errorbar(load, mean, yerr=[[x[3] - x[4] for x in r], [x[5] - x[3] for x in r]],
                    marker="o", capsize=3, label=name)
    ax.set_xlabel("average cluster load (%), about 1.025 x p x 100")
    ax.set_ylabel("mean job slowdown (lower is better)")
    ax.set_title(f"Slowdown at different loads, {len(SEEDS)} unseen seeds, 95% CI")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../results/figures/load_sweep.png", dpi=150)
    plt.savefig("../results/figures/load_sweep.pdf")
    print("saved load_sweep.csv, load_sweep.png, load_sweep.pdf")
