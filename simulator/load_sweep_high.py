"""Case 1 of ptsneededpondering.md, point 2: the agent trained at p = 0.7 (at most one job per timestep)
is tested at loads up to 190%, with several arrivals per timestep (Poisson(lam), WorkloadGenerator.sample_many).
No retraining. Load is about 1.025 * lam. All loads use the same arrival model, so the curve is self-consistent,
but it is not the same arrival model as load_sweep.py (one Bernoulli arrival per timestep).
Writes results/tables/load_sweep_high.csv and results/figures/load_sweep_high.png"""
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import SchedulerEnv
from policy import PolicyNet
from evaluate import random_policy, sjf_policy, make_agent_policy, run_episode, mean_ci

LOADS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]       # percent
SEEDS = range(1000, 1100)                                   # unseen, same for all policies


class MultiArrivalEnv(SchedulerEnv):
    """SchedulerEnv with Poisson(lam) arrivals per timestep. Only _arrive is changed."""

    def __init__(self, lam, **kw):
        self.lam = lam                                      # must exist before SchedulerEnv.__init__ calls reset()
        super().__init__(**kw)

    def _arrive(self):
        if self.t < self.episode_len:
            self.queue.extend(self.gen.sample_many(self.t, self.lam))   # WorkloadGenerator.sample_many


def evaluate_multi(policy, lam):
    env = MultiArrivalEnv(lam)
    rng = np.random.default_rng(123)                        # policy RNG, separate from the env RNG
    out = [run_episode(env, policy, s, rng) for s in SEEDS]     # run_episode, evaluate.py
    sd, rw, cap = zip(*out)
    return np.array(sd), np.array(rw), int(sum(cap))


if __name__ == "__main__":
    torch.manual_seed(0)
    net = PolicyNet()
    net.load_state_dict(torch.load("../results/models/policy.pt"))
    net.eval()
    policies = [("Random", random_policy), ("SJF", sjf_policy),
                ("Agent (sample)", make_agent_policy(net, "sample"))]

    rows = []
    for load in LOADS:
        lam = load / 100 / 1.025
        for name, pol in policies:
            sd, rw, cap = evaluate_multi(pol, lam)
            m, h = mean_ci(sd)
            rows.append([round(lam, 3), load, name, round(m, 3), round(m - h, 3), round(m + h, 3), cap])
            print(f"load~{load:3d}% lam={lam:.3f}  {name:15s} {m:7.3f} [{m - h:7.3f}, {m + h:7.3f}] cap hits {cap}", flush=True)

    with open("../results/tables/load_sweep_high.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam", "load_pct", "policy", "mean_slowdown", "ci_low", "ci_high", "cap_hits"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, _ in policies:
        r = [x for x in rows if x[2] == name]
        ax.errorbar([x[1] for x in r], [x[3] for x in r],
                    yerr=[[x[3] - x[4] for x in r], [x[5] - x[3] for x in r]],
                    marker="o", capsize=3, label=name)
    ax.set_xlabel("average cluster load (%), about 1.025 x lam x 100")
    ax.set_ylabel("mean job slowdown (lower is better)")
    ax.set_title(f"Slowdown up to 190% load, {len(SEEDS)} unseen seeds, 95% CI")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../results/figures/load_sweep_high.png", dpi=150)
    plt.savefig("../results/figures/load_sweep_high.pdf")
    print("saved load_sweep_high.csv, load_sweep_high.png, load_sweep_high.pdf")
