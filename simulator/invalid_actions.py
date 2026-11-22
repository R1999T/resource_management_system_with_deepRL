"""How often does the agent pick an invalid action (empty slot, or job that cannot be placed)?
Compares two saved models (small: 10x10x100, final: 20x10x300) on training seeds and unseen seeds.
Writes ../results/tables/invalid_actions.csv"""
import csv
import numpy as np
import torch
from simulator import SchedulerEnv
from policy import PolicyNet
from evaluate import make_agent_policy, valid_mask

MODELS = [("small (10x10x100)", "../results/models/policy_small.pt"),
          ("final (20x10x300)", "../results/models/policy.pt")]
SEED_SETS = [("train seeds 0-9 (x10)", [s for _ in range(10) for s in range(10)]),
             ("unseen seeds 1000-1099", list(range(1000, 1100)))]


def run(policy, seeds):
    env = SchedulerEnv()
    rng = np.random.default_rng(123)
    calls = invalid = fit_calls = fit_invalid = timesteps = 0
    for s in seeds:
        env.reset(seed=s)
        done = False
        while not done:
            mask = valid_mask(env)                       # evaluate.py
            a = policy(env, rng)
            calls += 1
            bad = not bool(mask[a])                      # empty slot or job cannot be placed
            invalid += bad
            if mask[1:].any():                           # at least one job could have been started
                fit_calls += 1
                fit_invalid += bad                       # invalid action (void is not counted here)
            _, _, done = env.step(a)
        timesteps += env.t
    n = len(seeds)
    return [calls / n, timesteps / n, 100 * invalid / calls,
            100 * fit_invalid / max(fit_calls, 1)]


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = []
    print(f"{'model':18s} {'mode':7s} {'seeds':24s} {'step calls':>10s} {'timesteps':>9s} {'invalid %':>9s} {'invalid % (job could start)':>28s}")
    for mname, path in MODELS:
        net = PolicyNet()
        net.load_state_dict(torch.load(path))
        net.eval()
        for mode in ("sample", "argmax"):
            pol = make_agent_policy(net, mode)
            for sname, seeds in SEED_SETS:
                r = run(pol, seeds)
                rows.append([mname, mode, sname] + [round(v, 1) for v in r])
                print(f"{mname:18s} {mode:7s} {sname:24s} {r[0]:10.1f} {r[1]:9.1f} {r[2]:9.1f} {r[3]:28.1f}")
    with open("../results/tables/invalid_actions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "seeds", "step_calls_per_episode", "timesteps_per_episode",
                    "invalid_pct", "invalid_pct_when_a_job_could_start"])
        w.writerows(rows)
    print("saved ../results/tables/invalid_actions.csv")
