import csv
import numpy as np
import torch
from scipy import stats
from simulator import SchedulerEnv
from encoder import encode
from policy import PolicyNet

# Policy signature for everything below: policy(env, rng) -> action in 0..M
# (see OUR_DECISIONS_baseline_padding.md, evaluation section)


def random_policy(env, rng):                       # same as run_demo.py
    return int(rng.integers(0, env.M + 1))


def sjf_policy(env, rng):                          # same as run_demo.py
    best, best_dur = 0, None
    for i, job in enumerate(env.queue[:env.M], start=1):
        if env.cluster.find_start(job) == 0 and (best_dur is None or job.duration < best_dur):
            best, best_dur = i, job.duration
    return best


def valid_mask(env):
    """True for actions that env.step would accept: void (0) and slots with a job that can be placed."""
    mask = torch.zeros(env.M + 1, dtype=torch.bool)
    mask[0] = True
    for i, job in enumerate(env.queue[:env.M], start=1):
        mask[i] = env.cluster.find_start(job) is not None    # Cluster.find_start, simulator.py
    return mask


def make_agent_policy(net, mode, masked=False):
    """Wrap a PolicyNet (policy.py) as policy(env, rng). mode = 'argmax' or 'sample'.
    masked=True (extension, not in the paper): invalid actions get logit -inf."""
    def policy(env, rng):
        x = encode(env.get_state())                # env.get_state (SchedulerEnv), encode (encoder.py)
        with torch.no_grad():
            logits = net(torch.as_tensor(x))       # PolicyNet.forward
            if masked:
                logits = logits.masked_fill(~valid_mask(env), float("-inf"))
            if mode == "argmax":
                return int(torch.argmax(logits))   # most probable action
            return int(torch.distributions.Categorical(logits=logits).sample())   # sample like in training
    return policy


def run_episode(env, policy, seed, rng):
    """One episode on the jobset `seed`. Returns (mean slowdown, total reward, hit_cap)."""
    env.reset(seed=seed)                           # SchedulerEnv.reset: same seed, same jobs
    total, done = 0.0, False
    while not done:
        _, r, done = env.step(policy(env, rng))    # SchedulerEnv.step
        total += r
    slowdown = np.mean([(j.finish - j.arrival) / j.duration for j in env.finished])
    hit_cap = bool(env.queue or env.running)       # stopped by max_t with jobs still unfinished
    return slowdown, total, hit_cap


def evaluate(policy, seeds, p_arrival=0.7):
    env = SchedulerEnv(p_arrival=p_arrival)        # SchedulerEnv, simulator.py
    rng = np.random.default_rng(123)               # policy RNG, separate from the env RNG
    out = [run_episode(env, policy, s, rng) for s in seeds]
    sd, rw, cap = zip(*out)
    return np.array(sd), np.array(rw), int(sum(cap))


def mean_ci(x, level=0.95):
    """mean and half-width of the CI: t * std / sqrt(n)."""
    n = len(x)
    half = stats.t.ppf((1 + level) / 2, n - 1) * np.std(x, ddof=1) / np.sqrt(n)
    return float(np.mean(x)), float(half)


if __name__ == "__main__":
    import sys
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seeds = range(1000, 1000 + n_seeds)            # unseen: training used seeds 0..9
    torch.manual_seed(0)                           # for the 'sample' mode

    net = PolicyNet()
    net.load_state_dict(torch.load("../results/models/policy.pt"))
    net.eval()

    policies = [("Random", random_policy),
                ("SJF", sjf_policy),
                ("Agent (argmax)", make_agent_policy(net, "argmax")),
                ("Agent (sample)", make_agent_policy(net, "sample")),
                ("Agent (argmax, masked)", make_agent_policy(net, "argmax", masked=True)),
                ("Agent (sample, masked)", make_agent_policy(net, "sample", masked=True))]

    rows = []
    print(f"{n_seeds} unseen seeds ({seeds[0]}..{seeds[-1]}), same seeds for every policy")
    print(f"{'policy':24s} {'mean slowdown':>14s} {'95% CI':>18s} {'mean reward':>12s} {'cap hits':>9s}")
    for name, pol in policies:
        sd, rw, cap = evaluate(pol, seeds)
        m, h = mean_ci(sd)
        rows.append([name, n_seeds, round(m, 3), round(m - h, 3), round(m + h, 3),
                     round(float(rw.mean()), 2), cap])
        print(f"{name:24s} {m:14.3f} [{m - h:7.3f}, {m + h:7.3f}] {rw.mean():12.2f} {cap:9d}")

    with open("../results/tables/evaluation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n_seeds", "mean_slowdown", "ci_low", "ci_high",
                    "mean_total_reward", "cap_hits"])
        w.writerows(rows)
    print("saved evaluation_results.csv")
