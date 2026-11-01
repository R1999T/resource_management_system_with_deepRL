import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import SchedulerEnv, CAPACITY


def random_policy(env, rng):
    return int(rng.integers(0, env.M + 1))          # uniform over {0, 1, ..., M}


def sjf_policy(env, rng):
    """Shortest Job First: start the shortest visible job that fits right now, else void."""
    best, best_dur = 0, None
    for i, job in enumerate(env.queue[:env.M], start=1):
        if env.cluster.find_start(job) == 0 and (best_dur is None or job.duration < best_dur):
            best, best_dur = i, job.duration
    return best


def run_episode(policy, seed=0, p=0.7):
    env = SchedulerEnv(p_arrival=p, seed=seed)       # same seed -> same job arrivals
    rng = np.random.default_rng(123)
    log = {"t": [], "waiting": [], "util": [], "cum_reward": []}
    cum, done = 0.0, False
    while not done:
        t_before = env.t
        _, r, done = env.step(policy(env, rng))
        cum += r
        if env.t != t_before:                        # time moved: record one point per timestep
            log["t"].append(env.t)
            log["waiting"].append(len(env.queue))
            log["util"].append(1 - env.cluster.free[0].mean() / CAPACITY)
            log["cum_reward"].append(cum)
    slow = [(j.finish - j.arrival) / j.duration for j in env.finished]
    return log, float(np.mean(slow)), len(slow)


results = {}
for name, pol in [("Random", random_policy), ("SJF", sjf_policy)]:
    log, avg_slow, n = run_episode(pol)
    results[name] = log
    print(f"{name:7s}: jobs finished = {n}, average slowdown = {avg_slow:.2f}, "
          f"final cumulative reward = {log['cum_reward'][-1]:.1f}")

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
for name, log in results.items():
    axes[0].plot(log["t"], log["waiting"], label=name)
    axes[1].plot(log["t"], log["util"], label=name)
    axes[2].plot(log["t"], log["cum_reward"], label=name)
axes[0].set_ylabel("Waiting jobs")
axes[1].set_ylabel("Mean utilization (now)")
axes[2].set_ylabel("Cumulative reward")
axes[2].set_xlabel("Timestep")
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend()
fig.suptitle("Sample path of one episode (same job arrivals)")
fig.tight_layout()
fig.savefig("sample_path.pdf")
fig.savefig("sample_path.png", dpi=200)
print("saved sample_path.pdf and sample_path.png")