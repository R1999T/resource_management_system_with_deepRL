import numpy as np
from simulator import SchedulerEnv

env = SchedulerEnv(p_arrival=0.7, seed=0)
rng = np.random.default_rng(1)

for ep in range(5):
    env.reset()
    total, done = 0.0, False
    while not done:
        a = int(rng.integers(0, env.M + 1))      # random action in {0, 1, ..., M}
        _, r, done = env.step(a)
        total += r
    slow = [(j.finish - j.arrival) / j.duration for j in env.finished]
    all_done = not env.queue and not env.running
    print(f"episode {ep}: jobs finished = {len(slow)}, total reward = {total:.3f}, "
          f"sum slowdown = {sum(slow):.3f}, min slowdown = {min(slow):.2f}, all done = {all_done}")
    if all_done:
        assert abs(total + sum(slow)) < 1e-9, "reward != -sum of slowdowns"
print("check passed: total reward equals minus the sum of slowdowns")