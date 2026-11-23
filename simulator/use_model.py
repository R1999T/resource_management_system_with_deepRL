"""Demo: use the trained agent on one episode.
Run from the simulator/ folder:  python use_model.py [seed] [argmax|sample]
Defaults: seed 1000, sample."""
import sys
import numpy as np
import torch
from simulator import SchedulerEnv
from policy import PolicyNet
from evaluate import make_agent_policy

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
mode = sys.argv[2] if len(sys.argv) > 2 else "sample"

net = PolicyNet()                                                 # policy.py
net.load_state_dict(torch.load("../results/models/policy.pt"))    # learned weights
net.eval()
policy = make_agent_policy(net, mode)                             # evaluate.py: policy(env, rng) -> action
env = SchedulerEnv(p_arrival=0.7)                                 # simulator.py
rng = np.random.default_rng(0)

env.reset(seed=seed)
total, done, step = 0.0, False, 0
print(f"seed {seed}, mode {mode}. First 10 decisions (action 0 = wait, i = start the job in slot i):")
while not done:
    queue_len, t = len(env.queue), env.t
    action = policy(env, rng)                                     # state -> encode -> network -> action
    _, reward, done = env.step(action)                            # SchedulerEnv.step
    total += reward
    if step < 10:
        print(f"  step {step:2d}: time {t:2d}, jobs waiting {queue_len:2d}, action {action:2d}, reward {reward:6.2f}")
    step += 1

slowdowns = [(j.finish - j.arrival) / j.duration for j in env.finished]
print(f"jobs finished: {len(env.finished)}, mean slowdown: {np.mean(slowdowns):.2f}, total reward: {total:.1f}")
