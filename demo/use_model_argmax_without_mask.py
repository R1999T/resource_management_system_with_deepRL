"""Demo: trained agent, ARGMAX mode, WITHOUT mask. This is the setting that gets stuck:
the network keeps picking an empty slot, nothing is started, and the episode only stops at the 500-timestep cap.
Prints the first 10 decisions, then the summary (look at "hit cap").
Run from anywhere:  python demo/use_model_argmax_without_mask.py [seed]
Default seed 1000."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent                     # resource_management_system_with_deepRL/
sys.path.insert(0, str(ROOT / "simulator"))                       # so simulator.py, policy.py, evaluate.py can be imported
from simulator import SchedulerEnv
from policy import PolicyNet
from evaluate import make_agent_policy

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

net = PolicyNet()                                                 # policy.py
net.load_state_dict(torch.load(ROOT / "results" / "models" / "policy.pt"))
net.eval()
policy = make_agent_policy(net, "argmax", masked=False)           # evaluate.py
env = SchedulerEnv(p_arrival=0.7)                                 # simulator.py
rng = np.random.default_rng(0)

env.reset(seed=seed)
total, done, step = 0.0, False, 0
print(f"seed {seed}, mode argmax, WITHOUT mask. First 10 decisions (action 0 = wait, i = start the job in slot i):")
while not done:
    queue_len, t = len(env.queue), env.t                          # jobs waiting and current time, before the action
    durations = [j.duration for j in env.queue]                   # T of each waiting job
    scheduled_before = len(env.running) + len(env.finished)       # jobs started so far
    action = policy(env, rng)                                     # state -> encode -> network -> action
    _, reward, done = env.step(action)                            # SchedulerEnv.step
    total += reward
    scheduled = len(env.running) + len(env.finished)
    if step < 10:
        placed = "yes" if scheduled > scheduled_before else "no"  # a valid action starts one job
        print(f"  step {step:2d}: time {t:2d}, jobs waiting {queue_len:2d} T={durations}, action {action:2d}, "
              f"reward {reward:6.2f}, valid {placed:3s}, scheduled so far {scheduled:2d}")
    step += 1

slowdowns = [(j.finish - j.arrival) / j.duration for j in env.finished]
print(f"step calls: {step}, final time: {env.t}, jobs still waiting: {len(env.queue)}")
print(f"jobs finished: {len(env.finished)}, hit cap: {bool(env.queue or env.running)}, "
      f"mean slowdown (finished jobs only): {np.mean(slowdowns):.2f}, total reward: {total:.1f}")
