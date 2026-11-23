"""Demo: trained agent, ARGMAX mode, WITH invalid-action mask (extension, not in the paper).
Same policy.pt: it was NOT trained with the mask. Before each choice the invalid actions get logit -inf
(evaluate.py: valid_mask and make_agent_policy(..., masked=True)).
Shows the network's own top action next to the action taken, and the probabilities before and after the mask once.
Run from anywhere:  python demo/use_model_argmax_mask.py [seed]
Default seed 1000."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent                     # resource_management_system_with_deepRL/
sys.path.insert(0, str(ROOT / "simulator"))                       # so simulator.py, policy.py, evaluate.py can be imported
from simulator import SchedulerEnv
from encoder import encode
from policy import PolicyNet
from evaluate import make_agent_policy, valid_mask

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

net = PolicyNet()                                                 # policy.py
net.load_state_dict(torch.load(ROOT / "results" / "models" / "policy.pt"))
net.eval()
policy = make_agent_policy(net, "argmax", masked=True)            # evaluate.py
env = SchedulerEnv(p_arrival=0.7)                                 # simulator.py
rng = np.random.default_rng(0)

env.reset(seed=seed)
total, done, step, top_invalid, shown = 0.0, False, 0, 0, False
print(f"seed {seed}, mode argmax, WITH mask. First 10 decisions (action 0 = wait, i = start the job in slot i):")
while not done:
    t, durations = env.t, [j.duration for j in env.queue]         # time and T of each waiting job, before the action
    with torch.no_grad():
        logits = net(torch.as_tensor(encode(env.get_state())))    # PolicyNet.forward: raw scores
    mask = valid_mask(env)                                        # True = an action env.step would accept
    raw_top = int(torch.argmax(logits))                           # what the network alone would pick
    top_invalid += int(not mask[raw_top])
    if not shown and not mask[raw_top]:                           # once: show what the mask does to the probabilities
        before = torch.softmax(logits, 0)
        after = torch.softmax(logits.masked_fill(~mask, float("-inf")), 0)
        print(f"  (step {step}: the network's top action {raw_top} is invalid)")
        print(f"    valid actions : {[i for i in range(len(mask)) if mask[i]]}")
        print(f"    prob. before  : {np.round(before.numpy(), 2)}")
        print(f"    prob. after   : {np.round(after.numpy(), 2)}")
        shown = True
    scheduled_before = len(env.running) + len(env.finished)
    action = policy(env, rng)                                     # masked argmax
    _, reward, done = env.step(action)                            # SchedulerEnv.step
    total += reward
    if step < 10:
        placed = "yes" if len(env.running) + len(env.finished) > scheduled_before else "no"
        print(f"  step {step:2d}: time {t:2d}, waiting T={durations}, network top {raw_top:2d}, "
              f"taken {action:2d}, started a job {placed:3s}, reward {reward:6.2f}")
    step += 1

slowdowns = [(j.finish - j.arrival) / j.duration for j in env.finished]
print(f"jobs finished: {len(env.finished)}, hit cap: {bool(env.queue or env.running)}, "
      f"mean slowdown: {np.mean(slowdowns):.2f}, total reward: {total:.1f}, "
      f"steps where the network's top action was invalid: {top_invalid} of {step}")
