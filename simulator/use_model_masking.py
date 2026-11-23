"""Demo: the trained agent with invalid-action masking (extension, not in the paper).
The network is the same policy.pt. It was NOT trained with the mask. The mask is applied only when an action is chosen:
invalid actions get logit -inf before argmax or softmax (evaluate.py: make_agent_policy(..., masked=True)).
Shows, for the first 10 decisions, the network's own top action next to the action actually taken, and prints the
probabilities before and after the mask once. At the end the same seed is run again without the mask, for contrast.
Run from the simulator/ folder:  python use_model_masking.py [seed] [argmax|sample]
Defaults: seed 1000, argmax (argmax is the mode that got stuck without the mask)."""
import sys
import numpy as np
import torch
from simulator import SchedulerEnv
from encoder import encode
from policy import PolicyNet
from evaluate import make_agent_policy, valid_mask, run_episode

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
mode = sys.argv[2] if len(sys.argv) > 2 else "argmax"
torch.manual_seed(0)                                              # for the 'sample' mode

net = PolicyNet()                                                 # policy.py
net.load_state_dict(torch.load("../results/models/policy.pt"))    # same weights as everywhere else
net.eval()
masked_policy = make_agent_policy(net, mode, masked=True)         # evaluate.py
plain_policy = make_agent_policy(net, mode, masked=False)         # evaluate.py
env = SchedulerEnv(p_arrival=0.7)                                 # simulator.py
rng = np.random.default_rng(0)

env.reset(seed=seed)
total, done, step, top_invalid, shown = 0.0, False, 0, 0, False
print(f"seed {seed}, mode {mode}, WITH mask. First 10 decisions "
      f"(action 0 = wait, i = start the job in slot i):")
while not done:
    t, durations = env.t, [j.duration for j in env.queue]         # time and T of each waiting job, before the action
    with torch.no_grad():
        logits = net(torch.as_tensor(encode(env.get_state())))    # PolicyNet.forward: raw scores
    mask = valid_mask(env)                                        # evaluate.py: True = action env.step would accept
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
    action = masked_policy(env, rng)                              # masked choice
    _, reward, done = env.step(action)                            # SchedulerEnv.step
    total += reward
    if step < 10:
        placed = "yes" if len(env.running) + len(env.finished) > scheduled_before else "no"
        print(f"  step {step:2d}: time {t:2d}, waiting T={durations}, network top {raw_top:2d}, "
              f"taken {action:2d}, valid {placed:3s}, reward {reward:6.2f}")
    step += 1

slowdowns = [(j.finish - j.arrival) / j.duration for j in env.finished]
print(f"WITH mask   : jobs finished {len(env.finished)}, hit cap {bool(env.queue or env.running)}, "
      f"mean slowdown {np.mean(slowdowns):.2f}, total reward {total:.1f}, "
      f"steps where the network's top action was invalid {top_invalid} of {step}")

sd, rw, cap = run_episode(env, plain_policy, seed, rng)           # evaluate.py: same seed, no mask
print(f"WITHOUT mask: hit cap {cap}, mean slowdown {sd:.2f}, total reward {rw:.1f}")
