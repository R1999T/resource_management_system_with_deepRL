import numpy as np
import torch
from simulator import SchedulerEnv
from encoder import encode
from policy import PolicyNet


def rollout(env, net, seed):
    """Run one episode on the jobset given by `seed`.
    Returns one log-prob and one reward per env.step() call."""
    state = env.reset(seed=seed)
    log_probs, rewards = [], []
    done = False
    while not done:
        a, logp = net.act(encode(state))     # network picks an action
        state, r, done = env.step(a)         # env gives the next state and reward
        log_probs.append(logp)
        rewards.append(r)
    return log_probs, rewards


def compute_returns(rewards, gamma=1.0):
    """v_t = r_t + gamma * v_(t+1), computed from the last step backwards."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def compute_advantages(all_returns):
    """all_returns: list of N return lists (same jobset, different lengths).
    Baseline b_t = mean over episodes of v_t, with short episodes padded by zeros
    (see OUR_DECISIONS_baseline_padding.md). Returns one advantage list per episode."""
    maxlen = max(len(r) for r in all_returns)
    padded = [np.concatenate([r, np.zeros(maxlen - len(r))]) for r in all_returns]
    baseline = np.mean(padded, axis=0)
    return [np.array(r) - baseline[:len(r)] for r in all_returns]


def update(net, optimizer, all_log_probs, all_advantages):
    """One gradient step. loss = -sum(log_prob * advantage) over all steps of all episodes.
    A positive advantage pushes that action's probability up, a negative one pushes it down."""
    loss = 0.0
    for log_probs, advs in zip(all_log_probs, all_advantages):
        loss = loss - (torch.stack(log_probs) * torch.as_tensor(advs, dtype=torch.float32)).sum()
    optimizer.zero_grad()      # clear old gradients (torch.optim.Optimizer)
    loss.backward()            # gradients of loss w.r.t. all weights (torch Tensor)
    optimizer.step()           # RMSprop changes the weights (torch.optim.RMSprop)
    return loss.item()


if __name__ == "__main__":

    #1.how return is computed
    print(compute_returns([-4, -1, -2]))          # expect [-7.0, -3.0, -2.0]
    print(compute_returns([-1, -2, -3], 0.5))     # expect [-2.75, -3.5, -3.0]

    #2.how advantage is computed
    advs = compute_advantages([[-9, -6, -3], [-15, -12, -9, -6, -3]])
    print(advs[0])                                # expect [3. 3. 3.]
    print(advs[1])                                # expect [-3. -3. -3. -3. -1.5]

    env = SchedulerEnv()
    net = PolicyNet()

    for run in range(2):                      # same seed twice -> same jobs
        log_probs, rewards = rollout(env, net, seed=0)
        returns = compute_returns(rewards)
        print(f"run {run}: step calls = {len(rewards)}, "
              f"len(log_probs) = {len(log_probs)}, timesteps = {env.t}, "
              f"jobs generated = {env.gen.next_id}, "
              f"total reward = {sum(rewards):.2f}, v_0 = {returns[0]:.2f}")

    
    #3. update check: one fixed state, action 3 with advantage +1 -> its probability should rise
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
    x = torch.as_tensor(encode(env.reset(seed=0)))
    p_before = torch.softmax(net(x), dim=0)[3].item()
    for _ in range(20):
        logp = torch.log_softmax(net(x), dim=0)[3]
        update(net, optimizer, [[logp]], [np.array([1.0])])
    p_after = torch.softmax(net(x), dim=0)[3].item()
    print(f"p(action 3): before = {p_before:.3f}, after = {p_after:.3f}")   # expect after > before
