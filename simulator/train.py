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


if __name__ == "__main__":
    print(compute_returns([-4, -1, -2]))          # expect [-7.0, -3.0, -2.0]
    print(compute_returns([-1, -2, -3], 0.5))     # expect [-2.75, -3.5, -3.0]

    env = SchedulerEnv()
    net = PolicyNet()

    for run in range(2):                      # same seed twice -> same jobs
        log_probs, rewards = rollout(env, net, seed=0)
        returns = compute_returns(rewards)
        print(f"run {run}: step calls = {len(rewards)}, "
              f"len(log_probs) = {len(log_probs)}, timesteps = {env.t}, "
              f"jobs generated = {env.gen.next_id}, "
              f"total reward = {sum(rewards):.2f}, v_0 = {returns[0]:.2f}")
