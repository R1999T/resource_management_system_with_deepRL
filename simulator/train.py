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


if __name__ == "__main__":
    env = SchedulerEnv()
    net = PolicyNet()

    for run in range(2):                      # same seed twice -> same jobs
        log_probs, rewards = rollout(env, net, seed=0)
        print(f"run {run}: step calls = {len(rewards)}, "
              f"len(log_probs) = {len(log_probs)}, timesteps = {env.t}, "
              f"jobs generated = {env.gen.next_id}, "
              f"total reward = {sum(rewards):.2f}")
