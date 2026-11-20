import torch
import torch.nn as nn
from torch.distributions import Categorical

N_IN = 4460      # 400 cluster + 10 * 400 slots + 60 backlog
N_HIDDEN = 20    # paper, Sec. 4.1: one fully connected hidden layer with 20 neurons
N_OUT = 11       # actions 0..10 (0 = void)


class PolicyNet(nn.Module):
    def __init__(self, n_in=N_IN, n_hidden=N_HIDDEN, n_out=N_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),                      # activation not given in the paper: our choice
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)                  # logits; softmax happens inside Categorical

    def act(self, x):
        """x: encoded state (numpy or tensor). Returns (action int, log-prob tensor)."""
        dist = Categorical(logits=self(torch.as_tensor(x)))
        a = dist.sample()
        return int(a), dist.log_prob(a)


if __name__ == "__main__":
    from simulator import SchedulerEnv
    from encoder import encode

    net = PolicyNet()
    print("parameters:", sum(p.numel() for p in net.parameters()))   # expect 89451

    env = SchedulerEnv(seed=0)
    x = encode(env.get_state()) #get all the things from the queue
    probs = torch.softmax(net(torch.as_tensor(x)), dim=0)
    print("probabilities sum:", float(probs.sum().item()))                   # expect 1.0
    print("probabilities:", probs.detach().numpy().round(3))          # roughly 0.09 each at start
    a, logp = net.act(x)
    print("sampled action:", a, "log-prob:", float(logp))             # action in 0..10
