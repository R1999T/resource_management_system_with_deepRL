# Points needing pondering

Open points that we found while working and that need more thought before the report. Add new points at the end.

## 1. Argmax agent gets stuck at the end of an episode

**What we saw.** On 100 unseen seeds (1000 to 1099) the trained agent (20 jobsets x 10 episodes x 300 iterations) was evaluated in two modes. Sampling gave mean slowdown 1.874 and 0 cap hits. Argmax gave mean slowdown 1.751 but all 100 episodes hit the 500-timestep cap. The argmax slowdown only counts finished jobs, so it looks better than it is. Its mean reward (-104.42) is worse than sampling (-65.89), and reward counts every waiting job.

**What the diagnostic showed** (seeds 1000, 1001, 1002, model `results/models/policy.pt`):
- At the end of the episode one job is left in slot 1 and the cluster is empty, so the job fits (`fitting slots: [1]`).
- Argmax picks action 2 again and again (498 of 531 step calls on seed 1000). Slot 2 is empty, so this is an invalid action. Time moves forward and nothing is scheduled.
- Between 464 and 484 step calls per episode were steps where a job fit but the agent did not pick a fitting slot.
- The episode only stops at the cap of 500 timesteps.

**Dummy example.** Queue has only j1 in slot 1 (duration 2, demand [3,1]). The cluster is empty. The network gives these probabilities:

    action 0 (void):   0.05
    action 1 (j1):     0.30
    action 2 (empty):  0.45
    action 3 (empty):  0.20

Argmax takes the largest, action 2. It is invalid, so time moves one step and j1 gets nothing. The next state is the same, so the probabilities are the same and argmax picks action 2 again, until the cap. With sampling, action 1 has a 30% chance each step, so j1 starts after a few steps and the episode ends normally.

**Our reading (a guess, not tested).** The network learned that action 2 is usually good. States with a nearly empty queue are rare in training, so the network has not learned to avoid empty slots there.

**What we do about it (proposed).**
- The main agent number in the report is the sampling one.
- Argmax is reported as a failure mode, with the cap-hit count and the reward, not only the slowdown.
- We did not mask invalid actions, because that would be our own change to the paper's agent and would hide this weakness (see `OUR_DECISIONS_baseline_padding.md`, evaluation section, point 5).

**Still open.** Is the cause really the rarity of near-empty states, or something else (for example too little training)? A test: train longer, or check the probabilities of the network in such a state directly.

## 2. Load sweep stops at about 102%, the paper goes to 190%

**What we saw.** `simulator/load_sweep.py` varies the arrival probability p from 0.1 to 1.0 and plots slowdown against load (results in `results/tables/load_sweep.csv` and `results/figures/load_sweep.png`). The paper's Figure 4 covers loads from 10% to 190% (Section 4.1 and 4.2). Our sweep ends at about 102%.

**Why.** In our simulator `SchedulerEnv._arrive()` adds at most one job per timestep, with probability p, so p cannot be above 1. Our load formula is load = p x E[T] x E[demand per resource] / capacity = p x 4.1 x 2.5 / 10 = 1.025 p. At p = 1.0 this gives about 102%, and that is the maximum.

**Dummy example.** p = 0.5 gives load = 1.025 x 0.5, about 51%. To reach 190% we would need p = 1.85, which is not a probability.

**Our reading (a guess, not checked).** The paper text we read says only that jobs arrive by a Bernoulli process and that the average load varies from 10% to 190%. With the paper's own job sizes (dominant demand 0.25r to 0.5r, other resource 0.05r to 0.1r, durations as in Section 4.1) and at most one job per timestep, our arithmetic gives load of about 0.92 p, which is below 100% even at p = 1. So the paper must define load in a different way, or allow more arrivals per timestep. We could not tell which from the text.

**What we do about it (proposed).**
- Report the sweep up to about 102% and say clearly that this is a limit of our simulator.
- Do not claim to reproduce the paper's Figure 4 above 100% load.

**Still open.** How did the paper reach 190%? Places to check: the authors' released code (`hongzimao/deeprm`, how it sets the arrival rate) or the instructor. If we need loads above 100%, `WorkloadGenerator` and `_arrive()` must allow more than one arrival per timestep. That is a change in the coding chat.
