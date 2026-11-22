# Our decision: baseline with episodes of different length

This note records a choice that the paper does not specify. We made it ourselves and it should be mentioned in the report.

## The problem

The training algorithm (paper, Figure 3, page 4) runs N episodes on the same jobset and uses a time-dependent baseline:

    b_t = (1/N) * sum over i of v_t^i

The jobs are the same in all N episodes, but the episodes do not have the same length. The agent picks different actions, so some episodes finish earlier. In our untrained runs on seed 0, one episode took 118 `env.step()` calls and another took 123.

For a step t that lies beyond the end of a short episode, that episode has no return v_t. The paper does not say what to use in its place.

Example with N = 2:

    episode 1 (3 steps): [-9, -6, -3]
    episode 2 (5 steps): [-15, -12, -9, -6, -3]

At steps 3 and 4 only episode 2 exists, so the average at those steps is not defined.

## The fix we use

Pad the returns of each short episode with zeros up to the length of the longest episode, then average step by step. A finished episode has no more penalty to collect, so a return of 0 after its end is a natural value.

    episode 1 padded: [-9, -6, -3, 0, 0]
    episode 2:        [-15, -12, -9, -6, -3]
    baseline:         [-12, -9, -6, -3, -1.5]

The advantage of an episode is then taken only over its own length:

    advantage_t = v_t - b_t, for t = 1 .. length of that episode

## Where this comes from

- The paper text does not describe this case.
- The authors' released code does it this way: `hongzimao/deeprm`, file `pg_re.py`, lines 194 to 202. It computes the discounted returns, pads them with zeros to the longest episode (`np.zeros(maxlen - len(ret))`), takes `np.mean(padded_rets, axis=0)` as the baseline, and uses `ret - baseline[:len(ret)]` as the advantage.
- In the report, state that this follows the authors' released code and not the paper text.

## Related detail

A "step" in the baseline is one `env.step()` call, not one timestep of the cluster. A valid job placement is a step in which time does not move. The authors' code also counts steps this way.

---

# Our decisions: evaluation (`evaluate.py`)

These choices are about how we compare the trained agent with Random and SJF. The paper does not describe them in detail, so they are ours and belong in the report.

## 1. Evaluate on unseen jobsets

**Decision:** training uses seeds 0 to 9. Evaluation uses 100 different seeds (1000 to 1099).

**Problem it solves:** the training metrics (mean slowdown 2.17, mean reward -76.93 in the 10 x 10 x 100 run) are measured on jobsets the agent has already seen many times. The agent may do well there only because it fits those jobsets. Unseen seeds test whether it generalizes.

## 2. Same seeds for all three policies

**Decision:** Random, SJF and the trained agent all run on exactly the same 100 seeds.

**Problem it solves:** a different jobset can be easier or harder (more long jobs, more arrivals). If the policies saw different jobsets, a difference in slowdown could come from the jobsets and not from the policy. Arrivals do not depend on the agent's actions, and `reset(seed)` gives the same jobs every time, so the only difference left is the policy.

## 3. Confidence intervals, not just the mean

**Decision:** for each policy we report the mean slowdown over the 100 episodes and a 95% confidence interval.

    CI = mean +/- t * (std / sqrt(n))

with n = 100 episodes (t is about 1.98, close to 1.96).

**Problem it solves:** one run gives one noisy number. A mean alone cannot tell whether a gap between two policies is real or luck. The course also requires confidence intervals for claims of improvement. If the interval of the agent does not overlap the interval of SJF, we can say the difference is real. A larger n makes std / sqrt(n) smaller, so the interval gets narrower.

**Note:** 95% is a convention that we choose. It does not come from the data.

## 4. One common policy signature

**Decision:** every policy has the form `policy(env, rng) -> action`, as `random_policy` and `sjf_policy` in `run_demo.py` already do. The trained agent gets a small wrapper that calls `env.get_state()`, `encode`, the network, and returns the action.

**Problem it solves:** one `evaluate` function can then run all three without special cases, and the comparison code is the same for all.

## 5. Agent acts with argmax and with sampling, both reported

**Decision:** in evaluation the agent is run in two modes: (a) argmax, the most probable action in every state, and (b) sampling from the action probabilities, as in training. Both results go in the table.

**Why:** the paper text, as we read it, does not say which one is used when the agent is tested. So this is our choice and the report should say so.

**Problem it mitigates:**
- Argmax is deterministic, so one episode per seed is enough. But it can get stuck. If the most probable action in some state is void (0), the agent may do nothing for the whole episode. The slowdown is then very large.
- Sampling is less likely to get stuck because other actions have some probability, but it adds randomness, so more episodes are needed.
- Running both lets us check one mode against the other. If the two results disagree strongly, we know something is wrong and can look at it.

**Alternative that we rejected:** masking void or invalid actions when a job fits. This would make the agent look better, but it is our own change and not part of the paper. It would hide a weakness of the learned policy.

## 6. Count episodes that hit the safety cap

**Decision:** for each policy and mode we count how many episodes reached the 500-timestep cap of the simulator, and report that count.

**Problem it solves:** the cap stops an episode that would never end, but the slowdown of such an episode is not a normal result. Without a count, a stuck agent would be hidden inside the mean. With the count, the reader can see how often it happened. It costs almost nothing, since it is only a counter.

## 7. Metric

**Decision:** the main number is the mean slowdown per episode, taken over `env.finished`. Total reward per episode is also stored. Total reward equals minus the sum of slowdowns, so it carries the same information in a different scale.

**Why:** the paper compares policies on average job slowdown (Figure 4), so we compare on the same quantity.
