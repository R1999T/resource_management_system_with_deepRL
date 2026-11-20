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
