# AVD894 Course Project: DeepRM (Mao et al., 2016)

Course project for AVD894. Paper: H. Mao, M. Alizadeh, I. Menache, S. Kandula,
*Resource Management with Deep Reinforcement Learning*, HotNets 2016.

## Status

| Item | State |
|---|---|
| Task 1: simplified discrete MDP + value iteration | Notebook done
| Task 2: simulator | Not started |
| Term paper, presentation | completed yet to update in repository |

## Repository contents

| File | Purpose |
|---|---|
| `task1_value_iteration.ipynb` | Task 1: 4-state scheduling MDP, value iteration, convergence plot |
| `README.md` | This file, including the AI interaction log |

Run: `pip install numpy matplotlib`, then open the notebook and run all cells.

## Task 1 model (our simplification, not the paper's model)

- One CPU, one waiting-job slot, job duration `T = 2`.
- States: 0 = Free/no job, 1 = Free/job waiting, 2 = Busy/no job, 3 = Busy/job waiting.
- Actions: wait, start (start is valid only in state 1).
- Busy job finishes w.p. 0.5 per step; a new job arrives w.p. 0.5 when the slot is empty.
- Reward: `-1/T` per job in the system per step (paper, Sec. 3.2 "Rewards"). Discount `gamma = 0.9`.

**Known issue:** in state 1, `wait` and `start` have exactly equal value, because a waiting job
blocks new arrivals and keeps the job count low. The model must be adjusted (for example,
penalise turned-away arrivals) before the Task 1 submission. Only `P` and `r` change.

## AI usage log

Tool: Claude (Anthropic), used as a study partner and coding aide. We are responsible for
everything submitted. Fill in the date and commit hash for each entry when committing.

| # | Date / commit | Prompt (summary) | AI output | How we checked it |
|---|---|---|---|---|
| 1 | | Help choose a paper from the course list (little time, laptop-only compute) | Shortlist of 3 papers; recommended Mao et al. | Read the paper intro and Sec. 3; confirmed the simulator is small enough for an i7 / 16 GB laptop |
| 2 | | Explain the intro examples (job scheduling, bitrate adaptation, relay selection, VM placement, congestion control) | Plain-language analogies | Compared against the paper text; AI first answered from memory and mentioned a "project management" example that is not in the paper, which we caught |
| 3 | | Explain Sec. 2 background: policy, policy gradient, REINFORCE, baseline | Step-by-step explanation with toy two-action examples and tables | Re-derived the toy numbers by hand; matched Eq. (1), Eq. (2) and Fig. 3 of the paper |
| 4 | | Explain Sec. 3.1 and 3.2: model, state (Fig. 2), action trick, reward | Walkthrough, tiny worked episode | Checked each reading against the paper text and Fig. 2; verified that total reward equals minus the sum of slowdowns |
| 5 | | Build a Task 1 MDP in our class format (transition, reward, policy tables) | Deterministic 4-state job-ordering MDP, then a 4-state looping MDP with wait/start matrices | Hand value-iteration on the small MDP (V = -1, -1, Q = -2.5 vs -4); checked every transition row sums to 1 |
| 6 | | Create the Task 1 notebook | `task1_value_iteration.ipynb` | Ran all cells (converges in 170 sweeps); found a tie between wait and start in state 1, so the model needs fixing |

### Corrections made after AI output

- Kubernetes scheduler code snippet was given from memory and was not checked against the repository; do not cite it without verifying.
- The Task 1 notebook first claimed `start` was optimal in state 1; running the code showed a tie, and the text was corrected.

## References

1. H. Mao et al., Resource Management with Deep Reinforcement Learning, HotNets-XV, 2016.
2. R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction.