# AVD894 Course Project: DeepRM (Mao et al., 2016)

Course project for AVD894. Paper: H. Mao, M. Alizadeh, I. Menache, S. Kandula,
*Resource Management with Deep Reinforcement Learning*, HotNets 2016.

## Status

| Item | State |
|---|---|
| Task 1: simplified discrete MDP + policy/value iteration | Done (see Task 1 below) |
| Task 2: simulator (Python) | Code done, report pending |
| Term paper, presentation, RL agent | Not started |

## Repository contents

| File | Purpose |
|---|---|
| `<task1 file name>` | Task 1: 4-state scheduling MDP solved by policy iteration |
| `task1_value_iteration.ipynb` | Task 1: same MDP solved by value iteration, convergence plot |
| `simulator.py` | Task 2: `Job`, `Cluster`, `WorkloadGenerator`, `SchedulerEnv` |
| `test_workload.py` | Checks the workload against the paper's numbers |
| `test_env.py` | Checks total reward equals minus the sum of slowdowns |
| `run_demo.py` | Random and SJF policies, sample-path plot (`sample_path.pdf`) |
| `HANDOFF.md` | Project summary for a teammate (and their Claude chat) to learn the code |
| `README.md` | This file, including the AI interaction log |

Run: `pip install numpy matplotlib`, then `python test_workload.py`, `python test_env.py`, `python run_demo.py`.

## Task 1 model (our simplification, not the paper's model)

- One CPU, one waiting-job slot, job duration `T = 2`, discount `gamma = 0.9`.
- States: 0 = Free/no job, 1 = Free/job waiting, 2 = Busy/no job, 3 = Busy/job waiting.
- Actions: wait, start (valid only in state 1).
- Busy job finishes w.p. 0.5 per step; a new job arrives w.p. 0.5 when the slot is empty.
- Reward: `-1/T` per job in the system per step (paper, Sec. 3.2 "Rewards").

## Task 2 simulator (follows paper Sec. 3.1, 3.2, 4.1)

- Two resources (CPU, memory), cluster image with 20 lookahead timesteps.
- Arrivals: Bernoulli with probability `p` per step. 80% of jobs last 1-3 steps, 20% last 10-15.
- Each job has one dominant resource and one minor resource.
- State: first `M = 10` waiting jobs plus a backlog count. Actions: `{0 (void), 1..M}`; time stays frozen until a void or invalid action.
- Reward: `-sum(1/T_j)` over all jobs in the system. Episode: 50 timesteps of arrivals, run until all jobs finish.

**Simplifications (our choices, to list in the report):**
- Integer units: capacity 10 per resource; dominant demand 3-5 units, minor demand 1 unit.
- The paper's backlog cap of 60 jobs is not enforced.
- Safety cap of 500 timesteps per episode.

## AI usage log

Tool: Claude (Anthropic), used as a study partner and coding aide. We are responsible for
everything submitted. Fill in the date and commit hash for each entry when committing.

| # | Date / commit | Prompt (summary) | AI output | How we checked it |
|---|---|---|---|---|

| 1 | | Explain the intro examples (job scheduling, bitrate adaptation, relay selection, VM placement, congestion control) | Plain-language analogies | Compared against the paper text; AI first answered from memory and mentioned a "project management" example that is not in the paper, which we caught |
| 2 | | Explain Sec. 2 background: policy, policy gradient, REINFORCE, baseline | Step-by-step explanation with toy two-action examples and tables | Re-derived the toy numbers by hand; matched Eq. (1), Eq. (2) and Fig. 3 of the paper |
| 3 | | Explain Sec. 3.1 and 3.2: model, state (Fig. 2), action trick, reward | Walkthrough, tiny worked episode | Checked each reading against the paper text and Fig. 2; verified that total reward equals minus the sum of slowdowns |
| 4 | | Build a Task 1 MDP in our class format (transition, reward, policy tables) | Deterministic job-ordering MDP, then a 4-state looping MDP with wait/start matrices | Hand value-iteration on the small MDP (V = -1, -1, Q = -2.5 vs -4); checked every transition row sums to 1 |
| 5 | | Create the Task 1 value-iteration notebook | `task1_value_iteration.ipynb` | Ran all cells (converges in 170 sweeps); found a tie between wait and start in state 1, so the model needed fixing |
| 6 | | Design the Task 2 simulator from the paper | Design in 6 parts (workload, cluster image, step rule, episode, demo, agent interface); suggested integer units | Compared each rule with Sec. 3.1, 3.2 and 4.1; worked the placement rule by hand on a seat example |
| 7 | | Code chunk 1: `Job` and `Cluster` | Classes with `find_start`, `place`, `advance` | Ran the seat example; output matched the hand calculation |
| 8 | | Code chunk 2: workload generator | `WorkloadGenerator` and `test_workload.py` | 100,000-step test matched the paper's numbers (rate 0.699, short fraction 0.802, mean duration 4.09, mean demand 2.50) |
| 9 | | Code chunk 3: environment | `SchedulerEnv` with `reset`/`step` and `test_env.py` | Identity check passed on 5 random episodes (total reward = minus sum of slowdowns); explained each rule in our own words |
| 10 | | Code chunk 4: demo | `run_demo.py`: random and SJF policies, sample-path plot | AI ran it first (SJF slowdown lower than random); our run: <fill in> |
| 11 | | Scope and time estimate for Task 2 | Required parts from the course PDF, a rough 5-7 hour estimate | Compared the scope with the Task 2 text in the course PDF; the hour estimate is a rough guess |
| 12 | | Plain-English commit messages for each code chunk | Draft messages with short summary and detail lines | Edited to match what each commit actually contains |
| 13 | | Update README and AI log; steps to set up the environment for a teammate | Updated `README.md`; short setup steps | Checked the file list and run commands against the repository |

### Corrections made after AI output

- Kubernetes scheduler code snippet was given from memory and was not checked against the repository; do not cite it without verifying.
- The Task 1 notebook first claimed `start` was optimal in state 1; running the code showed a tie, and the text was corrected.
- The `start` row for state 1 in the Task 1 policy-iteration code was changed by us so the CPU becomes Busy with certainty.
- AI first stated that the Task 1 deadline (Oct 29) was the current day when it was Oct 31; it noticed and corrected this later. Confirm the deadline with the instructor.

## References

1. H. Mao et al., Resource Management with Deep Reinforcement Learning, HotNets-XV, 2016.
2. R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction.