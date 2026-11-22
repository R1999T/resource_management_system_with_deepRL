# AVD894 Course Project: DeepRM (Mao et al., 2016)

Course project for AVD894. Paper: H. Mao, M. Alizadeh, I. Menache, S. Kandula,
*Resource Management with Deep Reinforcement Learning*, HotNets 2016.

## Status

| Item | State |
|---|---|
| Task 1: simplified discrete MDP + policy/value iteration | Done (see Task 1 below) |
| Task 2: simulator (Python) | Code done, report pending |
| RL agent: encoder, policy network, training, evaluation with 95% confidence intervals | Done (see `CONTEXT.md` and `README.md` section 3) |
| Extensions: invalid-action masking, loads up to 190% with Poisson arrivals | Done |
| Term paper, presentation | Not started |

## Repository contents

| File | Purpose |
|---|---|
| `<task1 file name>` | Task 1: 4-state scheduling MDP solved by policy iteration |
| `task1_value_iteration.ipynb` | Task 1: same MDP solved by value iteration, convergence plot |
| `simulator.py` | Task 2: `Job`, `Cluster`, `WorkloadGenerator`, `SchedulerEnv` |
| `test_workload.py` | Checks the workload against the paper's numbers |
| `test_env.py` | Checks total reward equals minus the sum of slowdowns |
| `run_demo.py` | Random and SJF policies, sample-path plot (`results/figures/sample_path.pdf`) |
| `encoder.py`, `policy.py` | State encoder (4460 numbers) and `PolicyNet` (4460-20-11) |
| `train.py` | REINFORCE training with a per-jobset baseline |
| `evaluate.py`, `plot_eval.py` | Random, SJF and the agent on unseen seeds with 95% CI; bar chart |
| `invalid_actions.py` | How often the agent picks an invalid action |
| `load_sweep.py`, `load_sweep_high.py` | Slowdown at different loads (up to 102%, and up to 190% with Poisson arrivals) |
| `results/` | `models/`, `logs/`, `tables/`, `figures/` |
| `CONTEXT.md` | Project summary for a teammate (and their Claude chat) to learn the code |
| `OUR_DECISIONS_baseline_padding.md`, `ptsneededpondering.md` | Our choices that are not in the paper, and open points |
| `README.md` | How to run and verify everything |
| `AI_iteraction_log.md` | This file: status and the AI interaction log |

How to run: see `README.md`.

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
| 14 | `63c3f56`, `83e8ade`, `b3e382b` (teammate's commits) | Design the state encoder and the policy network following the paper | AI explained and drafted `encoder.py` and `policy.py` in small chunks after our go-ahead; we ran and edited the code | Compared sizes with Sec. 4.1 (20 hidden neurons, 11 outputs, 89,451 parameters, so 4460 inputs); ran `python encoder.py` (vector length 4460) and `python policy.py` (89451 parameters, probabilities sum to 1). We replaced a tensor-to-float conversion that gave a warning by `.item()` |
| 15 | `1ffa2b2`, `7e80bba`, `2feaca3`, `ff2a3d4` | Episode rollout, returns v_t, baseline and advantages; what to do when episodes of one jobset have different lengths | `rollout`, `compute_returns`, `compute_advantages` with zero padding; `OUR_DECISIONS_baseline_padding.md` | Hand examples ([-4, -1, -2] gives returns [-7, -3, -2]; two episodes of 3 and 5 steps give advantages [3, 3, 3] and [-3, -3, -3, -3, -1.5]). AI reported that the authors' released code (`pg_re.py`, lines 194 to 202) pads with zeros the same way; we have not re-checked that file, so verify before citing it |
| 16 | `e5d6699` | Policy-gradient update (loss and RMSprop step) | `update` in `train.py` | Test on one fixed state: the probability of action 3 with advantage +1 rose from 0.090 to 0.267 in 20 updates. We worked the sign of the loss by hand (advantage -3 and log-prob -2.0 gives a loss term of -6) |
| 17 | `fe7f851`, `960ad10` | Training loop (paper Fig. 3), model saving, a timing log | `train` and the command `python train.py train jobsets episodes iterations`; `training_log.csv` | Small run (5 x 5 x 20): mean slowdown fell from 5.98 to 3.30. Final run (20 x 10 x 300) took 3530 s and ended at mean slowdown 1.70 on the training seeds. These numbers are on training seeds, so they are not a fair comparison |
| 18 | `baf7b0a` | Evaluate Random, SJF and the agent on unseen seeds with confidence intervals | Design discussed first (same seeds for all policies, 95% CI formula, argmax against sampling); `evaluate.py`, `plot_eval.py` | Ran 100 unseen seeds (1000 to 1099). Cross-checked that mean reward divided by the number of jobs matches the mean slowdown (for example 185 / 35 about 5.3 for Random). We confirmed the CI idea on a small example by hand |
| 19 | `baf7b0a`, teammate's `1d0bb40` | Why the argmax agent hits the 500-timestep cap | A throw-away diagnostic script on seeds 1000 to 1002, later `invalid_actions.py` | The diagnostic showed one job left, an empty cluster and action 2 (an empty slot) chosen again and again. `invalid_actions.py` gave about 61% invalid actions in sampling mode and about 94% in argmax mode when a job could start. Our reading (the network uses invalid actions as "wait") is a guess and is not tested |
| 20 | `baf7b0a` | Extension: mask invalid actions at evaluation time | `valid_mask` and the `masked` option in `evaluate.py` | Cap hits fell from 100 to 0, but the mean slowdown rose from 1.87 to 2.21 (the confidence intervals do not overlap). We kept this result as it is |
| 21 | teammate's `2633bf7`, `1c502e5`; `81ac8fb` | Slowdown at different loads; loads up to 190% | `load_sweep.py` (up to 102%, one arrival per step at most); `load_sweep_high.py` and `WorkloadGenerator.sample_many` (Poisson arrivals) | AI pointed out that our simulator cannot exceed about 102% load with one Bernoulli arrival per step (load is about 1.025 x p). We checked the load formula by hand (4.1 x 2.5 / 10). AI reviewed the diff of `sample_many` (same job rules as `sample`, only the number of arrivals differs, only `_arrive` is overridden) and the results. The Poisson model is not the paper's model |
| 22 | `ea7bca6`, `e5f117b`, `b82813a` | Write our decisions, open points, a handoff note and the README section for the agent | `OUR_DECISIONS_baseline_padding.md`, `ptsneededpondering.md`, `CONTEXT.md`, `README.md` | Copied the numbers from `results/tables/` and `training_log.csv`, and corrected one sentence in `CONTEXT.md` after checking that the SJF and sampling-agent confidence intervals overlap |
| 23 | the commits above | Help with git: commit order, merging with the teammate's commits (`git pull --rebase --autostash`), splitting work so the teammate commits some files | Step-by-step commands | Checked `git status`, `git log` and file comparisons with `origin/master` before removing local copies of files that the teammate had pushed |

### Corrections made after AI output

- Kubernetes scheduler code snippet was given from memory and was not checked against the repository; do not cite it without verifying.
- The Task 1 notebook first claimed `start` was optimal in state 1; running the code showed a tie, and the text was corrected.
- The `start` row for state 1 in the Task 1 policy-iteration code was changed by us so the CPU becomes Busy with certainty.
- AI first stated that the Task 1 deadline (Oct 29) was the current day when it was Oct 31; it noticed and corrected this later. Confirm the deadline with the instructor.
- AI first proposed 128 hidden neurons; the paper says 20. It also proposed the backlog as one number (4401 inputs); we use 60 cells (4460 inputs), which is inferred from the paper's 89,451 parameters and is not stated in the paper. AI also said the paper states that job slot images and the cluster image have the same size; the paper does not state this explicitly.
- AI edited `train.py` for timing and chose the settings of the load sweep without asking first; we asked it to ask before such decisions.
- AI first wrote decisions for the report ("what we do about it") in `ptsneededpondering.md` without asking; they are marked as proposed.
- AI said from memory that the Tetris Packer heuristic takes the largest alignment first. The paper text says "increasing order" and we do not have the Tetris paper, so this is open. Tetris* and Packer are not implemented.
- A first metric in `invalid_actions.py` (steps with one job left in the queue) did not check that the job could be placed; we replaced it with "steps where a job could be started".

task1:

Why this fixes the issue:

• No Absorbing Loops: In your original wait matrix, once you hit FREE_JOB_WAITING, you could never leave it if you kept choosing WAIT. Changing it to 0.95 and 0.05 allows algorithms to evaluate alternative pathways safely.
• Maintains Intent: The system still behaves exactly how you originally designed it 95% of the time, meaning your reward optimization policy will yield the correct logical behavior without breaking the math background.


## References

1. H. Mao et al., Resource Management with Deep Reinforcement Learning, HotNets-XV, 2016.
2. R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction.