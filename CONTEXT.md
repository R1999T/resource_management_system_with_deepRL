# CONTEXT: DeepRM course project (handoff note)

Last updated: 2026-11-22.

This note lets a second chat (the "dub session") answer concept questions about the project without seeing the coding chat. The coding chat writes the code. The second chat only explains. Read this note, then read the code files named below when a question needs them.

## 1. Project in one paragraph

Course project for AVD894 (reinforcement learning, IIST). The paper is Mao, Alizadeh, Menache, Kandula (2016), "Resource Management with Deep Reinforcement Learning" (DeepRM, HotNets). A cluster has two resources (CPU, memory). Jobs arrive over time and wait in a queue. A scheduler decides which waiting job to start. DeepRM trains a small neural network with a policy gradient method (REINFORCE with a baseline) to schedule jobs so that the average job slowdown is small. The project reproduces this and compares the trained agent with simple rules (Random, Shortest Job First).

Slowdown of a job = (completion time - arrival time) / job duration. It is at least 1. Smaller is better.

## 2. Course deliverables and dates

| Item | Marks | Due |
|---|---|---|
| Task 1 (small discrete MDP solved with policy iteration) | 5 | Oct 26 |
| Task 2 (simulator) | 5 | Nov 2 |
| Code on GitHub with proper commit history and an AI log | 10 | Nov 25 (moved from Nov 23, as the student reports) |
| Term paper, 6 pages, NeurIPS style, LaTeX | 10 | Nov 25 (moved from Nov 23, as the student reports) |
| Presentation, 10 minutes | 10 | Nov 25 to 27 |

Rules that matter for the work: claims of improvement need confidence intervals (a mean alone is not accepted). The term paper needs an "Extensions" section. Use of AI must be logged at each commit. The student must be able to explain everything in a viva. It is a team project; a teammate (GitHub user abhishek1466) also commits to the same repository.

## 3. Status

| Part | State |
|---|---|
| Task 1 notebook and report | Done |
| Task 2 simulator (`simulator.py`) | Done, tests pass |
| Task 2 report | Unknown whether submitted |
| State encoder, policy network, training (`encoder.py`, `policy.py`, `train.py`) | Done |
| Evaluation on unseen jobsets against Random and SJF with 95% confidence intervals (`evaluate.py`) | Done |
| Extension 1: invalid-action masking (evaluation time only) | Done |
| Extension 2: loads up to 190% with Poisson arrivals | Done |
| Results at different loads (paper Figure 4 style) | Done, two versions (see section 6) |
| Tetris* and Packer baselines | Not done on purpose. The student decided they are not needed for now. |
| Git repository | Exists, being committed step by step. Remote: GitHub `R1999T/resource_management_system_with_deepRL`. |
| README update for the new scripts, AI log rows for the new commits | Not done |
| Term paper, presentation | Not started |

The code part is considered finished (2026-11-22). What is left is committing, the AI log, the README, the paper and the slides.

## 4. Files

Folder `resource_management_system_with_deepRL/`. Run every script from inside `simulator/` (paths to `../results/` are relative).

| File | What it holds |
|---|---|
| `simulator/simulator.py` | `Job`, `Cluster`, `WorkloadGenerator` (with `sample` and the extension `sample_many`), `SchedulerEnv`. |
| `simulator/encoder.py` | `encode(state)`: state dict to a vector of 4460 numbers. |
| `simulator/policy.py` | `PolicyNet` (torch): 4460 -> 20 -> 11. `act(x)` samples an action and returns its log-probability. |
| `simulator/train.py` | `rollout`, `compute_returns`, `compute_advantages`, `update`, `train`. Run `python train.py train [jobsets episodes iterations]`. |
| `simulator/evaluate.py` | `random_policy`, `sjf_policy`, `valid_mask`, `make_agent_policy(net, mode, masked)`, `run_episode`, `evaluate`, `mean_ci`. Run `python evaluate.py 100`. |
| `simulator/plot_eval.py` | Bar chart from `results/tables/evaluation_results.csv`. |
| `simulator/invalid_actions.py` | How often the agent picks an invalid action. |
| `simulator/load_sweep.py` | Slowdown against load, p from 0.1 to 1.0 (about 10% to 102% load). |
| `simulator/load_sweep_high.py` | Same up to 190% load, with Poisson arrivals (`MultiArrivalEnv`). |
| `simulator/run_demo.py`, `test_env.py`, `test_workload.py` | Demo of Random and SJF on one episode, simulator checks. |
| `results/models/` | `policy.pt` (final, 20x10x300), `policy_small.pt` (10x10x100). |
| `results/logs/`, `results/tables/`, `results/figures/` | Training logs, result tables (csv) and figures (png, pdf). |

Other files in the project folder: `README.md`, `OUR_DECISIONS_baseline_padding.md` (our choices, including the evaluation ones), `ptsneededpondering.md` (open points), `AI_iteraction_log.md`, the Task 1 notebook.

## 5. How the simulator works (`simulator.py`)

- **Cluster:** an array `free` of shape (20, 2). Row 0 is now, row 1 is the next timestep, and so on, 20 rows ahead. Each entry is the number of free units of a resource at that time. Capacity is 10 units per resource.
- **Cluster methods:** `find_start(job)` returns the first row where the job fits for its whole duration (or None). `place(job, t)` subtracts the job's demand from those rows. `advance()` drops row 0 and adds a fully free row at the bottom.
- **WorkloadGenerator.sample(t):** a job arrives with probability p per timestep (at most one job per step). Duration is 1 to 3 (80%) or 10 to 15 (20%). One dominant resource has demand 3 to 5 units, the other resource 1 unit.
- **SchedulerEnv state** (`get_state()`): a dict with `free` (the (20, 2) array), `slots` (a list of (duration, demand) for the first M = 10 waiting jobs) and `backlog` (the number of waiting jobs beyond the first 10).
- **Actions:** 0 = void, i from 1 to 10 = start the job in slot i. 11 actions in total.
- **step(action):** a valid action places the job (it may start at a later row than now), the reward is 0 and time does not move. A void or invalid action (empty slot, or a job that cannot be placed) moves time by one step: the reward is minus the sum of 1/duration over all jobs in the system (waiting or running), the cluster advances, and a new job may arrive.
- **Episode:** jobs arrive during the first 50 timesteps (`episode_len`). The episode ends when no job is waiting or running after that, or at a safety cap of 500 timesteps (`max_t`).
- **Why the reward works:** each job adds 1/duration for every timestep it is in the system, so its total is (completion - arrival)/duration = its slowdown. So the total reward of an episode is minus the sum of the slowdowns.
- **reset(seed=None):** a seed makes the same jobset appear again. Arrivals do not depend on the agent's actions, so the same seed gives the same jobs for any policy.
- **Important detail:** one call to `step()` is not one timestep. Placing a job is a step in which time does not move.
- **Load:** the average load is about 1.025 x p (mean duration 4.1, mean demand 2.5 units per resource out of 10). With one arrival at most per timestep, p cannot exceed 1, so load is at most about 102%.

## 6. How the agent works, and results

**Encoder** (`encode(state)`), vector length 4460 = 400 + 4000 + 60: cluster grid (20 rows, 2 resources, 10 units, 1 = used) 400, ten job slot grids of the same shape 4000, backlog as 60 cells (first `backlog` cells are 1) 60.

**Network** (`PolicyNet`): `Linear(4460, 20)`, ReLU, `Linear(20, 11)`. 89451 parameters, which matches the paper. 11 outputs are scores (logits), softmax gives probabilities.

**Training** (`train.py`): `rollout` runs one episode. `compute_returns` gives v_t = r_t + gamma v_(t+1), gamma = 1. `compute_advantages` uses a time-dependent baseline per jobset (mean over the episodes of that jobset, short episodes padded with zeros). `update` does loss = minus sum of (log-prob x advantage), RMSprop with learning rate 0.001. One weight update per iteration across all jobsets. Paper values are 100 jobsets, 20 episodes, 1000 iterations. Ours is smaller (CPU only).

Call map for `train.py`: `train` calls `rollout`, `compute_returns`, `compute_advantages`, `update`. `rollout` calls `SchedulerEnv.reset`, `get_state`, `encode`, `PolicyNet.act`, `SchedulerEnv.step`.

**Training runs** (`results/logs/training_log.csv`, numbers are on the training seeds 0 to 9, not a fair comparison):

| Run | Jobsets x episodes x iterations | Time | Last mean reward | Last mean slowdown |
|---|---|---|---|---|
| Small | 5 x 5 x 20 | 49 s | -118.19 | 3.30 |
| Medium (`policy_small.pt`) | 10 x 10 x 100 | 579 s | -76.93 | 2.17 |
| Final (`policy.pt`) | 20 x 10 x 300 | 3530 s | -59.96 | 1.70 |

**Evaluation** (`evaluate.py`, final model, 100 unseen seeds 1000 to 1099, same seeds for every policy, load p = 0.7, 95% confidence intervals from the t distribution):

| Policy | Mean slowdown | 95% CI | Mean total reward | Cap hits (episodes stopped at 500 timesteps) |
|---|---|---|---|---|
| Random | 5.299 | [5.108, 5.490] | -184.92 | 0 |
| SJF | 1.693 | [1.589, 1.798] | -59.24 | 0 |
| Agent, argmax | 1.751 | [1.612, 1.891] | -104.42 | 100 |
| Agent, sample | 1.874 | [1.744, 2.004] | -65.89 | 0 |
| Agent, argmax, masked | 2.209 | [2.036, 2.382] | -77.75 | 0 |
| Agent, sample, masked | 2.213 | [2.040, 2.385] | -77.89 | 0 |

The slowdown of an episode counts only finished jobs. For argmax, jobs stay unfinished until the cap, so its slowdown 1.751 looks better than it is. The total reward counts every waiting job and shows the real cost.

Reading: the agent is much better than Random. SJF has the lower mean slowdown at this load (1.693 against 1.874 for the sampling agent), but the two confidence intervals overlap a little ([1.589, 1.798] and [1.744, 2.004]), so at p = 0.7 the gap is not clearly established. The paper reports DeepRM comparable to or better than SJF. Our agent was trained much less.

**Argmax problem** (details in `ptsneededpondering.md`, point 1): the agent in argmax mode gets stuck. At the end of an episode one job waits, the cluster is empty, and argmax picks action 2 again and again (an empty slot, so it is invalid and only time moves) until the cap. `invalid_actions.py` data (final model, steps where a job could be started): the agent picked an invalid action in about 61% of them in sampling mode and about 94% in argmax mode, on training and unseen seeds alike. The medium model shows the same.

**Extension 1, masking** (`valid_mask`, `masked=True`): the logit of every invalid action is set to minus infinity before argmax or sampling. This removes the cap hits (100 to 0) but the slowdown gets worse (2.21 against 1.87 for sampling). A guess, not tested: the agent uses invalid actions as "wait", which keeps resources free for later short jobs, and masking takes that away.

**Load sweeps.**
- `load_sweep.py` (Bernoulli arrivals, p from 0.1 to 1.0, load 10% to 102%): the agent is close to SJF at 50% and 60% load, worse than SJF at low load (10%: 1.29 vs 1.02) and clearly worse at high load (102%: 3.40 vs 2.11). Random is 5 to 6 everywhere.
- `load_sweep_high.py` (Extension 2, Poisson arrivals `sample_many`, load 10% to 190%): same picture. At 170% and 190% the agent is about equal to or slightly worse than Random (190%: agent 13.85, Random 12.77, SJF 6.92). This arrival model is not the paper's (the paper says Bernoulli) and differs from `load_sweep.py`. The agent was trained only at p = 0.7.

## 7. Choices that are ours, not in the paper

| Choice | Where | Note |
|---|---|---|
| Integer units: capacity 10 per resource, dominant demand 3 to 5, other demand 1 | simulator | The paper uses fractions of capacity. |
| Backlog cap of 60 not enforced in the simulator | simulator | The encoder shows at most 60 backlog cells. |
| ReLU activation | policy | The paper does not name it. |
| Backlog as 60-cell image, vector size 4460 | encoder | Inferred from the 89,451 parameters in the paper. Say "inferred". |
| Zero padding of returns when episodes differ in length | train | Matches the authors' released code (`hongzimao/deeprm`, `pg_re.py`), not stated in the paper text. |
| gamma = 1 | train | Needed so that reward = minus slowdown. |
| Smaller training | train | CPU only. |
| Evaluation design: unseen seeds, same seeds for all policies, 95% CI from the t distribution, argmax and sampling both reported, cap hits counted | evaluate | See `OUR_DECISIONS_baseline_padding.md`. The paper does not say whether the agent samples or uses argmax at test time (we did not see it stated). |
| Masking of invalid actions | evaluate | Extension, not in the paper. |
| Poisson arrivals for loads above 102% | simulator | Extension, not in the paper. |

Decisions about what goes into the report are not final. Items marked "proposed" in `ptsneededpondering.md` were not approved yet.

## 8. Paper facts already checked (page numbers of the HotNets PDF)

- Page 3, Section 3.1: state is cluster images, job slot images for the first M jobs, and a backlog count.
- Page 4, Section 3.2 and 3.3: actions, time-freeze rule, reward = minus sum of 1/T_j, training algorithm in Figure 3.
- Page 5, Section 4.1: one hidden layer with 20 neurons, 89,451 parameters, images 20 timesteps long, M = 10, backlog 60, 100 jobsets, N = 20 episodes per jobset, RMSprop with learning rate 0.001, 1000 iterations.
- Page 5, Section 4.2 "Comparables": SJF allocates jobs in increasing order of duration. Packer allocates jobs "in increasing order of alignment between job demands and resource availability (same as the packing heuristic in [17])". Tetris* combines short-job preference and packing in one score. All are work-conserving. [17] is Grandl et al., "Multi-resource packing for cluster schedulers", SIGCOMM 2014. The paper gives no formulas. We did not implement Packer and Tetris*.
- Figure 4: DeepRM, Tetris*, SJF, Packer and Random at loads from 10% to 190%.

## 9. Next steps

1. Commit the remaining files, one logical chunk per commit, with AI log rows.
2. Update `README.md` with how to run the new scripts.
3. Term paper (6 pages, NeurIPS style, with an "Extensions" section) and presentation.
4. Optional: learning curve plot from `results/logs/history.npy` (paper Figure 6 style), a test of why masking hurts (probability of void in masked states), Tetris* and Packer.

## 10. How to help the student (style)

- Language: Hinglish. Short messages. One small step at a time, then ask one question and wait. Do not dump information.
- Always give one small dummy example with numbers. Say which function belongs to which class and file.
- The terminal where the student works does not render LaTeX, so plain text math in code blocks works best there. A chat window may render LaTeX.
- Guesses are welcome. If an answer is wrong, point to where the mistake likely is and let the student retry. Check the numbers before saying it is wrong.
- Cross-check against the paper first, and say when something is not from the paper. Do not put anything into the report without the student's approval.
