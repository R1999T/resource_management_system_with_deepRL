import numpy as np
from dataclasses import dataclass, field
from typing import Optional

CAPACITY = 10      # units per resource (our integer-unit choice)
HORIZON = 20       # lookahead rows in the cluster image (paper, Sec. 4.1)
N_RES = 2          # CPU and memory


@dataclass
class Job:
    job_id: int
    arrival: int                 # timestep the job arrived
    duration: int                # T_j
    demand: np.ndarray           # units needed per resource, e.g. [4, 1]
    start: Optional[int] = None  # timestep the job starts running
    finish: Optional[int] = None # timestep the job completes


class Cluster:
    """Free units per future timestep. Row 0 = now, row 1 = next step, ..."""

    def __init__(self, horizon=HORIZON, capacity=CAPACITY, n_res=N_RES):
        self.horizon = horizon
        self.capacity = capacity
        self.free = np.full((horizon, n_res), capacity, dtype=int)

    def find_start(self, job):
        """First row offset where the job fits for its whole duration, else None."""
        for t in range(self.horizon - job.duration + 1):
            window = self.free[t:t + job.duration]       # rows the job would occupy
            if np.all(window >= job.demand):             # every row, every resource
                return t
        return None

    def place(self, job, t):
        """Reserve the job's demand in rows t .. t+duration-1."""
        self.free[t:t + job.duration] -= job.demand

    def advance(self):
        """Time moves one step: drop row 0, add a fully free row at the bottom."""
        self.free = np.vstack([self.free[1:],
                               np.full((1, self.free.shape[1]), self.capacity, dtype=int)])

class WorkloadGenerator:
    """Creates jobs following the workload in the paper (Sec. 4.1), in integer units."""

    def __init__(self, p_arrival, rng=None):
        self.p = p_arrival                      # probability a job arrives each timestep
        self.rng = rng if rng is not None else np.random.default_rng()
        self.next_id = 0

    def sample(self, t):
        """Return a new Job with probability p, otherwise None."""
        if self.rng.random() >= self.p:
            return None

        # 80% short jobs (1-3 steps), 20% long jobs (10-15 steps)
        if self.rng.random() < 0.8:
            duration = int(self.rng.integers(1, 4))
        else:
            duration = int(self.rng.integers(10, 16))

        # one dominant resource (3-5 units), the other needs 1 unit
        demand = np.ones(N_RES, dtype=int)
        dominant = int(self.rng.integers(0, N_RES))
        demand[dominant] = int(self.rng.integers(3, 6))

        job = Job(self.next_id, arrival=t, duration=duration, demand=demand)
        self.next_id += 1
        return job


class SchedulerEnv:
    """Action 0 = void; action i (1..M) = schedule the job in slot i."""

    def __init__(self, p_arrival=0.7, M=10, episode_len=50, max_t=500, seed=None):
        self.p = p_arrival
        self.M = M
        self.episode_len = episode_len
        self.max_t = max_t
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.cluster = Cluster()
        self.gen = WorkloadGenerator(self.p, self.rng)
        self.queue = []      # waiting jobs, in arrival order (first M are visible slots)
        self.running = []    # scheduled jobs that have not finished yet
        self.finished = []
        self._arrive()
        return self.get_state()

    def _arrive(self):
        if self.t < self.episode_len:
            job = self.gen.sample(self.t)
            if job is not None:
                self.queue.append(job)

    def get_state(self):
        slots = [(j.duration, j.demand.copy()) for j in self.queue[:self.M]]
        backlog = max(0, len(self.queue) - self.M)
        return {"free": self.cluster.free.copy(), "slots": slots, "backlog": backlog}

    def step(self, action):
        # 1) try to schedule a job (time stays frozen)
        if 1 <= action <= self.M and action - 1 < len(self.queue):
            job = self.queue[action - 1]
            offset = self.cluster.find_start(job)
            if offset is not None:
                self.cluster.place(job, offset)
                job.start = self.t + offset
                job.finish = job.start + job.duration
                self.queue.pop(action - 1)
                self.running.append(job)
                return self.get_state(), 0.0, False

        # 2) void or invalid action: time moves forward one step
        in_system = self.queue + [j for j in self.running if j.finish > self.t]
        reward = -sum(1.0 / j.duration for j in in_system)

        self.cluster.advance()
        self.t += 1
        done_now = [j for j in self.running if j.finish <= self.t]
        for j in done_now:
            self.running.remove(j)
            self.finished.append(j)
        self._arrive()

        done = (self.t >= self.episode_len and not self.queue and not self.running) \
               or self.t >= self.max_t
        return self.get_state(), reward, done





if __name__ == "__main__":
    # Test: the seat example from our chat (CPU column only varies)
    c = Cluster(horizon=6)
    c.free[:, 0] = [3, 6, 6, 10, 10, 10]

    j1 = Job(1, arrival=0, duration=2, demand=np.array([4, 1]))
    t1 = c.find_start(j1)
    print("j1 starts at row offset:", t1)           # expect 1 (second row)
    c.place(j1, t1)
    print("CPU free after j1:", c.free[:, 0])       # expect [3 2 2 10 10 10]

    j2 = Job(2, arrival=0, duration=3, demand=np.array([3, 1]))
    print("j2 starts at row offset:", c.find_start(j2))   # expect 3 (fourth row)

    c.advance()
    print("CPU free after advance:", c.free[:, 0])  # expect [2 2 10 10 10 10]