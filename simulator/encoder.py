import numpy as np
from simulator import CAPACITY, HORIZON, N_RES

M = 10              # visible job slots
BACKLOG_CELLS = 60  # backlog image width (paper sets backlog to 60 jobs)


def _grid(used):
    """used: int array (HORIZON, N_RES) -> 0/1 grid (HORIZON, N_RES, CAPACITY)."""
    return (np.arange(CAPACITY) < used[:, :, None]).astype(np.float32)


def encode(state):
    """State dict from env.get_state() -> flat float32 vector of length 4460."""
    cluster = _grid(CAPACITY - state["free"])

    slots = np.zeros((M, HORIZON, N_RES, CAPACITY), dtype=np.float32)
    for i, (duration, demand) in enumerate(state["slots"]):
        used = np.zeros((HORIZON, N_RES), dtype=int)
        used[:duration] = demand                     # first `duration` rows need `demand` units
        slots[i] = _grid(used)

    backlog = (np.arange(BACKLOG_CELLS) < state["backlog"]).astype(np.float32)   # first `backlog` cells = 1
    return np.concatenate([cluster.ravel(), slots.ravel(), backlog])


if __name__ == "__main__":
    from simulator import SchedulerEnv

    env = SchedulerEnv(seed=0)
    x = encode(env.get_state())
    print("vector length:", x.shape[0])               # expect 4460

    # hand example: empty cluster, one job (T=3, demand [4, 1]) in slot 1
    state = {"free": np.full((HORIZON, N_RES), CAPACITY),
             "slots": [(3, np.array([4, 1]))],
             "backlog": 3}
    x = encode(state)
    slot1 = x[400:800].reshape(HORIZON, N_RES, CAPACITY)
    print("cluster all zero:", x[:400].sum() == 0)     # expect True
    for name, r in [("CPU", 0), ("memory", 1)]:
        print(f"slot 1 {name} grid (rows 0-3):")
        for t in range(4):
            print(f"  row {t}:", " ".join(str(int(v)) for v in slot1[t, r]))
    print("slot 1 row 3 sum:", slot1[3].sum())         # expect 0.0 (job ended after 3 rows)
    print("slot 2 sum:", x[800:1200].sum())            # expect 0.0 (empty slot)
    print("backlog first 5 cells:", x[-60:][:5])       # expect [1. 1. 1. 0. 0.]
    print("backlog sum:", x[-60:].sum())               # expect 3.0
