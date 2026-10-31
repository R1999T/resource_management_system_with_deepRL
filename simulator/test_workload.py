import numpy as np
from simulator import WorkloadGenerator

N = 100000
gen = WorkloadGenerator(p_arrival=0.7, rng=np.random.default_rng(0))
jobs = [j for t in range(N) if (j := gen.sample(t)) is not None]

durs = np.array([j.duration for j in jobs])
dem = np.array([j.demand for j in jobs])

print("arrival rate:", len(jobs) / N)                              # about 0.7
print("fraction short (1-3):", np.mean(durs <= 3))                 # about 0.8
print("mean duration:", durs.mean())                               # about 4.1
print("mean demand per resource:", dem.mean(axis=0))               # about [2.5 2.5]
print("dominant demands seen:", sorted(set(dem.max(axis=1).tolist())))  # [3, 4, 5]