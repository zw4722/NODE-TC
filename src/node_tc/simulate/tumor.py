from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .dataset import SimulatedDataset, SimulatedSample


@dataclass
class GompertzDynamics:
    rho: float
    k_cap: float  # carrying capacity

    def __call__(self, t: float, V: np.ndarray) -> np.ndarray:
        # V shape: (1,)
        v = np.clip(V[0], 1e-8, None)
        dv = self.rho * v * np.log(self.k_cap / v)
        return np.array([dv], dtype=float)

    def __repr__(self) -> str:
        return f"Gompertz(rho={self.rho:.3f}, Kcap={self.k_cap:.3f})"


def _sample_rho_by_cluster(k: int, num_clusters: int, rng: np.random.Generator) -> float:
    """
    ρ 的区间抽样规则：
    slow[0.15,0.25], mid[0.25,0.35], fast[0.35,0.55]
    - K=2: slow vs fast
    - K=3: slow vs mid vs fast
    - K=4: slow×lowK, slow×highK, fast×lowK, fast×highK（ρ只分 slow/fast）
    """
    slow = (0.15, 0.25)
    mid = (0.25, 0.35)
    fast = (0.35, 0.55)

    if num_clusters == 2:
        lo, hi = slow if k == 0 else fast
    elif num_clusters == 3:
        lo, hi = [slow, mid, fast][k]
    elif num_clusters == 4:
        # 0,1 -> slow ; 2,3 -> fast
        lo, hi = slow if k in (0, 1) else fast
    else:
        # 兜底：循环
        buckets = [slow, mid, fast]
        lo, hi = buckets[k % 3]
    return float(rng.uniform(lo, hi))


def _sample_kcap_by_cluster(k: int, num_clusters: int, rng: np.random.Generator) -> float:
    """
    Kcap 的区间抽样规则：
    low[4,7], high[10,15]
    - K=2 或 3：默认都用 high（避免额外维度干扰），你也可改成 mix
    - K=4：0,2 -> low ; 1,3 -> high （与 slow/fast 交叉）
    """
    low = (4.0, 7.0)
    high = (10.0, 15.0)

    if num_clusters == 4:
        lo, hi = low if k in (0, 2) else high
    else:
        lo, hi = high
    return float(rng.uniform(lo, hi))


def simulate(
    num_patients: int = 1000,
    num_clusters: int = 3,
    obs_dim: int = 5,          # D=5
    static_dim: int = 3,       # d_s=3
    num_time_interval: tuple[int, int] = (4, 12),  # 4..11 (右开区间)
    time_max: float = 10.0,    # T_max=10
    missing_rate: float = 0.1, # p_miss
    noise_std: float = 0.1,    # σ
    seed: int = 42,
) -> tuple[SimulatedDataset, List[GompertzDynamics]]:
    assert obs_dim >= 1, "obs_dim must be >= 1"
    assert 0.0 <= missing_rate < 1.0
    assert num_clusters in (2, 3, 4), "benchmark 默认 K∈{2,3,4}"

    rng = np.random.default_rng(seed)

    # 1) cluster dynamics pool (ρ_k, Kcap_k)
    dynamics_pool: List[GompertzDynamics] = []
    for k in range(num_clusters):
        rho_k = _sample_rho_by_cluster(k, num_clusters, rng)
        kcap_k = _sample_kcap_by_cluster(k, num_clusters, rng)
        dynamics_pool.append(GompertzDynamics(rho=rho_k, k_cap=kcap_k))

    # 2) assign clusters (uniform π)
    true_k = rng.integers(0, num_clusters, size=num_patients)

    # 3) static covariates s_i ~ N(0, I_3)
    static_vars = rng.normal(0, 1, size=(num_patients, static_dim)) if static_dim > 0 else None

    # 4) initial V0 influenced by static vars (log-scale)
    #    让 V0 合理落在 (0, Kcap) 内
    #    logV0 = base + s_i beta + ε
    beta = rng.normal(0, 0.25, size=(static_dim,)) if static_dim > 0 else None
    eps0 = rng.normal(0, 0.2, size=(num_patients,))
    base = -0.2  # 控制初始量级
    logV0 = np.full((num_patients,), base, dtype=float)
    if static_dim > 0:
        logV0 = logV0 + static_vars @ beta  # type: ignore[operator]
    logV0 = logV0 + eps0
    V0_raw = np.exp(logV0)  # positive

    # 5) biomarker mapping: x = a*log(V)+b + subject_shift
    #    x0 = log(V)  (主维度)
    #    x1..xD-1 = linear transforms of log(V)
    a = np.ones((obs_dim,), dtype=float)
    b = np.zeros((obs_dim,), dtype=float)
    if obs_dim > 1:
        a[1:] = rng.uniform(0.6, 1.4, size=(obs_dim - 1,))
        b[1:] = rng.normal(0, 0.3, size=(obs_dim - 1,))
    subject_shift = rng.normal(0, 0.15, size=(num_patients, obs_dim))

    # 6) simulate each patient
    num_time_points = rng.integers(num_time_interval[0], num_time_interval[1], size=num_patients)

    samples: list[SimulatedSample] = []
    for i in range(num_patients):
        k_i = int(true_k[i])
        dyn = dynamics_pool[k_i]

        # clip V0 to (tiny, 0.8*Kcap) to avoid starting near cap
        V0 = float(np.clip(V0_raw[i], 1e-4, 0.8 * dyn.k_cap))
        y0 = np.array([V0], dtype=float)

        n_t = int(num_time_points[i])

        # irregular t (include 0)
        t_eval = np.sort(rng.uniform(0.0, time_max, size=n_t - 1))
        t_eval = np.concatenate(([0.0], t_eval))

        # integrate Gompertz
        sol = solve_ivp(
            fun=dyn,
            t_span=(0.0, time_max + 1e-3),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
        )
        if not sol.success:
            continue

        V_true = sol.y.T[:, 0]  # (n_t,)
        logV_true = np.log(np.clip(V_true, 1e-8, None))

        # build clean multivariate trajectory z(t) in R^D (use clean obs as latent)
        z_clean = (a[None, :] * logV_true[:, None]) + b[None, :] + subject_shift[i][None, :]

        # noisy observations
        x_noisy = z_clean + rng.normal(0.0, noise_std, size=z_clean.shape)

        # missingness (MCAR over timepoints, keep at least t=0)
        if missing_rate > 0:
            keep = rng.random(size=(x_noisy.shape[0],)) >= missing_rate
            keep[0] = True
            t_obs = t_eval[keep]
            x_obs = x_noisy[keep]
        else:
            t_obs = t_eval
            x_obs = x_noisy

        samples.append(
            SimulatedSample(
                id=i,
                true_cluster=k_i,
                static_vars=(static_vars[i] if static_dim > 0 else None),
                t=t_obs,
                obs=x_obs,
                t_=t_eval,
                obs_=z_clean,   # true latent trajectory in R^D
            )
        )

    return SimulatedDataset(samples=samples), dynamics_pool
