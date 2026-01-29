from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.integrate import solve_ivp

from .dataset import SimulatedDataset, SimulatedSample


@dataclass
class CoupledTumorDynamicsD:
    """
    D维广义 Lotka-Volterra 交互系统 (gLV):
    dx_i/dt = r_i * x_i * ln(K_i / x_i) + x_i * (M @ x)_i
    """
    r: np.ndarray        # (D,)
    K: np.ndarray        # (D,)
    M: np.ndarray        # (D, D) 交互矩阵

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        # 使用 1e-6 确保数值稳定，防止 log(0)
        x_safe = np.maximum(x, 1e-6)
        
        # 基础增长项 (Gompertz)
        growth = self.r * x_safe * np.log(self.K / x_safe)
        
        # 多维交互项 (M @ x 为其它指标对当前维度的影响)
        interact = x_safe * (self.M @ x_safe)
        
        return growth + interact

def simulate(
    num_patients: int = 1000,
    num_clusters: int = 3,
    obs_dim: int = 5,
    static_dim: int = 3,
    num_time_interval: tuple[int, int] = (4, 12),
    time_max: float = 10.0,
    missing_rate: float = 0.1,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[SimulatedDataset, List[CoupledTumorDynamicsD]]:
    
    assert obs_dim >= 2, "obs_dim must be >= 2 for interaction dynamics"
    rng = np.random.default_rng(seed)
    D = obs_dim

    # 1) 定义 Cluster 动力学池 (强化可分性)
    dynamics_pool: List[CoupledTumorDynamicsD] = []
    for k in range(num_clusters):
        r = rng.uniform(0.15, 0.35, size=D)
        K = rng.uniform(10.0, 16.0, size=D)
        
        # 系统性差异：调整不同簇的基础演化速率
        if k == 0:      # 清除型: 肿瘤慢，免疫快
            r[0] *= 0.8; r[1] *= 1.2; K[0] *= 0.9; K[1] *= 1.1
        elif k == 1:    # 逃逸型: 肿瘤快，免疫慢
            r[0] *= 1.2; r[1] *= 0.8; K[0] *= 1.1; K[1] *= 0.9
        
        M = np.zeros((D, D))
        # 核心交互: x0(肿瘤) vs x1(免疫)
        if k == 0:    M[0, 1], M[1, 0] = -0.45, +0.10
        elif k == 1:  M[0, 1], M[1, 0] = -0.05, +0.02
        else:         M[0, 1], M[1, 0] = -0.30, +0.30
            
        if D > 2:
            M[1, 2:] = rng.uniform(-0.06, 0.06, size=D-2)
            M[2:, 1] = rng.uniform(-0.06, 0.06, size=D-2)
            
            # 其它指标间的稀疏交互：先计算再清零对角线
            sparse_mask = rng.random((D-2, D-2)) < 0.2
            M_others = rng.uniform(-0.04, 0.04, size=(D-2, D-2)) * sparse_mask
            np.fill_diagonal(M_others, 0.0)
            M[2:, 2:] = M_others

        # 补丁：全局自抑制项（gLV稳定性的关键）
        np.fill_diagonal(M, -0.02)
        dynamics_pool.append(CoupledTumorDynamicsD(r=r, K=K, M=M))

    # 2) 样本生成逻辑
    true_k = rng.integers(0, num_clusters, size=num_patients)
    static_vars = rng.normal(0, 1, size=(num_patients, static_dim)) if static_dim > 0 else None
    
    # 调小 subject_shift 以防掩盖动力学信号
    subject_shift = rng.normal(0, 0.08, size=(num_patients, D))

    samples: list[SimulatedSample] = []
    for i in range(num_patients):
        k_i = int(true_k[i])
        dyn = dynamics_pool[k_i]
        static_i = static_vars[i] if static_vars is not None else None
        
        # 补丁：初始值 x0_init 的前两维加 Cluster 偏移
        x0_init = rng.uniform(1.2, 3.2, size=D)
        if k_i == 0:    x0_init[0] *= 0.85; x0_init[1] *= 1.15
        elif k_i == 1:  x0_init[0] *= 1.15; x0_init[1] *= 0.85
        
        if static_i is not None:
            x0_init += static_i @ rng.normal(0, 0.15, size=(static_dim, D))
        x0_init = np.maximum(x0_init, 0.5)

        # 补丁：确保 t_eval 严格递增且保持采样点数
        n_t = rng.integers(*num_time_interval)
        t_raw = np.sort(rng.uniform(0.0, time_max, size=n_t - 1))
        t_eval = np.concatenate(([0.0], t_raw))
        eps = 1e-6
        for j in range(1, len(t_eval)):
            if t_eval[j] <= t_eval[j-1]:
                t_eval[j] = t_eval[j-1] + eps
        t_eval = np.clip(t_eval, 0.0, time_max)
        
        sol = solve_ivp(
            dyn, (0.0, time_max), x0_init, 
            t_eval=t_eval, method="RK45", 
            rtol=1e-4, atol=1e-6
        )
        if not sol.success: continue

        z_clean = sol.y.T 
        # 观测值包含 shift，但 z_clean (真值) 保持纯净
        x_noisy = (z_clean + subject_shift[i]) + rng.normal(0.0, noise_std, size=z_clean.shape)
        
        if missing_rate > 0:
            keep = rng.random(size=x_noisy.shape[0]) >= missing_rate
            keep[0] = True
            t_obs, x_obs = t_eval[keep], x_noisy[keep]
        else:
            t_obs, x_obs = t_eval, x_noisy

        samples.append(SimulatedSample(
            id=i, true_cluster=k_i, static_vars=static_i,
            t=t_obs, obs=x_obs, t_=t_eval, obs_=z_clean
        ))

    return SimulatedDataset(samples=samples), dynamics_pool