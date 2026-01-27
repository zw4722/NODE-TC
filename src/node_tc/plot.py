from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    # 仅用于类型检查，运行时不会导入，从而避免触发 estimator->trainer 的链式导入
    from .estimator import NODETrajectoryCluster


class TrajectoriesPlotter:
    def __init__(
        self,
        t: np.ndarray,
        observations: np.ndarray,
        trajectories: Callable[[np.ndarray], np.ndarray] | None = None,
        trajectories_dfdt: Callable[[float, np.ndarray], np.ndarray] | None = None,
        nodetc_estimator: "NODETrajectoryCluster" | None = None,
        nodetc_estimator_k: int | None = None,
        num_intervals: int = 100,
    ) -> None:
        assert t.ndim == 1, f"t should be 1D array, got shape {t.shape}"
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        assert observations.ndim == 2, (
            f"observations should be 2D array, got shape {observations.shape}"
        )

        if (
            trajectories is not None
            or trajectories_dfdt is not None
            or nodetc_estimator is not None
        ):
            assert (
                (trajectories is not None)
                + (trajectories_dfdt is not None)
                + (nodetc_estimator is not None)
            ) == 1, (
                "Only one of trajectories, trajectories_dfdt, or nodetc_estimator should be provided"
            )

            t_min, t_max = float(np.min(t)), float(np.max(t))
            t_eval = np.linspace(t_min, t_max, num_intervals)

            if trajectories is not None:
                x = trajectories(t_eval)

            elif trajectories_dfdt is not None:
                # 初始值用最小时间点对应的观测
                y0 = observations[t == t_min][0]
                x = solve_ivp(
                    trajectories_dfdt,
                    [t_min, t_max],
                    y0,
                    t_eval=t_eval,
                ).y.T

            else:
                # nodetc_estimator is not None
                assert nodetc_estimator_k is not None, (
                    "nodetc_estimator_k must be provided when using nodetc_estimator"
                )

                # ✅ 更稳：估计 device（优先用 estimator.device，没有就用 model 参数 device）
                device = getattr(nodetc_estimator, "device", None)
                if device is None:
                    # estimator 里一般有 model_，取第一个参数的 device
                    device = next(nodetc_estimator.model_.parameters()).device

                y0 = torch.tensor(
                    observations[t == t_min][0],
                    dtype=torch.float32,
                    device=device,
                )
                t_eval_t = torch.tensor(t_eval, dtype=torch.float32, device=device)

                with torch.no_grad():
                    nodetc_estimator.model_.eval()
                    x = (
                        nodetc_estimator.model_.solve_ivp(
                            nodetc_estimator_k, y0, t_eval_t
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

            assert x.shape[1] == observations.shape[1], (
                "trajectories and observations must have the same number of dimensions"
            )

        self.t = t
        self.observations = observations
        self.ndim = self.observations.shape[1]
        self.flag_has_trajectories = (
            trajectories is not None
            or trajectories_dfdt is not None
            or nodetc_estimator is not None
        )
        if self.flag_has_trajectories:
            self.t_eval = t_eval
            self.x = x

    def plot_trajectories(self, fig: Figure | None = None) -> Figure:
        n_row = int(np.sqrt(self.ndim))
        n_col = int(np.ceil(self.ndim / n_row))

        if fig is None:
            fig = plt.figure(figsize=(n_col * 4, n_row * 3))

        axs = fig.subplots(ncols=n_col, nrows=n_row)
        axs = axs.flatten()

        for i in range(self.ndim):
            ax = axs[i]
            ax.plot(self.t, self.observations[:, i], "o", label=f"Dim {i + 1}")
            if self.flag_has_trajectories:
                ax.plot(self.t_eval, self.x[:, i], "-b", label=f"Dim {i + 1}")
            ax.grid(True)

        fig.tight_layout()
        return fig

    def plot_trajectories_2d(self, fig: Figure | None = None) -> Figure:
        if fig is None:
            fig = plt.figure(figsize=(3 * self.ndim, 3 * self.ndim))

        axs = fig.subplots(ncols=self.ndim, nrows=self.ndim, squeeze=False)
        for i in range(self.ndim):
            for j in range(self.ndim):
                if i == j:
                    axs[i, j].axis("off")
                    continue
                ax = axs[i, j]
                ax.scatter(
                    self.observations[:, j],
                    self.observations[:, i],
                    color="black",
                    label="Observations",
                )
                if self.flag_has_trajectories:
                    ax.plot(
                        self.x[:, j],
                        self.x[:, i],
                        "-r",
                        label="Trajectory",
                    )
                ax.set_xlabel(f"Dim {j + 1}")
                ax.set_ylabel(f"Dim {i + 1}")
                ax.grid(True)

        fig.tight_layout()
        return fig

