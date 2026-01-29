from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypedDict, NotRequired

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SimulatedSample:
    """
    存储单个模拟个体的数据。

    Attributes:
        true_cluster: 真实簇标签
        t: 观测时间点 (n_obs,)
        obs: 观测值 (n_obs, D)
        id: 个体ID（建议为 0..N-1 连续）
        static_vars: 静态协变量 (d_s,) 或 None
        t_: 真值时间点 (n_true,) 或 None
        obs_: 真值潜在状态/无噪声轨迹 (n_true, Dz) 或 None
    """
    true_cluster: int
    t: np.ndarray
    obs: np.ndarray
    id: int
    static_vars: np.ndarray | None = None
    t_: np.ndarray | None = None
    obs_: np.ndarray | None = None


@dataclass
class SimulatedDataset:
    samples: list[SimulatedSample]

    def __post_init__(self):
        self.true_k = np.array([p.true_cluster for p in self.samples], dtype=int)
        # 修改处：移除 .tolist()，直接传入 ndarray
        self.num_clusters = len(set(self.true_k))

        self.n_static_vars = 0
        for sample in self.samples:
            if sample.static_vars is not None:
                self.n_static_vars = int(sample.static_vars.shape[0])
                break

        if self.n_static_vars > 0:
            self.static_vars = np.stack(
                [
                    p.static_vars
                    if p.static_vars is not None
                    else np.full((self.n_static_vars,), np.nan, dtype=float)
                    for p in self.samples
                ],
                axis=0,
            )

    def __repr__(self) -> str:
        d = int(self.samples[0].obs.shape[1]) if len(self.samples) > 0 else -1
        return (
            f"SimulatedDataset(num_patients={len(self.samples)}, "
            f"num_clusters={self.num_clusters}, obs_dim={d})"
        )

    def to_csv(self, dir: str | Path) -> None:
        dir = Path(dir)
        dir.mkdir(exist_ok=True, parents=True)

        indice = np.array([p.id for p in self.samples], dtype=int)

        # meta.csv
        df_meta = pd.DataFrame(self.true_k[:, None], index=indice, columns=["label"])
        if self.n_static_vars > 0:
            df_static_vars = pd.DataFrame(
                self.static_vars,
                index=indice,
                columns=[f"static_{i}" for i in range(self.n_static_vars)],
            )
            df_meta = pd.concat([df_meta, df_static_vars], axis=1)
        df_meta.to_csv(dir / "meta.csv")

        # observations.csv
        obs_arr = []
        obs_index = []
        obs_dim = int(self.samples[0].obs.shape[1])
        for s in self.samples:
            obs_arr.append(np.concatenate([s.t[:, None], s.obs], axis=1))
            obs_index.extend([s.id] * len(s.t))
        obs_arr = np.concatenate(obs_arr, axis=0)

        pd.DataFrame(
            obs_arr,
            index=np.array(obs_index, dtype=int),
            columns=["t"] + [f"x{i}" for i in range(obs_dim)],
        ).to_csv(dir / "observations.csv")

        # true_observations.csv（可选）
        has_true = all((s.t_ is not None and s.obs_ is not None) for s in self.samples)
        if has_true:
            z_dim = int(self.samples[0].obs_.shape[1])  # type: ignore[union-attr]
            true_arr = []
            true_index = []
            for s in self.samples:
                t_ = s.t_  # type: ignore[assignment]
                z_ = s.obs_  # type: ignore[assignment]
                true_arr.append(np.concatenate([t_[:, None], z_], axis=1))
                true_index.extend([s.id] * len(t_))
            true_arr = np.concatenate(true_arr, axis=0)

            pd.DataFrame(
                true_arr,
                index=np.array(true_index, dtype=int),
                columns=["t"] + [f"z{i}" for i in range(z_dim)],
            ).to_csv(dir / "true_observations.csv")

    @classmethod
    def read_csv(cls, dir: str | Path) -> SimulatedDataset:
        dir = Path(dir)
        df_meta = pd.read_csv(dir / "meta.csv", index_col=0)
        df_obs = pd.read_csv(dir / "observations.csv", index_col=0)

        true_path = dir / "true_observations.csv"
        df_obs_ = pd.read_csv(true_path, index_col=0) if true_path.exists() else None

        samples: list[SimulatedSample] = []
        for ind, df_i in df_obs.groupby(level=0):
            # pandas index 可能是 np.int64
            ind_int = int(ind)

            df_i = df_i.sort_values("t")
            t_i = df_i["t"].to_numpy(dtype=float)
            obs_i = df_i.filter(regex=r"^x\d+$").to_numpy(dtype=float)

            if df_obs_ is not None:
                df_obs__i = df_obs_.loc[[ind]].sort_values("t")  # 保留为 DataFrame
                t__i = df_obs__i["t"].to_numpy(dtype=float)
                obs__i = df_obs__i.filter(regex=r"^z\d+$").to_numpy(dtype=float)
            else:
                t__i, obs__i = None, None

            df_meta_i = df_meta.loc[ind_int]
            if isinstance(df_meta_i, pd.DataFrame):
                df_meta_i = df_meta_i.iloc[0]

            if any(col.startswith("static_") for col in df_meta_i.index):
                static_vars_i = df_meta_i.filter(regex=r"^static_\d+$").to_numpy(dtype=float)
            else:
                static_vars_i = None

            samples.append(
                SimulatedSample(
                    id=ind_int,
                    true_cluster=int(df_meta_i["label"]),
                    t=t_i,
                    obs=obs_i,
                    static_vars=static_vars_i,
                    t_=t__i,
                    obs_=obs__i,
                )
            )

        return SimulatedDataset(samples=samples)


class SimuItem(TypedDict):
    id: int
    t: torch.Tensor
    x: torch.Tensor
    y: int
    z: NotRequired[torch.Tensor]


class SimulatedDatasetForTorch(Dataset):
    def __init__(
        self,
        samples: list[SimulatedSample] | SimulatedDataset,
        transform: Callable[[SimuItem], SimuItem] | None = None,
    ):
        if isinstance(samples, SimulatedDataset):
            samples = samples.samples
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SimuItem:
        s = self.samples[index]
        res: SimuItem = {
            "id": int(s.id),
            "t": torch.tensor(s.t, dtype=torch.float32),
            "x": torch.tensor(s.obs, dtype=torch.float32),
            "y": int(s.true_cluster),
        }
        if s.static_vars is not None:
            res["z"] = torch.tensor(s.static_vars, dtype=torch.float32)

        if self.transform is not None:
            res = self.transform(res)
        return res
