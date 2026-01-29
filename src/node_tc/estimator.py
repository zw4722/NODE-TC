from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib.figure import Figure

from .simulate.dataset import SimulatedDataset, SimulatedDatasetForTorch, SimuItem
from .model import NODETC
from .trainer import EMTrainer


class TrajectoryDatasetCollateFunc:
    def __init__(
        self,
        time_key: str = "t",
        obs_key: str = "x",
        id_key: str = "id",
        label_key: str | None = None,
        static_vars_key: str | None = None,
    ) -> None:
        self.time_key = time_key
        self.obs_key = obs_key
        self.label_key = label_key
        self.static_vars_key = static_vars_key
        self.id_key = id_key

    def __call__(
        self, batch: list[SimuItem]
    ) -> dict[str, torch.Tensor]:
        obs_i = batch[0][self.obs_key]
        assert torch.is_tensor(obs_i), f"obs_i is not a torch tensor: {obs_i}"

        t, _ = torch.cat([sample[self.time_key] for sample in batch]).unique().sort()
        obs = torch.zeros((len(batch), len(t), obs_i.shape[1]), dtype=torch.float32)
        mask = torch.zeros((len(batch), len(t)), dtype=torch.float32)
        for i, sample in enumerate(batch):
            t_i = sample[self.time_key]
            obs_i = sample[self.obs_key]
            ind = torch.searchsorted(t, t_i)
            obs[i, ind, :] = obs_i
            mask[i, ind] = 1.0


        res = {
            "t": t,
            "x": obs,
            "mask": mask,
            "id": torch.tensor(
                [sample[self.id_key] for sample in batch], dtype=torch.long
            ),
        }
        if (res["t"][1:] - res["t"][:-1]).min() == 0:
            # fmt: off
            import ipdb; ipdb.set_trace()
            # fmt: on

        if self.label_key is not None:
            if self.label_key not in batch[0]:
                raise ValueError(f"label_key {self.label_key} not in batch[0]")
            res["y"] = torch.tensor(
                [sample[self.label_key] for sample in batch], dtype=torch.long
            )
        if self.static_vars_key is not None:
            if self.static_vars_key not in batch[0]:
                raise ValueError(
                    f"static_vars_key {self.static_vars_key} not in batch[0]"
                )
            res["z"] = torch.stack([sample[self.static_vars_key] for sample in batch], dim=0)

        return res


class NODETrajectoryCluster:
    def __init__(
        self,
        num_clusters: int,
        batch_size: int = 64,
        num_workers: int = 0,
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        bn: bool = False,
        adjoint: bool = False,
        device: str | torch.device = "cuda",
        update_nn_params_epochs_every_round: int = 1,
    ) -> None:
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.bn = bn
        self.adjoint = adjoint
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.update_nn_params_epochs_every_round = update_nn_params_epochs_every_round

        self.params_ = {
            "num_clusters": num_clusters,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "bn": bn,
            "adjoint": adjoint,
            "device": str(device),
            "update_nn_params_epochs_every_round": update_nn_params_epochs_every_round,
        }

    def fit(
        self,
        dataset: Dataset | SimulatedDataset | SimulatedDatasetForTorch,
        time_key: str = "t",
        obs_key: str = "x",
        id_key: str = "id",
        label_key: str | None = None,
        static_vars_key: str | None = None,
    ) -> None:
        """训练模型"""

        if isinstance(dataset, SimulatedDataset):
            dataset = SimulatedDatasetForTorch(dataset)

        collate_fn = TrajectoryDatasetCollateFunc(
            time_key=time_key,
            obs_key=obs_key,
            id_key=id_key,
            label_key=label_key,
            static_vars_key=static_vars_key,
        )
        loader_shuffle = DataLoader(
            dataset,  # type: ignore
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        # loader_noshuffle = DataLoader(
        #     dataset,  # type: ignore
        #     num_workers=self.num_workers,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     collate_fn=collate_fn,
        # )

        batch1 = next(iter(loader_shuffle))
        obs_dim = batch1[obs_key].shape[-1]
        static_dim = 0 if static_vars_key is None else batch1[static_vars_key].shape[-1]

        # 初始化模型
        self.model_ = NODETC(
            obs_dim=obs_dim,
            latent_dim=obs_dim,
            static_dim=static_dim,
            num_clusters=self.num_clusters,
            bn=self.bn,
            adjoint=self.adjoint,
            init_state_encoder=static_dim > 0,
            activation=torch.nn.GELU,
        )

        # 初始化训练器
        self.trainer_ = EMTrainer(
            model=self.model_,
            loader=loader_shuffle,
            num_epochs=self.num_epochs,
            lr=self.learning_rate,
            device=self.device,
            update_nn_params_epochs_every_round=self.update_nn_params_epochs_every_round,
        )

        # 训练模型, 记录训练历史
        history = self.trainer_.train()
        self.history_ = pd.DataFrame.from_records(history)

        # 得到最终属于各自集群的概率
        responsibilities = self.trainer_.e_step(loader_shuffle, False)
        self.responsibilities_ = responsibilities.cpu().numpy()

    def plot_vector_field(self) -> Figure:
        return self.trainer_.plot_vector_field()

    def save_model(self, fn: str | Path) -> None:
        res = {
            "model": self.model_.state_dict(),
            "params": self.params_,
        }
        torch.save(res, fn)

    @classmethod
    def load_model(
        cls, fn: str | Path, device: str | torch.device = "cuda"
    ) -> "NODETrajectoryCluster":
        res = torch.load(fn)
        estimator = cls(**res["params"])
        estimator.model_.load_state_dict(res["model"])
        return estimator

    def save_history(self, fn: str | Path) -> None:
        self.history_.to_csv(fn)
