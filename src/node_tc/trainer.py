from __future__ import annotations

from typing import overload, Literal
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import NODETC


def compute_entropy(probabilities: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """计算熵"""
    res = -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)
    if normalize:
        res = res / torch.log(
            torch.tensor(probabilities.shape[1], dtype=torch.float32, device=probabilities.device)
        )
    return res


class EMTrainer:
    def __init__(
        self,
        model: NODETC,
        loader: DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr: float = 1e-2,
        num_epochs: int = 50,
        update_nn_params_epochs_every_round: int = 1,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.update_nn_params_epochs_every_round = update_nn_params_epochs_every_round

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self._obs_start: np.ndarray | None = None
        self._obs_end: np.ndarray | None = None

        # ✅ 关键修复：建立 id -> row 映射，禁止 responsibilities[batch["id"]] 这种写法
        self._id2row = self._build_id2row(loader)

    def _build_id2row(self, loader: DataLoader) -> dict[int, int]:
        ds = loader.dataset
        # 兼容你 SimulatedDatasetForTorch：里面有 samples，sample.id 是原始 id
        if not hasattr(ds, "samples"):
            raise AttributeError(
                "DataLoader.dataset has no attribute 'samples'. "
                "Please ensure dataset has 'samples' with each sample having an 'id'."
            )
        id2row: dict[int, int] = {}
        for row, sample in enumerate(ds.samples):
            sid = int(sample.id)
            if sid in id2row:
                raise ValueError(f"Duplicated sample id found in dataset: {sid}")
            id2row[sid] = row
        return id2row

    def _batch_ids_to_rows(self, batch_ids: torch.Tensor) -> torch.Tensor:
        # batch_ids: (B,)
        ids_list = batch_ids.detach().cpu().numpy().tolist()
        rows = [self._id2row[int(i)] for i in ids_list]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    @overload
    def e_step(self, loader: DataLoader, return_true_clusters: Literal[False]) -> torch.Tensor: ...
    @overload
    def e_step(
        self, loader: DataLoader, return_true_clusters: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def e_step(
        self, loader: DataLoader, return_true_clusters: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        responsibilities, sid, true_clusters = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="E-Step:", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                llk = self.model.get_log_likelihoods(batch)

                joint_log_prob = llk + self.model.log_prior  # type: ignore
                log_marginal_prob = torch.logsumexp(joint_log_prob, dim=1, keepdim=True)
                log_respon = joint_log_prob - log_marginal_prob
                respon = torch.exp(log_respon)

                responsibilities.append(respon)
                sid.append(batch["id"])
                if return_true_clusters:
                    true_clusters.append(batch["y"])

        responsibilities = torch.cat(responsibilities, dim=0)
        sid = torch.cat(sid, dim=0)

        # ✅ 统一按 id 排序，保证 responsibilities 的“行顺序”可复现
        indice = torch.argsort(sid)
        responsibilities = responsibilities[indice]

        if return_true_clusters:
            true_clusters = torch.cat(true_clusters, dim=0)
            true_clusters = true_clusters[indice]
            return responsibilities, true_clusters

        return responsibilities

    def update_nn_params(self, loader: DataLoader, responsibilities: torch.Tensor) -> float:
        self.model.train()
        loss_epoch = 0.0
        for batch in tqdm(loader, desc="M-Step(Update NN params):", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            llk = self.model.get_log_likelihoods(batch)

            row_idx = self._batch_ids_to_rows(batch["id"])
            respon_k = responsibilities[row_idx]  # ✅ 正确索引

            loss = self.model.compute_loss(respon_k.detach(), llk)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
        return loss_epoch / max(len(loader), 1)

    def update_pi(self, responsibilities: torch.Tensor):
        pi_new = torch.mean(responsibilities, dim=0)
        self.model.log_prior.data = torch.log(pi_new + 1e-6)

    def update_cov(self, loader: DataLoader, responsibilities: torch.Tensor):
        residue_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        weight_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for batch in tqdm(loader, desc="M-Step(Update cov):", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                residue_i = self.model.get_residue(batch)

                row_idx = self._batch_ids_to_rows(batch["id"])
                respon_i = responsibilities[row_idx]  # ✅ 正确索引

                respon_i_weight = respon_i[..., None] * batch["mask"][:, None, :]
                residue_sum = residue_sum + (residue_i.pow(2) * respon_i_weight[..., None]).sum(dim=(0, 2))
                weight_sum = weight_sum + respon_i_weight.sum(dim=(0, 2))

        var = residue_sum / weight_sum[:, None]
        self.model.log_vars.data = torch.log(var.clamp(min=1e-6))

    def find_observation_range(self):
        obs_start_list, obs_end_list = [], []
        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Finding observation range:", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                obs = batch["x"]
                mask = batch["mask"]

                i, j = torch.nonzero(mask == 1.0, as_tuple=True)
                obs_start_list.append(obs[i, j].min(dim=0).values)
                obs_end_list.append(obs[i, j].max(dim=0).values)

        self._obs_start = torch.stack(obs_start_list, dim=0).min(dim=0)[0].detach().cpu().numpy()
        self._obs_end = torch.stack(obs_end_list, dim=0).max(dim=0)[0].detach().cpu().numpy()

    def train(self) -> list[dict[str, float | int]]:
        self.find_observation_range()

        best_ari = -1.0
        best_model = None
        best_epoch = -1

        history: list[dict[str, float | int]] = []
        responsibilities = self.e_step(self.loader, False)

        for epoch in tqdm(range(self.num_epochs), desc="Training: "):
            self.update_pi(responsibilities)
            self.update_cov(self.loader, responsibilities)

            for _ in range(self.update_nn_params_epochs_every_round):
                loss = self.update_nn_params(self.loader, responsibilities)
                tqdm.write(f"Epoch {epoch + 1} | Loss: {loss:.4f}")

            responsibilities, true_clusters = self.e_step(self.loader, True)

            pred_clusters = torch.argmax(responsibilities, dim=1).cpu().numpy()
            true_clusters_np = true_clusters.cpu().numpy()

            ari = adjusted_rand_score(true_clusters_np, pred_clusters)
            entropy = compute_entropy(responsibilities).mean().item()
            tqdm.write(f"Epoch {epoch + 1} | Adjusted Rand Index: {ari:.4f}, Entropy: {entropy:.4f}")

            history.append({"epoch": epoch + 1, "ari": float(ari), "entropy": float(entropy)})

            if ari >= best_ari:
                best_model = deepcopy(self.model.state_dict())
                best_ari = ari
                best_epoch = epoch + 1

        tqdm.write(f"Best ARI: {best_ari:.4f} at epoch {best_epoch}")
        if best_model is not None:
            self.model.load_state_dict(best_model)

        tqdm.write("训练完成!")
        return history

