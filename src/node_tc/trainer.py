from typing import overload, Literal
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .model import NODETC


def compute_entropy(probabilities: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """计算预测概率的熵，用于监控聚类结果的确定性"""
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

    def e_step(self, loader: DataLoader) -> dict[str, torch.Tensor]:
        """
        高性能 E-Step：
        1. 遍历计算所有样本的后验概率。
        2. 在 GPU 上通过排序确保 responsibilities[i] 对应 id=i 的样本。
        """
        self.model.eval()
        all_respon, all_ids, all_y = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="E-Step", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                llk = self.model.get_log_likelihoods(batch)
                
                # 计算联合对数概率并归一化得到后验 w_ik
                joint_log_prob = llk + self.model.log_prior
                log_marginal_prob = torch.logsumexp(joint_log_prob, dim=1, keepdim=True)
                respon = torch.exp(joint_log_prob - log_marginal_prob)

                all_respon.append(respon)
                all_ids.append(batch["id"])
                if "y" in batch:
                    all_y.append(batch["y"])

        # 🚀 性能优化点：在内存中只执行一次排序对齐
        responsibilities = torch.cat(all_respon, dim=0)
        u_ids = torch.cat(all_ids, dim=0)
        # 得到排序索引，保证后续索引时 id 与 row 严格匹配
        sort_idx = torch.argsort(u_ids)
        
        res = {"resp": responsibilities[sort_idx]}
        if all_y:
            res["y"] = torch.cat(all_y, dim=0)[sort_idx]
            
        return res

    def update_nn_params(self, loader: DataLoader, responsibilities: torch.Tensor) -> float:
        """M-Step (Part 1): 更新 Neural ODE 的网络权重"""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="M-Step(NN)", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            llk = self.model.get_log_likelihoods(batch)
            
            # ✅ 高性能索引：直接使用 GPU Tensor 索引，无 CPU 往返开销
            target_resp = responsibilities[batch["id"]]
            
            # 计算 Q 函数（负值作为 loss）
            loss = self.model.compute_loss(target_resp.detach(), llk)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def update_pi_and_cov(self, loader: DataLoader, responsibilities: torch.Tensor):
        """M-Step (Part 2): 更新混合系数 pi 和观测方差 sigma"""
        # 1. 更新 pi
        pi_new = torch.mean(responsibilities, dim=0)
        self.model.log_prior.data = torch.log(pi_new + 1e-6)

        # 2. 更新 sigma (协方差)
        res_sum = torch.zeros_like(self.model.log_vars)
        w_sum = torch.zeros(self.model.num_clusters, device=self.device)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="M-Step(Cov)", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                residue = self.model.get_residue(batch) # (B, K, T, D)
                mask = batch["mask"] # (B, T)
                
                target_resp = responsibilities[batch["id"]]
                # 结合责任分数和时间掩码
                combined_w = target_resp[..., None] * mask[:, None, :] # (B, K, T)
                
                res_sum += (residue.pow(2) * combined_w[..., None]).sum(dim=(0, 2))
                w_sum += combined_w.sum(dim=(0, 2))

        new_var = res_sum / w_sum[:, None].clamp(min=1e-6)
        self.model.log_vars.data = torch.log(new_var.clamp(min=1e-6))

    def train(self) -> list[dict]:
        self.find_observation_range() # 绘图初始化
        
        history = []
        best_ari = -1.0
        best_state = None

        # 初始 E-Step
        e_output = self.e_step(self.loader)
        resp = e_output["resp"]

        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            # M-Step: 先更新统计参数，再更新神经网络参数
            self.update_pi_and_cov(self.loader, resp)
            
            epoch_loss = 0.0
            for _ in range(self.update_nn_params_epochs_every_round):
                epoch_loss = self.update_nn_params(self.loader, resp)

            # E-Step: 更新下一轮的责任分配分数
            e_output = self.e_step(self.loader)
            resp = e_output["resp"]

            # 指标记录（仅当存在 y 标签时计算 ARI/NMI）
            metrics = {"epoch": epoch + 1, "loss": epoch_loss}
            metrics["entropy"] = compute_entropy(resp).mean().item()

            if "y" in e_output:
                true_y = e_output["y"].cpu().numpy()
                pred_y = resp.argmax(dim=1).cpu().numpy()
                metrics["ari"] = adjusted_rand_score(true_y, pred_y)
                metrics["nmi"] = normalized_mutual_info_score(true_y, pred_y)
                
                # 记录并保存最佳模型（基于 ARI）
                if metrics["ari"] > best_ari:
                    best_ari = metrics["ari"]
                    best_state = deepcopy(self.model.state_dict())
            
            # --- 仅修改此打印逻辑区块 ---
            log_str = f"Epoch {epoch+1:02d} | Loss: {metrics['loss']:.4f} | Entropy: {metrics['entropy']:.4f}"
            if "ari" in metrics:
                # 在此处补上 NMI 的显示
                log_str += f" | ARI: {metrics['ari']:.4f} | NMI: {metrics['nmi']:.4f}"
            tqdm.write(log_str)
            # --------------------------
            
            history.append(metrics)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        print("\nTraining completed.")
        return history

    def find_observation_range(self):
        """确定绘图坐标轴范围"""
        starts, ends = [], []
        with torch.no_grad():
            for batch in self.loader:
                x, m = batch["x"], batch["mask"]
                valid_x = x[m == 1.0]
                if valid_x.numel() > 0:
                    starts.append(valid_x.min(dim=0)[0])
                    ends.append(valid_x.max(dim=0)[0])
        if starts:
            self._obs_start = torch.stack(starts).min(dim=0)[0].cpu().numpy()
            self._obs_end = torch.stack(ends).max(dim=0)[0].cpu().numpy()

    def plot_vector_field(self) -> Figure:
        """隐空间向量场可视化 (仅限 D=2)"""
        if self.model.obs_dim != 2:
            print("Vector field plot is only supported for 2D observation space.")
            return plt.figure()

        start, end = self._obs_start, self._obs_end
        fig, axs = plt.subplots(1, self.model.num_clusters, figsize=(6 * self.model.num_clusters, 6))
        if self.model.num_clusters == 1: axs = [axs]

        n_steps = 20
        x_range = np.linspace(start[0], end[0], n_steps)
        y_range = np.linspace(start[1], end[1], n_steps)
        X, Y = np.meshgrid(x_range, y_range)
        grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=-1), 
                                  dtype=torch.float32, device=self.device)

        for i in range(self.model.num_clusters):
            dz_dt = self.model.ode_funcs[i](torch.tensor(0.0).to(self.device), grid_points)
            u = dz_dt[:, 0].reshape(n_steps, n_steps).detach().cpu().numpy()
            v = dz_dt[:, 1].reshape(n_steps, n_steps).detach().cpu().numpy()

            ax = axs[i]
            ax.streamplot(X, Y, u, v, color="red", linewidth=0.7, arrowstyle="->")
            ax.set_title(f"Cluster {i + 1}")
            ax.set_xlim(start[0], end[0])
            ax.set_ylim(start[1], end[1])
            ax.set_aspect("equal")

        fig.tight_layout()
        return fig
