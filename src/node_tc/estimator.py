import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI

from .model import NODETC
from .simulate.dataset import SimulatedDatasetForTorch


class NODETrajectoryCluster:
    def __init__(
        self,
        num_clusters: int = 3,
        obs_dim: int = 5,
        static_dim: int = 3,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 100,
        device=None,
    ):
        self.num_clusters = num_clusters
        self.obs_dim = obs_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NODETC(
            num_clusters=num_clusters,
            obs_dim=obs_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.lr = lr
        self.epochs = epochs
        self.history = []

        # ✅ optimizer & mixture weights π 放在 estimator 自己管理，避免 trainer 循环导入
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.pi = torch.full((self.num_clusters,), 1.0 / self.num_clusters, dtype=torch.float32, device=self.device)

    def fit(self, dataset, batch_size: int = 64, shuffle: bool = True):
        """
        batch_size: ✅ 真正生效：每 batch_size 个样本 optimizer.step() 一次（梯度累积）
        """
        torch_dataset = dataset if isinstance(dataset, SimulatedDatasetForTorch) else SimulatedDatasetForTorch(dataset)

        n = len(torch_dataset)
        print(f"==> 任务启动 | 设备: {self.device} | 簇数: {self.num_clusters} | N={n} | batch_size={batch_size}")
        print(f"{'Epoch':<6} | {'Loss':<10} | {'ARI':<8} | {'NMI':<8} | {'Status'}")
        print("-" * 60)

        main_pbar = tqdm(range(self.epochs), desc="Overall", position=0, leave=True)

        for epoch in main_pbar:
            # 0) index 顺序（支持 shuffle）
            indices = np.arange(n)
            if shuffle:
                np.random.shuffle(indices)

            # 1) E-Step
            weights, labels_true, preds = self._run_estep(torch_dataset, indices)

            # 2) M-Step (mini-batch)
            avg_loss = self._run_mstep(torch_dataset, indices, weights, batch_size=batch_size)

            # 3) metrics
            ari_val = ARI(labels_true, preds)
            nmi_val = NMI(labels_true, preds)

            metrics = {"epoch": int(epoch + 1), "loss": float(avg_loss), "ari": float(ari_val), "nmi": float(nmi_val)}
            self.history.append(metrics)

            tqdm.write(f"{epoch+1:<6} | {avg_loss:<10.4f} | {ari_val:<8.3f} | {nmi_val:<8.3f} | OK")
            main_pbar.set_postfix({"ARI": f"{ari_val:.3f}", "Loss": f"{avg_loss:.4f}"})

        return self.history

    def _run_estep(self, dataset: SimulatedDatasetForTorch, indices: np.ndarray):
        self.model.eval()

        weights_list = []
        labels_true = []
        preds = []

        with torch.no_grad():
            for idx in tqdm(indices, desc="  [E-Step]", position=1, leave=False):
                item = dataset[int(idx)]

                # 兼容没有 z 的情况
                if "z" in item:
                    s_i = item["z"].unsqueeze(0).to(self.device)  # (1, static_dim)
                else:
                    s_i = torch.zeros((1, self.static_dim), dtype=torch.float32, device=self.device)

                t_i = item["t"].to(self.device)   # (T,)
                y_i = item["x"].to(self.device)   # (T, D)
                labels_true.append(int(item["y"]))

                log_probs = []
                for k in range(self.num_clusters):
                    pred_k = self.model(s_i, t_i, k).squeeze(1)  # (T, D) or (T, ?)
                    mask = ~torch.isnan(y_i)

                    if not torch.isfinite(pred_k).all():
                        mse = torch.tensor(1e6, device=self.device)
                    else:
                        mse = torch.mean((pred_k[mask] - y_i[mask]) ** 2) if mask.any() else torch.tensor(0.0, device=self.device)

                    log_probs.append(-mse + torch.log(self.pi[k] + 1e-9))

                w_i = torch.softmax(torch.stack(log_probs), dim=0)  # (K,)
                weights_list.append(w_i)
                preds.append(int(torch.argmax(w_i).item()))

        weights = torch.stack(weights_list, dim=0)  # (N, K) in "indices 顺序"
        labels_true = np.array(labels_true, dtype=int)
        preds = np.array(preds, dtype=int)
        return weights, labels_true, preds

    def _run_mstep(self, dataset: SimulatedDatasetForTorch, indices: np.ndarray, weights: torch.Tensor, batch_size: int = 64):
        """
        ✅ 真 mini-batch：每 batch_size 个样本 step 一次
        注意：weights 的行顺序与 indices 一致，所以这里用 row_id 对应 weights[row_id]
        """
        self.model.train()

        total_loss = 0.0
        valid_samples = 0

        self.optimizer.zero_grad(set_to_none=True)

        # row_id: 0..N-1 对应 weights 的行
        for row_id, idx in enumerate(tqdm(indices, desc="  [M-Step]", position=1, leave=False)):
            item = dataset[int(idx)]

            if "z" in item:
                s_i = item["z"].unsqueeze(0).to(self.device)
            else:
                s_i = torch.zeros((1, self.static_dim), dtype=torch.float32, device=self.device)

            t_i = item["t"].to(self.device)
            y_i = item["x"].to(self.device)
            mask = ~torch.isnan(y_i)

            # 对该样本的 EM 加权损失
            sample_loss = torch.tensor(0.0, device=self.device)
            is_valid = True

            for k in range(self.num_clusters):
                w_ik = weights[row_id, k]
                if w_ik > 1e-4:
                    pred_k = self.model(s_i, t_i, k).squeeze(1)
                    if not torch.isfinite(pred_k).all():
                        is_valid = False
                        break
                    mse = torch.mean((pred_k[mask] - y_i[mask]) ** 2) if mask.any() else torch.tensor(0.0, device=self.device)
                    sample_loss = sample_loss + w_ik * mse

            if is_valid:
                # ✅ 关键：按 batch_size 归一化梯度（否则 batch_size 改变会改变有效学习率）
                (sample_loss / float(batch_size)).backward()
                total_loss += float(sample_loss.detach().cpu().item())
                valid_samples += 1

            # ✅ 每 batch_size 个样本更新一次
            if (row_id + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # 处理最后一个不足 batch 的尾巴
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # ✅ 更新混合权重 π（weights 是当前 epoch 的后验）
        with torch.no_grad():
            self.pi = weights.mean(dim=0).to(self.device)

        return total_loss / max(valid_samples, 1)

    def save_history(self, path):
        pd.DataFrame(self.history).to_csv(path, index=False)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
