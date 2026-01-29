from typing import Type

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint


class MLP(nn.Module):
    def __init__(
        self,
        inpt_dim: int,
        oupt_dim: int = 1,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        net = []
        for i, o in zip([inpt_dim] + hiddens[:-1], hiddens):
            net.append(nn.Linear(i, o))
            if bn:
                net.append(nn.BatchNorm1d(o))
            net.append(activation())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        net.append(nn.Linear(hiddens[-1], oupt_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    """学习ODE动态的神经网络 f(z, t; theta)"""

    def __init__(
        self,
        inpt_dim: int,
        oupt_dim: int = 1,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
        autonomous: bool = True,
    ):
        super(ODEFunc, self).__init__()

        self.autonomous = autonomous
        self.net = MLP(
            inpt_dim if self.autonomous else inpt_dim + 1,
            oupt_dim,
            hiddens,
            activation,
            bn,
            dropout,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.autonomous:
            return self.net(z)

        return self.net(torch.cat([z, t.unsqueeze(1)], dim=1))


class InitStateEncoder(nn.Module):
    """根据静态变量和首次观测推断初始状态 z0"""

    def __init__(
        self,
        obs_dim: int,
        static_dim: int,
        latent_dim: int,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
    ):
        super(InitStateEncoder, self).__init__()
        self.net = MLP(
            obs_dim + static_dim,
            latent_dim,
            hiddens=hiddens,
            activation=activation,
            bn=bn,
            dropout=dropout,
        )

    def forward(
        self, first_obs: torch.Tensor, static_vars: torch.Tensor
    ) -> torch.Tensor:
        input_data = torch.cat([first_obs, static_vars], dim=1)
        return self.net(input_data)


class NODETC(nn.Module):
    """动态混合神经ODE模型"""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        static_dim: int,
        num_clusters: int,
        hiddens: list[int] = [64, 64],
        activation: Type[nn.Module] = nn.Tanh,
        bn: bool = False,
        dropout: float = 0.0,
        autonomous: bool = True,
        adjoint: bool = True,
        init_state_encoder: bool = False,
        method: str | None = None,
        options: dict | None = None,
    ):
        super(NODETC, self).__init__()
        self.num_clusters = num_clusters
        self.obs_dim = obs_dim
        self.adjoint = adjoint
        self.init_state_encoder = init_state_encoder
        self.method = method
        self.options = options

        # 为每个簇创建一个独立的ODE函数
        self.ode_funcs = nn.ModuleList(
            [
                ODEFunc(
                    latent_dim, latent_dim, hiddens, activation, bn, dropout, autonomous
                )
                for _ in range(num_clusters)
            ]
        )

        # 创建一个共享的编码器
        if self.init_state_encoder:
            self.encoder = InitStateEncoder(
                obs_dim, static_dim, latent_dim, hiddens, activation, bn, dropout
            )
        else:
            self.init_state = nn.Parameter(torch.randn(latent_dim))

        # 如果观测维度和潜在维度不同，需要一个解码器
        if obs_dim != latent_dim:
            self.decoder = nn.Linear(latent_dim, obs_dim)
        else:
            self.decoder = nn.Identity()

        # 模型参数
        # 簇权重 (logits形式，通过softmax保证和为1)
        self.register_buffer(
            "log_prior", torch.log_softmax(torch.zeros(num_clusters), dim=0)
        )

        # 每个簇的协方差 (这里简化为对角阵，学习log(sigma^2))
        self.register_buffer("log_vars", torch.randn(num_clusters, obs_dim))

    def solve_ivp(self, k: int, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        func = self.ode_funcs[k]
        if self.adjoint:
            pred_z = odeint_adjoint(
                func, z0, t, method=self.method, options=self.options
            )  # (T, B, D)
        else:
            pred_z = odeint(
                func, z0, t, method=self.method, options=self.options
            )  # (T, B, D)
        assert torch.is_tensor(pred_z), "ODE solver must return a tensor"
        if z0.ndim == 2:
            pred_z = pred_z.permute(1, 0, 2)  # (B, T, D)
        return pred_z

    def forward(self, patient_data: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        t = patient_data["t"]  # (T,)
        obs = patient_data["x"]  # (B, T, D)

        if self.init_state_encoder:
            z0 = obs[:, 0]  # (B, D)
            static = patient_data["z"]  # (B, D)
            z0 = self.encoder(z0, static)
        else:
            z0 = self.init_state.repeat(obs.shape[0], 1)

        preds = []
        for k in range(self.num_clusters):
            # 1. 求解ODE得到潜在轨迹
            pred_z = self.solve_ivp(k, z0, t)

            # 2. 解码到观测空间
            pred_i = self.decoder(pred_z)
            preds.append(pred_i)

        return preds

    def get_residue(self, patient_data: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = patient_data["x"]  # (B, T, D)
        mask = patient_data["mask"]  # (B, T)
        preds = self.forward(patient_data)

        residue = []
        for pred_obs in preds:
            # 3. 计算残差
            residue_k = (obs - pred_obs) * mask[..., None]
            residue.append(residue_k)

        return torch.stack(residue, dim=1)  # (B, K, T, D)

    def get_log_likelihoods(
        self, patient_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算单个患者数据在每个簇下的对数似然"""
        obs = patient_data["x"]  # (B, T, D)
        mask = patient_data["mask"]  # (B, T)

        preds = self.forward(patient_data)

        log_likelihoods_k = []
        for k, pred_obs in enumerate(preds):
            # 3. 计算高斯对数似然
            # 使用torch.distributions简化计算
            dist = torch.distributions.Normal(
                loc=pred_obs,
                scale=torch.exp(0.5 * self.log_vars[k]),  # type:ignore
            )
            # 假设各维度独立，对数似然是各维度和各时间点之和
            log_p = dist.log_prob(obs).mul(mask[..., None]).sum(dim=(1, 2))
            # log_p = log_p / (mask.sum(dim=1) * self.obs_dim)
            log_likelihoods_k.append(log_p)

        return torch.stack(log_likelihoods_k, dim=1)  # (B, K)

    def compute_loss(
        self, responsibilities: torch.Tensor, log_likelihoods: torch.Tensor
    ) -> torch.Tensor:
        """计算M-step的损失函数 (Q函数)"""
        # 加权的对数似然
        weighted_log_likelihood = (responsibilities * log_likelihoods).sum(dim=1).mean()

        # EM算法的目标是最大化Q函数，等价于最小化其负值
        return -weighted_log_likelihood