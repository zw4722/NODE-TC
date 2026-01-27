import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        # 使用 Tanh 或 ELU 代替 Softplus 往往在 ODE 中更稳定
        # 因为它们的导数有界，不容易导致数值爆炸
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(), 
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, x):
        # 肿瘤增长动力学通常不直接依赖绝对时间 t，而是依赖当前状态 x
        return self.net(x)

class NODETC(nn.Module):
    def __init__(self, num_clusters, obs_dim, static_dim, hidden_dim=64):
        super().__init__()
        self.num_clusters = num_clusters
        self.obs_dim = obs_dim
        
        # 编码器：将静态变量映射为 ODE 的初始状态 z(t0)
        self.encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # K 个不同的 ODE 函数，每个代表一个聚类簇的动力学
        self.ode_funcs = nn.ModuleList([
            ODEFunc(obs_dim, hidden_dim) for _ in range(num_clusters)
        ])

    def forward(self, s_i, t_steps, cluster_idx):
        """
        s_i: [1, static_dim]
        t_steps: [T]
        cluster_idx: int
        """
        # 1. 编码初始状态
        z0 = self.encoder(s_i) # [1, obs_dim]
        
        # 2. 调用自适应步长求解器 dopri5 (Dormand-Prince)
        # dopri5 比 rk4 更聪明，它会自动根据误差调整步长
        # 如果动力学变剧烈，它会缩小步长；平缓时放大步长，从而兼顾速度和精度
        try:
            zt = odeint(
                self.ode_funcs[cluster_idx], 
                z0, 
                t_steps, 
                method='dopri5', 
                rtol=1e-3, 
                atol=1e-4
            )
        except Exception as e:
            # 捕获可能的积分器崩溃
            print(f"ODE Integration Error at cluster {cluster_idx}: {e}")
            # 返回一个全零张量防止梯度直接断掉，或者进行其他错误处理
            return torch.zeros((len(t_steps), 1, self.obs_dim)).to(s_i.device)

        return zt # 输出形状 [T, 1, obs_dim]