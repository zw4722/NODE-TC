import argparse
import json
from pathlib import Path
from datetime import datetime

# 同时导入两种模拟机制
from node_tc.simulate import linear, tumor


def main():
    parser = argparse.ArgumentParser(description="轨迹聚类数据模拟脚本")

    # --- 核心机制切换参数 ---
    parser.add_argument(
        "--mechanism",
        type=str,
        default="linear",
        choices=["linear", "tumor"],
        help="数据生成机制。linear: 基础线性动力学；tumor: 肿瘤-免疫交互系统(ODE耦合)",
    )

    # --- 通用参数 ---
    parser.add_argument("--data_dir", type=str, default="./data/simulate/",
                        help="输出数据的基础目录路径")
    parser.add_argument("--num_patients", type=int, default=1000,
                        help="模拟样本数")
    parser.add_argument("--num_clusters", type=int, default=3, choices=[2, 3, 4],
                        help="潜在簇数")
    parser.add_argument("--obs_dim", type=int, default=2,
                        help="观测维度。tumor 至少 2 维(肿瘤+免疫)")
    parser.add_argument("--static_dim", type=int, default=3,
                        help="静态协变量维度")
    parser.add_argument("--missing_rate", type=float, default=0.1,
                        help="缺失率 (0~1)")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="高斯噪声标准差")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_time_interval", type=int, nargs=2, default=[4, 12],
                        help="观测次数区间 [min, max)，如 [4,12] 表示 4..11 次")
    parser.add_argument("--plot", action="store_true",
                        help="是否保存可视化图（可选）")

    args = parser.parse_args()

    # --- 路径：mechanism/ + timestamp 子目录，避免覆盖 ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.data_dir) / args.mechanism
    save_path = base_dir / f"{args.mechanism}_{stamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    # --- 机制分发 ---
    if args.mechanism == "linear":
        print(">>> 正在运行原有线性机制 (linear)...")
        dataset = linear.simulate(
            num_patients=args.num_patients,
            num_clusters=args.num_clusters,
            obs_dim=args.obs_dim,
            static_dim=args.static_dim,
            noise_std=args.noise_std,
            missing_rate=args.missing_rate,
            seed=args.seed,
            num_time_interval=tuple(args.num_time_interval),
        )
        dynamics_pool = None
    else:
        print(">>> 正在运行新版肿瘤动力学机制 (tumor)...")
        # 保证 tumor 的 obs_dim >= 2，同时让 config 与实际一致
        args.obs_dim = max(2, args.obs_dim)

        dataset, dynamics_pool = tumor.simulate(
            num_patients=args.num_patients,
            num_clusters=args.num_clusters,
            obs_dim=args.obs_dim,
            static_dim=args.static_dim,
            noise_std=args.noise_std,
            missing_rate=args.missing_rate,
            seed=args.seed,
            num_time_interval=tuple(args.num_time_interval),
        )

    # --- 保存数据（兼容你现有 dataset.to_csv） ---
    dataset.to_csv(save_path)

    # --- 保存 args.json（兼容 train.py 读取） ---
    with open(save_path / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"Successfully simulated {args.num_patients} samples via '{args.mechanism}' mechanism.")
    print(f"Data saved to: {save_path}")

    if args.plot:
        print(f"Plotting is not implemented yet. Target dir: {save_path / 'plots/'}")
        # TODO: plot logic


if __name__ == "__main__":
    main()
