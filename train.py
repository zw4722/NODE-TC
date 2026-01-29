from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import json

# 修改：直接从顶层包导入，遵循解耦规范
from node_tc.simulate.dataset import SimulatedDataset, SimulatedDatasetForTorch
from node_tc import NODETrajectoryCluster


def main():
    parser = ArgumentParser(description="NODE-TC 模型训练脚本：用于处理模拟或真实的轨迹聚类任务")
    
    # --- 路径与环境配置 ---
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/experiment/",
        help="实验结果（模型、日志、参数）的保存根目录"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/simulate/tumor_latest/",
        help="数据集目录, 需包含 data.csv 和 args.json"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["simulate", "real"],
        default="simulate",
        help="数据来源类型: simulate (带标签) 或 real (无标签)"
    )

    # --- 超参数配置 ---
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="聚类数量 (K)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="学习率"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="EM 算法总迭代轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小"
    )

    # --- 模型架构与算法细节 ---
    parser.add_argument(
        "--bn",
        action="store_true",
        help="是否使用批归一化"
    )
    parser.add_argument(
        "--adjoint",
        action="store_true",
        help="是否使用伴随方法"
    )
    parser.add_argument(
        "--update_nn_params_epochs_every_round",
        type=int,
        default=2,
        help="每轮 M-step 更新神经网络的轮数"
    )
    
    args = parser.parse_args()

    # --- 路径处理：增加时间戳，防止实验记录被覆盖 ---
    data_dir = Path(args.data_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f"{Path(args.save_dir).name}_{timestamp}"
    save_dir = Path(args.save_dir).parent / save_name

    # --- 数据集加载与预处理 ---
    if args.data_type == "simulate":
        print(f">>> 读取模拟数据集: {data_dir}")
        simu_data = SimulatedDataset.read_csv(data_dir)

        # 健壮的时间归一化处理
        with open(data_dir / "args.json", "r") as f:
            simu_args = json.load(f)
        
        # 兼容 tumor 和 linear 的时间区间设定
        intervals = simu_args.get("num_time_interval", [4, 12])
        max_t = float(intervals[1] - 1)

        def transform(sample):
            # 将时间缩放到 [0, 1] 提升数值稳定性
            sample["t"] = sample["t"] / max_t
            return sample

        dataset = SimulatedDatasetForTorch(simu_data, transform)
    else:
        raise NotImplementedError("暂不支持真实数据集")

    # --- 模型初始化 ---
    model = NODETrajectoryCluster(
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        bn=args.bn,
        adjoint=args.adjoint,
        update_nn_params_epochs_every_round=args.update_nn_params_epochs_every_round,
    )

    # --- 核心训练过程 ---
    print(f">>> 开始 EM 训练 (目标簇数: {args.num_clusters})...")
    # fit 方法内部会调用 EMTrainer，后者已被修改为自动计算并记录 ARI 和 NMI
    model.fit(
        dataset,
        time_key="t",
        obs_key="x",
        id_key="id",
        label_key="y" if args.data_type == "simulate" else None,
        static_vars_key="z",
    )

    # --- 结果持久化 ---
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练配置
    config_to_save = vars(args)
    config_to_save["train_timestamp"] = timestamp
    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        
    # 保存模型状态
    model.save_model(save_dir / "model.pt")
    # 保存训练历史 (内部包含 epoch, loss, ari, nmi, entropy 等列)
    model.save_history(save_dir / "history.csv")
    
    print(f"\n[完成] 结果已保存至: {save_dir}")


if __name__ == "__main__":
    main()