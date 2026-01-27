from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import json

from node_tc.simulate.dataset import SimulatedDataset, SimulatedDatasetForTorch
from node_tc.estimator import NODETrajectoryCluster  # ✅ 避免 node_tc/__init__.py 循环导入


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./results/tumor/")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_type", type=str, choices=["simulate", "real"], default="simulate")

    # ✅ 这些参数要和 estimator.py 的 __init__ 对齐
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--obs_dim", type=int, default=5)
    parser.add_argument("--static_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)

    # ✅ 真正生效的 batch_size（在 fit 里做梯度累积）
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_shuffle", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir.rstrip("/") + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}/")

    # --- 数据集加载 ---
    if args.data_type == "simulate":
        simu_data = SimulatedDataset.read_csv(data_dir)

        # ✅ 用数据本身最大时间归一化（对 tumor / linear / 真实数据都通用）
        t_global_max = max(float(s.t.max()) for s in simu_data.samples if len(s.t) > 0)

        def transform(item):
            item["t"] = item["t"] / float(t_global_max + 1e-8)
            return item

        dataset = SimulatedDatasetForTorch(simu_data, transform)
    else:
        raise NotImplementedError("暂不支持真实数据集")

    # --- 模型训练 ---
    model = NODETrajectoryCluster(
        num_clusters=args.num_clusters,
        obs_dim=args.obs_dim,
        static_dim=args.static_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
    )

    model.fit(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
    )

    # --- 结果保存 ---
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    model.save_model(save_dir / "model.pt")
    model.save_history(save_dir / "history.csv")


if __name__ == "__main__":
    main()

