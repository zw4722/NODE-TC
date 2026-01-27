from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json

from node_tc.simulate import tumor
from node_tc.plot import TrajectoriesPlotter


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/simulate/tumor/",
                        help="数据保存路径前缀")
    parser.add_argument("--num_patients", type=int, default=1000)
    parser.add_argument("--num_clusters", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--obs_dim", type=int, default=5)
    parser.add_argument("--static_dim", type=int, default=3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_time_interval", type=int, nargs=2, default=[4, 12],
                        help="观测次数区间（右开），默认 4..11 用 [4,12]")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir.rstrip("/") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"数据保存路径: {data_dir}")

    simu_data, dynamics = tumor.simulate(
        num_patients=args.num_patients,
        num_clusters=args.num_clusters,
        obs_dim=args.obs_dim,
        static_dim=args.static_dim,
        missing_rate=args.missing_rate,
        noise_std=args.noise_std,
        seed=args.seed,
        num_time_interval=tuple(args.num_time_interval),
        time_max=10.0,
    )
    simu_data.to_csv(data_dir)

    with open(data_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.plot:
        print("绘制生成数据集的轨迹图...")
        sample = simu_data.samples[0]
        plotter = TrajectoriesPlotter(
            t=sample.t,
            observations=sample.obs,
            trajectories_dfdt=None,  # tumor dynamics 非线性且 1D，这里可不传
        )
        fig = plotter.plot_trajectories()
        fig.savefig(data_dir / "sampled_trajectory.png")


if __name__ == "__main__":
    main()
