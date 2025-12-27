import json
import argparse
import numpy as np
import pandas as pd  
from trainer import train

def load_json(path):
    with open(path) as f:
        return json.load(f)


def setup_parser():
    parser = argparse.ArgumentParser(description='Evaluate LK-CIL across random seeds.')
    parser.add_argument('--config', type=str, default='./exps/lkcil_cxr14.json', help='Path to JSON configuration.')
    return parser


def main():
    args = setup_parser().parse_args()
    config = load_json(args.config)
    args = vars(args)
    args.update(config)

    seeds = [1993, 2024, 4202]

    collection = {
        "Avg AUC": [],
        "Avg ACC": [],
        "mAP": [],
        "Forget (AUC)": [],
        "Forget (ACC)": [],
        "BWT (AUC)": [],
        "BWT (ACC)": []
    }

    print(f"Start running experiments on {len(seeds)} seeds: {seeds}...")

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 20} Running Seed {seed} ({i + 1}/{len(seeds)}) {'=' * 20}")
        args["seed"] = seed

        metrics = train(args)

        collection["Avg AUC"].append(metrics["avg_auc"])
        collection["Avg ACC"].append(metrics["avg_acc"])
        collection["mAP"].append(metrics["map"])
        collection["Forget (AUC)"].append(metrics["forget_auc"])
        collection["Forget (ACC)"].append(metrics["forget_acc"])
        collection["BWT (AUC)"].append(metrics["bwt_auc"])
        collection["BWT (ACC)"].append(metrics["bwt_acc"])

        print(f"Seed {seed} Completed.")
        print(f"Result: AUC={metrics['avg_auc']:.4f}, ACC={metrics['avg_acc']:.4f}, mAP={metrics['map']:.4f}")

    print("\n\n")
    print("=" * 60)
    print(f"FINAL AGGREGATED RESULTS OVER {len(seeds)} SEEDS")
    print("=" * 60)

    final_results = {}
    for key, values in collection.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_results[key] = f"{mean_val * 100:.2f} ± {std_val * 100:.2f}"

    try:
        df = pd.DataFrame(list(final_results.items()), columns=["Metric", "Mean ± Std (%)"])
        print(df.to_string(index=False))
    except ImportError:
        print(f"{'Metric':<20} | {'Mean ± Std (%)'}")
        print("-" * 40)
        for k, v in final_results.items():
            print(f"{k:<20} | {v}")

    print("=" * 60)

    print("\nRaw Data (for copying):")
    for k, v in collection.items():
        v_str = [f"{x:.4f}" for x in v]
        print(f"{k}: {v_str}")


if __name__ == '__main__':
    main()
