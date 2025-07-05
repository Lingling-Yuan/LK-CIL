import json
import argparse
import numpy as np
from trainer import train

def load_json(path):
    with open(path) as f:
        return json.load(f)

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate LK-CIL across random seeds.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./exps/lkcil.json',
        help='Path to JSON configuration.'
    )
    return parser

def main():
    args = setup_parser().parse_args()
    config = load_json(args.config)
    args = vars(args)
    args.update(config)

    seeds = [1993, 2024, 4202]
    auc_list = []
    auc_matrices = []

    for seed in seeds:
        print(f"\nRunning for seed {seed}")
        args["seed"] = seed
        metrics = train(args)

        avg_auc = metrics.get("avg_auc", np.nan)
        auc_list.append(avg_auc)
        print(f"Seed {seed}: Average AUC = {avg_auc:.4f}")

        matrix = metrics.get("auc_matrix")
        if matrix is not None:
            auc_matrices.append(matrix)
            print("  auc_matrix collected.")
        else:
            print("  auc_matrix is None.")

    print("\n==== Across seeds AUC ====")
    print(f"Mean AUC : {np.nanmean(auc_list):.4f} (std: {np.nanstd(auc_list):.4f})")


if __name__ == '__main__':
    main()
