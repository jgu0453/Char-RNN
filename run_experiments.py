import argparse
import itertools
import subprocess
import sys
from pathlib import Path


DATASETS = [
    ("shakespeare", "data/shakespeare.txt"),
    ("sherlock", "data/sherlock.txt"),
]
MODEL_TYPES = ["gru", "lstm"]
HIDDEN_SIZES = [128, 256]
TEMPERATURES = [0.5, 0.8, 1.0]


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Char-RNN experiment matrix across datasets, model types, and hidden sizes."
    )
    parser.add_argument("--n_epochs", type=int, default=1200)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_len", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--predict_len", type=int, default=700)
    parser.add_argument("--prime_str", type=str, default="Wh")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all commands without executing.",
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("samples").mkdir(exist_ok=True)

    for (dataset_name, dataset_path), model_type, hidden_size in itertools.product(
        DATASETS, MODEL_TYPES, HIDDEN_SIZES
    ):
        run_name = f"{dataset_name}_{model_type}_h{hidden_size}"
        train_cmd = [
            sys.executable,
            "train.py",
            dataset_path,
            "--run_name",
            run_name,
            "--model",
            model_type,
            "--hidden_size",
            str(hidden_size),
            "--n_layers",
            str(args.n_layers),
            "--n_epochs",
            str(args.n_epochs),
            "--print_every",
            str(args.print_every),
            "--batch_size",
            str(args.batch_size),
            "--chunk_len",
            str(args.chunk_len),
            "--learning_rate",
            str(args.learning_rate),
            "--sample_temperature",
            "0.8",
            "--sample_len",
            "500",
            "--prime_str",
            args.prime_str,
        ]
        run_cmd(train_cmd, dry_run=args.dry_run)

        checkpoint = f"results/{run_name}.pt"
        for temperature in TEMPERATURES:
            sample_path = f"samples/{run_name}_temp{temperature}.txt"
            gen_cmd = [
                sys.executable,
                "generate.py",
                checkpoint,
                "--prime_str",
                args.prime_str,
                "--predict_len",
                str(args.predict_len),
                "--temperature",
                str(temperature),
                "--output_path",
                sample_path,
            ]
            run_cmd(gen_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
