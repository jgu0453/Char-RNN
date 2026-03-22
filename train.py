import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from char_rnn_model import N_CHARACTERS, CharRNN, random_training_set, read_text, time_since
from generate import generate


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: CharRNN,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    text: str,
    text_len: int,
    chunk_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    inp, target = random_training_set(
        text=text,
        text_len=text_len,
        chunk_len=chunk_len,
        batch_size=batch_size,
        device=device,
    )

    hidden = model.init_hidden(batch_size=batch_size, device=device)
    optimizer.zero_grad()
    loss = 0.0

    for c in range(chunk_len):
        output, hidden = model(inp[:, c], hidden)
        loss = loss + criterion(output, target[:, c])

    loss = loss / chunk_len
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to text file")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--chunk_len", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--samples_dir", type=str, default="samples")
    parser.add_argument("--sample_temperature", type=float, default=0.8)
    parser.add_argument("--sample_len", type=int, default=600)
    parser.add_argument("--prime_str", type=str, default="Wh")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem.lower()
    run_name = args.run_name or f"{dataset_name}_{args.model}_h{args.hidden_size}"

    results_dir = Path(args.results_dir)
    samples_dir = Path(args.samples_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    text, text_len = read_text(str(dataset_path))
    if text_len < args.chunk_len + 2:
        raise ValueError(
            f"Dataset too short ({text_len} chars). Increase data size or reduce --chunk_len."
        )

    model = CharRNN(
        input_size=N_CHARACTERS,
        hidden_size=args.hidden_size,
        output_size=N_CHARACTERS,
        model=args.model,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses: list[tuple[int, float]] = []

    print(f"Training run: {run_name}")
    print(f"Dataset: {dataset_path} ({text_len} characters)")
    print(f"Device: {device}")

    wall_start = time.time()
    progress = tqdm(range(1, args.n_epochs + 1), desc="epochs")
    for epoch in progress:
        loss = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            text=text,
            text_len=text_len,
            chunk_len=args.chunk_len,
            batch_size=args.batch_size,
            device=device,
        )
        losses.append((epoch, loss))
        progress.set_postfix(loss=f"{loss:.4f}")

        if epoch % args.print_every == 0:
            elapsed = time_since(wall_start)
            print(f"[{elapsed}] epoch={epoch}/{args.n_epochs} loss={loss:.4f}")

    checkpoint_path = results_dir / f"{run_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "model_type": args.model,
                "hidden_size": args.hidden_size,
                "n_layers": args.n_layers,
                "n_characters": N_CHARACTERS,
                "dataset": str(dataset_path),
                "run_name": run_name,
                "seed": args.seed,
            },
        },
        checkpoint_path,
    )

    loss_csv_path = results_dir / f"{run_name}_loss.csv"
    with loss_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        writer.writerows(losses)

    sample_text = generate(
        model=model,
        prime_str=args.prime_str,
        predict_len=args.sample_len,
        temperature=args.sample_temperature,
        device=device,
    )
    sample_path = samples_dir / f"{run_name}_temp{args.sample_temperature}.txt"
    sample_path.write_text(sample_text, encoding="utf-8")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved loss log: {loss_csv_path}")
    print(f"Saved sample: {sample_path}")


if __name__ == "__main__":
    main()
