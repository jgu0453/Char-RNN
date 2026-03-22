import argparse
from pathlib import Path

import torch

from char_rnn_model import ALL_CHARACTERS, CHAR_TO_INDEX, CharRNN, char_tensor


def generate(
    model: CharRNN,
    prime_str: str,
    predict_len: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    hidden = model.init_hidden(batch_size=1, device=device)
    prime_input = char_tensor(prime_str, device=device).unsqueeze(0)
    predicted = prime_str

    with torch.no_grad():
        for i in range(len(prime_str) - 1):
            _, hidden = model(prime_input[:, i], hidden)

        inp = prime_input[:, -1]
        for _ in range(predict_len):
            output, hidden = model(inp, hidden)
            output_dist = output.squeeze().div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1).item()
            predicted_char = ALL_CHARACTERS[top_i]
            predicted += predicted_char
            inp = torch.tensor([CHAR_TO_INDEX[predicted_char]], dtype=torch.long, device=device)

    return predicted


def load_model(checkpoint_path: str, device: torch.device) -> CharRNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = CharRNN(
        input_size=config["n_characters"],
        hidden_size=config["hidden_size"],
        output_size=config["n_characters"],
        model=config["model_type"],
        n_layers=config["n_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("-p", "--prime_str", type=str, default="A")
    parser.add_argument("-l", "--predict_len", type=int, default=400)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    output = generate(
        model=model,
        prime_str=args.prime_str,
        predict_len=args.predict_len,
        temperature=args.temperature,
        device=device,
    )

    print(output)
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

