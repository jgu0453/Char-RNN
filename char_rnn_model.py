import math
import random
import string
import time
from pathlib import Path

import torch
import torch.nn as nn
import unidecode

ALL_CHARACTERS = string.printable
N_CHARACTERS = len(ALL_CHARACTERS)
CHAR_TO_INDEX = {ch: idx for idx, ch in enumerate(ALL_CHARACTERS)}


def read_text(path: str) -> tuple[str, int]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = unidecode.unidecode(text)
    return text, len(text)


def char_tensor(text: str, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.zeros(len(text), dtype=torch.long)
    for i, ch in enumerate(text):
        tensor[i] = CHAR_TO_INDEX.get(ch, 0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def random_training_set(
    text: str,
    text_len: int,
    chunk_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = torch.zeros(batch_size, chunk_len, dtype=torch.long, device=device)
    target = torch.zeros(batch_size, chunk_len, dtype=torch.long, device=device)
    max_start = text_len - chunk_len - 1
    if max_start <= 0:
        raise ValueError(
            f"Text is too short for chunk_len={chunk_len}. "
            f"Need at least {chunk_len + 2} characters."
        )

    for batch_idx in range(batch_size):
        start_idx = random.randint(0, max_start)
        chunk = text[start_idx : start_idx + chunk_len + 1]
        inp[batch_idx] = char_tensor(chunk[:-1], device=device)
        target[batch_idx] = char_tensor(chunk[1:], device=device)
    return inp, target


def time_since(start_time: float) -> str:
    elapsed = time.time() - start_time
    minutes = math.floor(elapsed / 60)
    seconds = int(elapsed - (minutes * 60))
    return f"{minutes}m {seconds}s"


class CharRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        model: str = "gru",
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        else:
            raise ValueError(f"Unsupported model type: {model}")

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(
        self, input_step: torch.Tensor, hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        batch_size = input_step.size(0)
        encoded = self.encoder(input_step)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.model == "lstm":
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            return h0, c0
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

