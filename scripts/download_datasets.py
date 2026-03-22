from pathlib import Path
from urllib.request import urlopen

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
SHERLOCK_URL = "https://sherlock-holm.es/stories/plain-text/cnus.txt"


def download(url: str, output_path: Path) -> None:
    print(f"Downloading {url} -> {output_path}")
    with urlopen(url) as response:
        data = response.read().decode("utf-8", errors="ignore")
    output_path.write_text(data, encoding="utf-8")


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    download(SHAKESPEARE_URL, data_dir / "shakespeare.txt")
    download(SHERLOCK_URL, data_dir / "sherlock.txt")
    print("Done.")


if __name__ == "__main__":
    main()

