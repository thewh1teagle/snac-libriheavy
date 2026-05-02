import argparse
import json
from pathlib import Path

import librosa
import torch
from snac import SNAC
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a folder of WAV files into SNAC codes as JSONL."
    )
    parser.add_argument("folder", type=Path, help="Folder containing .wav files.")
    parser.add_argument("output_jsonl", type=Path, help="Output JSONL file.")
    parser.add_argument(
        "--model",
        default="hubertsiuzdak/snac_24khz",
        help="SNAC model name or local model directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on.",
    )
    return parser.parse_args()


def wav_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def encode_file(model: SNAC, path: Path, device: torch.device) -> list[list[int]]:
    audio, _ = librosa.load(path, sr=model.sampling_rate, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        codes = model.encode(audio_tensor)

    return [code.squeeze(0).detach().cpu().tolist() for code in codes]


def main() -> None:
    args = parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {args.folder}")

    files = wav_files(args.folder)
    if not files:
        raise SystemExit(f"No WAV files found in: {args.folder}")

    device = torch.device(args.device)
    model = SNAC.from_pretrained(args.model).eval().to(device)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for path in tqdm(files, desc="Encoding WAV files", unit="file"):
            record = {
                "filename": path.name,
                "codes": encode_file(model, path, device),
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
