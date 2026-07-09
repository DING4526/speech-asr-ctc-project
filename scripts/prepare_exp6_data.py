from __future__ import annotations

import argparse
from pathlib import Path


def read_manifest(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno} should contain utterance id and value")
        items[parts[0]] = parts[1]
    return items


def prepare_split(raw_root: Path, out_root: Path, split: str) -> tuple[int, int]:
    split_dir = raw_root / "split" / split
    wavs = read_manifest(split_dir / "wav.scp")
    texts = read_manifest(split_dir / "pinyin")

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_lines = []
    text_lines = []
    removed = 0
    for utt_id, wav_rel in wavs.items():
        text = texts.get(utt_id)
        wav_path = raw_root / wav_rel
        if text is None or not wav_path.exists() or wav_path.stat().st_size <= 0:
            removed += 1
            continue
        wav_lines.append(f"{utt_id}\t{wav_path.as_posix()}")
        text_lines.append(f"{utt_id}\t{text}")

    (out_dir / "wav.scp").write_text("\n".join(wav_lines) + "\n", encoding="utf-8")
    (out_dir / "pinyin").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return len(wav_lines), removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare copied Exp6 Lab6 ASR data for this project")
    parser.add_argument("--raw-root", default="data/lab6", help="Copied Exp6 dataset root with Wave/ and split/")
    parser.add_argument("--out-root", default="data/lab6_clean", help="Output root for cleaned manifests")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not (raw_root / "Wave").exists():
        raise FileNotFoundError(f"Missing audio folder: {raw_root / 'Wave'}")
    if not (raw_root / "split").exists():
        raise FileNotFoundError(f"Missing split folder: {raw_root / 'split'}")

    for split in ["train", "dev", "test"]:
        kept, removed = prepare_split(raw_root, out_root, split)
        print(f"{split}: kept={kept}, removed={removed}")

    print(f"Clean manifests written to {out_root}")


if __name__ == "__main__":
    main()
