from pathlib import Path
import argparse
import shutil

def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab6-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--out-root", default="data/lab6")
    args = parser.parse_args()

    lab6_root = Path(args.lab6_root)
    audio_root = Path(args.audio_root)
    out_root = Path(args.out_root)

    split_root = lab6_root / "dataset" / "split"
    if not split_root.exists():
        raise FileNotFoundError(f"Cannot find Lab6 split folder: {split_root}")

    for split in ["train", "dev", "test"]:
        src_dir = split_root / split
        src_wav = src_dir / "wav.scp"
        src_txt = src_dir / "pinyin"

        if not src_wav.exists():
            raise FileNotFoundError(f"Missing {src_wav}")
        if not src_txt.exists():
            raise FileNotFoundError(f"Missing {src_txt}")

        dst_dir = out_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        wav_out = []
        for line in read_lines(src_wav):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, wav_path = parts
            wav_path = wav_path.replace("\\", "/")
            p = Path(wav_path)
            if not p.is_absolute():
                p = audio_root / p
            wav_out.append(f"{utt_id}\t{p.as_posix()}")

        (dst_dir / "wav.scp").write_text("\n".join(wav_out) + "\n", encoding="utf-8")
        shutil.copyfile(src_txt, dst_dir / "pinyin")

        print(f"{split}: wrote {dst_dir / 'wav.scp'} and {dst_dir / 'pinyin'}")

    print("Done. Imported Lab6 manifests to data/lab6")

if __name__ == "__main__":
    main()
