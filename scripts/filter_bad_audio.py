from pathlib import Path
import argparse
import soundfile as sf

def load_text(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            d[parts[0]] = parts[1]
    return d

def clean_split(src_root, dst_root, split):
    src_dir = Path(src_root) / split
    dst_dir = Path(dst_root) / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    wav_lines = Path(src_dir / "wav.scp").read_text(encoding="utf-8").splitlines()
    text_dict = load_text(src_dir / "pinyin")

    good_wavs = []
    good_texts = []
    bad = []

    for line in wav_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue

        utt_id, wav_path = parts
        if utt_id not in text_dict:
            continue

        try:
            info = sf.info(wav_path)
            if info.frames <= 0:
                raise RuntimeError("empty audio")
            good_wavs.append(f"{utt_id}\t{wav_path}")
            good_texts.append(f"{utt_id}\t{text_dict[utt_id]}")
        except Exception as e:
            bad.append((utt_id, wav_path, str(e)))

    (dst_dir / "wav.scp").write_text("\n".join(good_wavs) + "\n", encoding="utf-8")
    (dst_dir / "pinyin").write_text("\n".join(good_texts) + "\n", encoding="utf-8")

    print(f"{split}: kept={len(good_wavs)}, removed_bad={len(bad)}")
    for item in bad[:20]:
        print("BAD:", item[0], item[1], item[2])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/lab6")
    parser.add_argument("--dst", default="data/lab6_clean")
    args = parser.parse_args()

    for split in ["train", "dev", "test"]:
        clean_split(args.src, args.dst, split)

    print("Done. Clean data saved to data/lab6_clean")

if __name__ == "__main__":
    main()
