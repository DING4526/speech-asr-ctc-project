from pathlib import Path
import argparse

def load_wavs(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            d[parts[0]] = parts[1]
    return d

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/lab6")
    args = parser.parse_args()

    root = Path(args.root)
    for split in ["train", "dev", "test"]:
        wav_file = root / split / "wav.scp"
        txt_file = root / split / "pinyin"

        wavs = load_wavs(wav_file)
        texts = load_text(txt_file)

        matched = set(wavs) & set(texts)
        missing_audio = []
        for utt_id in matched:
            if not Path(wavs[utt_id]).exists():
                missing_audio.append(wavs[utt_id])

        print(f"{split}: wav={len(wavs)}, text={len(texts)}, matched={len(matched)}, missing_audio={len(missing_audio)}")
        if missing_audio:
            print("  first missing:", missing_audio[0])

if __name__ == "__main__":
    main()
