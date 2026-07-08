from pathlib import Path
import argparse
import shutil

def take_lines(src, dst, n):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(encoding="utf-8").splitlines()
    selected = [x for x in lines if x.strip()][:n]
    dst.write_text("\n".join(selected) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/lab6")
    parser.add_argument("--dst", default="data/lab6_small")
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    nums = {"train": args.train, "dev": args.dev, "test": args.test}

    if dst.exists():
        shutil.rmtree(dst)

    for split, n in nums.items():
        take_lines(src / split / "wav.scp", dst / split / "wav.scp", n)
        take_lines(src / split / "pinyin", dst / split / "pinyin", n)
        print(f"{split}: {n} samples written.")

    print("Done. Small Lab6 dataset saved to data/lab6_small")

if __name__ == "__main__":
    main()
