from __future__ import annotations

from pathlib import Path
import yaml


def main():
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    root = Path("data/toy")
    base["data"] = {
        "train_wav_scp": str(root / "train" / "wav.scp"),
        "train_text": str(root / "train" / "pinyin"),
        "dev_wav_scp": str(root / "dev" / "wav.scp"),
        "dev_text": str(root / "dev" / "pinyin"),
        "test_wav_scp": str(root / "test" / "wav.scp"),
        "test_text": str(root / "test" / "pinyin"),
    }
    base["vocab_path"] = "data/toy/vocab.txt"
    base["output_dir"] = "checkpoints/toy"
    base["train"]["epochs"] = 1
    base["train"]["batch_size"] = 4
    base["feature"]["n_mels"] = 40
    base["model"]["input_dim"] = 40
    base["model"]["d_model"] = 32
    base["model"]["nhead"] = 4
    base["model"]["num_layers"] = 1
    base["model"]["dim_feedforward"] = 64
    base["train"]["num_threads"] = 1
    Path("configs/toy.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("Wrote configs/toy.yaml")


if __name__ == "__main__":
    main()
