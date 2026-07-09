from __future__ import annotations

import argparse
import csv
import subprocess
import time
import urllib.request
from pathlib import Path


NOISE_TEST_FILES = [
    "AirConditioner_1.wav",
    "AirConditioner_10.wav",
    "AirConditioner_2.wav",
    "AirConditioner_3.wav",
    "AirConditioner_4.wav",
    "AirConditioner_5.wav",
    "AirConditioner_6.wav",
    "AirConditioner_7.wav",
    "AirConditioner_8.wav",
    "AirConditioner_9.wav",
    "AirportAnnouncements_1.wav",
    "AirportAnnouncements_2.wav",
    "AirportAnnouncements_3.wav",
    "AirportAnnouncements_4.wav",
    "Babble_1.wav",
    "Babble_2.wav",
    "Babble_3.wav",
    "Babble_4.wav",
    "Babble_5.wav",
    "Babble_6.wav",
    "Babble_7.wav",
    "Babble_8.wav",
    "CopyMachine_1.wav",
    "CopyMachine_2.wav",
    "Munching_1.wav",
    "Munching_2.wav",
    "Munching_3.wav",
    "Munching_4.wav",
    "Munching_5.wav",
    "Munching_6.wav",
    "Munching_7.wav",
    "Munching_8.wav",
    "Neighbor_1.wav",
    "Neighbor_2.wav",
    "Neighbor_3.wav",
    "Neighbor_4.wav",
    "Neighbor_5.wav",
    "Neighbor_6.wav",
    "Neighbor_7.wav",
    "Neighbor_9.wav",
    "ShuttingDoor_1.wav",
    "ShuttingDoor_2.wav",
    "ShuttingDoor_3.wav",
    "ShuttingDoor_4.wav",
    "ShuttingDoor_5.wav",
    "ShuttingDoor_6.wav",
    "ShuttingDoor_7.wav",
    "ShuttingDoor_8.wav",
    "Typing_1.wav",
    "Typing_2.wav",
    "VacuumCleaner_1.wav",
]

RAW_BASE_URL = "https://raw.githubusercontent.com/microsoft/MS-SNSD/master/noise_test"


def category_for(filename: str) -> str:
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[0]


def download_one_urllib(url: str, out_path: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "speech-asr-ctc-project"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"empty response: {url}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    tmp_path.write_bytes(data)
    tmp_path.replace(out_path)


def download_one_curl(url: str, out_path: Path, timeout: float) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    command = [
        "curl.exe",
        "-L",
        "--fail",
        "--max-time",
        str(int(timeout)),
        "-o",
        str(tmp_path),
        url,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"empty response: {url}")
    tmp_path.replace(out_path)


def download_one(url: str, out_path: Path, timeout: float, transport: str) -> None:
    if transport == "curl":
        download_one_curl(url, out_path, timeout=timeout)
        return
    if transport == "urllib":
        download_one_urllib(url, out_path, timeout=timeout)
        return
    raise ValueError(f"unsupported transport: {transport}")


def write_manifest(out_dir: Path, rows: list[dict]) -> None:
    fields = ["filename", "category", "path", "bytes", "source_url"]
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the lightweight MS-SNSD noise_test subset.")
    parser.add_argument("--out-dir", default="data/noise/ms_snsd_noise_test")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--transport", choices=["curl", "urllib"], default="curl")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    for index, filename in enumerate(NOISE_TEST_FILES, start=1):
        out_path = out_dir / filename
        url = f"{RAW_BASE_URL}/{filename}"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{index:02d}/{len(NOISE_TEST_FILES)}] skip existing {filename}")
        else:
            for attempt in range(1, args.retries + 1):
                try:
                    print(f"[{index:02d}/{len(NOISE_TEST_FILES)}] download {filename} (attempt {attempt})")
                    download_one(url, out_path, timeout=args.timeout, transport=args.transport)
                    break
                except Exception as exc:
                    if attempt >= args.retries:
                        failures.append((filename, str(exc)))
                    else:
                        time.sleep(min(2 * attempt, 8))
            if failures and failures[-1][0] == filename:
                continue

        rows.append(
            {
                "filename": filename,
                "category": category_for(filename),
                "path": str(out_path),
                "bytes": out_path.stat().st_size if out_path.exists() else 0,
                "source_url": url,
            }
        )

    write_manifest(out_dir, rows)
    print(f"Wrote manifest to {out_dir / 'manifest.csv'}")
    print(f"Available noise files: {len(rows)}")
    if failures:
        for filename, error in failures:
            print(f"FAILED {filename}: {error}")
        raise SystemExit(f"Failed to download {len(failures)} files")


if __name__ == "__main__":
    main()
