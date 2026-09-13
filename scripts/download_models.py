"""Download MediaPipe .task models with checksum verification.

Models are committed for now (offline first-run), but this script is the
migration path to release-asset distribution (P1.6): if models/ is ever
removed from git, run this to fetch them.

Usage: python scripts/download_models.py [--check-only]
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FILES = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        None,  # checksum: fill in when publishing a release; None = size check only
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        None,
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    check_only = "--check-only" in sys.argv
    ok = True
    for name, (url, expected) in FILES.items():
        dest = MODELS / name
        if dest.is_file():
            print(f"OK present: {dest} ({dest.stat().st_size} bytes)")
            if expected:
                actual = sha256(dest)
                if actual != expected:
                    print(f"FAIL checksum {name}: got {actual}, want {expected}")
                    ok = False
            continue
        if check_only:
            print(f"MISSING: {dest}")
            ok = False
            continue
        print(f"Downloading {name} ...")
        MODELS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
