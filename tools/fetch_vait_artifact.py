"""Download, join, and verify the versioned VAIT artifact before extraction."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/2333repeat/validation-aligned-intervention-transfer/releases/download"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    release = json.loads((ROOT / "evidence/vait600/release.json").read_text(encoding="utf-8"))
    target = ROOT / "artifacts/vait600"
    if target.exists():
        raise SystemExit("artifacts/vait600 already exists; verify it with tools/replay_vait_recent_snapshot.py --verify-only")
    downloads = ROOT / "artifacts" / "vait600_download"
    downloads.mkdir(parents=True, exist_ok=True)
    for part in release["parts"]:
        path = downloads / part["name"]
        if path.exists() and (path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]):
            raise ValueError(f"Existing downloaded part is invalid: {path}")
        if not path.exists():
            pending = path.with_suffix(path.suffix + ".partial")
            url = f"{BASE}/{release['tag']}/{part['name']}"
            request = urllib.request.Request(url, headers={"User-Agent": "VAIT-reproduction"})
            with urllib.request.urlopen(request, timeout=120) as response, pending.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if pending.stat().st_size != part["bytes"] or digest(pending) != part["sha256"]:
                raise ValueError(f"Downloaded part failed verification: {part['name']}")
            pending.replace(path)
        print(f"Verified {part['name']}", flush=True)
    temporary = downloads / release["asset_name"]
    with temporary.open("wb") as out:
        for part in release["parts"]:
            with (downloads / part["name"]).open("rb") as source:
                shutil.copyfileobj(source, out)
    if temporary.stat().st_size != release["bytes"] or digest(temporary) != release["sha256"]:
        raise ValueError("Joined artifact checksum does not match the release manifest")
    with zipfile.ZipFile(temporary) as archive:
        for info in archive.infolist():
            path = (ROOT / info.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Archive contains a path outside artifacts/vait600")
            if path.exists():
                raise FileExistsError(path)
        archive.extractall(ROOT)
    temporary.unlink()
    for part in release["parts"]:
        (downloads / part["name"]).unlink()
    downloads.rmdir()
    print(f"Artifact verified and extracted to {target}")


if __name__ == "__main__":
    main()
