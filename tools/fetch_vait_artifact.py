"""Download the versioned VAIT artifact and verify its checksum before extraction."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = json.loads((ROOT/'evidence/vait600/release.json').read_text(encoding='utf-8'))
    target = ROOT/'artifacts/vait600'
    if target.exists():
        raise SystemExit('artifacts/vait600 already exists; verify it with tools/replay_vait_recent_snapshot.py --verify-only')
    temporary = ROOT/'artifacts'/release['asset_name']
    temporary.parent.mkdir(exist_ok=True)
    if not temporary.exists():
        part = temporary.with_suffix('.part')
        request = urllib.request.Request(release['asset_url'], headers={'User-Agent':'VAIT-reproduction'})
        with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        part.replace(temporary)
    with temporary.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != release['sha256']:
        raise ValueError('Downloaded artifact checksum does not match the release manifest')
    with zipfile.ZipFile(temporary) as archive:
        for info in archive.infolist():
            path = (ROOT/info.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError('Archive contains a path outside artifacts/vait600')
            if path.exists():
                raise FileExistsError(path)
        archive.extractall(ROOT)
    temporary.unlink()
    print(f'Artifact verified and extracted to {target}')


if __name__ == '__main__':
    main()
