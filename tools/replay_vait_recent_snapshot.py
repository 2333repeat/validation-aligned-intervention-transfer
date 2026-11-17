"""Verify or replay the published 600-policy artifact without downloading market data."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_artifact(artifact):
    artifact = artifact.resolve()
    manifest = json.loads((artifact / 'MANIFEST.json').read_text(encoding='utf-8'))
    for record in manifest:
        path = (artifact / record['path']).resolve()
        if not path.is_relative_to(artifact):
            raise ValueError('Artifact manifest path escapes its root')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {record["path"]}')
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=ROOT / 'artifacts/vait600')
    parser.add_argument('--output', type=Path, default=ROOT / 'replay_output')
    parser.add_argument('--workers', type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    artifact = args.artifact_root.resolve()
    count = verify_artifact(artifact)
    print(f'Verified {count} artifact files.', flush=True)
    if args.verify_only:
        return
    out = args.output.resolve()
    if out.is_relative_to(artifact) or artifact.is_relative_to(out):
        raise ValueError('Replay output must be separate from the frozen artifact directory')
    snapshot = artifact / 'snapshot'
    cfg = json.loads((snapshot / 'protocol.json').read_text(encoding='utf-8'))
    cfg.update(output_root=str(out), policy_root=str(artifact / 'Policy-Value'),
               archive_manifest=str(artifact / 'archive_manifest.json'))
    out.mkdir(parents=True, exist_ok=True)
    for market in cfg['markets']:
        folder = out / 'data' / market
        folder.mkdir(parents=True, exist_ok=True)
        for name in ['future_ohlcv.csv', 'future_source.json']:
            src, dst = snapshot / 'data' / market / name, folder / name
            if dst.exists() and dst.read_bytes() != src.read_bytes():
                raise ValueError(f'Existing input differs: {dst}')
            if not dst.exists():
                shutil.copy2(src, dst)
    config = out / 'replay_config.json'
    if config.exists() and json.loads(config.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Existing replay configuration differs')
    config.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    stages = [('prepare_vait_recent', [])]
    if not args.prepare_only:
        stages += [('run_vait_recent', ['--stage','pretest','--workers',str(args.workers)]),
                   ('evaluate_vait_recent',['--workers',str(args.workers)]),
                   ('aggregate_vait_recent',[])]
    for module, extra in stages:
        subprocess.run([sys.executable, '-m', 'TradingCircuitDiscovery.experiments.'+module,
                        '--config', str(config), *extra], cwd=ROOT, env=env, check=True)
    print(f'Completed: {out}')


if __name__ == '__main__':
    main()
