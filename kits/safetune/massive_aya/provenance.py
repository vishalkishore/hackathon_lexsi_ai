"""Record versions and source identity without exposing credentials."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
MODEL_REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True)
    return {'utc_timestamp': datetime.now(timezone.utc).isoformat(),
            'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
            'uv_lock_sha256': sha256(ROOT / 'uv.lock'),
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p)
                              for p in sorted((ROOT / 'aya_massive_en_hi').rglob('*.py'))},
            'versions': {name: version(name) for name in
                         ('torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate')}}


def model_identity(model, revision):
    """Require an immutable Hub revision or a merge with recorded base identity."""
    import json
    import re
    path = Path(model)
    if path.is_dir():
        record = json.loads((path / 'merge_manifest.json').read_text())
        if record['model_revision'] != revision:
            raise ValueError('Merged base revision differs from requested revision')
        return record['base_model'], revision
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Provide a pinned 40-character model revision')
    return model, revision
