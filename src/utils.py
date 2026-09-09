"""Persistência de artefatos e rastreabilidade."""
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def output_dirs(root: Path) -> None:
    for folder in ('data/processed', 'models', 'reports/figures', 'reports/metrics'):
        (root / folder).mkdir(parents=True, exist_ok=True)
