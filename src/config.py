from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict, root: Path) -> None:
    out = cfg['output']
    for key in ('matches_dir', 'warped_dir', 'logs_dir'):
        (root / out[key]).mkdir(parents=True, exist_ok=True)
