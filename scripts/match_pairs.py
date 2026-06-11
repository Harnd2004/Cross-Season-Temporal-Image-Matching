import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, ensure_dirs
from src.run_pipeline import process_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summer', required=True)
    ap.add_argument('--winter', required=True)
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    ensure_dirs(cfg, ROOT)
    result = process_pair(args.summer, args.winter, cfg, ROOT, save_outputs=True)
    print(result)


if __name__ == '__main__':
    main()
