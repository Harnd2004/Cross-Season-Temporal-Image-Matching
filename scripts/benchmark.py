import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, ensure_dirs
from src.data_io import list_paired_images
from src.run_pipeline import process_pair
from src.profile import save_latency_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    cfg['runtime']['save_visualizations'] = False
    ensure_dirs(cfg, ROOT)
    pairs = list_paired_images(ROOT / cfg['data']['summer_dir'], ROOT / cfg['data']['winter_dir'], set(cfg['data']['extensions']))
    records = [process_pair(s, w, cfg, ROOT, save_outputs=False) for s, w in pairs]
    save_latency_report(records, ROOT / cfg['output']['logs_dir'] / 'latency_report.csv', ROOT / cfg['output']['logs_dir'] / 'summary.json')
    print(f'Benchmarked {len(records)} pairs')


if __name__ == '__main__':
    main()
