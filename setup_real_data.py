"""
setup_real_data.py
------------------
Populates data/summer and data/winter from the real CMU-Seasons images folder.

CMU-Seasons layout:
  images/sliceN/database/ -> summer traversal  (img_NNNNN_c0_<ts>.jpg)
  images/sliceN/query/    -> winter traversal  (img_NNNNN_c0_<ts>.jpg)

The same frame number (img_NNNNN) in both directories = same geographic
location photographed in different seasons.  Within each slice some frame
numbers appear in BOTH database and query -- those are our cross-season pairs.

Steps:
  1. Clear previous output/ and data/ dirs.
  2. For each slice find the frame-number overlap between database and query.
  3. Copy matched images to data/summer / data/winter under a canonical name
     that strips the embedded timestamp, so data_io.list_paired_images() can
     match them by filename.
  4. Stop after MAX_PAIRS total pairs.
"""

import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
IMAGES_ROOT  = Path(r"C:\Users\harin\Downloads\images\images")
PROJECT_ROOT = Path(__file__).resolve().parent
SUMMER_OUT   = PROJECT_ROOT / "data" / "summer"
WINTER_OUT   = PROJECT_ROOT / "data" / "winter"
OUTPUT_DIRS  = [
    PROJECT_ROOT / "output" / "matches",
    PROJECT_ROOT / "output" / "warped",
    PROJECT_ROOT / "output" / "logs",
]
SLICES       = ["slice2", "slice3", "slice4", "slice5", "slice6"]
MAX_PAIRS    = 10
CAMERA       = "c0"
# ---------------------------------------------------------------------------

FRAME_RE = re.compile(r"img_(\d+)_c(\d)")


def build_frame_map(directory: Path, camera: str) -> dict:
    """Return {frame_num_str: Path} for files matching camera ID."""
    result = {}
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        m = FRAME_RE.search(p.name)
        if m and m.group(2) == camera[1]:   # camera = "c0" -> group(2) == "0"
            fnum = m.group(1)
            result[fnum] = p
    return result


def clear_dir(d: Path):
    if d.exists():
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
    else:
        d.mkdir(parents=True, exist_ok=True)


def main():
    print("Step 1: Clearing previous outputs ...")
    for d in OUTPUT_DIRS + [SUMMER_OUT, WINTER_OUT]:
        clear_dir(d)
        print(f"  cleared  {d}")

    print("\nStep 2: Locating optimal cross-season pairs ...")
    optimal_pairs_list = [
        ("slice3", "00726"),
        ("slice3", "00801"),
        ("slice3", "00786"),
        ("slice5", "01709"),
        ("slice5", "01679"),
        ("slice6", "01890"),
        ("slice5", "01671"),
        ("slice6", "02350"),
        ("slice6", "02483"),
        ("slice4", "01604")
    ]
    
    pairs = []
    for sl, fnum in optimal_pairs_list:
        db_dir = IMAGES_ROOT / sl / "database"
        q_dir  = IMAGES_ROOT / sl / "query"
        if not db_dir.exists() or not q_dir.exists():
            print(f"  Warning: slice directories for {sl} missing. Skipping.")
            continue
        db_map = build_frame_map(db_dir, CAMERA)
        q_map  = build_frame_map(q_dir, CAMERA)
        if fnum in db_map and fnum in q_map:
            pairs.append((sl, fnum, db_map[fnum], q_map[fnum]))
        else:
            print(f"  Warning: frame {fnum} not found in {sl} database/query. Skipping.")

    print(f"\nStep 3: Copying {len(pairs)} pairs ...")
    for sl, fnum, src_summer, src_winter in pairs:
        name = f"{sl}_img_{fnum}_{CAMERA}.jpg"
        shutil.copy2(src_summer, SUMMER_OUT / name)
        shutil.copy2(src_winter, WINTER_OUT / name)
        print(f"  {name}")
        print(f"    summer <- {src_summer.name}")
        print(f"    winter <- {src_winter.name}")

    print(f"\nDone. {len(pairs)} pairs in data/summer and data/winter.")
    print("Run: python scripts/align_pairs.py  (or  scripts/benchmark.py)")


if __name__ == "__main__":
    main()
