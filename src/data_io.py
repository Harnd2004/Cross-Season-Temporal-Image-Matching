from pathlib import Path
import cv2


def list_paired_images(summer_dir, winter_dir, extensions):
    summer_dir = Path(summer_dir)
    winter_dir = Path(winter_dir)
    summer_map = {p.name: p for p in summer_dir.iterdir() if p.suffix.lower() in extensions}
    winter_map = {p.name: p for p in winter_dir.iterdir() if p.suffix.lower() in extensions}
    names = sorted(set(summer_map).intersection(winter_map))
    return [(summer_map[n], winter_map[n]) for n in names]


def read_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f'Unable to read image: {path}')
    return img


def resize_max(img, max_size=1280):
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale == 1.0:
        return img, 1.0
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale
