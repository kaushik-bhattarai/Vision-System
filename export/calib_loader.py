"""INT8 calibration data for the detector: letterboxed calibration batches ({"images": NCHW float32})."""
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent          # export/
ROOT = HERE.parent                              # project root (config/ lives here)
_det = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())["detector"]
H, W, B = _det["height"], _det["width"], _det["batch_size"]
files = sorted(glob.glob(str(HERE / "calib_images" / "*.jpg")))


def letterbox(img):
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = round(w * s), round(h * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W, 3), np.uint8)   # nvinfer pads with 0
    t, l = (H - nh) // 2, (W - nw) // 2
    out[t:t + nh, l:l + nw] = img
    return out

def load_data():
    for i in range(0, len(files) - B + 1, B):
        x = np.stack([letterbox(cv2.imread(f))[:, :, ::-1] for f in files[i:i + B]])
        x = x.astype(np.float32) / 255.0
        yield {"images": np.ascontiguousarray(x.transpose(0, 3, 1, 2))}