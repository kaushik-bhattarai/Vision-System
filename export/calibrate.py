#!/usr/bin/env python3
"""Build the INT8 calibration cache for the detector in config.yaml via polygraphy.
Run from the project root in the APP venv (polygraphy + tensorrt + opencv):   python export/calibrate.py [--force]
Images come from export/calib_images/*.jpg.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
det = cfg["detector"]
H, W = det["height"], det["width"]

batch = det["batch_size"]
if not batch:  # 0 = auto: largest camera group
    cameras_file = Path(cfg["cameras_file"])
    cj = json.loads((cameras_file if cameras_file.is_absolute() else ROOT / cameras_file).read_text())
    groups = cj.get("groups") or [cj["cameras"]]
    batch = max(len(g) for g in groups)

stem = f"{det['name']}_{H}x{W}"
onnx = ROOT / "models" / f"{stem}.onnx"
cache = ROOT / "models" / f"{stem}_calib.cache"

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="recalibrate even if the cache exists")
a = ap.parse_args()

if not onnx.exists():
    sys.exit(f"{onnx.name} not found - run export/export_models.py first")
n_img = len(glob.glob(str(HERE / "calib_images" / "*.jpg")))
if n_img < batch:
    sys.exit(f"only {n_img} images in export/calib_images/, need at least one batch ({batch})")
if cache.exists() and not a.force:
    print(f"{cache.name} already exists (use --force to rebuild)")
    sys.exit(0)
cache.unlink(missing_ok=True)  # polygraphy would otherwise reuse it

shape = f"images:[{batch},3,{H},{W}]"
with tempfile.TemporaryDirectory() as tmp:
    cmd = ["polygraphy", "convert", str(onnx), "--int8",
           "--data-loader-script", str(HERE / "calib_loader.py"),
           "--calibration-cache", str(cache),
           "--trt-min-shapes", f"images:[1,3,{H},{W}]",
           "--trt-opt-shapes", shape, "--trt-max-shapes", shape,
           "-o", os.path.join(tmp, "throwaway.engine")]
    print(" ".join(cmd))
    env = dict(os.environ, CALIB_BATCH=str(batch))
    sys.exit(subprocess.run(cmd, env=env).returncode)