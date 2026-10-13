"""Derives every detector file name from config.yaml and writes the pgie config that nvinfer actually loads."""
from __future__ import annotations

from pathlib import Path

from .config import ROOT, _fail

NETWORK_MODE = {"fp32": 0, "int8": 1, "fp16": 2}
GENERATED = ROOT / "config" / "_pgie_generated.txt"  # git-ignored


def det_paths(a):
    """(onnx, engine, calib cache) for the detector settings in `a`."""
    stem = f"{a.det_name}_{a.det_height}x{a.det_width}"
    onnx = ROOT / "models" / f"{stem}.onnx"
    engine = ROOT / "models" / f"{onnx.name}_b{a.det_batch}_gpu0_{a.det_precision}.engine"
    calib = ROOT / "models" / f"{stem}_calib.cache"  # size in the name: a stale cache is never reused
    return onnx, engine, calib


def write_pgie(a, precision=None, dest=None, conf=None):
    """Fill the template detector.config with the derived keys and write it to `dest` (default: _pgie_generated.txt).
    precision: override detector.precision (used by training/accuracy.py); conf: override pre-cluster-threshold."""
    precision = precision or a.det_precision
    if precision not in NETWORK_MODE:
        _fail("detector.precision must be int8, fp16 or fp32")
    saved, a.det_precision = a.det_precision, precision  # det_paths() names the engine after the precision
    try:
        onnx, engine, calib = det_paths(a)
    finally:
        a.det_precision = saved

    if not onnx.exists():
        _fail(f"{onnx.name} not found - run: python export/export_models.py --m1 <best.pt>")
    labels = ROOT / "models" / "labels.txt"
    if not labels.exists():
        _fail("models/labels.txt not found - run: python export/export_models.py --m1 <best.pt>")
    nc = len([ln for ln in labels.read_text().splitlines() if ln.strip()])

    override = {
        "onnx-file": onnx,
        "model-engine-file": engine,
        "labelfile-path": labels,
        "batch-size": a.det_batch,
        "network-mode": NETWORK_MODE[precision],
        "num-detected-classes": nc,
        "custom-lib-path": ROOT / "parser" / "libnvdsparse_yolo26.so",
    }
    if precision == "int8":
        if not calib.exists():
            _fail(f"{calib.name} missing - run: python export/calibrate.py")
        override["int8-calib-file"] = calib

    out = []
    for ln in Path(a.pgie_template).read_text().splitlines():
        s = ln.strip()
        key = s.split("=", 1)[0].strip()
        if not s.startswith("#") and key in override:
            continue  # template value dropped, injected below
        if conf is not None and not s.startswith("#") and key == "pre-cluster-threshold":
            ln = f"pre-cluster-threshold={conf}"
        out.append(ln)
        if s == "[property]":
            out += [f"{k}={v}" for k, v in override.items()]

    dest = Path(dest) if dest else GENERATED
    dest.write_text("\n".join(out) + "\n")  # absolute paths, so the file location doesn't matter
    return dest