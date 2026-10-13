#!/usr/bin/env python3
"""Accuracy of the detector (M1) per TensorRT precision. Compares fp32 / fp16 / int8 on the held-out validation split made by train.py.

Run in the APP venv (pyds available).  Needs: pip install opencv-python-headless
Model name, size and batch come from detector.* in config/config.yaml; the per-precision nvinfer config is generated
by dsbench/models.py (write_pgie), so the engines are the same files run.py uses.
Detections are cached in bench_out/accuracy/dets_<precision>.json.
The first run of each precision builds a TensorRT engine.
INT8 needs the calibration cache (python export/calibrate.py).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent      # training/
ROOT = HERE.parent                          # project root (dsbench/, config/, models/ live here)
sys.path.insert(0, str(ROOT))

NETWORK_MODE = {"fp32": 0, "int8": 1, "fp16": 2}
IMG_DIR = HERE / "dataset_split" / "images" / "val"
LBL_DIR = HERE / "dataset_split" / "labels" / "val"
OUT = ROOT / "bench_out" / "accuracy"
FRAMES = OUT / "frames"


def _deployed_conf():
    """Operating point = pre-cluster-threshold of the deployed pgie config (falls back to 0.30)."""
    import yaml
    from dsbench.config import read_nvinfer
    det = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())["detector"]
    return float(read_nvinfer(ROOT / det["config"]).get("pre-cluster-threshold", 0.30))


OP_CONF = _deployed_conf()


def list_images():
    imgs = sorted(p for p in IMG_DIR.glob("*") if p.suffix.lower() == ".jpg")
    if not imgs:
        sys.exit(f"no .jpg images in {IMG_DIR} (run train.py first, it creates dataset_split/)")
    return imgs


# DeepStream detection run (one precision)
def run_detector(prec):
    import cv2
    os.environ["USE_NEW_NVSTREAMMUX"] = "no"  # legacy muxer: needs width/height
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import pyds
    from dsbench.config import load_config
    from dsbench.gstutil import iter_list
    from dsbench.models import write_pgie

    a = load_config(make_pgie=False)  # accuracy.py builds its own per-precision pgie below    imgs = list_images()
    FRAMES.mkdir(parents=True, exist_ok=True)
    for f in FRAMES.glob("*.jpg"):
        f.unlink()
    sizes = []
    for i, p in enumerate(imgs):
        (FRAMES / f"{i:06d}.jpg").symlink_to(p.resolve())
        h, w = cv2.imread(str(p)).shape[:2]
        sizes.append((w, h))
    mw, mh = sizes[0]
    if len(set(sizes)) > 1:
        print(f"  [warn] images have different sizes; muxer uses {mw}x{mh} (first image), boxes are rescaled")

    # detector config for THIS precision; threshold lowered to 0.001 so the full PR curve can be computed
    pgie_cfg = write_pgie(a, precision=prec, dest=ROOT / "config" / f"_pgie_eval_{prec}.txt", conf=0.001)
    Gst.init(None)
    desc = (f"multifilesrc location={FRAMES}/%06d.jpg index=0 caps=image/jpeg,framerate=30/1 ! jpegdec ! "
            f"nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! mux.sink_0 "
            f"nvstreammux name=mux batch-size=1 width={mw} height={mh} batched-push-timeout=40000 ! "
            f"nvinfer name=pgie config-file-path={pgie_cfg} ! fakesink sync=false")
    pipe = Gst.parse_launch(desc)
    dets = {}

    def cb(pad, info, _):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf)) if buf is not None else None
        if bm is None:
            return Gst.PadProbeReturn.OK
        for fm in iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            idx = fm.frame_num
            if not 0 <= idx < len(imgs):
                continue
            ow, oh = sizes[idx]
            sx, sy = ow / mw, oh / mh
            rows = []
            for o in iter_list(fm.obj_meta_list, pyds.NvDsObjectMeta.cast):
                r = o.rect_params
                rows.append([r.left * sx, r.top * sy, (r.left + r.width) * sx, (r.top + r.height) * sy,
                             float(o.confidence), int(o.class_id)])
            dets[idx] = rows
        return Gst.PadProbeReturn.OK

    pipe.get_by_name("pgie").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, cb, None)
    loop = GLib.MainLoop()

    def on_bus(_b, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {msg.src.get_name()}: {err.message}\n  {dbg}")
            loop.quit()
        return True

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)
    print(f"[{prec}] running {len(imgs)} images through nvinfer (engine build may take minutes)...")
    pipe.set_state(Gst.State.PLAYING)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    Path(pgie_cfg).unlink(missing_ok=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dets_{prec}.json").write_text(json.dumps(
        {"images": [p.name for p in imgs], "sizes": sizes, "dets": {str(k): v for k, v in dets.items()}}))
    print(f"[{prec}] {len(dets)}/{len(imgs)} frames processed, "
          f"{sum(len(v) for v in dets.values())} raw detections")


# metrics (numpy only)
def box_iou(a, b):
    """a [m,4], b [n,4] xyxy -> [m,n]"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def match(pred, gt):
    """pred [n,6] x1y1x2y2 conf cls ; gt [m,5] cls x1y1x2y2 -> bool [n,10] (IoU 0.5:0.05:0.95), ultralytics-style."""
    correct = np.zeros((len(pred), 10), bool)
    if not len(pred) or not len(gt):
        return correct
    iou = box_iou(gt[:, 1:], pred[:, :4])
    same = gt[:, 0:1] == pred[:, 5][None]
    for k, t in enumerate(np.linspace(0.5, 0.95, 10)):
        m = np.argwhere((iou >= t) & same)
        if len(m):
            m = np.c_[m, iou[m[:, 0], m[:, 1]]]
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), k] = True
    return correct


def ap_curve(rec, prec):
    mrec, mpre = np.r_[0.0, rec, 1.0], np.r_[1.0, prec, 0.0]
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def load_gt(names, sizes):
    gts = []
    for name, (w, h) in zip(names, sizes):
        f = LBL_DIR / (Path(name).stem + ".txt")
        rows = []
        if f.exists():
            for ln in f.read_text().splitlines():
                p = ln.split()
                if len(p) == 5:
                    c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                    rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        gts.append(np.array(rows, float).reshape(-1, 5))
    return gts


def evaluate(blob, gts, nc):
    n = len(blob["images"])
    per_img = [np.array(blob["dets"].get(str(i), []), float).reshape(-1, 6) for i in range(n)]
    tp, conf, pcls, tcls = [], [], [], []
    tp30 = ndet30 = 0
    for p, g in zip(per_img, gts):
        tp.append(match(p, g))
        conf.append(p[:, 4])
        pcls.append(p[:, 5])
        tcls.append(g[:, 0])
        p30 = p[p[:, 4] >= OP_CONF]  # operating point: re-match with only the deployed detections
        tp30 += int(match(p30, g)[:, 0].sum())
        ndet30 += len(p30)
    tp, conf, pcls, tcls = np.concatenate(tp), np.concatenate(conf), np.concatenate(pcls), np.concatenate(tcls)
    aps = np.full((nc, 10), np.nan)
    for c in range(nc):
        nt = int((tcls == c).sum())
        if nt == 0:
            continue
        idx = pcls == c
        if not idx.any():
            aps[c] = 0.0
            continue
        t = tp[idx][np.argsort(-conf[idx])].astype(float)
        tpc, fpc = np.cumsum(t, 0), np.cumsum(1 - t, 0)
        rec, prec = tpc / nt, tpc / (tpc + fpc)
        aps[c] = [ap_curve(rec[:, k], prec[:, k]) for k in range(10)]
    ngt = len(tcls)
    return {"map50": float(np.nanmean(aps[:, 0])), "map5095": float(np.nanmean(aps)),
            "ap50_per_class": [None if np.isnan(v) else float(v) for v in aps[:, 0]],
            "precision_op": tp30 / max(ndet30, 1), "recall_op": tp30 / max(ngt, 1), "dets_op": ndet30, "gt": ngt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"], choices=list(NETWORK_MODE))
    ap.add_argument("--run", choices=list(NETWORK_MODE), help="(internal) run one precision and store detections")
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    if a.run:
        return run_detector(a.run)

    OUT.mkdir(parents=True, exist_ok=True)
    for prec in a.precisions:  # separate process per precision: clean GStreamer / TensorRT state
        if a.rerun or not (OUT / f"dets_{prec}.json").exists():
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run", prec])
            if r.returncode != 0 or not (OUT / f"dets_{prec}.json").exists():
                print(f"[{prec}] detection run failed, skipping")
                continue

    labels = [l.strip() for l in (ROOT / "models" / "labels.txt").read_text().splitlines() if l.strip()]
    res, gts = {}, None
    for prec in a.precisions:
        f = OUT / f"dets_{prec}.json"
        if not f.exists():
            continue
        blob = json.loads(f.read_text())
        gts = gts or load_gt(blob["images"], blob["sizes"])
        res[prec] = evaluate(blob, gts, len(labels))

    if not res:
        sys.exit("nothing to evaluate")
    ref = res.get("fp32")
    bar = "=" * 86
    first = next(iter(res.values()))
    print(f"\n{bar}\nDETECTOR ACCURACY - {len(gts)} val images, {first['gt']} ground-truth boxes\n{bar}")
    print(f"{'precision':<10}{'mAP50':>9}{'mAP50-95':>10}{'d mAP50-95':>12}{f'P@{OP_CONF}':>10}{f'R@{OP_CONF}':>10}")
    for prec, r in res.items():
        d = f"{(r['map5095'] - ref['map5095']) * 100:+.2f} pt" if ref and prec != "fp32" else "-"
        print(f"{prec:<10}{r['map50']:>9.4f}{r['map5095']:>10.4f}{d:>12}{r['precision_op']:>10.3f}{r['recall_op']:>10.3f}")
    print(f"\nAP50 per class ({OP_CONF} = deployed confidence threshold; P/R at IoU 0.5)")
    print(f"{'precision':<10}" + "".join(f"{n:>10}" for n in labels))
    for prec, r in res.items():
        print(f"{prec:<10}" + "".join(f"{'n/a' if v is None else format(v, '.4f'):>10}" for v in r["ap50_per_class"]))
    (OUT / "accuracy_summary.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {OUT / 'accuracy_summary.json'}")


if __name__ == "__main__":
    main()