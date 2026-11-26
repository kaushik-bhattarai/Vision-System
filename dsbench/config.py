"""Loads config.yaml and the cameras JSON into one flat settings object, resolves the camera groups, checks consistency,
and generates the detector's nvinfer config (see models.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

HUMAN = 0  # class id of "human" in the fine-tuned detector (0 human, 1 bike, 2 vehicle)
UNTRACKED = 0xFFFFFFFFFFFFFFFF
NAN = float("nan")


def mask_uri(uri: str) -> str:
    return re.sub(r"//[^@/]*@", "//***@", uri)


def _fail(msg):
    raise SystemExit(f"config: {msg}")


def _need(cfg, dotted):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or node.get(k) is None:
            _fail(f"config.yaml is missing '{dotted}'")
        node = node[k]
    return node


def _path(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def read_nvinfer(path) -> dict:
    """key=value pairs of an nvinfer config file."""
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "[")) and "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def _resolve_groups(spec, names):
    """Camera groups = fixed batches, as indices into the camera list. No 'groups' in the JSON -> one group of all."""
    if not spec:
        return [list(range(len(names)))]
    out = []
    for g in spec:
        bad = [x for x in g if x not in names]
        if bad:
            _fail(f"cameras file: group lists unknown camera {bad}")
        out.append([names.index(x) for x in g])
    if sorted(i for g in out for i in g) != list(range(len(names))):
        _fail("cameras file: 'groups' must list every camera exactly once")
    return out


def _check_models(a):
    """Reads the classifier crop size and checks settings that would waste or break the run."""
    biggest = max(len(g) for g in a.group_idx)
    if a.det_batch < biggest:
        _fail(f"detector.batch_size={a.det_batch} is smaller than the largest camera group ({biggest} cameras)")
    if a.det_batch > biggest:
        print(f"  [warn] detector.batch_size={a.det_batch} but the largest camera group has {biggest} cameras "
              f"(set detector.batch_size: 0 for auto)")
    try:
        sg = read_nvinfer(a.sgie)
    except OSError as e:
        _fail(f"cannot read classifier config: {e}")
    a.min_crop_w = int(sg.get("input-object-min-width", 0))
    a.min_crop_h = int(sg.get("input-object-min-height", 0))
    if a.tracker_width % 32 or a.tracker_height % 32:
        print(f"  [warn] tracker size {a.tracker_width}x{a.tracker_height} is not a multiple of 32")


def load_config(path=CONFIG_PATH, make_pgie=True):
    """make_pgie=False: skip generating config/_pgie_generated.txt (training/accuracy.py builds its own per precision)."""
    path = Path(path)
    if not path.exists():
        _fail(f"{path} not found")
    cfg = yaml.safe_load(path.read_text()) or {}
    g = lambda k: _need(cfg, k)  # noqa: E731
    a = SimpleNamespace(config_path=str(path))

    a.cameras_file = _path(g("cameras_file"))
    a.latency_ms, a.extra_surfaces = g("stream.rtsp_latency_ms"), g("stream.extra_surfaces")
    a.mux_width, a.mux_height = g("mux.width"), g("mux.height")
    a.mux_batch_size, a.push_timeout_us = g("mux.batch_size"), g("mux.push_timeout_us")

    # cameras first: detector.batch_size 0 = size of the largest group
    cams = Path(a.cameras_file)
    if not cams.exists():
        _fail(f"cameras file {cams} not found")
    cj = json.loads(cams.read_text())
    a.names = [c["name"] for c in cj["cameras"]]
    a.uris = [c["uri"] for c in cj["cameras"]]
    if len(set(a.names)) != len(a.names):
        _fail("cameras file: camera names must be unique")
    a.group_idx = _resolve_groups(cj.get("groups"), a.names)

    a.pgie_template = _path(g("detector.config"))
    a.det_name, a.det_width, a.det_height = g("detector.name"), g("detector.width"), g("detector.height")
    a.det_batch = int(g("detector.batch_size")) or max(len(x) for x in a.group_idx)
    a.det_precision = str(g("detector.precision")).lower()
    if a.det_precision not in ("int8", "fp16", "fp32"):
        _fail("detector.precision must be int8, fp16 or fp32")
    if a.det_width % 32 or a.det_height % 32:
        print(f"  [warn] detector size {a.det_width}x{a.det_height} is not a multiple of 32")

    a.tracker_lib, a.tracker_config = g("tracker.lib"), _path(g("tracker.config"))
    a.tracker_width, a.tracker_height = g("tracker.width"), g("tracker.height")
    a.sgie = _path(g("classifier.config"))
    a.stage_queue, a.gpu_id = g("pipeline.stage_queue"), g("pipeline.gpu_id")
    for k in ("slot_buffers", "token_at", "gather_ms", "turn_timeout_ms", "token_timeout_ms", "max_frame_age_ms",
              "offline_after"):
        setattr(a, k, g(f"scheduler.{k}"))
    if a.token_at not in ("m1", "final"):
        _fail("scheduler.token_at must be 'm1' or 'final'")
    a.tile_width, a.tile_height, a.osd_font = g("display.tile_width"), g("display.tile_height"), g("display.osd_font")
    for k in ("dwell_lost_s", "dwell_alert_s", "crowd_min_people", "crowd_distance", "cluster_min_people"):
        setattr(a, k, g(f"analytics.{k}"))
    r = cfg.get("roi") or {}
    a.roi_file = _path(r.get("file", "config/rois.json"))
    a.roi_out_dir = _path(r.get("out_dir", "bench_out/roi"))
    a.roi_classes = [int(x) for x in r.get("classes", [0, 1, 2])]
    a.roi_events = str(r.get("events", "both")).lower()
    if a.roi_events not in ("both", "enter", "exit"):
        _fail("roi.events must be both, enter or exit")
    a.roi_confirm, a.roi_lost_s = int(r.get("confirm_frames", 2)), float(r.get("lost_s", 5))
    a.roi_jpeg = int(r.get("jpeg_quality", 90))
    a.warmup, a.stall_s, a.startup_s = g("run.warmup"), g("run.stall_s"), g("run.startup_s")
    a.out_dir = _path(g("run.out_dir"))

    _check_models(a)

    a.pgie = None  # the nvinfer config that pipeline.py loads
    if make_pgie:
        from .models import write_pgie  # imported here: models.py imports this module

        a.pgie = str(write_pgie(a))
        print(f"[config] detector {a.det_name}_{a.det_height}x{a.det_width}  batch {a.det_batch}  {a.det_precision}"
              f"  -> {Path(a.pgie).name}")
    return a
