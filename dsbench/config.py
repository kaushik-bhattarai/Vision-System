"""Loads config.yaml and the cameras JSON into one flat settings object, resolves the camera groups, checks consistency."""
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
    """Warn about the mismatches that make TensorRT rebuild the engine (or silently run something else)."""
    biggest = max(len(g) for g in a.group_idx)
    try:
        pg = read_nvinfer(a.pgie)
    except OSError as e:
        _fail(f"cannot read detector config: {e}")
    if "batch-size" in pg and int(pg["batch-size"]) != a.det_batch:
        print(f"  [warn] {a.pgie}: batch-size={pg['batch-size']} but config.yaml detector.batch_size={a.det_batch}")
    if a.det_batch != biggest:
        print(f"  [warn] detector.batch_size={a.det_batch} but the largest camera group has {biggest} cameras")
    for key in ("onnx-file", "model-engine-file"):
        name = Path(pg.get(key, "")).name
        m = re.search(r"(\d+)x(\d+)", name)
        if m and (int(m.group(1)), int(m.group(2))) != (a.det_height, a.det_width):
            print(f"  [warn] {key} '{name}' says {m.group(1)}x{m.group(2)} (HxW) but config.yaml has "
                  f"{a.det_height}x{a.det_width}")
    eng = Path(pg.get("model-engine-file", "")).name
    if eng and f"_b{a.det_batch}_" not in eng:
        print(f"  [warn] engine file '{eng}' does not contain _b{a.det_batch}_ - the engine will be rebuilt on every start")
    try:
        sg = read_nvinfer(a.sgie)
    except OSError as e:
        _fail(f"cannot read classifier config: {e}")
    a.min_crop_w = int(sg.get("input-object-min-width", 0))
    a.min_crop_h = int(sg.get("input-object-min-height", 0))
    if a.tracker_width % 32 or a.tracker_height % 32:
        print(f"  [warn] tracker size {a.tracker_width}x{a.tracker_height} is not a multiple of 32")


def load_config(path=CONFIG_PATH):
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
    a.pgie, a.det_width, a.det_height, a.det_batch = (_path(g("detector.config")), g("detector.width"),
                                                      g("detector.height"), g("detector.batch_size"))
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
    a.warmup, a.stall_s, a.startup_s = g("run.warmup"), g("run.stall_s"), g("run.startup_s")
    a.out_dir = _path(g("run.out_dir"))

    cams = Path(a.cameras_file)
    if not cams.exists():
        _fail(f"cameras file {cams} not found")
    cj = json.loads(cams.read_text())
    a.names = [c["name"] for c in cj["cameras"]]
    a.uris = [c["uri"] for c in cj["cameras"]]
    if len(set(a.names)) != len(a.names):
        _fail("cameras file: camera names must be unique")
    a.group_idx = _resolve_groups(cj.get("groups"), a.names)
    _check_models(a)
    return a