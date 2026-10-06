"""Constants, command-line options, camera list and camera-group resolution."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# COCO class ids the pipeline cares about (the YOLO parser only emits these)
PERSON = 0
TWO_W = {1: "bicycle", 3: "motorcycle"}
FOUR_W = {2: "car", 5: "bus", 7: "truck"}
VEHICLES = {**TWO_W, **FOUR_W}
COCO_NAMES = {PERSON: "person", **VEHICLES}
UNTRACKED = 0xFFFFFFFFFFFFFFFF
NAN = float("nan")

STAGE_TAGS = {"m1": "M1_only", "track": "M1+track", "full": "M1+track+M2"}
SUB_MUX, SUB_TRACKER, SUB_MIN_CROP = (640, 360), (640, 384), 16  # substream defaults
MAIN_MUX, MAIN_TRACKER, MAIN_MIN_CROP = (1920, 1080), (960, 544), 32

_CHAN = re.compile(r"(/Streaming/Channels/\d*0)[12](?=$|[/?#])")


def pick_stream(uri: str, which: str) -> str:
    """Hikvision channel ids: x01 = main stream, x02 = substream. Other URIs are left alone."""
    if which == "asis":
        return uri
    return _CHAN.sub(lambda m: m.group(1) + ("2" if which == "sub" else "1"), uri)


def mask_uri(uri: str) -> str:
    return re.sub(r"//[^@/]*@", "//***@", uri)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepStream cascade benchmark with round-robin camera groups.",
                                allow_abbrev=False)
    c = p.add_argument_group("cameras")
    c.add_argument("--cameras", default=str(ROOT / "config/cameras.json"),
                   help="JSON file with cameras (+ optional groups); used when --uri is not given")
    c.add_argument("--uri", dest="uris", nargs="+", help="RTSP URLs or video files (overrides --cameras)")
    c.add_argument("--names", nargs="+", help="camera names matching --uri")
    c.add_argument("--groups", nargs="+", metavar="A,B", help="fixed batches by camera name, e.g. D4,D3 D2,D1")
    c.add_argument("--group-sizes", help="consecutive groups by size, e.g. 6,6,5")
    c.add_argument("--batch-size", type=int, default=0, help="cameras per group when no groups are given (0 = all)")
    c.add_argument("--stream", choices=["sub", "main", "asis"], default="sub",
                   help="sub: switch Hikvision .../x01 URLs to the substream .../x02 and use small sizes; "
                        "main: x01; asis: URLs untouched")
    c.add_argument("--latency-ms", type=int, default=400, help="RTSP jitter-buffer latency")
    c.add_argument("--file-loop", action="store_true", help="loop video files")
    c.add_argument("--extra-surfaces", type=int, default=4,
                   help="nvurisrcbin num-extra-surfaces (held frames pin decoder surfaces)")

    m = p.add_argument_group("models / pipeline")
    m.add_argument("--stages", choices=list(STAGE_TAGS), default="full")
    m.add_argument("--pgie", default=str(ROOT / "config/pgie_yolo26m.txt"))
    m.add_argument("--sgie", default=str(ROOT / "config/sgie_stage2.txt"))
    m.add_argument("--tracker-lib", default="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    m.add_argument("--tracker-config", default=str(ROOT / "config/tracker_accuracy.yml"))
    m.add_argument("--tracker-width", type=int, help="multiple of 32 (default 640 sub / 960 main)")
    m.add_argument("--tracker-height", type=int, help="multiple of 32 (default 384 sub / 544 main)")
    m.add_argument("--mux-width", type=int, help="default 640 sub / 1920 main")
    m.add_argument("--mux-height", type=int, help="default 360 sub / 1080 main")
    m.add_argument("--mux-batch-size", type=int, default=0, help="0 = size of the largest group")
    m.add_argument("--push-timeout-us", type=int, default=5000, help="legacy muxer batched-push-timeout")
    m.add_argument("--min-crop", type=int, help="smallest box side (px at mux size) that counts as a crop for M2")
    m.add_argument("--stage-queue", type=int, default=2, help="buffers in the queues between stages")
    m.add_argument("--no-queues", dest="queues", action="store_false", help="no queues between M1/tracker/M2")
    m.add_argument("--min-hits", type=int, default=3, help="frames a tracker ID must appear to count as unique")
    m.add_argument("--gpu-id", type=int, default=0)

    s = p.add_argument_group("round-robin scheduler")
    s.add_argument("--slot-buffers", type=int, default=1, help="frames each camera slot keeps (1 = newest only)")
    s.add_argument("--token-at", choices=["m1", "final"], default="m1",
                   help="m1: next group may enter M1 once the previous batch left M1; final: one group at a time")
    s.add_argument("--gather-ms", type=float, default=25, help="max wait for ALL cameras of a group (0 = whoever is ready)")
    s.add_argument("--turn-timeout-ms", type=float, default=50, help="safety net for an armed gate")
    s.add_argument("--token-timeout-ms", type=float, default=3000, help="release a lost token after this long")
    s.add_argument("--max-frame-age-ms", type=float, default=1000, help="older slot frames count as 'no frame'")
    s.add_argument("--offline-after", type=int, default=3, help="missed turns before a camera is skipped without waiting")

    d = p.add_argument_group("display")
    d.add_argument("--display", action=argparse.BooleanOptionalAction, default=None,
                   help="show cameras (default: on, except with --compare)")
    d.add_argument("--separate-windows", action="store_true", help="one window per camera")
    d.add_argument("--window-width", type=int, default=800)
    d.add_argument("--window-height", type=int, default=450)
    d.add_argument("--osd-font", type=int, help="label font size (default scales with mux width)")

    t = p.add_argument_group("tracker audit")
    t.add_argument("--no-audit", action="store_true")
    t.add_argument("--frag-window-s", type=float, default=2.0)
    t.add_argument("--frag-dist", type=float, default=1.0, help="max centre distance in object sizes")
    t.add_argument("--short-track-frames", type=int, default=15)

    r = p.add_argument_group("run control / output")
    r.add_argument("--compare", action="store_true", help="run m1, track and full one after another and compare")
    r.add_argument("--duration", type=float, help="seconds recorded after warm-up (0 = until Ctrl+C; default 60 with --compare)")
    r.add_argument("--warmup", type=int, default=50, help="batches per group discarded from the stats")
    r.add_argument("--stall-s", type=int, default=30, help="abort if no batch for this long while running")
    r.add_argument("--startup-s", type=int, default=1200, help="wait for the first batch (TensorRT engine build)")
    r.add_argument("--report-every", type=float, default=5)
    r.add_argument("--out-dir", default="bench_out")
    r.add_argument("--tag", default=None)
    return p


def resolve_groups(a, parser) -> list[list[int]]:
    """Camera groups = fixed batches. Returns lists of indices into a.uris."""
    n = len(a.uris)
    if a.group_sizes:
        try:
            sizes = [int(x) for x in a.group_sizes.split(",")]
        except ValueError:
            parser.error("--group-sizes must look like 6,6,5")
        if sum(sizes) != n or min(sizes) <= 0:
            parser.error(f"--group-sizes {sizes} adds up to {sum(sizes)} but there are {n} cameras")
        out, i = [], 0
        for s in sizes:
            out.append(list(range(i, i + s)))
            i += s
        return out
    if a.groups:
        out = []
        for spec in a.groups:
            names = [x.strip() for x in spec.split(",")]
            bad = [x for x in names if x not in a.names]
            if bad:
                parser.error(f"--groups: unknown camera {bad} (names: {a.names})")
            out.append([a.names.index(x) for x in names])
        if sorted(i for g in out for i in g) != list(range(n)):
            parser.error("--groups must list every camera exactly once")
        return out
    bs = a.batch_size or n
    return [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]


def parse_args(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)

    if a.uris is None:  # cameras come from the JSON file
        path = Path(a.cameras)
        if not path.exists():
            parser.error(f"no --uri given and {path} not found - copy config/cameras.example.json to config/cameras.json")
        cfg = json.loads(path.read_text())
        a.uris = [c["uri"] for c in cfg["cameras"]]
        a.names = a.names or [c["name"] for c in cfg["cameras"]]
        if not (a.groups or a.group_sizes or a.batch_size) and cfg.get("groups"):
            a.groups = [",".join(g) for g in cfg["groups"]]
    a.names = a.names or [f"cam{i + 1}" for i in range(len(a.uris))]
    if len(a.names) != len(a.uris):
        parser.error("--names must match the number of cameras")
    a.group_idx = resolve_groups(a, parser)

    sub = a.stream == "sub"
    a.mux_width = a.mux_width or (SUB_MUX if sub else MAIN_MUX)[0]
    a.mux_height = a.mux_height or (SUB_MUX if sub else MAIN_MUX)[1]
    a.tracker_width = a.tracker_width or (SUB_TRACKER if sub else MAIN_TRACKER)[0]
    a.tracker_height = a.tracker_height or (SUB_TRACKER if sub else MAIN_TRACKER)[1]
    a.min_crop = a.min_crop or (SUB_MIN_CROP if sub else MAIN_MIN_CROP)
    a.osd_font = a.osd_font or max(12, round(30 * a.mux_width / 1920))
    if a.display is None:
        a.display = not a.compare
    if a.duration is None:
        a.duration = 60 if a.compare else 0
    a.uris = [pick_stream(u, a.stream) for u in a.uris]
    if a.tracker_width % 32 or a.tracker_height % 32:
        print(f"[warn] tracker size {a.tracker_width}x{a.tracker_height} is not a multiple of 32")
    return a
