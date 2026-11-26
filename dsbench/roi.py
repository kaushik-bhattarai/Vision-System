"""Per-camera regions of interest: crossing detection (enter / exit) for people and vehicles, with timestamp and a saved image.
"""
from __future__ import annotations

import collections
import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import HUMAN
from .gstutil import Gst

EVENT_COLORS = {"enter": (0, 0, 255), "exit": (0, 200, 255)}  # BGR banner colour per event type


def category(cls):
    """Counter a class belongs to: people on their own, bikes and vehicles together."""
    return "person" if cls == HUMAN else "vehicle"


def point_in_polygon(x, y, pts):
    """Ray casting. pts = [(x, y), ...] in the same units as x, y."""
    inside, n = False, len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def load_zones(path, names, width, height):
    """-> {camera index: [(x_px, y_px), ...]} (pixel coordinates of the muxer output). Unknown cameras are reported."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as e:
        raise SystemExit(f"roi: {path} is not valid JSON: {e}")
    zones = {}
    for name, poly in data.items():
        pts = poly.get("polygon") if isinstance(poly, dict) else poly
        if name not in names:
            print(f"  [roi] warning: '{name}' in {path.name} is not a camera in the cameras file, ignored")
            continue
        if not pts or len(pts) < 3:
            print(f"  [roi] warning: zone of '{name}' has fewer than 3 points, ignored")
            continue
        zones[names.index(name)] = [(float(x) * width, float(y) * height) for x, y in pts]
    return zones


class RoiMonitor:
    def __init__(self, cams, a):
        self.cams = cams
        self.classes = set(a.roi_classes)
        self.events = a.roi_events          # both | enter | exit
        self.confirm = a.roi_confirm        # consecutive observations needed before a side change counts
        self.lost_s = a.roi_lost_s
        self.jpeg = a.roi_jpeg
        self.zones = load_zones(a.roi_file, cams, a.mux_width, a.mux_height)
        self.lock = threading.Lock()
        self.state = collections.defaultdict(dict)       # cam -> track id -> [inside, pending count, last seen]
        self.pending = collections.defaultdict(list)     # cam -> events waiting for a frame to be saved
        self.counts = collections.defaultdict(collections.Counter)  # cam name -> "person enter" / "vehicle exit" -> n
        self.saved = 0
        if not self.zones:
            print(f"[roi] no zones in {Path(a.roi_file).name}: ROI crossing detection is off "
                  f"(draw zones with: python tools/draw_roi.py)")
            self.q = self.thread = None
            return
        self.out = Path(a.roi_out_dir) / a.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out / "events.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "camera", "event", "category", "class", "track_id", "image"])
        self.q = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._saver, daemon=True, name="roi-saver")
        self.thread.start()
        print(f"[roi] zones on {len(self.zones)} camera(s): {', '.join(cams[c] for c in sorted(self.zones))} | "
              f"classes {sorted(self.classes)} | events {self.events} | images -> {self.out}")

    def has_zone(self, cam):
        return cam in self.zones

    # ---- decision (annotator probe) ----
    def update(self, cam, objs, now=None):
        """objs = [(track_id or None, class_id, label, x, y)]; (x, y) = bottom-centre of the box, in muxer pixels.
        Returns True while a tracked person / vehicle stands inside the zone (the zone is drawn red then)."""
        zone = self.zones[cam]
        now = now if now is not None else time.monotonic()
        st, found, any_inside = self.state[cam], [], False
        wall = datetime.now()
        for tid, cls, label, x, y in objs:
            if tid is None or cls not in self.classes:
                continue
            inside = point_in_polygon(x, y, zone)
            any_inside = any_inside or inside
            rec = st.get(tid)
            if rec is None:                     # first sighting: only remember the side, a track that appears inside is no crossing
                st[tid] = [inside, 0, now]
                continue
            rec[2] = now
            if inside == rec[0]:
                rec[1] = 0
                continue
            rec[1] += 1
            if rec[1] >= self.confirm:
                rec[0], rec[1] = inside, 0
                ev = "enter" if inside else "exit"
                if self.events in ("both", ev):
                    found.append({"time": wall, "camera": self.cams[cam], "event": ev, "class_id": cls,
                                  "label": label, "track_id": tid, "category": category(cls)})
        for tid in [t for t, r in st.items() if now - r[2] > self.lost_s]:
            del st[tid]
        if found:
            with self.lock:
                self.pending[cam].extend(found)
                for e in found:  # counted when decided, so a dropped image never loses a count
                    self.counts[self.cams[cam]][f"{e['category']} {e['event']}"] += 1
        return any_inside

    # ---- image grab (display branch, after the OSD) ----
    def attach_capture(self, cam, el):
        """el = last element of the camera's display branch (RGBA, system memory)."""
        el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._capture_cb(cam), None)

    def _capture_cb(self, cam):
        def cb(pad, info, _):
            if not self.pending[cam]:           # cheap unlocked check on every frame
                return Gst.PadProbeReturn.OK
            with self.lock:
                evs, self.pending[cam] = self.pending[cam], []
            buf, caps = info.get_buffer(), pad.get_current_caps()
            if not evs or buf is None or caps is None:
                return Gst.PadProbeReturn.OK
            s = caps.get_structure(0)
            w, h = s.get_value("width"), s.get_value("height")
            ok, mi = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.PadProbeReturn.OK
            try:
                raw = np.frombuffer(mi.data, np.uint8)
                frame = raw[: h * w * 4].reshape(h, w, 4).copy() if raw.size >= h * w * 4 else None
            finally:
                buf.unmap(mi)
            if frame is not None:
                try:
                    self.q.put_nowait((cam, evs, frame))
                except queue.Full:
                    print("[roi] warning: image queue full, crossing image dropped (event still counted)")
            return Gst.PadProbeReturn.OK
        return cb

    # ---- writing (saver thread) ----
    def _saver(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self._save(*item)
            except Exception as e:  # never let the saver die
                print(f"[roi] warning: could not save crossing image: {e}")

    def _save(self, cam, evs, frame_rgba):
        import cv2

        cam_name = self.cams[cam]
        img = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        h, w = img.shape[:2]
        scale = max(h / 720.0, 0.6)
        fs, th = 0.7 * scale, max(1, round(1.5 * scale))
        t0 = evs[0]["time"]
        lines = [f"CAMERA {cam_name}   {t0:%Y-%m-%d %H:%M:%S}.{t0.microsecond // 1000:03d}"]
        lines += [f"{e['event'].upper()}  {e['label']} #{e['track_id']}" for e in evs[:6]]
        if len(evs) > 6:
            lines.append(f"+{len(evs) - 6} more")
        lh = int(26 * scale)
        cv2.rectangle(img, (0, 0), (w, lh * len(lines) + 8), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            col = (255, 255, 255) if i == 0 else EVENT_COLORS[evs[i - 1]["event"]]
            cv2.putText(img, text, (8, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), EVENT_COLORS[evs[0]["event"]], max(2, round(3 * scale)))

        folder = self.out / cam_name
        folder.mkdir(parents=True, exist_ok=True)
        base = f"{cam_name}_{t0:%Y%m%d_%H%M%S}_{t0.microsecond // 1000:03d}"
        path, n = folder / f"{base}.jpg", 1
        while path.exists():
            path, n = folder / f"{base}_{n}.jpg", n + 1
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg])
        rel = path.relative_to(self.out).as_posix()
        with open(self.csv_path, "a", newline="") as f:
            wr = csv.writer(f)
            for e in evs:
                wr.writerow([f"{e['time']:%Y-%m-%d %H:%M:%S.%f}"[:-3], cam_name, e["event"], e["category"], e["label"], e["track_id"], rel])
        with self.lock:
            self.saved += 1
        print(f"[roi] {cam_name}: " + ", ".join(f"{e['event']} {e['label']} #{e['track_id']}" for e in evs) + f" -> {rel}")

    def close(self):
        if self.thread is not None:
            self.q.put(None)
            self.thread.join(5)

    def counter(self, cam):
        """{"person": {"enter": n, "exit": n}, "vehicle": {...}} of one camera (by index)."""
        c = self.counts.get(self.cams[cam], {})
        return {k: {"enter": c.get(f"{k} enter", 0), "exit": c.get(f"{k} exit", 0)} for k in ("person", "vehicle")}

    def counter_text(self, cam):
        c = self.counter(cam)
        return "   ".join(f"{k} in {v['enter']} out {v['exit']}" for k, v in c.items())

    def summary(self):
        """{camera: {"person": {"enter": n, "exit": n}, "vehicle": {...}}} for the cameras that have a zone."""
        if not self.zones:
            return {}
        with self.lock:
            return {self.cams[c]: self.counter(c) for c in sorted(self.zones)}
