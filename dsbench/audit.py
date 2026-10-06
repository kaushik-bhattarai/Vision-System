"""Tracker audit: follows every tracker ID per camera and flags fragmentation / duplicate tracks.

Events
  FRAG     a NEW id appears within window_s of a CONFIRMED track (>= min_hits frames) of the same kind vanishing
           nearby (centre distance <= dist x object size) that did not leave through the frame edge.
  DUP      a NEW id is born on top (IoU > 0.5) of another live track of the same kind in the same frame.
  REASSOC  an id that vanished came back with the SAME id within window_s (tracker re-associated: good).
"""
from __future__ import annotations

import collections
import csv
import math
import os
import threading
import time

import numpy as np

from .config import COCO_NAMES, PERSON, UNTRACKED

EDGE = 0.02  # a track whose last box touches this fraction of the frame border "left the scene"


def iou(a, b):
    """Boxes are (left, top, width, height)."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _kind(grp):
    return "person" if grp == 0 else "vehicle"


class TrackAudit:
    def __init__(self, cams, a, enabled=True):
        self.cams, self.W, self.H, self.enabled = cams, a.mux_width, a.mux_height, enabled
        self.min_hits, self.short_frames = a.min_hits, a.short_track_frames
        self.window_ns, self.dist = int(a.frag_window_s * 1e9), a.frag_dist
        self.lock = threading.Lock()
        self.t_ref = time.perf_counter_ns()
        self.events = []
        self.S = [dict(act={}, lost={}, done=[], frames=0, confirmed=0, frag=0, dup=0, reappear=0,
                       t_first=None, t_last=None) for _ in cams]

    def _at_edge(self, box):
        l, t, w, h = box
        return l <= EDGE * self.W or t <= EDGE * self.H or l + w >= (1 - EDGE) * self.W or t + h >= (1 - EDGE) * self.H

    def update(self, cam, objs, t_ns):
        """objs: [(oid, cls, left, top, w, h)] of one analysed frame. Returns {oid: track} for tracked objects."""
        if not self.enabled or not 0 <= cam < len(self.S):
            return {}
        S, recs, newborn = self.S[cam], {}, []
        with self.lock:
            S["frames"] += 1
            S["t_first"] = S["t_first"] or t_ns
            S["t_last"] = t_ns
            seen = {o[0] for o in objs if o[0] != UNTRACKED}
            for oid in [k for k in S["act"] if k not in seen]:  # vanished this frame: wait and see
                tr = S["act"].pop(oid)
                tr["edge"] = self._at_edge(tr["box"])
                S["lost"][oid] = tr
            for oid in [k for k, tr in S["lost"].items() if t_ns - tr["t1"] > self.window_ns]:
                S["done"].append(S["lost"].pop(oid))  # gone for good

            for oid, cls, l, t, w, h in objs:
                if oid == UNTRACKED:
                    continue
                tr = S["act"].get(oid)
                if tr is None:
                    tr = S["lost"].pop(oid, None)
                    if tr is not None:  # same ID came back after a short gap
                        S["reappear"] += 1
                    else:
                        tr = dict(id=oid, grp=0 if cls == PERSON else 1, cls=collections.Counter(), t0=t_ns, t1=t_ns,
                                  n=0, box=(l, t, w, h), edge=False, linked=None, flag=None, flag_until=0)
                        newborn.append(tr)
                    S["act"][oid] = tr
                tr["cls"][cls] += 1
                tr["t1"], tr["n"], tr["box"] = t_ns, tr["n"] + 1, (l, t, w, h)
                if tr["n"] == self.min_hits:
                    S["confirmed"] += 1
                recs[oid] = tr
            for tr in newborn:
                self._classify_newborn(cam, S, tr, recs, t_ns)
        return recs

    def _classify_newborn(self, cam, S, tr, recs, t_ns):
        b = tr["box"]
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        best = bd = None
        for oid0, tr0 in S["lost"].items():  # a confirmed track of the same kind that just vanished nearby?
            if tr0["grp"] != tr["grp"] or tr0["n"] < self.min_hits or tr0["edge"]:
                continue
            b0 = tr0["box"]
            d = math.hypot(cx - (b0[0] + b0[2] / 2), cy - (b0[1] + b0[3] / 2))
            if d <= self.dist * max(b0[2], b0[3], b[2], b[3]) and (bd is None or d < bd):
                best, bd = oid0, d
        if best is not None:
            tr0 = S["lost"].pop(best)
            S["done"].append(tr0)
            tr["linked"], tr["flag"], tr["flag_until"] = best, f"FRAG<-#{best}", 40
            S["frag"] += 1
            self._event(cam, "FRAG", best, tr, t_ns, (t_ns - tr0["t1"]) / 1e6, bd, tr0["n"])
            return
        for oid2, tr2 in recs.items():  # born on top of a live track?
            if oid2 != tr["id"] and tr2["grp"] == tr["grp"] and iou(b, tr2["box"]) > 0.5:
                tr["flag"], tr["flag_until"] = f"DUP~#{oid2}", 40
                S["dup"] += 1
                self._event(cam, "DUP", oid2, tr, t_ns, 0.0, 0.0, tr2["n"])
                return

    def _event(self, cam, kind, old_id, tr, t_ns, gap_ms, dist_px, old_frames):
        self.events.append(dict(cam=self.cams[cam], t_s=(t_ns - self.t_ref) / 1e9, kind=kind, old_id=old_id,
                                new_id=tr["id"], kind_of=_kind(tr["grp"]), gap_ms=gap_ms, dist_px=dist_px,
                                old_frames=old_frames))

    # ---- live text -----------------------------------------------------------------
    def header(self, cam, persons, vehicles):
        S = self.S[cam]
        pct = 100.0 * S["frag"] / max(S["confirmed"], 1)
        return [f"{self.cams[cam]}   P {persons}  V {vehicles}   IDs {S['confirmed']}",
                f"frag? {S['frag']} ({pct:.0f}%)   dup {S['dup']}   re-assoc {S['reappear']}"]

    def short_line(self, cam):
        S = self.S[cam]
        return f"{self.cams[cam]}: ids {S['confirmed']} frag? {S['frag']} dup {S['dup']} reassoc {S['reappear']}"

    # ---- end of run ----------------------------------------------------------------
    def _all_tracks(self, cam):
        S = self.S[cam]
        return S["done"] + list(S["lost"].values()) + list(S["act"].values())

    def summary(self):
        out = []
        with self.lock:
            for c, S in enumerate(self.S):
                tracks = self._all_tracks(c)
                ended = S["done"] + list(S["lost"].values())
                conf = [t for t in tracks if t["n"] >= self.min_hits]
                lives = np.asarray([(t["t1"] - t["t0"]) / 1e9 for t in ended if t["n"] >= self.min_hits], float)
                dur_min = (S["t_last"] - S["t_first"]) / 6e10 if S["t_first"] else 0.0
                out.append({
                    "cam": self.cams[c], "frames": S["frames"], "ids_total": len(tracks), "ids_confirmed": len(conf),
                    "flicker_ids": len(tracks) - len(conf),
                    "short_confirmed": sum(t["n"] < self.short_frames for t in conf),
                    "frag_suspects": S["frag"], "frag_pct": 100.0 * S["frag"] / max(len(conf), 1),
                    "dup_births": S["dup"], "reassoc_same_id": S["reappear"],
                    "class_flip_tracks": sum(len(t["cls"]) > 1 for t in conf),
                    "life_median_s": float(np.median(lives)) if lives.size else None,
                    "life_p90_s": float(np.percentile(lives, 90)) if lives.size else None,
                    "new_ids_per_min": len(conf) / dur_min if dur_min > 0 else None})
        return out

    def write_csv(self, out_dir, tag):
        with self.lock:
            rows = [{"cam": self.cams[c], "id": t["id"], "kind": _kind(t["grp"]),
                     "classes": "+".join(f"{COCO_NAMES.get(k, k)}:{v}" for k, v in t["cls"].most_common()),
                     "frames": t["n"], "first_s": (t["t0"] - self.t_ref) / 1e9, "last_s": (t["t1"] - self.t_ref) / 1e9,
                     "life_s": (t["t1"] - t["t0"]) / 1e9, "linked_from": "" if t["linked"] is None else t["linked"]}
                    for c in range(len(self.S)) for t in self._all_tracks(c)]
            events = list(self.events)
        for name, data in ((f"tracks_{tag}.csv", sorted(rows, key=lambda r: (r["cam"], r["first_s"]))),
                           (f"frag_events_{tag}.csv", events)):
            if data:
                with open(os.path.join(out_dir, name), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0]))
                    w.writeheader()
                    w.writerows(data)
