"""Per-group measurements: one timing record per batch."""
from __future__ import annotations

import collections
import threading

import numpy as np

from .config import NAN

STAGES = ("m1", "trk", "m2")


def describe(vals):
    """mean / p50 / p95 / p99 / max of the finite values, or None when there are none."""
    a = np.asarray([v for v in vals if v is not None and v == v], float)
    if not a.size:
        return None
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


class Metrics:
    """One instance per camera group. Times are perf_counter ns; ntp / wall are epoch ns."""

    def __init__(self, cams, group, warmup, min_w, min_h):
        self.group, self.cams, self.n = group, cams, len(cams)
        self.warmup, self.min_w, self.min_h = warmup, min_w, min_h
        self.lock = threading.Lock()
        self.pending = {}  # batch key -> {stamp name: ns}
        self.recs, self.batches_seen = [], 0
        self.last_final = self.t_first = None
        self.frames_rec = [0] * self.n  # frames per camera after warm-up (used for fps)
        self.batch_fill, self.combo = collections.Counter(), collections.Counter()

    def stamp(self, key, name, t_ns):
        with self.lock:
            self.pending.setdefault(key, {})[name] = t_ns
            if len(self.pending) > 200:  # a batch got lost somewhere: don't leak
                for k in sorted(self.pending)[:100]:
                    self.pending.pop(k, None)

    def final(self, key, t_ns, wall_ns, frames):
        """Called when a batch leaves M2. Returns the record, or None during warm-up."""
        with self.lock:
            st = self.pending.pop(key, {})
        self.batches_seen += 1
        rec, elem_sum = {}, 0.0
        for name in STAGES:  # per-element time (in-pad -> out-pad); the rest is queue wait
            t_in = st.get(f"{name}_in")
            t_out = t_ns if name == STAGES[-1] else st.get(f"{name}_out")
            rec[f"{name}_ms"] = NAN if t_in is None or t_out is None else (t_out - t_in) / 1e6
            elem_sum += rec[f"{name}_ms"]
        rec["proc_ms"] = (t_ns - (st.get("m1_in") or t_ns)) / 1e6
        rec["queue_ms"] = max(0.0, rec["proc_ms"] - elem_sum) if elem_sum == elem_sum else NAN

        ntps = [f["ntp"] for f in frames if f["ntp"] and abs(f["ntp"] - wall_ns) < 60e9]
        rec["e2e_ms"] = (wall_ns - min(ntps)) / 1e6 if ntps else NAN  # mux entry -> result
        rec["interval_ms"] = (t_ns - self.last_final) / 1e6 if self.last_final else NAN
        self.last_final = t_ns
        rec["batch_frames"] = len(frames)
        rec["eligible_objs"] = sum(w >= self.min_w and h >= self.min_h for f in frames for w, h in f["objs"])

        if self.batches_seen <= self.warmup:
            return None
        first = self.t_first is None
        self.t_first = self.t_first or t_ns
        rec["group"], rec["tick"], rec["t_rel_s"] = self.group, self.batches_seen, (t_ns - self.t_first) / 1e9
        self.batch_fill[len(frames)] += 1
        self.combo[tuple(sorted(f["src"] for f in frames))] += 1
        if not first:  # the first recorded batch only starts the clock, so its frames are not counted in the fps
            for f in frames:
                if 0 <= f["src"] < self.n:
                    self.frames_rec[f["src"]] += 1
        self.recs.append(rec)
        return rec