"""Per-group measurements: batch timing, live/unique counts and frame-arrival gaps."""
from __future__ import annotations

import collections
import threading

import numpy as np

from .config import FOUR_W, NAN, PERSON, TWO_W, UNTRACKED, VEHICLES

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

    def __init__(self, cams, stages, group, min_hits, warmup, min_crop):
        self.group, self.cams, self.n = group, cams, len(cams)
        self.active = [s for s, on in zip(STAGES, (True, stages != "m1", stages == "full")) if on]
        self.min_hits, self.warmup, self.min_crop = min_hits, warmup, min_crop
        self.lock = threading.Lock()
        self.pending = {}  # batch key -> {stamp name: ns}
        self.recs, self.batches_seen = [], 0
        self.last_final = self.t_first = None
        self.frames_rec = [0] * self.n  # frames after warm-up (used for fps)
        self.live = [{"person": 0, "4W": 0, "2W": 0} for _ in range(self.n)]
        self.peak = [{"person": 0, "vehicle": 0} for _ in range(self.n)]
        self.hits = [collections.Counter() for _ in range(self.n * 2)]  # idx = cam*2 + (0 person | 1 vehicle)
        self.unique = [set() for _ in range(self.n * 2)]
        self.vcls = [collections.defaultdict(collections.Counter) for _ in range(self.n)]
        self.m2_labels = collections.Counter()
        self.batch_fill, self.combo = collections.Counter(), collections.Counter()
        self.last_ntp, self.gaps = [None] * self.n, [[] for _ in range(self.n)]

    @property
    def has_tracker(self):
        return "trk" in self.active

    def stamp(self, key, name, t_ns, wall_ns=None, ntps=None):
        with self.lock:
            d = self.pending.setdefault(key, {})
            d[name] = t_ns
            if wall_ns is not None:
                d["wall_in"] = wall_ns
            if ntps is not None:
                d["ntps"] = ntps
            if len(self.pending) > 200:  # a batch got lost somewhere: don't leak
                for k in sorted(self.pending)[:100]:
                    self.pending.pop(k, None)

    def final(self, key, t_ns, wall_ns, frames):
        """Called when a batch leaves the last stage. Returns the record, or None during warm-up."""
        with self.lock:
            st = self.pending.pop(key, {})
        self.batches_seen += 1
        last, rec, elem_sum = self.active[-1], {}, 0.0
        for name in STAGES:  # per-element time (in-pad -> out-pad); the rest is queue wait
            if name not in self.active:
                rec[f"{name}_ms"] = 0.0
                continue
            t_in = st.get(f"{name}_in")
            t_out = t_ns if name == last else st.get(f"{name}_out")
            rec[f"{name}_ms"] = NAN if t_in is None or t_out is None else (t_out - t_in) / 1e6
            elem_sum += rec[f"{name}_ms"]
        rec["proc_ms"] = (t_ns - (st.get("m1_in") or t_ns)) / 1e6
        rec["queue_ms"] = max(0.0, rec["proc_ms"] - elem_sum) if elem_sum == elem_sum else NAN

        ntps = [f["ntp"] for f in frames if f["ntp"] and abs(f["ntp"] - wall_ns) < 60e9]
        if ntps:
            rec["skew_ms"] = (max(ntps) - min(ntps)) / 1e6
            rec["wait_before_m1_ms"] = ((st.get("wall_in") or wall_ns) - min(ntps)) / 1e6
            rec["e2e_ms"] = (wall_ns - min(ntps)) / 1e6
        else:
            rec["skew_ms"] = rec["wait_before_m1_ms"] = rec["e2e_ms"] = NAN
        rec["interval_ms"] = (t_ns - self.last_final) / 1e6 if self.last_final else NAN
        self.last_final = t_ns
        rec["batch_frames"] = len(frames)
        rec["m2_eligible_objs"] = self._count(frames)
        rec["live_persons_total"] = sum(l["person"] for l in self.live)
        rec["live_vehicles_total"] = sum(l["4W"] + l["2W"] for l in self.live)
        for c, name in enumerate(self.cams):
            rec[f"{name}_persons"] = self.live[c]["person"]
            rec[f"{name}_vehicles"] = self.live[c]["4W"] + self.live[c]["2W"]

        if self.batches_seen <= self.warmup:
            return None
        self.t_first = self.t_first or t_ns
        rec["tick"], rec["t_rel_s"] = self.batches_seen, (t_ns - self.t_first) / 1e9
        self.batch_fill[len(frames)] += 1
        self.combo[tuple(sorted(f["src"] for f in frames))] += 1
        for f in frames:
            s = f["src"]
            if not 0 <= s < self.n:
                continue
            self.frames_rec[s] += 1
            if f["ntp"]:
                if self.last_ntp[s]:
                    self.gaps[s].append((f["ntp"] - self.last_ntp[s]) / 1e6)
                self.last_ntp[s] = f["ntp"]
        self.recs.append(rec)
        return rec

    def _count(self, frames):
        """Update live / peak / unique counters; returns the number of objects big enough for M2."""
        eligible = 0
        for f in frames:
            s = f["src"]
            if not 0 <= s < self.n:
                continue
            p = v = w4 = w2 = 0
            for cls, oid, w, h, label in f["objs"]:
                if cls == PERSON:
                    p, grp = p + 1, 0
                elif cls in VEHICLES:
                    v, grp = v + 1, 1
                    w4 += cls in FOUR_W
                    w2 += cls in TWO_W
                else:
                    continue
                eligible += w >= self.min_crop and h >= self.min_crop
                if label:
                    self.m2_labels[label] += 1
                if oid != UNTRACKED:
                    idx = s * 2 + grp
                    self.hits[idx][oid] += 1
                    if self.hits[idx][oid] >= self.min_hits:
                        self.unique[idx].add(oid)
                    if grp:
                        self.vcls[s][oid][VEHICLES[cls]] += 1
            self.live[s] = {"person": p, "4W": w4, "2W": w2}
            self.peak[s]["person"] = max(self.peak[s]["person"], p)
            self.peak[s]["vehicle"] = max(self.peak[s]["vehicle"], v)
        return eligible

    def count_summary(self):
        trk, cams, tot = self.has_tracker, [], collections.Counter()
        for c in range(self.n):
            ids = self.unique[c * 2 + 1]
            by = collections.Counter(self.vcls[c][i].most_common(1)[0][0] for i in ids if i in self.vcls[c])
            d = {"cam": self.cams[c], "live_persons": self.live[c]["person"], "live_4W": self.live[c]["4W"],
                 "live_2W": self.live[c]["2W"], "peak_persons": self.peak[c]["person"],
                 "peak_vehicles": self.peak[c]["vehicle"],
                 "unique_persons": len(self.unique[c * 2]) if trk else None,
                 "unique_vehicles": len(ids) if trk else None, "unique_vehicles_by_class": dict(by)}
            cams.append(d)
            tot["live_persons"] += d["live_persons"]
            tot["live_vehicles"] += d["live_4W"] + d["live_2W"]
        return {"cameras": cams, "total": dict(tot)}
