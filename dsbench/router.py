"""Splits the global batches coming out of the shared pipeline into the per-group Metrics and checks batch integrity."""
from __future__ import annotations

import collections
import threading
import time

from .config import NAN


class Router:
    def __init__(self, groups, mets, gates, sched):
        self.mets, self.gates, self.sched = mets, gates, sched
        self.cam_group = {s: (g, j) for g, idxs in enumerate(groups) for j, s in enumerate(idxs)}
        self.lock = threading.Lock()
        self.fifo = collections.deque()  # batch keys in the order they left M1
        self.last_m1_out = None
        self.m1_gaps = []
        self.m1_batches = self.overlapped = self.after_m1 = self.inflight_max = 0
        self.mixed = self.dup_src = self.order_bad = self.fnum_bad = 0
        self.last_fnum = {}

    def split(self, frames):
        """{group: [frames with group-local 'src' and global 'gsrc']}"""
        per = collections.defaultdict(list)
        for f in frames:
            gj = self.cam_group.get(f["src"])
            if gj is not None:
                per[gj[0]].append(dict(f, src=gj[1], gsrc=f["src"]))
        return per

    def stage_in(self, label, key, frames, t):
        per = self.split(frames)
        for g in per:
            self.mets[g].stamp(key, f"{label}_in", t)
        if label == "m1":
            with self.lock:
                self.m1_batches += 1
                self.mixed += len(per) > 1  # a batch must hold the cameras of ONE group only
                self.dup_src += len({f["src"] for f in frames}) != len(frames)
                if self.last_m1_out is not None:
                    self.m1_gaps.append((t - self.last_m1_out) / 1e6)
                self.overlapped += self.after_m1 > 0  # previous batch is still in tracker / M2

    def stage_out(self, label, key, frames, t, wall, is_m1, is_last):
        per, srcs = self.split(frames), [f["src"] for f in frames]
        if is_m1:
            with self.lock:
                self.last_m1_out = t
                self.fifo.append(key)
                self.after_m1 += 1
                self.inflight_max = max(self.inflight_max, self.after_m1)
            self.sched.release("m1", srcs)
        if not is_last:
            for g in per:
                self.mets[g].stamp(key, f"{label}_out", t)
            return
        with self.lock:
            expected = self.fifo.popleft() if self.fifo else None
            if expected != key:  # batches must reach the end in the order they left M1
                self.order_bad += 1
                if expected is not None:
                    self.fifo.appendleft(expected)
                try:
                    self.fifo.remove(key)
                except ValueError:
                    pass
            self.after_m1 = max(0, self.after_m1 - 1)
            for f in frames:
                last = self.last_fnum.get(f["src"])
                self.fnum_bad += last is not None and f["num"] <= last
                self.last_fnum[f["src"]] = f["num"]
        for g, fr in per.items():
            ages = [self.gates[f["gsrc"]].pop_age() for f in fr]  # always pop: keeps the FIFO aligned
            rec = self.mets[g].final(key, t, wall, fr)
            if rec is not None:
                ok = [x for x in ages if x == x]
                slot, e2e = (max(ok) if ok else NAN), rec.get("e2e_ms", NAN)
                rec["slot_age_ms"] = slot
                rec["e2e_true_ms"] = slot + e2e if slot == slot and e2e == e2e else NAN
        self.sched.release("final", srcs)