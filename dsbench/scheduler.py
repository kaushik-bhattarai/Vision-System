"""Round-robin scheduling: a latest-frame slot + gate per camera, and the scheduler that releases one group at a time.

Per camera:  decoder -> leaky queue (size 1, the "slot") -> [gate] -> muxer
The queue's streaming thread blocks in a src-pad probe while the gate is closed, holding the OLDEST frame; the queue
keeps the NEWEST behind it. When the gate is armed and a newer frame is already waiting, the held frame is dropped
and the loop repeats, so the frame that finally passes is the newest one.
"""
from __future__ import annotations

import collections
import threading
import time

from .config import NAN
from .gstutil import Gst

FIRST_FRAME_S = 30  # how long to wait for every camera's first frame before starting


class Gate:
    def __init__(self, name, queue, stop_ev):
        self.name, self.q, self.stop = name, queue, stop_ev
        self.cv = threading.Condition()
        self.tl = threading.Lock()  # guards t_arrive (decoder thread writes, gate thread reads)
        self.armed = self.passed = self.holding = False
        self.t_arrive = collections.OrderedDict()  # pts -> ns when the frame reached the slot
        self.t_last = None
        self.arrived = self.forwarded = self.stale = self.missed = self.skipped = self.miss_run = 0
        self.ages = []  # ms each released frame spent in the slot
        self.pass_ages = collections.deque(maxlen=64)  # same, consumed in FIFO order at the end of the pipeline

    def attach(self):
        self.q.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._on_arrive, None)
        self.q.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_gate, None)

    def _forget(self, pts):
        with self.tl:
            return self.t_arrive.pop(pts, None) if pts is not None else None

    def _on_arrive(self, pad, info, _):
        b, now = info.get_buffer(), time.perf_counter_ns()
        self.arrived += 1
        self.t_last = now
        if b is not None and b.pts != Gst.CLOCK_TIME_NONE:
            with self.tl:
                self.t_arrive[b.pts] = now
                while len(self.t_arrive) > 8:
                    self.t_arrive.popitem(last=False)
        return Gst.PadProbeReturn.OK

    def _on_gate(self, pad, info, _):
        b = info.get_buffer()
        pts = b.pts if b is not None else None
        with self.cv:
            self.holding = True
            while not self.armed:
                if self.stop.is_set():
                    self.holding = False
                    return Gst.PadProbeReturn.DROP  # shutting down: release the streaming thread
                self.cv.wait(0.05)
            if self.q.get_property("current-level-buffers") > 0:  # a newer frame is waiting: this one is stale
                self.stale += 1
                self._forget(pts)
                self.holding = False
                return Gst.PadProbeReturn.DROP  # stays armed; the queue hands us the next (newer) frame
            t0 = self._forget(pts)
            age = (time.perf_counter_ns() - t0) / 1e6 if t0 else NAN
            self.ages.append(age)
            self.pass_ages.append(age)
            self.armed, self.passed, self.holding = False, True, False
            self.forwarded += 1
            self.cv.notify_all()
        return Gst.PadProbeReturn.OK

    # ---- scheduler side ------------------------------------------------------------
    def has_fresh(self, now_ns, max_age_ms):
        """A recent frame is waiting in the slot (the camera is alive)."""
        t = self.t_last
        if t is None or (now_ns - t) / 1e6 > max_age_ms:
            return False
        return self.holding or self.q.get_property("current-level-buffers") > 0

    def arm(self):
        with self.cv:
            self.armed, self.passed = True, False
            self.cv.notify_all()

    def wait_passed(self, timeout):
        with self.cv:
            self.cv.wait_for(lambda: self.passed or self.stop.is_set(), max(0.0, timeout))
            return self.passed

    def disarm(self):
        """Close the gate again. Returns True if a frame was released in the meantime."""
        with self.cv:
            self.armed = False
            return self.passed

    def pop_age(self):
        try:
            return self.pass_ages.popleft()
        except IndexError:
            return NAN


class Scheduler(threading.Thread):
    """Per turn: GATHER fresh frames of one group -> ARM their gates -> wait for the TOKEN (batch left the stage)."""

    def __init__(self, a, groups, gates, stop_ev):
        super().__init__(daemon=True, name="rr-scheduler")
        self.a, self.groups, self.gates, self.stop_ev = a, groups, gates, stop_ev
        self.cv = threading.Condition()
        self.pending = set()  # cameras of the current turn whose frame has not left the token stage yet
        self.turns, self.last_arm = [], {}
        self.token_timeouts = self.skipped_turns = 0

    def release(self, stage, srcs):
        """Called by the probes when a batch leaves 'm1' or the last stage ('final')."""
        if stage == self.a.token_at:
            with self.cv:
                self.pending.difference_update(srcs)
                if not self.pending:
                    self.cv.notify_all()

    def wake(self):
        with self.cv:
            self.cv.notify_all()

    def _gather(self, cands):
        """Wait up to --gather-ms until every candidate holds a fresh frame. Returns the ready ones."""
        t_end = time.perf_counter() + self.a.gather_ms / 1000.0
        while True:
            now_ns = time.perf_counter_ns()
            ready = [c for c in cands if self.gates[c].has_fresh(now_ns, self.a.max_frame_age_ms)]
            if len(ready) == len(cands) or time.perf_counter() >= t_end or self.stop_ev.is_set():
                return ready
            time.sleep(0.0005)

    def _wait_first_frames(self):
        t0 = time.perf_counter()
        while not self.stop_ev.is_set() and time.perf_counter() - t0 < FIRST_FRAME_S:
            if all(g.arrived > 0 for g in self.gates):
                return
            time.sleep(0.05)

    def run(self):
        a = self.a
        self._wait_first_frames()
        turn = 0
        while not self.stop_ev.is_set():
            g = turn % len(self.groups)
            t_begin = time.perf_counter_ns()

            cands = []  # frame ready, or a live camera whose next frame is about to arrive
            for c in self.groups[g]:
                gate = self.gates[c]
                if gate.has_fresh(t_begin, a.max_frame_age_ms) or gate.miss_run < a.offline_after:
                    cands.append(c)
                else:
                    gate.skipped += 1  # offline camera: don't wait for it every turn

            armed = self._gather(cands) if cands else []  # 1) GATHER
            for c in cands:
                if c not in armed:
                    self.gates[c].miss_run += 1
                    self.gates[c].missed += 1
            if not armed:
                self.skipped_turns += 1
                turn += 1
                self.stop_ev.wait(0.002)
                continue

            with self.cv:  # 2) ARM only the ready cameras (pending is set BEFORE arming: a fast release can't be missed)
                self.pending = set(armed)
            t_arm = time.perf_counter_ns()
            for c in armed:
                self.gates[c].arm()
            deadline = time.perf_counter() + a.turn_timeout_ms / 1000.0
            passed = []
            for c in armed:
                gate = self.gates[c]
                ok = gate.wait_passed(deadline - time.perf_counter()) or gate.disarm()
                if ok:
                    gate.miss_run = 0
                    passed.append(c)
                else:
                    gate.miss_run += 1
                    gate.missed += 1
                    with self.cv:
                        self.pending.discard(c)
            t_gates = time.perf_counter_ns()

            with self.cv:  # 3) TOKEN
                if not self.cv.wait_for(lambda: not self.pending or self.stop_ev.is_set(), a.token_timeout_ms / 1000.0):
                    self.token_timeouts += 1
                    self.pending.clear()
            t_tok = time.perf_counter_ns()
            prev = self.last_arm.get(g)
            self.last_arm[g] = t_arm
            self.turns.append({"turn": turn, "group": g, "cands": len(cands), "armed": len(armed),
                               "passed": len(passed), "wait_frames_ms": (t_gates - t_begin) / 1e6,
                               "service_ms": (t_tok - t_gates) / 1e6, "turn_ms": (t_tok - t_arm) / 1e6,
                               "cycle_ms": (t_arm - prev) / 1e6 if prev else NAN})
            turn += 1
