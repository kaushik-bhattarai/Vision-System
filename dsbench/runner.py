"""Runs the benchmark: builds the pipeline, records until Ctrl+C (or a stall / error), prints the end-of-run summary."""
from __future__ import annotations

import signal
import sys
import threading
import time

from . import gstutil as gu
from . import report
from .config import mask_uri
from .gstutil import GLib, Gst
from .hardware import HwSampler
from .metrics import Metrics
from .osd import attach_annotator
from .pipeline import attach_probes, build_pipeline
from .router import Router
from .scheduler import Scheduler


def run_once(a):
    gu.require_deepstream()
    cams, groups = a.names, a.group_idx
    a.run_name = time.strftime("%Y%m%d_%H%M%S")
    print(f"[bench] mux {a.mux_width}x{a.mux_height}   detector {a.det_width}x{a.det_height} batch {a.det_batch} "
        f"{a.det_precision}   tracker {a.tracker_width}x{a.tracker_height}   min crop {a.min_crop_w}x{a.min_crop_h}px")
    for name, uri in zip(cams, a.uris):
        print(f"  {name:<10} {mask_uri(uri)}")
    print("[bench] round-robin order: " + "  ->  ".join(
        f"g{g}: " + "+".join(cams[i] for i in idxs) for g, idxs in enumerate(groups)) + "  ->  g0 ...")

    mets = [Metrics([cams[i] for i in idxs], g, a.warmup, a.min_crop_w, a.min_crop_h) for g, idxs in enumerate(groups)]
    stop_ev = threading.Event()
    pipe, gates, stages = build_pipeline(a, cams, stop_ev)
    for gate in gates:
        gate.attach()
    sched = Scheduler(a, groups, gates, stop_ev)
    router = Router(groups, mets, gates, sched)
    attach_probes(stages, router)
    attach_annotator(stages[-1][1], cams, a)

    loop, hw = GLib.MainLoop(), HwSampler(a.gpu_id)
    hw.start()
    state = {"stop_reason": "", "last_n": 0, "stall": 0, "rec": False, "rec_t0": None, "arrived0": None}

    def quit_(reason):
        state["stop_reason"] = state["stop_reason"] or reason
        loop.quit()
        return False

    def on_bus(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            quit_("EOS")
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {msg.src.get_name()}: {err.message}\n        {dbg}")
            quit_("error")
        elif msg.type == Gst.MessageType.WARNING:
            w, _d = msg.parse_warning()
            print(f"[warn] {msg.src.get_name()}: {w.message}")
        return True

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)

    def tick():  # every second: stall watchdog, start of the recording window
        n = sum(m.batches_seen for m in mets)
        state["stall"] = state["stall"] + 1 if n == state["last_n"] else 0
        state["last_n"] = n
        limit = a.stall_s if n > 0 else a.startup_s  # before the first batch the TensorRT engine may still be building
        if state["stall"] >= limit:
            print(f"[bench] no batches for {limit}s - cameras unreachable / engine build failed / pipeline stuck")
            return quit_("stall")
        firsts = [m.t_first for m in mets if m.t_first]
        if firsts and not state["rec"]:  # warm-up done: hardware, decoded-fps and slot-age stats start here
            state["rec"] = True
            state["rec_t0"] = min(firsts) / 1e9  # perf_counter seconds, same clock as the hardware samples
            state["arrived0"] = (time.perf_counter(), sum(g.arrived for g in gates))
            for g in gates:
                g.ages.clear()
            print("[bench] warm-up done, recording - stop with Ctrl+C")
        return True

    GLib.timeout_add_seconds(1, tick)
    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, lambda: quit_("interrupted"))
    print("[bench] starting pipeline (the first run builds the TensorRT engine(s) - can take several minutes)")
    print("[bench] stop with Ctrl+C or by closing the window; the summary is printed at the end")
    pipe.set_state(Gst.State.PLAYING)
    sched.start()
    try:
        loop.run()
    finally:
        stop_ev.set()  # releases every gate before the pipeline is torn down
        sched.wake()
        if sched.is_alive():
            sched.join(3)
        t_end, arrived1 = time.perf_counter(), sum(g.arrived for g in gates)
        hw.stop_ev.set()
        pipe.set_state(Gst.State.NULL)

    ok = [m for m in mets if len(m.recs) >= 5]
    if not ok:
        sys.exit(f"only {[len(m.recs) for m in mets]} batches recorded after warm-up "
                 f"({state['stop_reason']}); nothing to report")
    decoded_fps = None
    if state["arrived0"]:
        t0, n0 = state["arrived0"]
        decoded_fps = (arrived1 - n0) / (t_end - t0) if t_end > t0 else None

    Ss = [report.summarize(m) for m in ok]
    A = report.aggregate(Ss, ok, hw.summary(since=state["rec_t0"] or 0.0), decoded_fps)
    A["scheduler"] = report.sched_summary(a, sched, router, gates, cams, a.warmup * len(groups))
    A["stop_reason"] = state["stop_reason"]
    report.print_summary(A)
    report.write_outputs(a, ok, A)
    return A