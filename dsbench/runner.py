"""Runs one benchmark configuration (run_once) or the three-way comparison (compare)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

from . import gstutil as gu
from . import report
from .audit import TrackAudit
from .config import STAGE_TAGS, mask_uri
from .gstutil import GLib, Gst
from .hardware import DecoderTimer, HwSampler
from .metrics import Metrics
from .osd import attach_annotator
from .pipeline import attach_probes, build_pipeline
from .report import nmean
from .router import Router
from .scheduler import Scheduler


def make_tag(a, stages):
    # the stream is part of the tag so main-stream and substream results never overwrite each other
    return (a.tag + "_" if a.tag else "") + f"{a.stream}_" + STAGE_TAGS[stages]


def _warn_on_batch_mismatch(a, biggest):
    """The engine batch must equal the largest group, otherwise TensorRT rebuilds the engine."""
    try:
        for line in open(a.pgie):
            if line.strip().startswith("batch-size=") and int(line.split("=")[1]) != biggest:
                print(f"  [warn] {a.pgie}: {line.strip()} but the largest group has {biggest} cameras "
                      f"(edit batch-size and the engine file name, or the engine will be rebuilt)")
    except (OSError, ValueError):
        pass


def run_once(a):
    gu.require_deepstream()
    tag, cams, groups = make_tag(a, a.stages), a.names, a.group_idx
    print(f"[{tag}] stream: {a.stream}   mux {a.mux_width}x{a.mux_height}   tracker {a.tracker_width}x{a.tracker_height}"
          f"   min crop {a.min_crop}px   OSD font {a.osd_font}")
    for name, uri in zip(cams, a.uris):
        print(f"  {name:<6} {mask_uri(uri)}")
    print(f"[{tag}] round-robin order: " + "  ->  ".join(
        f"g{g}: " + "+".join(cams[i] for i in idxs) for g, idxs in enumerate(groups)) + "  ->  g0 ...")
    _warn_on_batch_mismatch(a, max(len(g) for g in groups))

    mets = [Metrics([cams[i] for i in idxs], a.stages, g, a.min_hits, a.warmup, a.min_crop)
            for g, idxs in enumerate(groups)]
    dec, stop_ev = DecoderTimer(), threading.Event()
    pipe, gates, stages = build_pipeline(a, cams, dec, stop_ev)
    for gate in gates:
        gate.attach()
    sched = Scheduler(a, groups, gates, stop_ev)
    router = Router(groups, mets, gates, sched)
    attach_probes(stages, router)
    audit = TrackAudit(cams, a, enabled=not a.no_audit and a.stages != "m1")
    if a.stages == "m1" and not a.no_audit:
        print("  [note] --stages m1 has no tracker: track audit disabled")
    if audit.enabled or a.display:
        attach_annotator(stages[-1][1], audit, cams, a)

    loop, hw = GLib.MainLoop(), HwSampler(a.gpu_id)
    hw.start()
    state = {"stop_reason": "", "last_n": 0, "stall": 0, "secs": 0, "snap": False}

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

    def tick():  # every second: decoder scan, stall watchdog, duration check
        state["secs"] += 1
        if state["secs"] in (3, 10, 30):
            dec.scan(pipe)
        n = sum(m.batches_seen for m in mets)
        state["stall"] = state["stall"] + 1 if n == state["last_n"] else 0
        state["last_n"] = n
        limit = a.stall_s if n > 0 else a.startup_s  # before the first batch the TensorRT engine may still be building
        if state["stall"] >= limit:
            print(f"[{tag}] no batches for {limit}s - cameras unreachable / engine build failed / pipeline stuck")
            return quit_("stall")
        firsts = [m.t_first for m in mets if m.t_first]
        if firsts and not state["snap"]:
            dec.snap0, state["snap"] = dec.snapshot(), True  # decoded-fps window starts when recording starts
        if firsts and (time.perf_counter_ns() - min(firsts)) / 1e9 >= a.duration > 0:
            return quit_("duration")
        return True

    def progress():
        s = hw.last()
        hwtxt = f"GPU {report.fmt(s[0], '{:.0f}')}%  NVDEC {report.fmt(s[3], '{:.0f}')}%"
        for m in mets:
            if not m.recs:
                print(f"[{tag} g{m.group}] warming up... batches seen {m.batches_seen}/{a.warmup}")
                continue
            r, w = m.recs[-1], m.recs[-20:]
            iv = nmean([x["interval_ms"] for x in w])
            print(f"[g{m.group} {'+'.join(m.cams)}] t={r['t_rel_s']:5.0f}s  revisit {iv:6.1f} ms ({1000 / iv:5.1f} fps/cam)  "
                  f"proc {nmean([x['proc_ms'] for x in w]):5.1f}  M1 {nmean([x['m1_ms'] for x in w]):5.1f}  "
                  f"trk {nmean([x['trk_ms'] for x in w]):5.1f}  M2 {nmean([x['m2_ms'] for x in w]):5.1f}  "
                  f"slot age {nmean([x.get('slot_age_ms') for x in w]):5.1f}  "
                  f"e2e {nmean([x.get('e2e_true_ms') for x in w]):6.1f}  "
                  f"live P {r['live_persons_total']} V {r['live_vehicles_total']}")
        w = sched.turns[-10:]
        if w:
            print(f"[sched] gather {nmean([x['wait_frames_ms'] for x in w]):5.1f} ms  service "
                  f"{nmean([x['service_ms'] for x in w]):5.1f} ms  skipped turns {sched.skipped_turns}  token timeouts "
                  f"{sched.token_timeouts}  overlapped {router.overlapped}/{router.m1_batches}  "
                  f"mixed {router.mixed}  order {router.order_bad}  {hwtxt}")
        if audit.enabled:
            print("[track] " + " | ".join(audit.short_line(c) for c in range(len(cams))))
        return True

    GLib.timeout_add_seconds(1, tick)
    GLib.timeout_add_seconds(int(a.report_every), progress)
    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, lambda: quit_("interrupted"))
    print(f"[{tag}] starting pipeline (first run builds the TensorRT engine(s) - can take several minutes)")
    print(f"[{tag}] stop with Ctrl+C or by closing the window; the full report is printed at the end")
    pipe.set_state(Gst.State.PLAYING)
    sched.start()
    try:
        loop.run()
    finally:
        stop_ev.set()  # releases every gate before the pipeline is torn down
        sched.wake()
        if sched.is_alive():
            sched.join(3)
        dec.snap1 = dec.snapshot()
        hw.stop_ev.set()
        pipe.set_state(Gst.State.NULL)

    ok = [m for m in mets if len(m.recs) >= 5]
    if not ok:
        sys.exit(f"only {[len(m.recs) for m in mets]} batches recorded after warmup "
                 f"({state['stop_reason']}); nothing to report")
    hwsum = hw.summary()
    Ss = [report.summarize(m, a, hwsum, f"{tag}_g{m.group}", m.recs[-1]["t_rel_s"], dec) for m in ok]
    for S in Ss:
        report.print_group(S)
    A = report.aggregate(Ss, ok, a, hwsum, tag, dec)
    A["scheduler"] = report.sched_summary(a, sched, router, gates, cams, a.warmup * len(groups))
    A["stream"] = {"which": a.stream, "mux": [a.mux_width, a.mux_height],
                   "tracker": [a.tracker_width, a.tracker_height], "min_crop": a.min_crop}
    A["track_audit"] = audit.summary() if audit.enabled else []
    report.print_overall(A)
    report.print_sched(A["scheduler"])
    if A["track_audit"]:
        report.print_audit(A["track_audit"], a)
    report.write_outputs(a, tag, ok, Ss, A, sched, audit)
    return A


def compare(a):
    """Re-run this script once per stage set (fresh process each time, GStreamer can't be re-initialised cleanly)."""
    argv = [x for x in sys.argv[1:] if x != "--compare"]
    argv += ["--duration", str(a.duration), "--display" if a.display else "--no-display"]
    entry = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.py")
    results = []
    for stages in STAGE_TAGS:
        tag = make_tag(a, stages)
        print(f"\n######## {tag} ########")
        subprocess.run([sys.executable, entry, *argv, "--stages", stages], check=False)
        path = os.path.join(a.out_dir, f"summary_{tag}.json")
        if os.path.exists(path):
            results.append(json.load(open(path)))
        else:
            print(f"(no summary for {tag} - that run failed; see output above)")
    if len(results) > 1:
        report.print_compare(results)
        with open(os.path.join(a.out_dir, f"compare_{a.stream}.json"), "w") as f:
            json.dump(results, f, indent=2)
