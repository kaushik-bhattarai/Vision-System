"""Turns the raw measurements into summaries, the short end-of-run console report and the output files."""
from __future__ import annotations

import collections
import csv
import json
import os
import shutil

import numpy as np

from .config import NAN
from .metrics import describe

METRICS = [
    ("Slot age when picked (decoder out -> gate)", "slot_age_ms"),
    ("M1  detector (batch)", "m1_ms"),
    ("Tracker", "trk_ms"),
    ("M2  second model (crops of the batch)", "m2_ms"),
    ("Queue wait between elements", "queue_ms"),
    ("PROCESSING per batch (M1 in -> M2 out)", "proc_ms"),
    ("END-TO-END mux entry -> result", "e2e_ms"),
    ("END-TO-END incl. slot age (decoder out -> result)", "e2e_true_ms"),
    ("REVISIT INTERVAL of a group (result -> result)", "interval_ms"),
]
STAGE_KEYS = (("M1 detector", "m1_ms"), ("Tracker", "trk_ms"), ("M2 secondary", "m2_ms"))
TICK_COLS = ["group", "tick", "t_rel_s", "batch_frames", "m1_ms", "trk_ms", "m2_ms", "queue_ms", "proc_ms", "e2e_ms",
             "slot_age_ms", "e2e_true_ms", "interval_ms", "eligible_objs"]


def fmt(v, f="{:.1f}", none="n/a"):
    return none if v is None else f.format(v)


# Summaries
def _metrics(recs):
    return {k: s for _, k in METRICS if (s := describe([r.get(k) for r in recs]))}


def stage_rows(S):
    """Per-element time per batch / per frame and busy % (fraction of wall time the element is working)."""
    rows = []
    for label, k in STAGE_KEYS:
        s = S["metrics"].get(k)
        if s:
            rows.append({"stage": label, "key": k, "mean_ms": s["mean"], "p95_ms": s["p95"], "max_ms": s["max"],
                         "ms_per_frame": s["mean"] / max(S["avg_batch_frames"], 1e-9),
                         "busy_pct": 100.0 * s["mean"] * S["batches_per_s"] / 1000.0})
    return rows


def summarize(m):
    recs = m.recs
    elapsed = recs[-1]["t_rel_s"]
    S = {"group": m.group, "cams": m.cams, "ticks": len(recs), "elapsed_s": elapsed, "metrics": _metrics(recs)}
    S["batches_per_s"] = 1000.0 / S["metrics"]["interval_ms"]["mean"]
    S["fps_per_cam"] = [c / elapsed for c in m.frames_rec]
    S["batch_fill"] = {str(k): v for k, v in sorted(m.batch_fill.items())}
    S["full_batch_pct"] = 100.0 * m.batch_fill.get(m.n, 0) / max(len(recs), 1)
    S["avg_batch_frames"] = float(np.mean([r["batch_frames"] for r in recs]))
    S["avg_eligible_objs"] = float(np.mean([r["eligible_objs"] for r in recs]))
    S["combos"] = {"+".join(m.cams[i] for i in k): v for k, v in m.combo.most_common()}
    S["stage"] = stage_rows(S)
    return S


def aggregate(Ss, mets, hw, decoded_fps):
    """Combine the camera groups into one record."""
    recs = [r for m in mets for r in m.recs]
    A = {"groups": Ss, "ticks": len(recs), "elapsed_s": max(S["elapsed_s"] for S in Ss), "hardware": hw,
         "metrics": _metrics(recs)}
    A["batches_per_s"] = sum(S["batches_per_s"] for S in Ss)
    A["full_batch_pct"] = sum(S["full_batch_pct"] * S["ticks"] for S in Ss) / max(sum(S["ticks"] for S in Ss), 1)
    A["avg_eligible_objs"] = float(np.mean([r["eligible_objs"] for r in recs]))
    A["fps_total"] = sum(sum(S["fps_per_cam"]) for S in Ss)
    A["decoded_fps_total"] = decoded_fps
    gpu = (hw.get("gpu_util_pct") or {}).get("mean")
    nvd = (hw.get("nvdec_util_pct") or {}).get("mean")
    A["gpu_ms_per_frame"] = 10.0 * gpu / A["fps_total"] if gpu is not None and A["fps_total"] else None  # util% x10 ms / fps
    A["nvdec_ms_per_frame"] = 10.0 * nvd / decoded_fps if nvd is not None and decoded_fps else None
    busy = collections.OrderedDict()
    for S in Ss:
        for r in S["stage"]:
            busy[r["stage"]] = busy.get(r["stage"], 0.0) + r["busy_pct"]
    A["stage_overall"] = [{"stage": k, "busy_pct_sum": v} for k, v in busy.items()]
    return A


def sched_summary(a, sched, router, gates, cams, warm_turns):
    turns = [t for t in sched.turns if t["turn"] >= warm_turns]
    sd = {"groups": len(a.group_idx), "group_sizes": [len(g) for g in a.group_idx], "token_at": a.token_at,
          "turns": len(turns), "skipped_turns": sched.skipped_turns, "token_timeouts": sched.token_timeouts,
          "gather_ms": a.gather_ms,
          "partial_turn_pct": 100.0 * sum(t["armed"] < t["cands"] for t in turns) / max(len(turns), 1)}
    for k in ("wait_frames_ms", "service_ms", "turn_ms"):
        sd[k] = describe([t.get(k) for t in turns])
    sd["m1_gap_ms"] = describe(router.m1_gaps[warm_turns:])
    sd["overlap_pct"] = 100.0 * router.overlapped / max(router.m1_batches, 1)
    sd["max_batches_after_m1"] = router.inflight_max
    sd["integrity"] = {"batches_mixing_groups": router.mixed, "same_camera_twice_in_batch": router.dup_src,
                       "batches_out_of_order": router.order_bad, "frame_numbers_not_increasing": router.fnum_bad}
    sd["cameras"] = [{"cam": cams[c], "decoded": gt.arrived, "analysed": gt.forwarded,
                      "use_pct": 100.0 * gt.forwarded / max(gt.arrived, 1), "stale_drops": gt.stale,
                      "missed": gt.missed, "skipped_offline": gt.skipped,
                      "slot_age_ms": describe(list(gt.ages))} for c, gt in enumerate(gates)]
    return sd


# Console output
def _hv(hw, k):
    v = hw.get(k)
    return f"mean {v['mean']:.1f} / max {v['max']:.1f}" if v else "n/a"


def print_summary(A):
    sd, hw, bar = A["scheduler"], A["hardware"], "=" * 92
    print(f"\n{bar}\nBENCHMARK SUMMARY - {sum(sd['group_sizes'])} cameras in {sd['groups']} groups {sd['group_sizes']}, "
          f"{A['ticks']} batches, {A['elapsed_s']:.0f} s recorded\n{bar}")
    print(f"GPU util %            : {_hv(hw, 'gpu_util_pct')}   (NVML, whole GPU, recording window only)")
    print(f"NVDEC util %          : {_hv(hw, 'nvdec_util_pct')}")
    print(f"GPU memory MB         : {_hv(hw, 'gpu_mem_mb')}")
    print(f"CPU util %            : {_hv(hw, 'cpu_pct')}")
    print(f"Frames analysed / s   : {A['fps_total']:.1f} (all cameras)   decoded / s: {fmt(A['decoded_fps_total'])}")
    print(f"GPU time per frame    : {fmt(A['gpu_ms_per_frame'], '{:.2f}')} ms     "
          f"NVDEC time per frame: {fmt(A['nvdec_ms_per_frame'], '{:.2f}')} ms")
    print(f"Full batches          : {A['full_batch_pct']:.1f}%   M2 load: {A['avg_eligible_objs']:.1f} eligible crops/batch\n")

    print(f"{'metric (ms)':<52}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}\n" + "-" * 92)
    for label, k in METRICS:
        s = A["metrics"].get(k)
        if s:
            print(f"{label:<52}{s['mean']:>8.1f}{s['p50']:>8.1f}{s['p95']:>8.1f}{s['p99']:>8.1f}{s['max']:>8.1f}")
    print("-" * 92)

    busy = {r["stage"]: r["busy_pct_sum"] for r in A["stage_overall"]}
    print(f"\n  {'stage':<16}{'ms/batch':>10}{'ms/frame':>10}{'busy %':>9}")
    for label, k in STAGE_KEYS:
        rows = [r for S in A["groups"] for r in S["stage"] if r["key"] == k]
        if rows:
            print(f"  {label:<16}{np.mean([r['mean_ms'] for r in rows]):>10.2f}"
                  f"{np.mean([r['ms_per_frame'] for r in rows]):>10.2f}{busy.get(label, 0.0):>9.1f}")

    gap = sd["m1_gap_ms"]
    print(f"\nScheduler : M1 idle gap mean {fmt(gap and gap['mean'])} ms, overlap {sd['overlap_pct']:.1f}% of batches, "
          f"partial turns {sd['partial_turn_pct']:.1f}%, skipped turns {sd['skipped_turns']}, "
          f"token timeouts {sd['token_timeouts']}")
    use = [c["use_pct"] for c in sd["cameras"]]
    worst = min(sd["cameras"], key=lambda c: c["use_pct"])
    print(f"Slots     : frames used / decoded per camera: mean {np.mean(use):.1f}%, lowest {worst['use_pct']:.1f}% "
          f"({worst['cam']})   (per-camera table in summary.json)")
    it = sd["integrity"]
    print(f"Integrity : mixed-group batches {it['batches_mixing_groups']} | same camera twice {it['same_camera_twice_in_batch']} "
          f"| out of order {it['batches_out_of_order']} | frame numbers not increasing {it['frame_numbers_not_increasing']}"
          f"   -> {'OK' if not sum(it.values()) else 'PROBLEM'}")


# Files
def write_outputs(a, mets, A):
    run_dir = os.path.join(a.out_dir, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "ticks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TICK_COLS, extrasaction="ignore")
        w.writeheader()
        for m in mets:
            w.writerows(m.recs)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(A, f, indent=2, default=str)
    shutil.copy(a.config_path, os.path.join(run_dir, "config_used.yaml"))
    print(f"\nticks.csv, summary.json, config_used.yaml -> {os.path.abspath(run_dir)}")