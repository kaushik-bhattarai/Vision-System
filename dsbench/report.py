"""Turns the raw measurements into summaries, console tables and output files."""
from __future__ import annotations

import collections
import csv
import json
import os

import numpy as np

from .config import NAN
from .metrics import describe

METRICS = [
    ("Slot age when picked (decoder out -> gate)", "slot_age_ms"),
    ("Frame age when M1 starts (mux entry -> M1)", "wait_before_m1_ms"),
    ("Arrival skew of frames inside a batch", "skew_ms"),
    ("M1  detector (batch)", "m1_ms"),
    ("Tracker", "trk_ms"),
    ("M2  second model (crops of the batch)", "m2_ms"),
    ("Queue wait between elements", "queue_ms"),
    ("PROCESSING per batch (M1 in -> last out)", "proc_ms"),
    ("END-TO-END mux entry -> result", "e2e_ms"),
    ("END-TO-END incl. slot age (decoder out -> result)", "e2e_true_ms"),
    ("REVISIT INTERVAL of this group (result -> result)", "interval_ms"),
]
ALWAYS = ("proc_ms", "m1_ms", "interval_ms")
STAGE_KEYS = (("M1 detector", "m1_ms"), ("Tracker", "trk_ms"), ("M2 secondary", "m2_ms"))


def fmt(v, f="{:.1f}", none="n/a"):
    return none if v is None else f.format(v)


def nmean(vals):
    v = [x for x in vals if x is not None and x == x]
    return float(np.mean(v)) if v else NAN


# ======================================================================================
# Summaries
# ======================================================================================
def stage_rows(S):
    """Per-element time per batch / per frame and busy % (fraction of wall time the element is working)."""
    rows = []
    for label, k in STAGE_KEYS:
        s = S["metrics"].get(k)
        if s:
            rows.append({"stage": label, "key": k, "mean_ms": s["mean"], "p95_ms": s["p95"], "max_ms": s["max"],
                         "ms_per_frame": s["mean"] / max(S["avg_batch_frames"], 1e-9),
                         "occupancy_pct": 100.0 * s["mean"] * S["batches_per_s"] / 1000.0})
    return rows


def summarize(m, a, hw, tag, elapsed, dec):
    recs = m.recs
    S = {"tag": tag, "group": m.group, "stages": a.stages, "batch": m.n, "cams": m.cams, "ticks": len(recs),
         "elapsed_s": elapsed, "min_crop": a.min_crop, "hardware": hw}
    S["metrics"] = {k: s for _, k in METRICS
                    if (s := describe([r.get(k) for r in recs])) and (s["max"] > 0 or k in ALWAYS)}
    S["batches_per_s"] = 1000.0 / S["metrics"]["interval_ms"]["mean"]
    S["fps_per_cam"] = [c / elapsed for c in m.frames_rec]
    S["batch_fill"] = {str(k): v for k, v in sorted(m.batch_fill.items())}
    S["full_batch_pct"] = 100.0 * m.batch_fill.get(m.n, 0) / max(len(recs), 1)
    for key, col in (("avg_batch_frames", "batch_frames"), ("avg_eligible_objs", "m2_eligible_objs"),
                     ("avg_live_persons", "live_persons_total"), ("avg_live_vehicles", "live_vehicles_total")):
        S[key] = float(np.mean([r[col] for r in recs]))
    S["combos"] = {"+".join(m.cams[i] for i in k): v for k, v in m.combo.most_common()}
    S["cam_gaps"] = []
    for c in range(m.n):
        g = np.asarray(m.gaps[c], float)
        if g.size:
            med = float(np.median(g))
            S["cam_gaps"].append({"cam": m.cams[c], "median_ms": med, "p99_ms": float(np.percentile(g, 99)),
                                  "max_ms": float(g.max()), "late": int((g > 1.5 * med).sum()), "frames": int(g.size + 1)})
    S["counts"] = m.count_summary()
    S["m2_top_labels"] = m.m2_labels.most_common(5)
    S["decoder"] = dec.stats(m.group, m.cams)
    S["stage"] = stage_rows(S)
    return S


def aggregate(Ss, mets, a, hw, tag, dec):
    """Combine the camera groups into one record (also what --compare reads)."""
    recs = [r for m in mets for r in m.recs]
    A = {"tag": tag, "stages": a.stages, "groups": [{"group": S["group"], "cams": S["cams"]} for S in Ss],
         "ticks": len(recs), "hardware": hw}
    A["metrics"] = {k: s for _, k in METRICS
                    if (s := describe([r.get(k) for r in recs])) and (s["max"] > 0 or k in ALWAYS)}
    A["batches_per_s"] = sum(S["batches_per_s"] for S in Ss)
    A["full_batch_pct"] = sum(S["full_batch_pct"] * S["ticks"] for S in Ss) / max(sum(S["ticks"] for S in Ss), 1)
    A["fps_total"] = sum(sum(S["fps_per_cam"]) for S in Ss)
    gpu, nvd = hw.get("gpu_util_pct(mean,max)"), hw.get("nvdec_util_pct(mean,max)")
    A["gpu_ms_per_frame"] = 10.0 * gpu[0] / A["fps_total"] if gpu and A["fps_total"] else None  # util% x10 ms / fps
    A["decoded_fps_total"] = dec.total_fps()
    A["nvdec_ms_per_frame"] = 10.0 * nvd[0] / A["decoded_fps_total"] if nvd and A["decoded_fps_total"] else None
    occ = collections.OrderedDict()
    for S in Ss:
        for r in S["stage"]:
            occ[r["stage"]] = occ.get(r["stage"], 0.0) + r["occupancy_pct"]
    tot = sum(occ.values()) or 1.0
    A["stage_overall"] = [{"stage": k, "busy_pct_sum": v, "share_pct": 100.0 * v / tot,
                           "est_gpu_pct": gpu[0] * v / tot if gpu else None} for k, v in occ.items()]
    return A


def sched_summary(a, sched, router, gates, cams, warm_turns):
    turns = [t for t in sched.turns if t["turn"] >= warm_turns]
    sd = {"groups": len(a.group_idx), "group_sizes": [len(g) for g in a.group_idx], "token_at": a.token_at,
          "turns": len(turns), "skipped_turns": sched.skipped_turns, "token_timeouts": sched.token_timeouts,
          "gather_ms": a.gather_ms,
          "partial_turn_pct": 100.0 * sum(t["armed"] < t["cands"] for t in turns) / max(len(turns), 1)}
    for k in ("wait_frames_ms", "service_ms", "turn_ms", "cycle_ms"):
        sd[k] = describe([t.get(k) for t in turns])
    sd["m1_gap_ms"] = describe(router.m1_gaps[warm_turns:])
    sd["overlap_pct"] = 100.0 * router.overlapped / max(router.m1_batches, 1)
    sd["max_batches_after_m1"] = router.inflight_max
    sd["integrity"] = {"batches_mixing_groups": router.mixed, "same_camera_twice_in_batch": router.dup_src,
                       "batches_out_of_order": router.order_bad, "frame_numbers_not_increasing": router.fnum_bad}
    sd["cameras"] = [{"cam": cams[c], "decoded": gt.arrived, "analysed": gt.forwarded,
                      "use_pct": 100.0 * gt.forwarded / max(gt.arrived, 1), "stale_drops": gt.stale,
                      "missed": gt.missed, "skipped_offline": gt.skipped,
                      "slot_age_ms": describe(gt.ages[a.warmup:] or gt.ages)} for c, gt in enumerate(gates)]
    return sd


# ======================================================================================
# Console output
# ======================================================================================
def _banner(title, width=92):
    bar = "=" * width
    print(f"\n{bar}\n{title}\n{bar}")


def _metric_table(rows, label_w=52):
    print(f"{'metric (ms)':<{label_w}}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}\n" + "-" * 92)
    for label, s in rows:
        if s:
            print(f"{label:<{label_w}}{s['mean']:>8.1f}{s['p50']:>8.1f}{s['p95']:>8.1f}{s['p99']:>8.1f}{s['max']:>8.1f}")
    print("-" * 92)


def print_group(S):
    _banner(f"GROUP {S['group']} [{S['tag']}]  stages={S['stages']}  cams {S['cams']}  "
            f"batches={S['ticks']}  wall={S['elapsed_s']:.1f}s")
    _metric_table([(label, S["metrics"].get(k)) for label, k in METRICS])
    iv, pr = S["metrics"]["interval_ms"], S["metrics"]["proc_ms"]
    print(f"Group visits/s       : {S['batches_per_s']:.2f}   FPS per camera analysed: "
          + ", ".join(f"{c}={f:.2f}" for c, f in zip(S["cams"], S["fps_per_cam"])))
    print(f"Revisit interval     : mean {iv['mean']:.1f} ms  p95 {iv['p95']:.1f} ms  max {iv['max']:.1f} ms")
    print(f"Batch processing     : mean {pr['mean']:.1f} ms  p95 {pr['p95']:.1f} ms")
    print(f"Batch fill           : {S['full_batch_pct']:.1f}% of batches had all {S['batch']} cameras   "
          f"(size histogram {S['batch_fill']})")
    print("Batch composition    : " + ", ".join(f"{k} x{v}" for k, v in list(S["combos"].items())[:6]))
    print("\nSTAGE BREAKDOWN - exact per-element time (probes); includes waiting for the shared GPU")
    print(f"  {'stage':<26}{'ms/batch':>9}{'p95':>8}{'max':>8}{'ms/frame':>10}{'busy %':>8}")
    for r in S["stage"]:
        print(f"  {r['stage']:<26}{r['mean_ms']:>9.2f}{r['p95_ms']:>8.2f}{r['max_ms']:>8.2f}"
              f"{r['ms_per_frame']:>10.2f}{r['occupancy_pct']:>8.1f}")
    if S["decoder"]:
        print("  NVDEC - time each frame spends inside the decoder (latency, not engine busy time):")
        for x in S["decoder"]:
            print(f"    {x['cam']:<6} mean {x['mean_ms']:.2f} ms  p95 {x['p95_ms']:.2f}  max {x['max_ms']:.2f}   "
                  f"{x['frames']} frames, decoded at {fmt(x['fps'])} fps")
    else:
        print("  NVDEC per-frame time: decoder element not found or PTS not preserved")
    if S["stages"] == "full":
        print(f"\nStage-2 load         : {S['avg_eligible_objs']:.1f} eligible crops/batch "
              f"(>={S['min_crop']}px persons+vehicles)")
        if S["m2_top_labels"]:
            print(f"Stage-2 sample labels: {S['m2_top_labels']}   (proves M2 output is attached)")
    for g in S["cam_gaps"]:
        print(f"  {g['cam']:<6} gap between analysed frames: median {g['median_ms']:.1f} ms  p99 {g['p99_ms']:.1f}  "
              f"max {g['max_ms']:.1f}   late (>1.5x median): {g['late']} of {g['frames']}")
    print("\nCOUNTS (persons / vehicles) - live = in frame now, unique = distinct tracker IDs seen >= min-hits frames")
    for c in S["counts"]["cameras"]:
        u = (f"unique P {c['unique_persons']} V {c['unique_vehicles']} {c['unique_vehicles_by_class']}"
             if c["unique_persons"] is not None else "unique n/a (no tracker)")
        veh, w4, w2 = c["live_4W"] + c["live_2W"], c["live_4W"], c["live_2W"]
        print(f"  {c['cam']:<8} live P {c['live_persons']:>3}  V {veh:>3} (4W {w4}, 2W {w2})   "
              f"peak P {c['peak_persons']} V {c['peak_vehicles']}   {u}")


def print_overall(A):
    _banner(f"OVERALL [{A['tag']}]  groups: " + "  |  ".join("+".join(g["cams"]) for g in A["groups"]), 96)
    hw = A["hardware"]

    def hv(k):
        v = hw.get(k)
        return f"mean {v[0]:.1f} / max {v[1]:.1f}" if v else "n/a"

    print(f"GPU compute+graphics util %   : {hv('gpu_util_pct(mean,max)')}   (NVML, whole GPU, all stages together)")
    print(f"NVDEC video decoder util %    : {hv('nvdec_util_pct(mean,max)')}   (NVML, separate engine)")
    print(f"GPU memory MB                 : {hv('gpu_mem_mb(mean,max)')}")
    print(f"CPU util %                    : {hv('cpu_pct(mean,max)')}")
    print(f"Frames analysed per second    : {A['fps_total']:.1f} (all cameras)   decoded per second: "
          f"{fmt(A['decoded_fps_total'])}")
    print(f"GPU time per analysed frame   : {fmt(A['gpu_ms_per_frame'], '{:.2f}')} ms   (GPU util x 10 / frames per second)")
    print(f"NVDEC time per decoded frame  : {fmt(A['nvdec_ms_per_frame'], '{:.2f}')} ms")
    print(f"\n  {'stage':<26}{'busy % (sum of groups)':>24}{'share of stage time':>21}{'est. GPU %':>12}")
    for r in A["stage_overall"]:
        print(f"  {r['stage']:<26}{r['busy_pct_sum']:>24.1f}{r['share_pct']:>20.1f}%{fmt(r['est_gpu_pct']):>12}")
    print("  busy % is exact per element; est. GPU % splits the whole-GPU figure by element time (an ESTIMATE).\n"
          "  Exact GPU cost per stage: run --compare (difference between m1 / track / full).")


def print_sched(sd):
    _banner(f"ROUND-ROBIN SCHEDULER  {sd['groups']} groups, sizes {sd['group_sizes']}, "
            f"token released when a batch leaves: {sd['token_at']}   gather window {sd['gather_ms']:.0f} ms")
    _metric_table([("Gather + gate release (waiting for frames to be ready)", sd["wait_frames_ms"]),
                   ("SERVICE time per turn (frames released -> token back)", sd["service_ms"]),
                   ("Turn time (arm -> token back)", sd["turn_ms"]),
                   ("CYCLE of a group (its arm -> its next arm); fps/camera = 1000/cycle", sd["cycle_ms"]),
                   ("M1 idle gap between consecutive batches", sd["m1_gap_ms"])])
    print(f"Turns recorded       : {sd['turns']}   turns skipped (no camera ready): {sd['skipped_turns']}   "
          f"token timeouts: {sd['token_timeouts']}")
    print(f"Partial turns        : {sd['partial_turn_pct']:.1f}% of turns left a camera out (no frame within the gather window)")
    print(f"Overlap              : {sd['overlap_pct']:.1f}% of batches entered M1 while the previous batch was still in "
          f"tracker/M2   (max batches past M1 at once: {sd['max_batches_after_m1']})")
    it = sd["integrity"]
    print(f"INTEGRITY (all must be 0): mixed-group batches {it['batches_mixing_groups']} | same camera twice in a batch "
          f"{it['same_camera_twice_in_batch']} | batches out of order {it['batches_out_of_order']} | frame numbers not "
          f"increasing {it['frame_numbers_not_increasing']}   -> {'OK' if not sum(it.values()) else 'PROBLEM'}")
    print("\nSLOTS - frames decoded vs analysed per camera, and how old the picked frame was")
    print(f"  {'cam':<8}{'decoded':>9}{'analysed':>10}{'used %':>8}{'stale':>7}{'missed':>8}{'offline':>9}"
          f"{'age mean':>10}{'p95':>8}{'max':>8}")
    for c in sd["cameras"]:
        s = c["slot_age_ms"] or {"mean": NAN, "p95": NAN, "max": NAN}
        print(f"  {c['cam']:<8}{c['decoded']:>9}{c['analysed']:>10}{c['use_pct']:>8.1f}{c['stale_drops']:>7}"
              f"{c['missed']:>8}{c['skipped_offline']:>9}{s['mean']:>10.1f}{s['p95']:>8.1f}{s['max']:>8.1f}")
    print("  stale = held frame discarded because a newer one was waiting; missed = no frame within the gather window;\n"
          "  offline = turns skipped without waiting. decoded - analysed = frames replaced in the slot on purpose.")


def print_audit(rows, a):
    _banner(f"TRACKER AUDIT  (min-hits {a.min_hits}, frag window {a.frag_window_s:.1f}s, "
            f"frag distance {a.frag_dist:.1f}x object size)", 120)
    print(f"{'cam':<6}{'frames':>8}{'IDs':>7}{'confirmed':>10}{'flicker':>8}{'short':>7}{'frag?':>7}{'frag%':>7}"
          f"{'dup':>6}{'reassoc':>9}{'clsflip':>8}{'life med s':>11}{'life p90 s':>11}{'new IDs/min':>12}")
    for r in rows:
        print(f"{r['cam']:<6}{r['frames']:>8}{r['ids_total']:>7}{r['ids_confirmed']:>10}{r['flicker_ids']:>8}"
              f"{r['short_confirmed']:>7}{r['frag_suspects']:>7}{r['frag_pct']:>7.1f}{r['dup_births']:>6}"
              f"{r['reassoc_same_id']:>9}{r['class_flip_tracks']:>8}{fmt(r['life_median_s']):>11}"
              f"{fmt(r['life_p90_s']):>11}{fmt(r['new_ids_per_min']):>12}")
    print("  frag?    new ID appeared where a confirmed track of the same kind had just vanished (not at the edge):\n"
          "           the tracker probably lost the object and started a new ID. Heuristic: check the CSV + video.\n"
          "  dup      new ID born on top of a live track (IoU>0.5): duplicate track or double detection.\n"
          f"  short    confirmed IDs that lived < {a.short_track_frames} analysed frames (many = fragmented tracks).\n"
          "  flicker  IDs seen fewer than min-hits frames (noise).   clsflip = tracks whose M1 class changed.\n"
          "  reassoc  an ID vanished and came back with the SAME id (good: the tracker bridged the gap).")


def print_compare(all_A):
    hdr = (f"{'config':<22}{'M1':>8}{'track':>8}{'M2':>8}{'queue':>8}{'proc':>8}{'proc p95':>10}{'interval':>10}"
           f"{'int p95':>9}{'batch/s':>9}{'full%':>7}{'GPU%':>7}{'NVDEC%':>8}{'ms/frm':>8}")
    _banner("COMPARISON - what do the tracker and the second model cost?", len(hdr))
    g = lambda A, k, f="mean": A["metrics"][k][f] if k in A["metrics"] else 0.0
    hw = lambda A, k: ((A.get("hardware") or {}).get(k) or [NAN])[0]
    print(hdr + "\n" + "-" * len(hdr))
    for A in all_A:
        print(f"{A['tag']:<22}{g(A, 'm1_ms'):>8.1f}{g(A, 'trk_ms'):>8.1f}{g(A, 'm2_ms'):>8.1f}{g(A, 'queue_ms'):>8.1f}"
              f"{g(A, 'proc_ms'):>8.1f}{g(A, 'proc_ms', 'p95'):>10.1f}{g(A, 'interval_ms'):>10.1f}"
              f"{g(A, 'interval_ms', 'p95'):>9.1f}{A['batches_per_s']:>9.1f}{A['full_batch_pct']:>7.1f}"
              f"{hw(A, 'gpu_util_pct(mean,max)'):>7.0f}{hw(A, 'nvdec_util_pct(mean,max)'):>8.0f}"
              f"{A.get('gpu_ms_per_frame') or NAN:>8.2f}")
    print("-" * len(hdr) + "   (times in ms; ms/frm = GPU time per analysed frame)")
    by = {A["stages"]: A for A in all_A}
    cost = lambda s: (by.get(s) or {}).get("gpu_ms_per_frame")
    print("\nGPU cost per analysed frame by difference (ms of GPU busy time per frame):")
    if cost("m1"):
        print(f"  M1 (+ mux/convert)  : {cost('m1'):.2f} ms")
    if cost("m1") and cost("track"):
        print(f"  Tracker             : {cost('track') - cost('m1'):+.2f} ms")
    if cost("track") and cost("full"):
        print(f"  M2                  : {cost('full') - cost('track'):+.2f} ms")
    base = all_A[0]
    for A in all_A[1:]:
        d_iv = g(A, "interval_ms") - g(base, "interval_ms")
        print(f"{A['tag']} vs {base['tag']}: revisit interval {d_iv:+.1f} ms "
              f"({100 * d_iv / max(g(base, 'interval_ms'), 1e-9):+.0f}%), M1 itself {g(A, 'm1_ms') - g(base, 'm1_ms'):+.1f} ms")
    print("(M1 getting slower when tracker/M2 are added = GPU contention between stages)")


# ======================================================================================
# Files
# ======================================================================================
def _write_csv(path, rows, extra=()):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + list(extra), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_outputs(a, tag, mets, Ss, A, sched, audit):
    os.makedirs(a.out_dir, exist_ok=True)
    for m, S in zip(mets, Ss):
        _write_csv(os.path.join(a.out_dir, f"ticks_{tag}_g{m.group}.csv"), m.recs, ("slot_age_ms", "e2e_true_ms"))
        _write_json(os.path.join(a.out_dir, f"summary_{tag}_g{m.group}.json"), S)
    if sched.turns:
        _write_csv(os.path.join(a.out_dir, f"turns_{tag}.csv"), sched.turns)
    if audit.enabled:
        audit.write_csv(a.out_dir, tag)
    _write_json(os.path.join(a.out_dir, f"summary_{tag}.json"), A)
    print(f"\nper-batch CSVs, per-group + overall summary JSON, turns CSV"
          f"{', tracker CSVs' if audit.enabled else ''} -> {os.path.abspath(a.out_dir)}")
