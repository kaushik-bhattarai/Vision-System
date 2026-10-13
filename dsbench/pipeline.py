"""Builds the single shared pipeline:
The muxer is shared by all cameras; the Scheduler decides which group's frames pass the gates, so every batch holds
the cameras of ONE group. 
"""
from __future__ import annotations

import time
from pathlib import Path

from . import gstutil as gu
from .gstutil import Gst
from .osd import add_display
from .scheduler import Gate


def _stage_elements(a):
    """[(label, element)]: m1 -> trk -> m2."""
    pgie = gu.mk("nvinfer", "pgie")
    pgie.set_property("config-file-path", str(Path(a.pgie).resolve()))
    trk = gu.mk("nvtracker", "tracker")
    trk.set_property("ll-lib-file", a.tracker_lib)
    trk.set_property("ll-config-file", a.tracker_config)
    for k, v in (("tracker-width", a.tracker_width), ("tracker-height", a.tracker_height), ("gpu-id", a.gpu_id)):
        gu.set_if(trk, k, v)
    sgie = gu.mk("nvinfer", "sgie")
    sgie.set_property("config-file-path", str(Path(a.sgie).resolve()))
    return [("m1", pgie), ("trk", trk), ("m2", sgie)]


def build_pipeline(a, cams, stop_ev):
    """Returns (pipeline, gates indexed by camera id, [(label, element)])."""
    gu.require_deepstream()
    Gst.init(None)
    pipe = Gst.Pipeline()

    mux = gu.mk("nvstreammux", "mux")
    for k, v in (("batch-size", a.mux_batch_size or max(len(g) for g in a.group_idx)), ("width", a.mux_width),
                 ("height", a.mux_height), ("batched-push-timeout", a.push_timeout_us),
                 ("live-source", 1), ("gpu-id", a.gpu_id), ("attach-sys-ts", True)):
        gu.set_if(mux, k, v)
    pipe.add(mux)
    if mux.find_property("batched-push-timeout") is None:
        print("  [note] this is the NEW nvstreammux (no batched-push-timeout / width / height). To use the legacy one "
              "this tool was tuned for: export USE_NEW_NVSTREAMMUX=no (if your DeepStream still supports it).")

    gates = [None] * len(a.uris)
    for g, idxs in enumerate(a.group_idx):
        for j, s in enumerate(idxs):
            tag = f"g{g}-{j:02d}"
            sb = gu.create_source_bin(tag, a.uris[s], a)
            pipe.add(sb)
            src = sb.get_by_name(f"src-{tag}")
            if src is not None and a.extra_surfaces > 0:
                gu.set_if(src, "num-extra-surfaces", a.extra_surfaces)
            q = gu.make_queue(f"slot_{cams[s]}", a.slot_buffers, leaky=True)
            pipe.add(q)
            gu.link_pads(sb.get_static_pad("src"), q.get_static_pad("sink"), f"source {cams[s]} to its slot")
            gu.link_pads(q.get_static_pad("src"), gu.request_pad(mux, f"sink_{s}"),  # pad index = global camera id
                         f"slot of {cams[s]} to the muxer")
            gates[s] = Gate(cams[s], q, stop_ev)

    stages = _stage_elements(a)
    chain = [mux, gu.make_queue("q_m1", a.stage_queue)]
    for k, (label, el) in enumerate(stages):
        if k > 0:
            chain.append(gu.make_queue(f"q_{label}", a.stage_queue))
        chain.append(el)
    gu.link_chain(pipe, chain)
    add_display(pipe, a, cams, chain[-1])
    return pipe, gates, stages


def attach_probes(stages, router):
    """Timestamp every batch entering / leaving each stage and hand it to the Router."""
    last_label = stages[-1][0]

    def probe_in(label):
        def cb(pad, info, _):
            buf = info.get_buffer()
            if buf:
                t = time.perf_counter_ns()
                key, frames = gu.parse_batch(buf)
                if frames:
                    router.stage_in(label, key, frames, t)
            return Gst.PadProbeReturn.OK
        return cb

    def probe_out(label):
        def cb(pad, info, _):
            buf = info.get_buffer()
            if buf:
                t, wall = time.perf_counter_ns(), time.time_ns()
                last = label == last_label
                key, frames = gu.parse_batch(buf, with_objs=last)
                if frames:
                    router.stage_out(label, key, frames, t, wall, label == "m1", last)
            return Gst.PadProbeReturn.OK
        return cb

    for label, el in stages:
        el.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe_in(label), None)
        el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_out(label), None)