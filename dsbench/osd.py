"""On-screen display: per-object labels, per-camera header, and the demux -> OSD -> window(s) display branch."""
from __future__ import annotations

import time

from . import gstutil as gu
from .config import COCO_NAMES, PERSON, UNTRACKED, VEHICLES
from .gstutil import Gst, pyds

PALETTE = [(0.12, 0.75, 1.0), (0.2, 1.0, 0.3), (1.0, 0.75, 0.0), (1.0, 0.4, 0.9), (0.6, 0.5, 1.0), (0.0, 1.0, 0.8),
           (1.0, 0.55, 0.15), (0.75, 1.0, 0.2), (0.3, 0.6, 1.0), (1.0, 0.9, 0.4), (0.5, 1.0, 0.6), (0.9, 0.5, 0.6)]
RED, WHITE, GREY = (1.0, 0.1, 0.1), (1.0, 1.0, 1.0), (0.6, 0.6, 0.6)


def _style_text(tp, text, font, bg):
    tp.display_text = text
    tp.font_params.font_name, tp.font_params.font_size = "Serif", font
    tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    tp.set_bg_clr = 1
    tp.text_bg_clr.set(*bg)


def attach_annotator(last_el, audit, cams, a):
    """Probe after the last stage: feeds the tracker audit and (with --display) writes the labels the OSD draws."""
    font, W = a.osd_font, a.mux_width
    sc = max(W / 1920.0, 0.5)  # line widths follow the mux resolution
    bw_hi, bw_mid, bw_lo = max(2, round(8 * sc)), max(2, round(5 * sc)), max(1, round(3 * sc))

    def draw(o, tr):
        oid, tracked = o.object_id, o.object_id != UNTRACKED
        name = o.obj_label or COCO_NAMES.get(o.class_id, f"cls{o.class_id}")
        parts = [f"{name} #{oid if tracked else '-'} " + (f"{o.confidence:.2f}" if o.confidence >= 0 else "trk")]
        m2 = gu.classifier_result(o)
        if m2 and m2[0]:
            parts.append(f"> {m2[0]} {m2[1]:.2f}")
        flag = None
        if tr is not None:
            parts.append(f"a{tr['n']}")
            if tr["flag"] and tr["n"] <= tr["flag_until"]:
                flag = tr["flag"]
                parts.append(flag)
        if flag and flag.startswith("FRAG"):
            col, bw = RED, bw_hi
        elif flag:
            col, bw = WHITE, bw_hi
        elif tracked:
            col, bw = PALETTE[int(oid) % len(PALETTE)], bw_mid
        else:
            col, bw = GREY, bw_lo
        r = o.rect_params
        r.border_width = bw
        r.border_color.set(*col, 1.0)
        tp = o.text_params
        tp.x_offset = int(max(0, min(r.left, W - 20)))
        tp.y_offset = int(max(0, r.top - font * 1.4))
        _style_text(tp, "  ".join(parts), font, (col[0] * 0.35, col[1] * 0.35, col[2] * 0.35, 0.85))

    def header(bm, fm, lines):
        dm = pyds.nvds_acquire_display_meta_from_pool(bm)
        dm.num_labels = len(lines)
        for i, text in enumerate(lines):
            tp = dm.text_params[i]
            tp.x_offset, tp.y_offset = 12, 10 + int(i * font * 1.6)
            _style_text(tp, text, font + 4, (0.0, 0.0, 0.0, 0.75))
        pyds.nvds_add_display_meta_to_frame(fm, dm)

    def cb(pad, info, _):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf)) if buf is not None else None
        if bm is None:
            return Gst.PadProbeReturn.OK
        t_ns = time.perf_counter_ns()
        for fm in gu.iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            objs = list(gu.iter_list(fm.obj_meta_list, pyds.NvDsObjectMeta.cast))
            cam = fm.source_id
            recs = audit.update(cam, [(o.object_id, o.class_id, o.rect_params.left, o.rect_params.top,
                                       o.rect_params.width, o.rect_params.height) for o in objs], t_ns)
            if a.display and 0 <= cam < len(cams):
                p = sum(o.class_id == PERSON for o in objs)
                v = sum(o.class_id in VEHICLES for o in objs)
                for o in objs:
                    draw(o, recs.get(o.object_id))
                header(bm, fm, audit.header(cam, p, v) if audit.enabled else [f"{cams[cam]}   P {p}  V {v}"])
        return Gst.PadProbeReturn.OK

    last_el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, cb, None)


def add_display(pipe, a, cams, prev):
    """nvstreamdemux splits each batch back into cameras; each camera gets its own leaky queue -> convert -> OSD, so a
    slow window never slows the analysis. Default: ONE window (tiles placed by a compositor);
    --separate-windows: one window per camera. Drawing costs GPU time: benchmark headless."""
    demux = gu.mk("nvstreamdemux", "demux")
    pipe.add(demux)
    prev.link(demux)
    w, h = a.window_width, a.window_height
    cols = 1 if len(cams) == 1 else 2
    rows = (len(cams) + cols - 1) // cols

    comp = None
    if not a.separate_windows:
        comp = gu.mk("compositor", "comp")
        gu.set_if(comp, "ignore-inactive-pads", True)
        gu.set_if(comp, "latency", 100 * Gst.MSECOND)
        size = gu.mk("capsfilter", "comp_size")
        size.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={w * cols},height={h * rows}"))
        vc, win = gu.mk("videoconvert", "vc_show"), gu.mk("autovideosink", "win_show")
        gu.set_if(win, "sync", False)
        pipe.add(comp)
        gu.link_chain(pipe, [comp, size, vc, win])

    for s, name in enumerate(cams):
        branch = [gu.make_queue(f"buf_disp_{name}", 1, leaky=True), gu.mk("nvvideoconvert", f"conv_{name}"),
                  gu.mk("nvdsosd", f"osd_{name}")]
        gu.set_if(branch[2], "display-text", 1)
        gu.set_if(branch[2], "display-bbox", 1)
        x, y = (s % cols) * w, (s // cols) * h
        if comp is not None:
            tosys, caps = gu.mk("nvvideoconvert", f"tosys_{name}"), gu.mk("capsfilter", f"sysfmt_{name}")
            caps.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={w},height={h}"))
            branch += [tosys, caps]
        else:
            sink = next((gu.mk(f, f"disp_{name}") for f in ("nveglglessink", "nv3dsink", "autovideosink")
                         if Gst.ElementFactory.find(f)), None)
            if sink is None:
                raise SystemExit("no video sink found for --display")
            gu.set_if(sink, "qos", False)
            for k, v in (("window-x", x), ("window-y", y + (s // cols) * 40), ("window-width", w), ("window-height", h)):
                gu.set_if(sink, k, v)
            branch.append(sink)
        pipe.add(branch[0])
        gu.link_pads(gu.request_pad(demux, f"src_{s}"), branch[0].get_static_pad("sink"), f"demux output of {name}")
        gu.link_chain(pipe, branch)
        if comp is not None:
            cpad = gu.request_pad(comp, "sink_%u")
            gu.link_pads(branch[-1].get_static_pad("src"), cpad, f"{name} to the display compositor")
            cpad.set_property("xpos", x)
            cpad.set_property("ypos", y)
        else:
            gu.set_if(branch[-1], "sync", False)