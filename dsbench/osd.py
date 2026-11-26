"""On-screen display: per-object labels, per-camera head count, dwell / crowd / cluster, and the
demux -> OSD -> one display window."""
from __future__ import annotations

import math
import time

from . import gstutil as gu
from .config import HUMAN, UNTRACKED
from .gstutil import Gst, pyds

PALETTE = [(0.12, 0.75, 1.0), (0.2, 1.0, 0.3), (1.0, 0.75, 0.0), (1.0, 0.4, 0.9), (0.6, 0.5, 1.0), (0.0, 1.0, 0.8),
           (1.0, 0.55, 0.15), (0.75, 1.0, 0.2), (0.3, 0.6, 1.0), (1.0, 0.9, 0.4), (0.5, 1.0, 0.6), (0.9, 0.5, 0.6)]
GREY = (0.6, 0.6, 0.6)
RED = (1.0, 0.1, 0.1)


def _style_text(tp, text, font, bg):
    tp.display_text = text
    tp.font_params.font_name, tp.font_params.font_size = "Serif", font
    tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    tp.set_bg_clr = 1
    tp.text_bg_clr.set(*bg)


def attach_annotator(last_el, cams, a, an, roi=None):
    """Probe after the last stage: feeds the people analytics and writes the labels the OSD draws.
    Label (on the box) = class, tracker id, dwell seconds (people), M2 classification.
    Crowd members get a red box.  Header = head count, crowd and cluster status of the camera."""
    font, W = a.osd_font, a.mux_width
    sc = max(W / 1920.0, 0.5)  # line widths follow the mux resolution
    bw_trk, bw_untrk = max(2, round(5 * sc)), max(1, round(3 * sc))

    def draw(o, dwell, crowd):
        oid = o.object_id
        tracked = oid != UNTRACKED
        name = o.obj_label or f"cls{o.class_id}"
        text = f"{name} #{oid}" if tracked else name
        if dwell is not None:
            text += f" {dwell:.0f}s"
        m2 = gu.classifier_result(o)
        if m2 and m2[0]:
            text += f" > {m2[0]}"
        col, bw = (PALETTE[int(oid) % len(PALETTE)], bw_trk) if tracked else (GREY, bw_untrk)
        if crowd:
            col, bw = RED, bw_trk
        r = o.rect_params
        r.border_width = bw
        r.border_color.set(*col, 1.0)
        tp = o.text_params
        tp.x_offset = int(max(0, min(r.left, W - 20)))
        tp.y_offset = int(max(0, r.top - font * 1.4))
        _style_text(tp, text, font, (col[0] * 0.35, col[1] * 0.35, col[2] * 0.35, 0.85))

    def header(bm, fm, text, alert):
        dm = pyds.nvds_acquire_display_meta_from_pool(bm)
        dm.num_labels = 1
        tp = dm.text_params[0]
        tp.x_offset, tp.y_offset = 12, 10
        _style_text(tp, text, font + 4, (0.6, 0.0, 0.0, 0.85) if alert else (0.0, 0.0, 0.0, 0.75))
        pyds.nvds_add_display_meta_to_frame(fm, dm)

    def zone_lines(bm, fm, pts, hot):
        """Draws the camera's ROI polygon (muxer pixels); red while someone is inside, yellow otherwise."""
        segs = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        for k in range(0, len(segs), 16):  # a display meta holds at most 16 lines
            dm = pyds.nvds_acquire_display_meta_from_pool(bm)
            chunk = segs[k:k + 16]
            dm.num_lines = len(chunk)
            for ln, ((x1, y1), (x2, y2)) in zip(dm.line_params, chunk):
                ln.x1, ln.y1, ln.x2, ln.y2 = int(x1), int(y1), int(x2), int(y2)
                ln.line_width = bw_trk
                ln.line_color.set(1.0, 0.1, 0.1, 1.0) if hot else ln.line_color.set(1.0, 0.9, 0.0, 1.0)
            pyds.nvds_add_display_meta_to_frame(fm, dm)

    def cb(pad, info, _):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf)) if buf is not None else None
        if bm is None:
            return Gst.PadProbeReturn.OK
        now = time.monotonic()
        for fm in gu.iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            cam = fm.source_id
            if not 0 <= cam < len(cams):
                continue
            objs = list(gu.iter_list(fm.obj_meta_list, pyds.NvDsObjectMeta.cast))
            people = [o for o in objs if o.class_id == HUMAN]
            res = an.update(cam, [(None if o.object_id == UNTRACKED else o.object_id,
                                   o.rect_params.left + o.rect_params.width / 2,
                                   o.rect_params.top + o.rect_params.height / 2,
                                   o.rect_params.height) for o in people], now)
            it = iter(zip(res["dwell"], res["in_crowd"]))
            for o in objs:
                dw, cr = next(it) if o.class_id == HUMAN else (None, False)
                draw(o, dw, cr)
            in_zone = False
            if roi is not None and roi.has_zone(cam):
                in_zone = roi.update(cam, [(None if o.object_id == UNTRACKED else o.object_id, o.class_id,
                                            o.obj_label or f"cls{o.class_id}",
                                            o.rect_params.left + o.rect_params.width / 2,
                                            o.rect_params.top + o.rect_params.height) for o in objs], now)
                zone_lines(bm, fm, roi.zones[cam], in_zone)
            txt = f"{cams[cam]}   Head count: {res['people']}"
            if roi is not None and roi.has_zone(cam):
                txt += ("   ROI OCCUPIED: " if in_zone else "   ROI: ") + roi.counter_text(cam)
            if res["crowds"]:
                txt += f"   CROWD x{res['crowds']} (max {res['crowd_max']})"
            if res["cluster"]:
                txt += "   CLUSTER"
            header(bm, fm, txt, bool(res["crowds"] or res["cluster"]))
        return Gst.PadProbeReturn.OK

    last_el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, cb, None)


def add_display(pipe, a, cams, prev, roi=None):
    """nvstreamdemux splits each batch back into cameras; each camera gets its own leaky queue -> convert -> OSD."""
    demux = gu.mk("nvstreamdemux", "demux")
    pipe.add(demux)
    prev.link(demux)
    w, h = a.tile_width, a.tile_height
    n = len(cams)  # grid: 1 camera -> 1 column, up to 4 -> 2 columns, more -> roughly a 16:9 shaped wall
    cols = 1 if n == 1 else 2 if n <= 4 else round(math.sqrt(n * 16 / 9))
    rows = (n + cols - 1) // cols

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
        # keep the FULL frame (no resize in the branch); the compositor pad below scales it to the tile size
        tosys, caps = gu.mk("nvvideoconvert", f"tosys_{name}"), gu.mk("capsfilter", f"sysfmt_{name}")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGBA"))
        branch = [gu.make_queue(f"buf_disp_{name}", 1, leaky=True), gu.mk("nvvideoconvert", f"conv_{name}"),
                  gu.mk("nvdsosd", f"osd_{name}"), tosys, caps]
        gu.set_if(branch[2], "display-text", 1)
        gu.set_if(branch[2], "display-bbox", 1)
        pipe.add(branch[0])
        gu.link_pads(gu.request_pad(demux, f"src_{s}"), branch[0].get_static_pad("sink"), f"demux output of {name}")
        gu.link_chain(pipe, branch)
        if roi is not None and roi.has_zone(s):  # only cameras with a zone grab crossing images
            roi.attach_capture(s, caps)
        cpad = gu.request_pad(comp, "sink_%u")
        gu.link_pads(branch[-1].get_static_pad("src"), cpad, f"{name} to the display compositor")
        cpad.set_property("xpos", (s % cols) * w)
        cpad.set_property("ypos", (s // cols) * h)
        cpad.set_property("width", w)    # compositor scales the whole frame into the tile
        cpad.set_property("height", h)
