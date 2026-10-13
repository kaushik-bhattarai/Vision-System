"""Small GStreamer / DeepStream helpers shared by the pipeline, the probes and the OSD."""
from __future__ import annotations

import sys

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    import pyds
except Exception:  # lets the pure-python parts (config, metrics, report) be imported without DeepStream
    gi = pyds = Gst = GLib = None


def require_deepstream():
    if Gst is None:
        sys.exit("GStreamer python bindings / pyds not found. Install DeepStream and pyds first (see README).")


def set_if(el, name, val) -> bool:
    """Set a property only if this DeepStream version has it (properties differ between releases)."""
    try:
        if el.find_property(name) is None:
            return False
        el.set_property(name, val)
        return True
    except Exception as e:
        print(f"  [warn] could not set {el.get_name()}.{name}={val}: {e}")
        return False


def mk(factory, name):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        sys.exit(f"cannot create GStreamer element '{factory}' - is DeepStream installed / plugin path set?")
    return el


def make_queue(name, max_buffers, leaky=False):
    """leaky=True: when full the OLDEST buffer is dropped, so the newest is always kept."""
    q = mk("queue", name)
    set_if(q, "max-size-buffers", max_buffers)
    set_if(q, "max-size-bytes", 0)
    set_if(q, "max-size-time", 0)
    if leaky and not (set_if(q, "leaky", 2) or set_if(q, "leaky", "downstream")):
        print(f"  [warn] could not make {name} leaky")
    return q


def request_pad(el, name):
    return (getattr(el, "request_pad_simple", None) or el.get_request_pad)(name)


def link_pads(src_pad, sink_pad, what):
    if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
        sys.exit(f"cannot link {what}")


def link_chain(pipe, elements):
    """Add elements[1:] to the pipeline (elements[0] is already in) and link them in order."""
    for el in elements[1:]:
        pipe.add(el)
    for x, y in zip(elements, elements[1:]):
        x.link(y)


#sources
def _on_new_pad(_src, pad, nbin):
    caps = pad.get_current_caps() or pad.query_caps(None)
    if "video" in caps.get_structure(0).get_name() and caps.get_features(0).contains("memory:NVMM"):
        if not nbin.get_static_pad("src").set_target(pad):
            sys.stderr.write("failed to link decoder src pad to source bin ghost pad\n")


def create_source_bin(tag, uri, a):
    """nvurisrcbin (decoder + RTSP reconnect) for an RTSP URL."""
    nbin = Gst.Bin.new(f"source-bin-{tag}")
    use_bin = Gst.ElementFactory.find("nvurisrcbin") is not None
    src = mk("nvurisrcbin" if use_bin else "uridecodebin", f"src-{tag}")
    src.set_property("uri", uri)
    if use_bin:
        set_if(src, "rtsp-reconnect-interval", 5)
        set_if(src, "latency", a.latency_ms)
        set_if(src, "select-rtp-protocol", 4)  # 4 = TCP only
        set_if(src, "cudadec-memtype", 0)
        set_if(src, "gpu-id", a.gpu_id)
    src.connect("pad-added", _on_new_pad, nbin)
    nbin.add(src)
    nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    return nbin


#metadata
def iter_list(l, cast):
    """Walk a pyds GList, tolerating the StopIteration pyds raises at the end."""
    while l is not None:
        try:
            yield cast(l.data)
            l = l.next
        except StopIteration:
            return


def classifier_result(obj):
    """(label, probability) of the last classifier attached to an object, or None."""
    res = None
    for cm in iter_list(obj.classifier_meta_list, pyds.NvDsClassifierMeta.cast):
        for li in iter_list(cm.label_info_list, pyds.NvDsLabelInfo.cast):
            res = (li.result_label, li.result_prob)
    return res


def parse_batch(buf, with_objs=False):
    """-> (key, frames). key identifies the batch; frames = [{src, num, ntp, objs}].
    objs (only with_objs): (width, height) of every object box."""
    bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if bm is None:
        return (), []
    frames = []
    for fm in iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
        f = {"src": fm.source_id, "num": fm.frame_num, "ntp": fm.ntp_timestamp, "objs": []}
        if with_objs:
            for o in iter_list(fm.obj_meta_list, pyds.NvDsObjectMeta.cast):
                r = o.rect_params
                f["objs"].append((r.width, r.height))
        frames.append(f)
    return tuple(sorted((f["src"], f["num"]) for f in frames)), frames