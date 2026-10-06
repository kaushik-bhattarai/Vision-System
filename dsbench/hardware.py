"""Hardware sampling (GPU / NVDEC / CPU via NVML + psutil) and per-frame decoder timing."""
from __future__ import annotations

import collections
import threading
import time

import numpy as np

from . import gstutil
from .gstutil import Gst


class HwSampler(threading.Thread):
    """Samples GPU util, GPU memory, CPU and NVDEC util twice a second."""

    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.samples, self.stop_ev = [], threading.Event()
        self.nvml = self.h = self.psutil = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(gpu)
        except Exception:
            pass
        try:
            import psutil

            psutil.cpu_percent(None)
            self.psutil = psutil
        except Exception:
            pass

    def run(self):
        while not self.stop_ev.wait(0.5):
            util = mem = dec = None
            if self.h is not None:
                try:
                    util = self.nvml.nvmlDeviceGetUtilizationRates(self.h).gpu
                    mem = self.nvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20
                except Exception:
                    pass
                try:  # NVDEC engine busy %, separate from the CUDA/graphics figure
                    dec = self.nvml.nvmlDeviceGetDecoderUtilization(self.h)[0]
                except Exception:
                    pass  # unsupported driver/GPU: use `nvidia-smi dmon -s u` (column dec)
            self.samples.append((util, mem, self.psutil.cpu_percent(None) if self.psutil else None, dec))

    def last(self):
        return self.samples[-1] if self.samples else (None,) * 4

    def summary(self):
        def col(i):
            v = [s[i] for s in self.samples if s[i] is not None]
            return (float(np.mean(v)), float(np.max(v))) if v else None

        return {"gpu_util_pct(mean,max)": col(0), "gpu_mem_mb(mean,max)": col(1), "cpu_pct(mean,max)": col(2),
                "nvdec_util_pct(mean,max)": col(3)}


class DecoderTimer:
    """Probes every nv decoder element: time a frame spends inside it (matched by PTS) and decoded fps."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seen, self.t_in = set(), {}
        self.samples = collections.defaultdict(list)  # (group, cam index in group) -> [ms]
        self.count = collections.defaultdict(int)
        self.snap0 = self.snap1 = None

    @staticmethod
    def _source_key(el):
        """Walk up to the 'source-bin-g<group>-<index>' bin -> (group, index)."""
        while el is not None:
            n = el.get_name()
            if n.startswith("source-bin-"):
                try:
                    g, i = n[len("source-bin-"):].split("-")
                    return int(g[1:]), int(i)
                except ValueError:
                    return None
            el = el.get_parent()
        return None

    def maybe_attach(self, el):
        try:
            factory = el.get_factory()
            fname = factory.get_name() if factory else ""
        except Exception:
            return
        if not (fname.startswith("nv") and "dec" in fname):  # nvv4l2decoder, nvdec ...
            return
        name = el.get_name()
        with self.lock:
            if name in self.seen:
                return
            self.seen.add(name)
            key = self._source_key(el) or ("?", len(self.seen))
            self.t_in[key] = {}
        sink, src = el.get_static_pad("sink"), el.get_static_pad("src")
        if sink is None or src is None:
            return

        def on_in(pad, info, _):
            b = info.get_buffer()
            if b is not None and b.pts != Gst.CLOCK_TIME_NONE:
                with self.lock:
                    d = self.t_in[key]
                    d[b.pts] = time.perf_counter_ns()
                    if len(d) > 512:
                        for k in list(d)[:256]:
                            d.pop(k, None)
            return Gst.PadProbeReturn.OK

        def on_out(pad, info, _):
            b = info.get_buffer()
            if b is not None:
                now = time.perf_counter_ns()
                with self.lock:
                    self.count[key] += 1
                    t0 = self.t_in[key].pop(b.pts, None)
                    if t0 is not None:
                        self.samples[key].append((now - t0) / 1e6)
            return Gst.PadProbeReturn.OK

        sink.add_probe(Gst.PadProbeType.BUFFER, on_in, None)
        src.add_probe(Gst.PadProbeType.BUFFER, on_out, None)
        print(f"  [nvdec] timing decoder {name} for source {key}")

    def scan(self, pipe):
        """Attach to decoders created after start-up (nvurisrcbin builds them lazily)."""
        try:
            for el in pipe.iterate_recurse():
                self.maybe_attach(el)
        except Exception as e:
            print(f"  [warn] decoder scan failed: {e}")

    def snapshot(self):
        with self.lock:
            return time.perf_counter(), dict(self.count)

    def fps(self, key):
        if not (self.snap0 and self.snap1):
            return None
        dt = self.snap1[0] - self.snap0[0]
        return (self.snap1[1].get(key, 0) - self.snap0[1].get(key, 0)) / dt if dt > 0 else None

    def total_fps(self):
        vals = [v for v in (self.fps(k) for k in list(self.count)) if v is not None]
        return sum(vals) if vals else None

    def stats(self, group, cams):
        out = []
        for i, cam in enumerate(cams):
            v = self.samples.get((group, i))
            if v:
                a = np.asarray(v, float)
                out.append({"cam": cam, "frames": int(a.size), "mean_ms": float(a.mean()),
                            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
                            "fps": self.fps((group, i))})
        return out
