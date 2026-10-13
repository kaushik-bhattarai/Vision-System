"""Hardware sampling: GPU / NVDEC / memory via NVML and CPU via psutil."""
from __future__ import annotations

import threading
import time

import numpy as np


class HwSampler(threading.Thread):
    """Samples GPU util, GPU memory, CPU and NVDEC util twice a second.(engine build and warm-up are excluded)."""

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
            cpu = self.psutil.cpu_percent(None) if self.psutil else None
            self.samples.append((time.perf_counter(), util, mem, cpu, dec))

    def summary(self, since=0.0):
        """mean / max of every signal for samples taken at perf_counter() >= since."""
        rows = [s[1:] for s in self.samples if s[0] >= since]

        def col(i):
            v = [r[i] for r in rows if r[i] is not None]
            return {"mean": float(np.mean(v)), "max": float(np.max(v))} if v else None

        return {"gpu_util_pct": col(0), "gpu_mem_mb": col(1), "cpu_pct": col(2), "nvdec_util_pct": col(3),
                "samples": len(rows)}