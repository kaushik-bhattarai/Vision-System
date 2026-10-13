# DeepStream cascade benchmark

Round-robin multi-camera pipeline: fine-tuned YOLO26m detector (M1) → tracker → second-stage classifier (M2).

Assumes Ubuntu 24.04, Python 3.12, DeepStream 9.1 (CUDA 13.2) installed natively, and `pyds` built from the
DeepStream sources (section 1).

All commands below are run from the project root.

## 1. App environment (venv)

`pyds` (the DeepStream Python bindings) is deprecated and NVIDIA no longer publishes wheels for DeepStream 9.x, so
it has to be built once from the sources installed with DeepStream
(project: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps, build steps in its `bindings/README.md`).
This project is tested on DeepStream 9.1 only.

```bash
cd ~/Vision-system
python3 -m venv --system-site-packages .venv      # system packages needed for GStreamer's "gi"
source .venv/bin/activate
pip install -r requirements.txt
```

Build the `pyds` wheel (once per DeepStream version), following `bindings/README.md`:

```bash
cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
# build as described in bindings/README.md; the wheel ends up in bindings/dist/
```

Install it into the venv:

```bash
pip install /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/dist/pyds-*-cp312-cp312-linux_x86_64.whl
python3 -c "import gi, pyds; print('pyds', pyds.__version__)"
```

If the sources are missing under `.../deepstream/sources/`, clone the repository from the link above into that folder.
If `gi` is missing: `sudo apt install python3-gi python3-gst-1.0 gstreamer1.0-libav gstreamer1.0-plugins-good`.

## 2. Export the models

`ultralytics` (PyTorch) must not go into the app venv: it can replace NumPy and break `pyds`. Run the export step in a
separate venv (`pip install -r export/requirements-export.txt`).

```bash
python3 export/export_models.py --m1 path/to/best.pt     # name and size come from detector.* in config.yaml
ls models/        # best_416x416.onnx  labels.txt  stage2_cls.onnx  stage2_labels.txt
```

If `detector.precision` is `int8`, run the calibration next, in the **app venv** (once per model size; skip it for
fp16 / fp32):

```bash
python3 export/calibrate.py            # models/<name>_<H>x<W>_calib.cache from export/calib_images/*.jpg; --force to redo
```

**Retrained `best.pt`** (same name and size): the old engine and calibration cache keep their names and would be
reused for the new weights. Delete them first:

```bash
rm models/best_416x416.onnx_b*_gpu0_*.engine models/best_416x416_calib.cache
python3 export/export_models.py --m1 path/to/new_best.pt
python3 export/calibrate.py            # int8 only
```

## 3. Build the parser

```bash
source .venv/bin/activate        # or your app venv
cd parser
CUDA_VER=13.2 make               # CUDA_VER = version of /usr/local/cuda-<ver>
cd ..
```

Produces `parser/libnvdsparse_yolo26.so`. If it fails with C++ errors in the headers:
`sed -i 's/-std=c++14/-std=c++17/' parser/Makefile` and run `make` again.

## 4. Configuration

**`config/config.yaml`** holds every tunable: cameras file, mux / detector / tracker sizes, queue sizes, scheduler timings,
display tile size, warm-up and watchdog limits.

The tracker config is copied from DeepStream:

```bash
cp /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml config/tracker_accuracy.yml
```

**`config/cameras.json`**:

```json
{
  "cameras": [
    {"name": "D1", "uri": "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102"},
    {"name": "D2", "uri": "rtsp://user:pass@192.168.1.11:554/Streaming/Channels/102"}
  ],
  "groups": [["D1", "D2"]]
}
```

`groups` are the round-robin batches (every camera exactly once); without it all cameras form one group.

### Benchmarking with many cameras (no real cameras needed)

`simulation/make_cameras_40.py` builds `config/cameras_40.json` (copies of your real cameras, one group per entry of
`GROUP_SIZES`) and `simulation/mediamtx.yml`, so each real camera is connected only once:

```bash
python3 simulation/make_cameras_40.py
cd simulation && ./mediamtx mediamtx.yml      # keep running; start it from inside simulation/
```

Then set `cameras_file: config/cameras_40.json` in `config/config.yaml`.

## 5. Run

```bash
python3 run.py          # opens the display window; Ctrl+C (or closing the window) stops and prints the summary
```

The first run builds the TensorRT engines: expect several minutes with no output. `run.py` stops with the exact
missing step if a model file or calibration cache is not there.

Each run writes `bench_out/<timestamp>/`:

| file | content |
|---|---|
| `summary.json` | full summary: overall, per group, scheduler, per-camera slot table, hardware |
| `ticks.csv` | one row per batch (all groups): stage times, queue wait, processing, end-to-end, revisit interval |
| `config_used.yaml` | copy of `config/config.yaml` of that run |

## 6. Detector accuracy per precision (optional)

```bash
python3 training/accuracy.py                       # fp32, fp16 and int8 on the validation split from train.py
python3 training/accuracy.py --precisions fp16 int8
```
