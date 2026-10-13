# DeepStream cascade benchmark

Round-robin multi-camera pipeline: fine-tuned YOLO26m detector (M1) → tracker → second-stage classifier (M2), always
the full cascade, with the camera view in one display window. At the end of a run it reports latency, throughput,
GPU/NVDEC load and scheduler health.

Assumes Ubuntu 24.04, Python 3.12, DeepStream 9.1 (CUDA 13.2) installed natively, and `pyds` available as a wheel.
Everything is configured in `config/config.yaml`; `run.py` takes no arguments. RTSP cameras only, substream URLs written as-is in
the cameras file.

All commands below are run from the project root.

## Repository layout

| path | purpose |
|---|---|
| `run.py`, `dsbench/`, `parser/`, `config/`, `models/` | the application (what you deploy) |
| `export/` | ONNX export and INT8 calibration (`export_models.py`, `calib_loader.py`, `calib_images/`) |
| `training/` | detector training and accuracy evaluation (`train.py`, `accuracy.py`, datasets, results) |
| `simulation/` | testing without real cameras: `make_cameras_40.py` + MediaMTX relay |
| `yolo26*.pt` (root) | pretrained weights; kept in the root so ultralytics finds them instead of re-downloading |

## 1. App environment (venv)

```bash
cd ~/Vision-system
python3 -m venv --system-site-packages .venv      # system packages needed for GStreamer's "gi"
source .venv/bin/activate
pip install -r requirements.txt
pip install /path/to/pyds-<version>-cp312-cp312-linux_x86_64.whl
python3 -c "import gi, pyds; print('pyds', pyds.__version__)"
```

If `gi` is missing: `sudo apt install python3-gi python3-gst-1.0 gstreamer1.0-libav gstreamer1.0-plugins-good`.
Already have a venv where `import pyds` works? Use that one and skip this section.

## 2. Export the models (separate venv)

Keep this out of the app venv: ultralytics installs PyTorch and can change NumPy.

```bash
python3 -m venv ~/export_venv
source ~/export_venv/bin/activate
pip install -r export/requirements-export.txt
python3 export/export_models.py --m1 path/to/best.pt --m1-imgsz 416 416
deactivate
ls models/        # best_416x416.onnx  labels.txt  stage2_cls.onnx  stage2_labels.txt
```

Own stage-2 classifier: add `--m2 my_cls.pt`. The detector classes are `0 human, 1 bike, 2 vehicle`; `labels.txt` is
written from the model itself. For a different class count set `num-detected-classes` in `config/pgie_yolo26m.txt`
and `operate-on-class-ids` in `config/sgie_stage2.txt`.

Training the detector yourself: `python3 training/train.py` (settings at the top of the file), then export the
`best.pt` it prints.

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
display tile size, warm-up and watchdog limits. Each key is commented in the file.

Three values must agree: the largest camera group, `detector.batch_size` in `config/config.yaml`, and `batch-size` in
`config/pgie_yolo26m.txt` (the engine file name there ends with the same number, e.g. `_b6_`). `detector.width/height`
must match the ONNX / engine. `run.py` warns at start-up on a mismatch; a mismatch makes TensorRT rebuild the engine on
every start. `export/calib_loader.py` (INT8 calibration batches with images from `export/calib_images/`; run it from
the export venv, it needs opencv and PyYAML, both installed with ultralytics) reads the same `detector.*` values.

The tracker config is copied from DeepStream:

```bash
cp /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml config/tracker_accuracy.yml
```

**`config/cameras.json`** (not in git; write `@` in a password as `%40`):

```json
{
  "cameras": [
    {"name": "D1", "uri": "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102"},
    {"name": "D2", "uri": "rtsp://user:pass@192.168.1.11:554/Streaming/Channels/102"}
  ],
  "groups": [["D1", "D2"]]
}
```

`groups` are the round-robin batches (every camera exactly once); without it all cameras form one group. The URLs are
used exactly as written, so put the substream URL there.

### Benchmarking with many cameras (no real cameras needed)

`simulation/make_cameras_40.py` builds `config/cameras_40.json` (copies of your real cameras, one group per entry of
`GROUP_SIZES`) and `simulation/mediamtx.yml`, so each real camera is connected only once:

```bash
python3 simulation/make_cameras_40.py
cd simulation && ./mediamtx mediamtx.yml      # keep running; start it from inside simulation/
```

Then set `cameras_file: config/cameras_40.json` in `config/config.yaml`. `mediamtx.yml` contains your camera
credentials and is git-ignored.

## 5. Run

```bash
python3 run.py          # opens the display window; Ctrl+C (or closing the window) stops and prints the summary
```

The first run builds the TensorRT engines: expect several minutes with no output. The terminal only shows start-up
information, GStreamer warnings/errors and, at the end, a short summary (hardware load over the recording window,
throughput, latency per stage, scheduler / slot health, batch integrity).

Each run writes `bench_out/<timestamp>/`:

| file | content |
|---|---|
| `summary.json` | full summary: overall, per group, scheduler, per-camera slot table, hardware |
| `ticks.csv` | one row per batch (all groups): stage times, queue wait, processing, end-to-end, revisit interval |
| `config_used.yaml` | copy of `config/config.yaml` of that run |