# DeepStream cascade benchmark

Round-robin multi-camera pipeline: YOLO26m detector (M1) → tracker → second-stage classifier (M2), with latency,
throughput, GPU/NVDEC load, counts and tracker-quality reports.

Assumes Ubuntu 24.04, Python 3.12, DeepStream 9.1 (CUDA 13.2) installed natively, and `pyds` available as a wheel.

## 1. App environment (venv)

```bash
cd ~/dsbench
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
pip install -r requirements-export.txt
python3 export_models.py
deactivate
ls models/        # yolo26m_384x640.onnx  coco_labels.txt  stage2_cls.onnx  stage2_labels.txt
```

Own stage-2 classifier: `python3 export_models.py --m2 my_cls.pt`.

## 3. Build the parser

```bash
source .venv/bin/activate        # or your app venv
cd parser
CUDA_VER=13.2 make               # CUDA_VER = version of /usr/local/cuda-<ver>
cd ..
```

Produces `parser/libnvdsparse_yolo26.so`. If it fails with C++ errors in the headers:
`sed -i 's/-std=c++14/-std=c++17/' parser/Makefile` and run `make` again.

## 4. Tracker config and cameras

```bash
cp /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml config/tracker_accuracy.yml
cp config/cameras.example.json config/cameras.json
nano config/cameras.json         # names, RTSP URLs, groups (write @ in a password as %40)
```

`batch-size` in `config/pgie_yolo26m.txt` must equal the largest camera group (default `2`), and the engine file name
there ends with the same number (`_b2_`). A mismatch rebuilds the engine on every start.

## 5. Run

```bash
python3 run.py                           # shows a window; Ctrl+C stops and prints the report
python3 run.py --no-display              # headless (use for benchmarking)
python3 run.py --no-display --duration 120   # record 120 s, then report
python3 run.py --compare                 # m1 vs track vs full: cost of tracker and M2
python3 run.py -h                        # all options
```

The first run builds the TensorRT engines: expect several minutes with no output. Results are also written to
`bench_out/` (summary JSON, per-batch CSV, tracker CSVs).
