#!/usr/bin/env python3
"""Export the ONNX files (+ label files) that DeepStream turns into TensorRT engines.

  M1  yolo26m.pt    -> models/yolo26m_<H>x<W>.onnx   (80 COCO classes, default 384x640, dynamic batch)
  M2  stage-2 model -> models/stage2_cls.onnx        (default: yolo26s-cls.pt stand-in, 224x224)

nvinfer compiles the ONNX to a TensorRT engine on the first run and caches it next to the model
(e.g. models/yolo26m_384x640.onnx_b2_gpu0_fp16.engine). Do NOT use ultralytics `format=engine` for DeepStream:
that file carries an Ultralytics metadata header and is not a plain TensorRT plan.

H and W must be multiples of 32. For 640x360 sources use 384x640 (12 px letterbox padding top/bottom).

  python export_models.py                      # M1 at 384x640 + stand-in M2
  python export_models.py --m1-imgsz 640 640   # square export
  python export_models.py --m2 my_cls.pt       # your own classifier as stage 2
  python export_models.py --skip-m2
"""
import argparse
import os
import shutil

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--m1", default="yolo26m.pt")
p.add_argument("--m1-imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
p.add_argument("--m2", default="yolo26s-cls.pt")
p.add_argument("--m2-imgsz", type=int, default=224)
p.add_argument("--static-batch", type=int, default=0, help="fixed batch instead of dynamic (if dynamic gives trouble)")
p.add_argument("--opset", type=int, default=17)
p.add_argument("--out", default="models")
p.add_argument("--skip-m2", action="store_true")
a = p.parse_args()

from ultralytics import YOLO  # noqa: E402  (imported late so --help works without it)

os.makedirs(a.out, exist_ok=True)


def export(weights, imgsz, dst_name=None):
    model = YOLO(weights)
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    dst_name = dst_name or f"{os.path.splitext(os.path.basename(weights))[0]}_{h}x{w}.onnx"
    opts = dict(format="onnx", imgsz=imgsz, opset=a.opset, simplify=True)
    opts.update(dynamic=False, batch=a.static_batch) if a.static_batch else opts.update(dynamic=True)
    dst = os.path.join(a.out, dst_name)
    shutil.move(model.export(**opts), dst)
    print(f"  {weights} -> {dst}")

    try:  # keep batch dynamic, pin H and W (nvinfer / TensorRT prefer this)
        import onnx

        onnx_model = onnx.load(dst)
        dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
        if len(dims) == 4:
            for i, v in ((2, h), (3, w)):
                dims[i].ClearField("dim_param")
                dims[i].dim_value = v
            onnx.save(onnx_model, dst)
        show = lambda vs: [(v.name, [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]) for v in vs]
        print("    inputs :", show(onnx_model.graph.input))
        print("    outputs:", show(onnx_model.graph.output))
    except Exception as e:
        print("    onnx post-process skipped:", e)
    return model, dst_name


m1, m1_name = export(a.m1, a.m1_imgsz)
with open(os.path.join(a.out, "coco_labels.txt"), "w") as f:
    f.write("\n".join(m1.names[i] for i in range(len(m1.names))) + "\n")
print(f"  labels: {len(m1.names)} classes -> {a.out}/coco_labels.txt")

if not a.skip_m2:
    m2, _ = export(a.m2, a.m2_imgsz, "stage2_cls.onnx")
    with open(os.path.join(a.out, "stage2_labels.txt"), "w") as f:  # nvinfer wants ';'-separated labels
        f.write(";".join(str(m2.names[i]) for i in range(len(m2.names))) + "\n")
    print(f"  labels: {len(m2.names)} classes -> {a.out}/stage2_labels.txt")

print(f"\nM1 config must point at:\n  onnx-file=../{a.out}/{m1_name}\n"
      f"  model-engine-file=../{a.out}/{m1_name}_b<batch>_gpu0_fp16.engine")
