#!/usr/bin/env python3
"""Build config/cameras_40.json: sources (one group per entry of GROUP_SIZES) made from the cameras in config/cameras.json
and a MediaMTX relay config (simulation/mediamtx.yml).
Run from the project root:  python simulation/make_cameras_40.py
Then start the relay from this folder:  cd simulation && ./mediamtx mediamtx.yml
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent      # simulation/
ROOT = HERE.parent

SOURCE_CONFIG = ROOT / "config" / "cameras.json"
OUT_CONFIG = ROOT / "config" / "cameras_40.json"
GROUP_SIZES = [6, 6, 6, 6, 6, 5, 5]
USE_RELAY = True
RELAY_HOST = "127.0.0.1:8554"             # MediaMTX RTSP address
MEDIAMTX_YML = HERE / "mediamtx.yml"
RELAY_SOURCE_TRANSPORT = "tcp"

base = json.loads(SOURCE_CONFIG.read_text())["cameras"]
total = sum(GROUP_SIZES)

cams = []
for i in range(total):
    b = base[i % len(base)]
    name = f"{b['name']}_{i // len(base) + 1:02d}"
    uri = f"rtsp://{RELAY_HOST}/{b['name']}" if USE_RELAY else b["uri"]
    cams.append({"name": name, "uri": uri})

groups, i = [], 0
for size in GROUP_SIZES:
    groups.append([c["name"] for c in cams[i:i + size]])
    i += size
OUT_CONFIG.write_text(json.dumps({"cameras": cams, "groups": groups}, indent=2))
print(f"{total} sources in {len(groups)} groups {GROUP_SIZES} -> {OUT_CONFIG}")
for g, names in enumerate(groups):
    print(f"  g{g}: {', '.join(names)}")

if USE_RELAY:
    lines = ["# MediaMTX: pulls each real camera once, serves any number of local copies", "paths:"]
    for b in base:
        lines += [f"  {b['name']}:", f"    source: {b['uri']}", "    sourceOnDemand: no",
                  f"    rtspTransport: {RELAY_SOURCE_TRANSPORT}"]
    MEDIAMTX_YML.write_text("\n".join(lines) + "\n")
    print(f"relay config -> {MEDIAMTX_YML}\n  start it with:  cd simulation && ./mediamtx mediamtx.yml")
else:
    print("direct mode: the real cameras must accept this many simultaneous connections")