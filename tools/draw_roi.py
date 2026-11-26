#!/usr/bin/env python3
"""Draw the region of interest (ROI) of one or more cameras in the browser.

    python tools/draw_roi.py                       # then open http://localhost:8765
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE = r"""<!doctype html><meta charset=utf-8><title>ROI editor</title>
<style>body{font:14px sans-serif;background:#1b1b1b;color:#eee;margin:12px}button,select{font:inherit;padding:4px 10px;margin-right:6px}
#wrap{margin-top:10px;display:inline-block;position:relative}canvas{max-width:96vw;cursor:crosshair;background:#444}
#msg{margin-left:10px;color:#8f8}.has{color:#fd0}</style>
<div><select id=cam></select><button id=undo>Undo point</button><button id=clear>Clear</button>
<button id=save>Save zone</button><button id=del>Delete zone</button><button id=snap>Refresh snapshot</button><span id=msg></span></div>
<div>Click the corners of the zone (3+). Cameras marked * already have a zone.</div>
<div id=wrap><canvas id=c></canvas></div>
<script>
const $=id=>document.getElementById(id), c=$('c'), g=c.getContext('2d'); let zones={}, pts=[], img=new Image(), cams=[];
function msg(t){$('msg').textContent=t}
function draw(){g.clearRect(0,0,c.width,c.height); if(img.complete&&img.naturalWidth)g.drawImage(img,0,0,c.width,c.height);
 else{g.fillStyle='#555';g.fillRect(0,0,c.width,c.height)}
 if(!pts.length)return; g.lineWidth=3;g.strokeStyle='#ff0';g.fillStyle='rgba(255,230,0,.2)';g.beginPath();
 pts.forEach((p,i)=>i?g.lineTo(p[0]*c.width,p[1]*c.height):g.moveTo(p[0]*c.width,p[1]*c.height));
 if(pts.length>2)g.closePath(),g.fill();g.stroke();g.fillStyle='#f00';
 pts.forEach(p=>{g.beginPath();g.arc(p[0]*c.width,p[1]*c.height,5,0,7);g.fill()})}
function fill(){const s=$('cam'),cur=s.value;s.innerHTML='';cams.forEach(n=>{const o=document.createElement('option');o.value=n;
 o.textContent=n+(zones[n]?' *':'');if(zones[n])o.className='has';s.appendChild(o)});if(cur)s.value=cur}
function load(){pts=(zones[$('cam').value]||[]).map(p=>p.slice());img=new Image();img.onload=()=>{c.width=img.naturalWidth;c.height=img.naturalHeight;draw()};
 c.width=960;c.height=540;draw();img.src='/snapshot?cam='+encodeURIComponent($('cam').value)+'&t='+Date.now()}
c.onclick=e=>{const r=c.getBoundingClientRect();pts.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw()};
$('undo').onclick=()=>{pts.pop();draw()};$('clear').onclick=()=>{pts=[];draw()};$('cam').onchange=load;$('snap').onclick=load;
async function post(body){const r=await fetch('/save',{method:'POST',body:JSON.stringify(body)});zones=await r.json();fill()}
$('save').onclick=async()=>{if(pts.length<3)return msg('need at least 3 points');await post({cam:$('cam').value,polygon:pts});msg('saved '+$('cam').value)};
$('del').onclick=async()=>{await post({cam:$('cam').value,polygon:null});pts=[];draw();msg('deleted '+$('cam').value)};
fetch('/info').then(r=>r.json()).then(d=>{cams=d.cams;zones=d.zones;fill();load()});
</script>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--image", action="append", default=[], metavar="CAM=FILE", help="use this picture instead of grabbing a frame")
    args = ap.parse_args()

    import yaml
    from dsbench.config import _path, mask_uri

    cfg = yaml.safe_load(Path(args.config).read_text())
    cj = json.loads(Path(_path(cfg["cameras_file"])).read_text())
    names = [c["name"] for c in cj["cameras"]]
    uris = dict(zip(names, (c["uri"] for c in cj["cameras"])))
    roi_path = Path(_path((cfg.get("roi") or {}).get("file", "config/rois.json")))
    images = dict(x.split("=", 1) for x in args.image)

    def read_zones():
        if roi_path.exists():
            raw = json.loads(roi_path.read_text())
            return {k: (v["polygon"] if isinstance(v, dict) else v) for k, v in raw.items()}
        return {}

    def snapshot(cam):
        import cv2
        if cam in images:
            return Path(images[cam]).read_bytes()
        cap = cv2.VideoCapture(uris[cam])
        frame = None
        try:
            for _ in range(5):
                ok, f = cap.read()
                if ok:
                    frame = f
        finally:
            cap.release()
        if frame is None:
            print(f"  could not grab a frame from {cam} ({mask_uri(uris[cam])}); using a blank canvas")
            return None
        return cv2.imencode(".jpg", frame)[1].tobytes()

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/info":
                self._send(json.dumps({"cams": names, "zones": read_zones()}).encode(), "application/json")
            elif u.path == "/snapshot":
                cam = parse_qs(u.query).get("cam", [""])[0]
                data = snapshot(cam) if cam in uris else None
                if data is None:
                    self._send(b"", "image/jpeg", 404)
                else:
                    self._send(data, "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/save":
                return self._send(b"not found", "text/plain", 404)
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if req.get("cam") not in uris:
                return self._send(b"unknown camera", "text/plain", 400)
            zones = read_zones()
            if req.get("polygon"):
                zones[req["cam"]] = [[round(min(max(x, 0), 1), 4), round(min(max(y, 0), 1), 4)] for x, y in req["polygon"]]
            else:
                zones.pop(req["cam"], None)
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_path.write_text(json.dumps({k: {"polygon": v} for k, v in zones.items()}, indent=2))
            print(f"  saved {roi_path.name}: zones on {', '.join(zones) or 'no camera'}")
            self._send(json.dumps(zones).encode(), "application/json")

        def log_message(self, *a):
            pass

    print(f"ROI editor on http://{args.host}:{args.port}  (zones -> {roi_path})  Ctrl+C to quit")
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
