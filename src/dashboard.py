import time
import threading
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO

CAMERA_SOURCE = 0
PERSON_LIMIT = 3
COOLDOWN_SECONDS = 5

app = Flask(__name__)
model = YOLO("yolov8n.pt")
alerts = []
state = {"alarm": False, "last_alert": 0.0}
lock = threading.Lock()


def add_alert(message):
    now = time.time()
    with lock:
        if now - state["last_alert"] < COOLDOWN_SECONDS:
            return
        state["last_alert"] = now
        state["alarm"] = True
        alerts.append({"time": datetime.now().strftime("%H:%M:%S"), "message": message})
        del alerts[:-50]


def frames():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(CAMERA_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.5)
            continue
        result = model(frame, verbose=False)[0]
        found = [model.names[int(c)] for c in result.boxes.cls]
        people = found.count("person")
        danger = [n for n in found if n in {"knife", "baseball bat"}]
        if danger:
            add_alert("Weapon detected: " + ", ".join(sorted(set(danger))))
        elif people >= PERSON_LIMIT:
            add_alert(f"Crowd detected: {people} people")
        _, buf = cv2.imencode(".jpg", result.plot())
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Surveillance Dashboard</title>
<style>
 body{font-family:Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0}
 header{padding:14px 20px;background:#1e293b;font-size:20px;font-weight:bold}
 .wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 .card{background:#1e293b;border-radius:10px;padding:12px;flex:1;min-width:300px}
 img{width:100%;border-radius:8px}
 #status{padding:10px;border-radius:8px;font-weight:bold;background:#166534;margin-bottom:10px}
 #status.alarm{background:#b91c1c}
 li{padding:6px 0;border-bottom:1px solid #334155}
 button{padding:8px 14px;border:0;border-radius:6px;background:#3b82f6;color:#fff;cursor:pointer}
</style></head><body>
<header>AI Crime Detection Surveillance</header>
<div class="wrap">
 <div class="card" style="flex:2"><h3>Live Camera</h3><img src="/video"></div>
 <div class="card"><h3>Status</h3>
  <div id="status">SAFE</div>
  <button onclick="fetch('/reset',{method:'POST'})">Reset Alarm</button>
  <h3>Alert Log</h3><ul id="log"></ul></div>
</div>
<script>
async function refresh(){
  const d = await (await fetch('/alerts')).json();
  const s = document.getElementById('status');
  s.textContent = d.alarm ? 'ALARM: ' + d.alerts[d.alerts.length-1].message : 'SAFE';
  s.className = d.alarm ? 'alarm' : '';
  document.getElementById('log').innerHTML =
    d.alerts.slice().reverse().map(a => '<li>' + a.time + ' - ' + a.message + '</li>').join('');
}
setInterval(refresh, 1000); refresh();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alerts")
def get_alerts():
    with lock:
        return jsonify({"alarm": state["alarm"], "alerts": alerts})


@app.route("/reset", methods=["POST"])
def reset():
    with lock:
        state["alarm"] = False
    return "", 204


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
