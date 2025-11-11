# fall-detect.py
import os, time, math, socket, datetime, threading, yaml, cv2, numpy as np
from collections import deque
from flask import Flask, send_from_directory
from tflite_runtime.interpreter import Interpreter
from notifiers import TelegramNotifier

# ---------- config ----------
with open("config.yaml") as f:
    C = yaml.safe_load(f)

cam_index      = C["camera_index"]
frame_w        = C["frame_w"]
frame_h        = C["frame_h"]
TARGET_FPS     = C["target_fps"]
show_window    = bool(C.get("show_window", False))  # default false for SSH/headless

conf_keypoint  = C["conf_keypoint"]
ema_alpha      = C["ema_alpha"]
down_vel_thresh= C["down_vel_thresh"]
tilt_deg_thresh= C["tilt_deg_thresh"]
low_height_frac= C["low_height_frac"]
still_secs     = C["still_secs"]
cooldown_secs  = C["cooldown_secs"]

# forward/back cues (hip-level, straight view)
ar_flip_delta   = C["ar_flip_delta"]
span_drop_frac  = C["span_drop_frac"]
span_window_sec = C["span_window_sec"]

# fisheye center crop
center_crop_frac = float(C.get("center_crop_frac", 0.0))

# clip + viewer
PREBUFFER_SEC = C["prebuffer_sec"]
POSTREC_SEC   = C["postrec_sec"]
CLIPS_DIR     = os.path.expanduser(C["clips_dir"])
VIEWER_PORT   = C["viewer_port"]
os.makedirs(CLIPS_DIR, exist_ok=True)

# latch (single alert)
rearm_when_standing = C.get("rearm_when_standing_secs", 10)
max_hold_secs       = C.get("max_hold_secs", 900)

# telegram
tg_cfg = C["alerts"]["telegram"]
TG = TelegramNotifier(tg_cfg["bot_token"], tg_cfg["chat_id"])


# ---------- helpers ----------
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "localhost"
    finally:
        s.close()
    return ip

def torso_tilt(kp):
    # returns degrees: 0 = vertical torso, 90 = horizontal
    idx = {"l_sh":5, "r_sh":6, "l_hip":11, "r_hip":12}
    def pick(i): return kp[i] if kp[i][2] >= conf_keypoint else None
    lsh, rsh, lhp, rhp = [pick(idx[k]) for k in ("l_sh","r_sh","l_hip","r_hip")]
    if None in (lsh, rsh, lhp, rhp): return None
    sh_mid = ((lsh[0]+rsh[0])/2, (lsh[1]+rsh[1])/2)
    hp_mid = ((lhp[0]+rhp[0])/2, (lhp[1]+rhp[1])/2)
    dy, dx = hp_mid[0]-sh_mid[0], hp_mid[1]-sh_mid[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def person_bbox(kp):
    xs = [p[1] for p in kp if p[2] >= conf_keypoint]
    ys = [p[0] for p in kp if p[2] >= conf_keypoint]
    if not xs or not ys: return None
    return (min(xs), min(ys), max(xs), max(ys))  # norm x1,y1,x2,y2

def ema(prev, new, a):
    return new if prev is None else (a*new + (1-a)*prev)

# ---------- clip recorder ----------
class Recorder:
    def __init__(self, w, h, fps):
        self.w, self.h, self.fps = w, h, fps
        self.pre = deque(maxlen=int(PREBUFFER_SEC*fps))
        self.writer = None
        self.out = None
        self.end = 0.0
        self.on = False
    def push(self, frame):
        self.pre.append(frame.copy())
        if self.on and self.writer:
            self.writer.write(frame)
    def start(self):
        if self.on: return
        name = datetime.datetime.now().strftime("fall_%Y%m%d_%H%M%S.mp4")
        path = os.path.join(CLIPS_DIR, name)
        for four in ("mp4v","avc1","XVID"):
            fourcc = cv2.VideoWriter_fourcc(*four)
            wr = cv2.VideoWriter(path, fourcc, self.fps, (self.w, self.h))
            if wr.isOpened():
                self.writer = wr; self.out = path; break
        if not self.writer:  # last resort AVI
            avi = os.path.splitext(path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(avi, fourcc, self.fps, (self.w, self.h))
            self.out = avi
        for fr in self.pre:
            self.writer.write(fr)
        self.end = time.time() + POSTREC_SEC
        self.on = True
        print(f"[REC] Started -> {self.out}")
    def step(self, frame):
        if self.on and time.time() >= self.end:
            try: self.writer.release()
            except: pass
            self.on = False
            print(f"[REC] Saved -> {self.out}")
            return self.out
        return None
        
# ---------- viewer ----------
app = Flask(__name__)
@app.get("/clips/")
def list_clips():
    files = sorted(f for f in os.listdir(CLIPS_DIR) if not f.startswith("."))
    return "<ul>" + "".join(f'<li><a href="/clips/{f}">{f}</a></li>' for f in files) + "</ul>"
@app.get("/clips/<path:n>")
def get_clip(n): return send_from_directory(CLIPS_DIR, n)

threading.Thread(target=lambda: app.run("0.0.0.0", port=VIEWER_PORT, debug=False, use_reloader=False),
                 daemon=True).start()
VIEW_URL = f"http://{get_ip()}:{VIEWER_PORT}/clips"
print("Clip viewer at:", VIEW_URL)

# ---------- model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "lite-model_movenet_singlepose_lightning_3.tflite")
inter = Interpreter(model_path=model_path, num_threads=4)
inter.allocate_tensors()
in_det  = inter.get_input_details()[0]
out_det = inter.get_output_details()[0]
in_h, in_w = in_det["shape"][1], in_det["shape"][2]

# ---------- camera (force V4L2 on /dev/video0) ----------
cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))  # hint; harmless if ignored
if not cap.isOpened():
    print(f"ERROR: could not open camera index {cam_index} with CAP_V4L2. Check /dev/video0 and user group 'video'.")
    raise SystemExit(1)

# ---------- state ----------
rec = Recorder(frame_w, frame_h, TARGET_FPS)
last_time = time.time()
ema_hip = None
last_hip = None
low_t = 0.0
latched = False
latch_since = 0.0
last_alert_t = -1e9
hist_ar, hist_span = deque(), deque()

print("Running? press T in the window for a Telegram test (if show_window=true).")
while True:
    ok, frame = cap.read()
    if not ok:
        print("WARN: cap.read() failed; exiting.")
        break
    t = time.time()
    dt = max(1e-3, t - last_time)
    last_time = t

    # fisheye center crop
    if 0.0 < center_crop_frac < 1.0:
        ch = int(frame_h * center_crop_frac)
        cw = int(frame_w * center_crop_frac)
        y0 = (frame_h - ch) // 2
        x0 = (frame_w - cw) // 2
        frame = frame[y0:y0+ch, x0:x0+cw]
        frame = cv2.resize(frame, (frame_w, frame_h))

    rec.push(frame)

    # pose inference
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h)).astype(np.float32)
    inp = np.expand_dims((resized - 127.5) / 127.5, 0)
    inter.set_tensor(in_det["index"], inp)
    inter.invoke()
    kp = inter.get_tensor(out_det["index"])[0, 0, :, :].tolist()  # (17,3): y,x,score

    # features
    tilt = torso_tilt(kp)
    bbox = person_bbox(kp)

    # hip velocity (px/s) with EMA smoothing
    hips = [kp[i][0] for i in (11, 12) if kp[i][2] >= conf_keypoint]
    hip_y = float(np.mean(hips)) if hips else None
    vy = 0.0
    if hip_y is not None:
        hip_px = hip_y * frame_h
        ema_hip = ema(ema_hip, hip_px, ema_alpha)
        vy = 0.0 if last_hip is None else (ema_hip - last_hip) / dt
        last_hip = ema_hip

    # bbox metrics
    low = False
    ar = None
    if bbox:
        bw = (bbox[2] - bbox[0]) * frame_w
        bh = (bbox[3] - bbox[1]) * frame_h
        if bh > 1: ar = bw / bh
        low = (bh / frame_h) <= low_height_frac

    ys = [p[0] for p in kp if p[2] >= conf_keypoint]
    span_px = (max(ys) - min(ys)) * frame_h if ys else None

    # short history for hip-level cues
    cutoff = t - span_window_sec
    if ar is not None: hist_ar.append((t, ar))
    if span_px is not None: hist_span.append((t, span_px))
    while hist_ar and hist_ar[0][0] < cutoff: hist_ar.popleft()
    while hist_span and hist_span[0][0] < cutoff: hist_span.popleft()

    def delta(seq):
        if len(seq) < 2: return 0.0, 0.0
        return seq[-1][1] - seq[0][1], seq[0][1]
    
    ar_jump, _     = delta(hist_ar)        # + if wider relative to height
    span_chg, s0   = delta(hist_span)      # - if vertical span shrinking
    span_drop = (-span_chg) / max(1.0, s0) if hist_span else 0.0

    # decision logic (hip-level, straight view)
    horizontal = (tilt is not None and tilt >= tilt_deg_thresh)
    sideways   = (vy > down_vel_thresh) and horizontal
    fwd_back   = ( (ar_jump >= ar_flip_delta) or (span_drop >= span_drop_frac) ) and horizontal

    # stillness gate
    low_t = low_t + dt if low else max(0.0, low_t - dt)
    sustained = low_t >= still_secs
    fall_likely = (sideways or fwd_back) and sustained

    # cooldown + latch (single alert)
    cooling = (t - last_alert_t) < cooldown_secs
    if latched:
        # rearm when standing for a while OR safety timeout
        if not low and low_t == 0.0 and (t - latch_since) >= rearm_when_standing:
            latched = False
            print("[Latch cleared]")
        elif (t - latch_since) >= max_hold_secs:
            latched = False
            print("[Latch timeout cleared]")

    if fall_likely and not cooling and not latched:
        last_alert_t = t
        latched = True
        latch_since = t
        rec.start()
        TG.send(f"? Possible fall detected @ {time.strftime('%H:%M:%S')}.\nRecording short clip?\nClips: {VIEW_URL}/")

    clip_path = rec.step(frame)
    if clip_path:
        TG.send(f"? Clip ready: {VIEW_URL}/{os.path.basename(clip_path)}")
    # preview (optional)
    if show_window:
        hud = frame.copy()
        if bbox:
            x1 = int(bbox[0]*frame_w); y1 = int(bbox[1]*frame_h)
            x2 = int(bbox[2]*frame_w); y2 = int(bbox[3]*frame_h)
            cv2.rectangle(hud, (x1,y1), (x2,y2), (0,255,0), 2)
        txt1 = f"tilt={None if tilt is None else int(tilt)} vy={int(vy)}"
        txt2 = f"low={int(low)} low_secs={low_t:.1f} ar?={0 if ar is None else round(ar_jump,2)} span?={round(span_drop,2)}"
        cv2.putText(hud, txt1, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(hud, txt2, (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("fall", hud)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        if k in (ord('t'), ord('T')): TG.send("? Test message from Pi")
    else:
        time.sleep(max(0.0, 1.0/TARGET_FPS - dt))

cap.release()
cv2.destroyAllWindows()
