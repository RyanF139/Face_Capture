import cv2
import time
import os
import math
import requests
import numpy as np 
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue
import json

load_dotenv()

# ==============================
# CONFIG
# ==============================
SERVICE_ID = os.getenv("SERVICE_ID")
ENDPOINT_URL = os.getenv("CCTV_ENDPOINT")

MODEL_PATH = os.getenv("MODEL_PATH")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))

ENABLE_RESIZE = os.getenv("ENABLE_RESIZE", "true").lower() == "true"
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", 640))
RESIZE_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 360))

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 400))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 300))

GRID_ROWS = int(os.getenv("GRID_ROWS", 2))
GRID_COLS = int(os.getenv("GRID_COLS", 2))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", 5))

# ==============================
# SAFE MODEL PATH
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model ONNX tidak ditemukan: {MODEL_PATH}")

print("MODEL_PATH:", MODEL_PATH, flush=True)

# ==============================
# FOLDER SETUP
# ==============================
FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("=== APP STARTED ===")
print("SERVICE_ID:", SERVICE_ID)

# ==============================
# GLOBAL STORAGE
# ==============================
preview_frames = {}
preview_lock = Lock()
active_cameras = {}
camera_lock = Lock()

# ==============================
# WEBHOOK ASYNC
# ==============================
queue = Queue(maxsize=200)

def webhook_worker():
    while True:
        item = queue.get()
        if item is None:
            break

        face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, cctv_id, client_id = item

        try:
            data_payload = {
                "timestamp": ts_iso,
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "channel_id": cctv_id,
                "client_id": client_id
            }

            log_payload = {
                "url": WEBHOOK_URL,
                "face_file": face_name,
                "frame_file": frame_name,
                "data": data_payload
            }
            print(f"\n[WEBHOOK] Sending {json.dumps(log_payload, indent=4)}")

            response = requests.post(
                WEBHOOK_URL,
                files=[
                    ("files", (face_name, face_bytes, "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=data_payload,
                timeout=10   # naikkan jadi 10 detik
            )

            print(f"[WEBHOOK] Status: {response.status_code}")

        except requests.exceptions.Timeout:
            print("[WEBHOOK] TIMEOUT")
        except requests.exceptions.ConnectionError:
            print("[WEBHOOK] CONNECTION ERROR")
        except Exception as e:
            print("[WEBHOOK] ERROR:", e)

        queue.task_done()

for _ in range(3):   # 3 worker paralel
    Thread(target=webhook_worker, daemon=True).start()

# ==============================
# HELPERS
# ==============================
def enforce_limit(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    if len(files) <= MAX_IMAGES:
        return
    files.sort(key=os.path.getmtime)
    for f in files[:len(files)-MAX_IMAGES]:
        os.remove(f)

def iso_name(prefix):
    ts = datetime.now().replace(microsecond=0).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prefix}_{ts}.jpg"

# ==============================
# CAMERA WORKER
# ==============================
class CameraWorker:

    def __init__(self, cctv_id, client_id, rtsp_url):
        self.cctv_id = cctv_id
        self.client_id = client_id
        self.rtsp_url = "rtsp://" + rtsp_url
        self.face_last_time = {}
        self.face_memory_timeout = SAVE_INTERVAL
        self.last_global_save = 0
        self.running = True

        print(f"[CAMERA START] {cctv_id}")

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (320, 320),
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000
        )

    def stop(self):
        print(f"[CAMERA STOP] {self.cctv_id}")
        self.running = False
        if self.cap:
            self.cap.release()

    def can_save(self, box):
        now = time.time()

        # GLOBAL PROTECTION (ANTI SPAM ABSOLUT)
        if now - self.last_global_save < SAVE_INTERVAL:
            return False

        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2

        bucket_x = cx // 150
        bucket_y = cy // 150

        face_id = (bucket_x, bucket_y)

        # jangan pakai size_bucket lagi (itu bikin spam)
        # size fluktuatif bikin face dianggap baru

        if face_id in self.face_last_time:
            if now - self.face_last_time[face_id] >= SAVE_INTERVAL:
                self.face_last_time[face_id] = now
                self.last_global_save = now
                return True
            return False
        else:
            self.face_last_time[face_id] = now
            self.last_global_save = now
            return True

    def run(self):
       while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue

            original = frame
            oh, ow = original.shape[:2]

            if ENABLE_RESIZE:
                resized = cv2.resize(original, (RESIZE_WIDTH, RESIZE_HEIGHT))
                sx = ow / RESIZE_WIDTH
                sy = oh / RESIZE_HEIGHT
            else:
                resized = original
                sx = sy = 1

            h, w = resized.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(resized)

            boxes = []
            if faces is not None and len(faces) > 0:

                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, fw, fh = largest_face[:4].astype(int)

                x = int(x * sx)
                y = int(y * sy)
                fw = int(fw * sx)
                fh = int(fh * sy)

                margin = 0.30
                extra_top = 0.40

                pad_w = int(fw * margin)
                pad_h = int(fh * margin)
                extra_h = int(fh * extra_top)

                x_new = x - pad_w
                y_new = y - pad_h - extra_h
                fw_new = fw + (2 * pad_w)
                fh_new = fh + (2 * pad_h) + extra_h

                h_img, w_img = original.shape[:2]

                x_new = max(0, x_new)
                y_new = max(0, y_new)

                if x_new + fw_new > w_img:
                    fw_new = w_img - x_new

                if y_new + fh_new > h_img:
                    fh_new = h_img - y_new

                boxes.append((x_new, y_new, fw_new, fh_new))

            frame_with_box = original.copy()
            for (x,y,fw,fh) in boxes:
                cv2.rectangle(frame_with_box, (x,y), (x+fw,y+fh), (0,255,0), 2)

            if ENABLE_VIEW:
                with preview_lock:
                    preview_frames[self.cctv_id] = frame_with_box.copy()

            for box in boxes:
                if self.can_save(box):
                    x,y,fw,fh = box
                    face_img = original[y:y+fh, x:x+fw]

                    face_name = iso_name("face")
                    frame_name = iso_name("frame")

                    face_path = os.path.join(FACE_FOLDER, face_name)
                    frame_path = os.path.join(FRAME_FOLDER, frame_name)

                    ret1, face_buffer = cv2.imencode(".jpg", face_img)
                    ret2, frame_buffer = cv2.imencode(".jpg", frame_with_box)

                    if ret1 and ret2:

                        with open(face_path, "wb") as f:
                            f.write(face_buffer.tobytes())

                        with open(frame_path, "wb") as f:
                            f.write(frame_buffer.tobytes())

                        enforce_limit(FACE_FOLDER)
                        enforce_limit(FRAME_FOLDER)

                        ts = face_name.replace(".jpg","").replace("face_","")

                        if WEBHOOK_URL and not queue.full():
                            queue.put((
                                face_buffer.tobytes(),
                                frame_buffer.tobytes(),
                                face_name,
                                frame_name,
                                ts,
                                box,
                                self.cctv_id,
                                self.client_id
                            ))

                        print(f"[{self.cctv_id}] Saved:", face_name)

# ==============================
# FETCH CAMERA LIST
# ==============================
def fetch_cameras():
    url = f"{ENDPOINT_URL}?service_id={SERVICE_ID}"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            print(f"[CAMERA API] Status error: {r.status_code}")
            return None

        data = r.json()

        if "data" not in data:
            print("[CAMERA API] Invalid response format")
            return None

        return data["data"]

    except requests.exceptions.Timeout:
        print("[CAMERA API] TIMEOUT")
    except requests.exceptions.ConnectionError:
        print("[CAMERA API] CONNECTION ERROR")
    except Exception as e:
        print("[CAMERA API] ERROR:", e)

    return None

# ==============================
# CAMERA MANAGER
# ==============================
def camera_manager():
    while True:
        start_time = time.time()

        try:
            cameras = fetch_cameras()

            if cameras is None:
                print("[CAMERA MANAGER] API down, retry in 10s")
                time.sleep(10)
                continue

            print(f"[CAMERA MANAGER] Reload camera list ({len(cameras)} cameras)")

            with camera_lock:
                existing = set(active_cameras.keys())
                incoming = set(cam["cctv_id"] for cam in cameras)

                # Tambah kamera baru
                for cam in cameras:
                    cid = cam["cctv_id"]
                    if cid not in existing:
                        print("[CAMERA ADDED]", cid)
                        worker = CameraWorker(cid, cam["client_id"], cam["stream_url"])
                        active_cameras[cid] = worker
                        #Thread(target=worker.run, daemon=True).start()
                        t = Thread(target=worker.run, daemon=True)
                        worker.thread = t
                        t.start()

                # Hapus kamera yang tidak ada lagi
                for cid in list(existing - incoming):
                    print("[CAMERA REMOVED]", cid)

                    worker = active_cameras.pop(cid, None)
                    if worker:
                        worker.stop()
                        if hasattr(worker, "thread"):
                            worker.thread.join(timeout=2)

                    with preview_lock:
                        preview_frames.pop(cid, None)

        except Exception as e:
            print("[CAMERA MANAGER ERROR]", e)

        # Hitung sisa waktu supaya interval presisi
        elapsed = time.time() - start_time
        sleep_time = max(0, CAMERA_REFRESH_INTERVAL - elapsed)
        time.sleep(sleep_time)

Thread(target=camera_manager, daemon=True).start()

print("System ready")

# ==============================
# GRID VIEW LOOP
# ==============================
if ENABLE_VIEW:
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

while True:
    if ENABLE_VIEW:
        frames = []

        with preview_lock:
            for f in preview_frames.values():
                frames.append(cv2.resize(f, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))

        total = GRID_ROWS * GRID_COLS

        while len(frames) < total:
            frames.append(
                255 * np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype="uint8")
            )

        rows = []
        idx = 0
        for r in range(GRID_ROWS):
            row = cv2.hconcat(frames[idx:idx+GRID_COLS])
            rows.append(row)
            idx += GRID_COLS

        grid = cv2.vconcat(rows)
        cv2.imshow("Face Detection", grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    time.sleep(0.03)

if ENABLE_VIEW:
    cv2.destroyAllWindows()

    # STOP ALL CAMERA WORKERS