# =========================================================
# MULTI VIDEO FACE CAPTURE (FULL STABLE FINAL VERSION)
# CPU ONLY (YuNet) — Source: MP4 URL / YouTube
# =========================================================
import cv2
import time
import os
import math
import glob
import requests
import numpy as np
import json
import subprocess
import shutil

from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue

load_dotenv()

# =========================================================
# ====================== CONFIG (.env) =====================
# =========================================================

SERVICE_ID       = os.getenv("SERVICE_ID")
ENDPOINT_URL     = os.getenv("VIDEO_ENDPOINT")

MODEL_PATH       = os.getenv("MODEL_PATH")

SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD", 0.6))
BLUR_THRESHOLD   = float(os.getenv("BLUR_THRESHOLD", 0))

SAVE_FOLDER      = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL    = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES       = int(os.getenv("MAX_IMAGES", 150))

TARGET_MAX_WIDTH  = int(os.getenv("RESIZE_WIDTH", 640))
TARGET_MAX_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 360))
MIN_SIZE_CAPTURE  = int(os.getenv("MIN_SIZE_CAPTURE", 0))

FRAME_FPS        = int(os.getenv("FRAME_FPS", 12))
CROP_PADDING     = float(os.getenv("CROP_PADDING", 0.40))

ENABLE_VIEW      = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH    = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT   = int(os.getenv("DISPLAY_HEIGHT", 800))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

DEBUG_MODE        = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_FOLDER = os.getenv("DEBUG_VIDEO_FOLDER", "./sample_video")
DEBUG_LOOP        = os.getenv("DEBUG_LOOP", "true").lower() == "true"

WEBHOOK_URL      = os.getenv("WEBHOOK_URL")
WEBHOOK_API      = os.getenv("WEBHOOK_API")

# Loop video saat selesai (untuk non-live source seperti MP4)
VIDEO_LOOP       = os.getenv("VIDEO_LOOP", "true").lower() == "true"

print("ENABLE_VIEW:", ENABLE_VIEW)

# =========================================================
# ================= SAFE MODEL PATH ========================
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model ONNX tidak ditemukan: {MODEL_PATH}")

print("MODEL_PATH:", MODEL_PATH)

# =========================================================
# ================= FOLDER SETUP ===========================
# =========================================================

FACE_FOLDER  = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("=== APP STARTED ===")
print("SERVICE_ID:", SERVICE_ID)

# =========================================================
# ================= GLOBAL STORAGE =========================
# =========================================================

preview_frames = {}
preview_lock   = Lock()

active_cameras = {}
camera_lock    = Lock()

queue = Queue(maxsize=300)

# =========================================================
# ================= WEBHOOK WORKER =========================
# =========================================================

def webhook_worker():
    while True:
        item = queue.get()
        if item is None:
            break

        face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, video_id, client_id = item

        try:
            data_payload = {
                "timestamp": ts_iso,
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "channel_id": video_id,
                "client_id": client_id
            }

            log_payload = {
                "url": WEBHOOK_URL,
                "face_file": face_name,
                "frame_file": frame_name,
                "data": data_payload
            }

            print(f"\n[WEBHOOK] Sending:\n{json.dumps(log_payload, indent=4)}")

            # response = requests.post(
            #     WEBHOOK_URL,
            #     files=[
            #         ("files", (face_name, face_bytes, "image/jpeg")),
            #         ("files", (frame_name, frame_bytes, "image/jpeg")),
            #     ],
            #     data=data_payload,
            #     timeout=10
            # )

            # print(f"[WEBHOOK] Status: {response.status_code}")

            # if response.status_code >= 400:
            #     print("[WEBHOOK] ERROR RESPONSE:", response.text)

        except requests.exceptions.Timeout:
            print("[WEBHOOK] TIMEOUT")
        except requests.exceptions.ConnectionError:
            print("[WEBHOOK] CONNECTION ERROR")
        except Exception as e:
            print("[WEBHOOK] ERROR:", e)

        finally:
            queue.task_done()


for _ in range(3):
    Thread(target=webhook_worker, daemon=True).start()


# =========================================================
# ================= HELPERS ================================
# =========================================================

def iso_name(prefix):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prefix}_{ts}.jpg"


def enforce_limit(folder):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )
    while len(files) > MAX_IMAGES:
        os.remove(files.pop(0))


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def resolve_stream_url(video_url: str) -> str:
    """
    Resolve final playable URL:
    - YouTube  → pakai yt-dlp untuk ambil direct stream URL
    - MP4/HTTP → langsung pakai (OpenCV bisa baca HTTP/HTTPS MP4)
    """
    if is_youtube_url(video_url):
        if shutil.which("yt-dlp") is None:
            print("[WARNING] yt-dlp tidak ditemukan. Install: pip install yt-dlp")
            return video_url  # fallback, mungkin gagal

        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                 "--get-url", video_url],
                capture_output=True, text=True, timeout=30
            )
            urls = result.stdout.strip().splitlines()
            if urls:
                print(f"[yt-dlp] Resolved YouTube URL: {video_url} -> {urls[0][:80]}...")
                return urls[0]
        except Exception as e:
            print(f"[yt-dlp] Error resolving {video_url}: {e}")

    # MP4 atau URL lain: langsung kembalikan
    return video_url


# =========================================================
# ================= DEBUG VIDEO LOADER =====================
# =========================================================

def load_debug_sources(folder):
    files = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        files += glob.glob(os.path.join(folder, ext))

    sources = []
    for i, path in enumerate(files):
        sources.append({
            "video_id": f"debug_{i}",
            "client_id": "debug",
            "video_url": path
        })

    print("[DEBUG MODE] Found", len(sources), "files")
    return sources


# =========================================================
# ================= CAMERA / VIDEO WORKER ==================
# =========================================================

class VideoWorker:

    def __init__(self, video_id: str, client_id: str, video_url: str):

        self.video_id   = video_id
        self.client_id  = client_id
        self.video_url  = video_url          # URL asli (untuk log)
        self.running    = True

        # Resolve URL (YouTube → direct stream, MP4 → langsung)
        self.stream_url = resolve_stream_url(video_url)

        self.cap = self._open_cap(self.stream_url)

        self.frame_interval = 1 / FRAME_FPS
        self.last_time      = 0
        self.bad            = 0
        self.max_bad        = 20
        self.last_save      = 0
        self.face_last_time = {}

        # YuNet detector (CPU only)
        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (320, 320),
            score_threshold=SCORE_THRESHOLD
        )
        # warmup
        self.detector.setInputSize((320, 320))
        self.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

        print(f"[VIDEO START] {video_id} | {video_url[:60]}...")


    def _open_cap(self, url: str) -> cv2.VideoCapture:
        """Buka VideoCapture dengan backend yang tepat."""
        if os.path.isfile(url):
            cap = cv2.VideoCapture(url)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap


    def destroy(self, reason: str = "done"):
        """
        Bersihkan resource worker dan kirim status webhook.
        reason: "done"  → video selesai secara natural
                "removed" → dihapus oleh video_manager
        """
        self.running = False

        # Lepas capture
        try:
            self.cap.release()
        except Exception:
            pass

        # Hapus dari preview
        with preview_lock:
            preview_frames.pop(self.video_id, None)

        print(f"[{self.video_id}] Worker destroyed. Reason: {reason}")

        # Kirim webhook status
        if WEBHOOK_URL:
            self._send_done_webhook(reason)


    def _send_done_webhook(self, reason: str = "done"):
        """Kirim notifikasi status selesai ke webhook (non-blocking)."""
        def _post():
            payload = {
                "video_id": self.video_id,
                "client_id": self.client_id,
                "status": reason,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            try:
                print(f"[WEBHOOK DONE] Sending: {json.dumps(payload)}")
                resp = requests.post(
                    WEBHOOK_API+"/webhook/receive-status-video",
                    json=payload,
                    timeout=10
                )
                print(f"[WEBHOOK DONE] Status: {resp.status_code}")
                if resp.status_code >= 400:
                    print("[WEBHOOK DONE] ERROR RESPONSE:", resp.text)
            except requests.exceptions.Timeout:
                print("[WEBHOOK DONE] TIMEOUT")
            except requests.exceptions.ConnectionError:
                print("[WEBHOOK DONE] CONNECTION ERROR")
            except Exception as e:
                print("[WEBHOOK DONE] ERROR:", e)

        Thread(target=_post, daemon=True).start()


    def resize_adaptive(self, frame):
        h, w = frame.shape[:2]
        if w > TARGET_MAX_WIDTH:
            scale = TARGET_MAX_WIDTH / w
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            return small, 1/scale, 1/scale
        return frame, 1.0, 1.0


    def run(self):

        while self.running:

            now = time.time()
            if now - self.last_time < self.frame_interval:
                time.sleep(0.005)
                continue

            self.last_time = time.time()

            ret, frame = self.cap.read()

            if not ret:
                self.bad += 1

                if self.bad >= self.max_bad:
                    print(f"[{self.video_id}] Stream ended / error. "
                          f"Loop={VIDEO_LOOP}")

                    if VIDEO_LOOP:
                        # Ulangi dari awal (untuk file MP4 lokal / VOD)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Untuk URL remote, buka ulang
                        if not os.path.isfile(self.stream_url):
                            self.cap.release()
                            time.sleep(2)
                            self.stream_url = resolve_stream_url(self.video_url)
                            self.cap = self._open_cap(self.stream_url)
                        self.bad = 0
                    else:
                        print(f"[{self.video_id}] Video selesai.")
                        self.destroy(reason="done")
                        # Hapus diri dari active_cameras
                        with camera_lock:
                            active_cameras.pop(self.video_id, None)

                continue

            self.bad = 0

            # ── Deteksi wajah ──
            resized, sx, sy = self.resize_adaptive(frame)
            h, w = resized.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(resized)

            boxes = []

            if faces is not None:
                for f in faces:
                    score = float(f[4])
                    if score < SCORE_THRESHOLD:
                        continue

                    x, y, fw, fh = f[:4].astype(int)
                    x  = int(x  * sx);  y  = int(y  * sy)
                    fw = int(fw * sx);  fh = int(fh * sy)

                    if fw < MIN_SIZE_CAPTURE:
                        continue

                    x  = max(0, x)
                    y  = max(0, y)
                    fw = min(fw, frame.shape[1] - x)
                    fh = min(fh, frame.shape[0] - y)

                    boxes.append((x, y, fw, fh))

            view = frame.copy()

            for box in boxes:
                x, y, fw, fh = box

                cv2.rectangle(view, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(view, f"C:{score:.2f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Cooldown per zona wajah
                bucket = (x // 150, y // 150)
                if bucket in self.face_last_time and \
                   time.time() - self.face_last_time[bucket] < SAVE_INTERVAL:
                    continue
                self.face_last_time[bucket] = time.time()

                # ── Crop + padding ──
                pad_w = int(fw * CROP_PADDING)
                pad_h = int(fh * CROP_PADDING)
                cx = max(0, x - pad_w)
                cy = max(0, y - pad_h)
                cw = min(frame.shape[1] - cx, fw + pad_w * 2)
                ch = min(frame.shape[0] - cy, fh + pad_h * 2)

                face_img = frame[cy:cy+ch, cx:cx+cw]

                # Blur check
                gray      = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < BLUR_THRESHOLD:
                    continue

                face_name  = iso_name("face")
                frame_name = iso_name("frame")

                fb1 = cv2.imencode(".jpg", face_img)[1].tobytes()
                fb2 = cv2.imencode(".jpg", view)[1].tobytes()

                open(os.path.join(FACE_FOLDER,  face_name),  "wb").write(fb1)
                open(os.path.join(FRAME_FOLDER, frame_name), "wb").write(fb2)

                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                if WEBHOOK_URL:
                    ts_iso = datetime.utcnow().isoformat() + "Z"
                    queue.put((fb1, fb2, face_name, frame_name,
                               ts_iso, box, self.video_id, self.client_id))

                print(f"[{self.video_id}] Saved: {face_name}")
                self.last_save = time.time()

            with preview_lock:
                preview_frames[self.video_id] = view


# =========================================================
# ================= FETCH VIDEO LIST =======================
# =========================================================

def fetch_videos():
    """
    Ambil daftar video dari API.
    Response format:
    {
      "ok": true,
      "data": [
        {
          "video_id":  "...",
          "client_id": "...",
          "video_url": "https://..."
        },
        ...
      ]
    }
    """
    try:
        r = requests.get(ENDPOINT_URL, timeout=10)
        payload = r.json()
        if payload.get("ok"):
            return payload["data"]
        print("[fetch_videos] Response ok=false:", payload)
        return []
    except Exception as e:
        print("[fetch_videos] ERROR:", e)
        return []


# =========================================================
# ================= VIDEO MANAGER ==========================
# =========================================================

def video_manager():
    """
    Polling API setiap CAMERA_REFRESH_INTERVAL detik.
    Tambah worker baru jika video_id belum ada.
    (Opsional: bisa ditambah logika hapus worker yang sudah tidak ada di list.)
    """
    while True:
        sources = load_debug_sources(DEBUG_VIDEO_FOLDER) if DEBUG_MODE \
                  else fetch_videos()

        with camera_lock:
            existing = set(active_cameras.keys())

            for c in sources:
                vid = c["video_id"]
                if vid not in existing:
                    w = VideoWorker(vid, c["client_id"], c["video_url"])
                    t = Thread(target=w.run, daemon=True)
                    t.start()
                    active_cameras[vid] = w
                    print(f"[MANAGER] Started worker: {vid}")

            # Hapus worker yang tidak ada di list terbaru
            current_ids = {c["video_id"] for c in sources}
            for vid in list(existing):
                if vid not in current_ids:
                    print(f"[MANAGER] Stopping removed video: {vid}")
                    active_cameras[vid].destroy(reason="removed")
                    del active_cameras[vid]

        time.sleep(CAMERA_REFRESH_INTERVAL)


Thread(target=video_manager, daemon=True).start()

print("System ready — waiting for frames...")

# =========================================================
# ================= MAIN DISPLAY LOOP ======================
# =========================================================

if ENABLE_VIEW:

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    while True:

        with preview_lock:
            frames = list(preview_frames.values())

        if not frames:
            time.sleep(0.1)
            continue

        rows = math.ceil(math.sqrt(len(frames)))
        cols = math.ceil(len(frames) / rows)

        tw = DISPLAY_WIDTH  // cols
        th = DISPLAY_HEIGHT // rows

        imgs = [cv2.resize(f, (tw, th)) for f in frames]

        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        grid = []
        idx  = 0
        for r in range(rows):
            grid.append(cv2.hconcat(imgs[idx:idx+cols]))
            idx += cols

        cv2.imshow("Face Detection", cv2.vconcat(grid))

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:
    print("Running headless mode")
    while True:
        time.sleep(1)