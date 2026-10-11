#!/usr/bin/env python3
"""Live fruit detection on the Mac, using a Logitech camera and YOLOE."""

import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOE

OBJECTS = [
    "apple",
    "banana",
    "orange",
    "lemon",
    "lime",
    "strawberry",
    "grape",
    "watermelon",
    "pineapple",
    "mango",
    "pear",
    "peach",
    "kiwi",
    "blueberry",
    "cherry",
    "raspberry",
    "plum",
    "avocado",
    "person",
    "cup",
    "bottle",
    "bowl",
    "cell phone",
    "laptop",
    "book",
    "chair",
    "backpack",
    "keyboard",
    "mouse",
    "scissors",
    "spoon",
    "fork",
    "remote",
    "clock",
    "potted plant",
    "dog",
    "cat",
]

MODEL_NAME = "yoloe-11m-seg.pt"
CONFIDENCE = 0.2

BLUSH = (255, 236, 242)
ROSE = (186, 55, 102)
INK = (92, 28, 48)
CARD = (255, 248, 250)
WHITE = (255, 255, 255)
BOX_COLORS = (
    (214, 62, 112),
    (232, 104, 146),
    (168, 48, 96),
    (240, 140, 170),
    (196, 72, 122),
    (142, 42, 84),
)
TITLE_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Didot.ttc", 42, index=0)
LABEL_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Futura.ttc", 20, index=2)
BODY_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Futura.ttc", 18, index=0)
SMALL_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Futura.ttc", 15, index=0)


def list_video_devices():
    """Return [(index, name), ...] from ffmpeg's AVFoundation listing."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    devices = []
    in_video = False
    pattern = re.compile(r"\[(\d+)\]\s+(.+)$")
    for line in (proc.stderr + proc.stdout).splitlines():
        if "video devices" in line:
            in_video = True
            continue
        if "audio devices" in line:
            break
        if not in_video:
            continue
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(2).strip()
        if name.lower().startswith("capture screen"):
            continue
        devices.append((int(match.group(1)), name))
    return devices


def find_logitech(devices):
    for index, name in devices:
        lowered = name.lower()
        if "logitech" in lowered or "brio" in lowered:
            return index, name
    return None


def wait_for_logitech():
    print("Looking for the Logitech camera.")
    print("Unplug it from the Uno Q hub and plug it straight into the Mac.")
    while True:
        devices = list_video_devices()
        found = find_logitech(devices)
        if found:
            print(f"Using {found[1]} (camera {found[0]}).")
            return found
        visible = ", ".join(name for _, name in devices) or "none"
        print(f"Logitech not connected. Cameras I can see: {visible}")
        time.sleep(2)


class LogitechCamera:
    """Read frames from the named Logitech device, not from OpenCV's camera index."""

    def __init__(self, index, name, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-f",
                "avfoundation",
                "-framerate",
                str(fps),
                "-pixel_format",
                "uyvy422",
                "-video_size",
                f"{width}x{height}",
                "-i",
                f"{index}:none",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.name = name

    def read(self):
        if self.proc.poll() is not None:
            return False, None
        raw = self.proc.stdout.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame.copy()

    def release(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def boxes_from(result):
    items = []
    boxes = result.boxes
    if boxes is None:
        return items
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = result.names[int(box.cls[0])]
        items.append((x1, y1, x2, y2, float(box.conf[0]), label))
    return items


def color_for(label):
    return BOX_COLORS[sum(ord(char) for char in label) % len(BOX_COLORS)]


def render_ui(frame, items):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for x1, y1, x2, y2, conf, label in items:
        color = color_for(label)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, outline=color, width=4)
        caption = f"{label}  {conf:.0%}"
        left, top, right, bottom = LABEL_FONT.getbbox(caption)
        chip_w, chip_h = right - left + 18, bottom - top + 10
        chip_y = y1 - chip_h - 6 if y1 - chip_h - 6 > 4 else y1 + 6
        draw.rounded_rectangle((x1, chip_y, x1 + chip_w, chip_y + chip_h), radius=8, fill=color)
        draw.text((x1 + 9, chip_y + 3), caption, font=LABEL_FONT, fill=WHITE)

    rows = items[:6]
    row_h = 28
    panel_w = 280
    panel_h = 52 + row_h * max(1, len(rows)) + (22 if len(items) > 6 else 8)
    panel_x = width - panel_w - 18
    panel_y = height - panel_h - 18
    draw.rounded_rectangle(
        (panel_x + 3, panel_y + 4, panel_x + panel_w + 3, panel_y + panel_h + 4),
        radius=16,
        fill=(214, 160, 178),
    )
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=16, fill=CARD)
    draw.text((panel_x + 16, panel_y + 12), "Detected", font=LABEL_FONT, fill=ROSE)

    if not rows:
        draw.text((panel_x + 16, panel_y + 46), "Nothing in frame", font=BODY_FONT, fill=INK)
    else:
        for index, (x1, y1, x2, y2, conf, label) in enumerate(rows):
            y = panel_y + 48 + index * row_h
            color = color_for(label)
            draw.rounded_rectangle((panel_x + 16, y + 4, panel_x + 30, y + 18), radius=4, fill=color)
            draw.text((panel_x + 40, y), label, font=BODY_FONT, fill=INK)
            accuracy = f"{conf:.0%}"
            acc_w = SMALL_FONT.getbbox(accuracy)[2]
            draw.text((panel_x + panel_w - 16 - acc_w, y + 1), accuracy, font=SMALL_FONT, fill=ROSE)
        if len(items) > 6:
            draw.text((panel_x + 16, panel_y + panel_h - 24), f"+{len(items) - 6} more", font=SMALL_FONT, fill=ROSE)

    header_h = 92
    footer_h = 48
    margin = 22
    canvas = Image.new("RGB", (width + margin * 2, height + header_h + footer_h), BLUSH)
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.rectangle((0, 0, canvas.width, header_h - 8), fill=ROSE)
    canvas_draw.text((margin + 8, 22), "Object Detection System", font=TITLE_FONT, fill=WHITE)
    canvas.paste(image, (margin, header_h))
    credit = "made by Afya"
    credit_w = BODY_FONT.getbbox(credit)[2]
    credit_x = (canvas.width - credit_w) // 2
    credit_y = header_h + height + 12
    canvas_draw.text((credit_x, credit_y), credit, font=BODY_FONT, fill=ROSE)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME}. Fruit names are encoded on the CPU, then detection runs on {device}.")
    model = YOLOE(MODEL_NAME)
    # MobileCLIP cannot load on Apple GPU, so name the fruits before moving the detector.
    model.set_classes(OBJECTS)

    index, name = wait_for_logitech()
    if "logitech" not in name.lower() and "brio" not in name.lower():
        raise SystemExit(f"Refusing to open {name}. Only the Logitech camera is allowed.")

    state = {"frame": None, "items": []}
    lock = threading.Lock()
    stop = threading.Event()

    def capture_loop():
        cap = LogitechCamera(index, name)
        misses = 0
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if misses < 15:
                        continue
                    print("Camera frame dropped. Reconnecting…")
                    cap.release()
                    time.sleep(0.3)
                    found = wait_for_logitech()
                    cap = LogitechCamera(found[0], found[1])
                    misses = 0
                    continue
                misses = 0
                with lock:
                    state["frame"] = frame
        finally:
            cap.release()

    def detect_loop():
        while not stop.is_set():
            with lock:
                frame = None if state["frame"] is None else state["frame"].copy()
            if frame is None:
                time.sleep(0.01)
                continue
            results = model.predict(frame, conf=CONFIDENCE, verbose=False, device=device)
            items = boxes_from(results[0])
            with lock:
                state["items"] = items

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=detect_loop, daemon=True).start()

    window = "Object Detection System"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"Window is open on {name}. The picture stays live while detection runs. Press q to quit.")

    try:
        while not stop.is_set():
            with lock:
                frame = state["frame"]
                items = list(state["items"])
            if frame is None:
                time.sleep(0.01)
                continue
            cv2.imshow(window, render_ui(frame, items))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop.set()
                break
    finally:
        stop.set()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
