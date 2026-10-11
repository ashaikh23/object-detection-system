# Object Detection System

Live object detection on a Mac, using a Logitech camera and [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe). The camera picture stays at full speed. Detection runs beside it and draws a box on each match, with the name and accuracy.

Made by Afya.

## What you see

The window has a title, a live picture, a colored box on each detection, and a key at the bottom right. The key lists what is in the frame and the model’s accuracy for each one.

Press `q` in the window to quit.

## What it detects

The model is open-vocabulary. `detect.py` asks it to look for these classes:

| Fruits | Other objects |
| --- | --- |
| apple, banana, orange, lemon, lime | person, cup, bottle, bowl |
| strawberry, grape, watermelon, pineapple | cell phone, laptop, book, chair |
| mango, pear, peach, kiwi | backpack, keyboard, mouse, scissors |
| blueberry, cherry, raspberry, plum, avocado | spoon, fork, remote, clock, potted plant, dog, cat |

Several objects can be boxed at the same time. To change the list, edit `OBJECTS` in `detect.py`.

## Requirements

- macOS with a Logitech camera plugged into the Mac
- Python 3.11 or newer
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
- Camera permission for the app you run it from (Terminal, Cursor, or Visual Studio Code)

Apple silicon is recommended. Detection uses the GPU through PyTorch MPS when it is available, and the CPU otherwise.

The interface uses Didot and Futura, which ship with macOS.

## Setup

```bash
cd fruit-cam
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python
```

Install ffmpeg if you do not already have it:

```bash
brew install ffmpeg
```

The first run downloads `yoloe-11m-seg.pt` and the MobileCLIP text encoder. That download is a few hundred megabytes and only happens once.

## Run

```bash
source .venv/bin/activate
python -u detect.py
```

The script finds a camera whose name contains “Logitech” or “BRIO” and refuses to open the built-in camera or a phone camera.

### Visual Studio Code

1. Open the `fruit-cam` folder.
2. Select the `.venv` interpreter: Command Palette, then **Python: Select Interpreter**.
3. Run `detect.py`.
4. If the picture never appears, allow Visual Studio Code under **System Settings → Privacy & Security → Camera**, then run it again.

## How it works

`detect.py` does three things at once:

1. ffmpeg reads the Logitech at 1280×720 and keeps only the newest frame.
2. YOLOE-11m classifies that frame against the names in `OBJECTS`. Text prompts are encoded on the CPU, because MobileCLIP cannot load on Apple’s GPU. The detector itself runs on MPS when it can.
3. The window draws the live frame immediately and paints the latest boxes on top, so the picture does not wait for the model.

Detections below a confidence of `0.2` are dropped. That threshold is `CONFIDENCE` in `detect.py`.

## Notes

Do not commit `.venv`, `yoloe-11m-seg.pt`, or `mobileclip_blt.ts`. They are local install and download files.

YOLOE is published by Ultralytics under the AGPL-3.0 license. This project calls that library. It does not include the model weights.
