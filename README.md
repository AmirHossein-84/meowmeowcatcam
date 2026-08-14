# Gesture Meme Detector

Point your webcam at yourself, make a face/hand gesture, get a cat meme back in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM, no install).

Two windows/panes side by side, streamer-style:
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the meme matching whatever gesture you're currently making

## Gestures

Checked in this order — when a pose could match more than one, the earlier one wins.

| # | Gesture | How to trigger |
|---|---|---|
| 1 | Fingers Together (muehehe) | Both hands up, index fingers only, tips touching |
| 2 | Two Hands On Head | Both hands up, above the top of your head |
| 3 | Crash-Out Cat | Both hands up beside your face, not above your head |
| 4 | Fist | One hand, all four fingers curled |
| 5 | Rockstar | Thumb + pinky out, shaka style |
| 6 | Shhh | Index finger only, tip resting on your mouth |
| 7 | One Finger Up | Index finger only, held away from your face |
| 8 | Hand Covering Face | Any hand shape sitting where your face just was |
| 9 | Hand Stretched Out | One open palm, all fingers extended, away from your face |
| 10 | Side Eye | No hands needed — turn your head 15°+ either way (real head-pose yaw) |
| 11 | Default | Nothing in particular, hands down |

Meme images live in `memes/`. A couple of gestures pick randomly between multiple images.

## Running it — desktop (Python)

Requires Python 3 and a webcam.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

Or just double-click **`Launch Gesture Meme.command`** once the venv above is set up — it runs `.venv/bin/python3 gesture_meme.py` for you.

Press `q` or `Esc` in the Camera window to quit.

## Running it — browser

No install needed, but the webcam API requires serving over HTTP (opening `index.html` directly as a `file://` URL will not get camera permission). From this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` and allow camera access. Models load from Google's hosted MediaPipe CDN at runtime, so nothing local is needed for the browser version.

## Live debug HUD

The Camera window always shows a small readout in the top-left corner:

```
gesture: sideEyeCat
yaw: +18.4 deg  (side-eye thr +/-15.0)
```

Useful for tuning the detection thresholds at the top of `gesture_meme.py` / `app.js` if a gesture is triggering too easily or not easily enough for your setup/lighting.

## Project layout

```
gesture_meme.py   desktop version (OpenCV + MediaPipe Python tasks API)
app.js            browser version (MediaPipe tasks-vision WASM)
index.html        browser UI shell
memes/            meme images (+ one video, unused for now)
models/           MediaPipe .task model files used by the desktop version
requirements.txt  Python dependencies
```
