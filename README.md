# Gesture Meme Detector

Turn the camera on, make some gestures listed below to have the cat meme pop up next to you!

It will show up as:
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the meme matching whatever gesture you're currently making

## Gestures

| # | Gesture | How to trigger |
|---|---|---|
| 1 | Fingers Together (muehehe) | Both hands up, index fingers only, tips touching |
| 2 | Devastated cat | Both hands up, above the top of your head |
| 3 | Crash out Cat | Both hands up beside your face, not above your head |
| 4 | Fist | One hand, all four fingers curled |
| 5 | Rockstar | Thumb + pinky out |
| 6 | Shhh | Index finger only, tip resting on your mouth |
| 7 | One Finger Up | Index finger only, held away from your face |
| 8 | Kidnap cat | Any hand shape sitting where your face just was |
| 9 | i HAVE NO MONIES | One open palm, all fingers extended, away from your face |
| 10 | Side Eye | Just turn your head |
| 11 | Default | Nothing in particular, hands down |
| 12 | Judgy cat while munching on food | Look down slightly sideways 
| 13 | Huh | Open your mouth and be shocked 
| 14 | O I I.A I | SPIN SPIN SPIN 

Meme images live in `memes/`. A couple of gestures pick randomly between multiple images.

## Running it — desktop (Python)

Requires Python 3 and a webcam.

Easiest way: just double click **`Launch Gesture Meme.command`**. First run takes a minute to set itself up (installs everything automatically), then launches straight away. Every run after that is instant.

**First time opening it:** macOS will warn "cannot be opened because it is from an unidentified developer" — this is normal for any downloaded script, not specific to this one. Right-click the file → **Open** → click **Open** in the dialog that appears. You only need to do this once.

Or manually, if you prefer Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

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
