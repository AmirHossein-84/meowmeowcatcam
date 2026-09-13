# Gesture Meme Detector

Turn the camera on, make a gesture, and the matching cat meme pops up next to you!

It shows:
- **Camera** — your webcam feed with hand landmarks, a live debug readout (`H` toggles), a gesture cheat-sheet (`?`), and a caption of the current gesture
- **Meme** — the meme matching whatever gesture you're currently making

Video is processed locally and never uploaded. The browser build fetches MediaPipe code/models from jsDelivr + Google Cloud Storage once at startup.

## Gestures (16 total, `gestures.json` is the source of truth)

Desktop implements all 16. Browser implements the 11 scoped subset — spin, dance, mouth-open, huh, and side-eye-down are desktop-only and marked as such in-app.

| Gesture | How to trigger | Meme |
|---|---|---|
| Rockstar / Shaka | Thumb + pinky out | `cat.jpg` |
| Default | Nothing in particular, hands down | `pokercat.jpg` |
| One Finger Up | Index finger only, held away from face | `profcat.jpg`, `professorcat.jpg` |
| Fist | One hand, all four fingers curled | `punchcat.jpg` |
| Shhh | Index finger only, tip on mouth | `shhcat.jpg` |
| Fingers Together (muehehe) | Both hands up, index fingers only, tips touching | `uwucat.jpg`, `uwucatt.jpg`, `fingers-together-muehehe.jpg` |
| Kidnap cat | Any hand shape where your face just was | `hand-cover-face.jpg` |
| Crash-out Cat | Both hands up beside face, **both fists** | `crashout-cat.jpg` |
| Devastated cat | Both hands up, above head | `two-hands-on-head.jpg` |
| No monies | One open palm, away from face | `hand-stretched-out-palm-up.jpg` |
| Side Eye | Turn your head sideways | `side-eye-cat.jpg` |
| Judgy cat (desktop, experimental) | Look down slightly sideways | `side-eye-down.png` |
| Laugh and point (desktop) | Mouth wide open WITH a hand visible | `laugh-and-point.jpg` |
| Huh (desktop) | Mouth open AND eyes wide, no hands | `huh.png` |
| Dance (desktop, experimental) | Two open palms, one top + one bottom | `two-palms-up.mov` |
| Spin (desktop, experimental) | SPIN in your chair | `spin-cat.mov` |

`memes/iunno-cat.jpg` is reserved for a future shrug gesture (not wired yet).

## Running it — desktop (Python, full 16 gestures)

Requires Python **>=3.11** and a webcam.

- macOS/Linux: double-click **`Launch Gesture Meme.command`**
- Windows: double-click **`Launch Gesture Meme.bat`**

First run creates `.venv` and installs `requirements.txt` (~90MB, a few minutes). Later runs are instant. Press `q` or `Esc` to quit, `H` for HUD, `?` for help.

Manual:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 gesture_meme.py
```

If the camera fails: check OS permissions, close Zoom/OBS/Teams (only one app can hold the camera), then retry. The app tries index 0 then 1.

## Running it — browser (11 gestures)

No install, but the webcam API needs HTTP (opening `index.html` as `file://` won't get camera permission):

```bash
python3 -m http.server 8901
```

Then open `http://localhost:8901/` and allow camera access. Or on Windows double-click **`Launch Web Meme.bat`** (serves port 8901, local only). Models load from the pinned MediaPipe CDN (`tasks-vision@0.10.14`) at runtime.

The page shows loading → permission → playing → error states (camera-denied / CDN-blocked / GPU-unsupported show a message + Retry instead of a black page). Press `?` for the gesture list.

## Live debug HUD

Desktop Camera window:

```
gesture: sideEyeCat
yaw: +18.4 deg  (side-eye thr +/-15.0)
flow mag / spin fraction / peak, jawOpen/eyeWide, smile/brow/wink, pitch
```

Browser shows gesture + yaw. Tune thresholds at the top of `gesture_meme.py` / `gestures.json` if a gesture misfires for your setup. Spin/huh/dance/side-eye-down thresholds are experimental — see the tuning comments in code before changing them; spin overrides everything and needs labeled sessions to retune safely.

`flow_debug_log.csv` appends every run (rotates at 5MB to `.1`); columns are `wall_ms,t_monotonic_ms,magnitude,coherence,score,fraction,peak_2s,gesture`.

## Project layout

```
gesture_meme.py   desktop version (OpenCV + MediaPipe Python tasks API)
app.js            browser version (MediaPipe tasks-vision WASM, 11-gesture subset)
index.html        browser UI shell (loading/error/help/caption states)
gestures.json     shared gesture spec (names, triggers, memes, thresholds)
memes/            meme images + 2 video memes (kebab-case filenames)
models/           MediaPipe .task files for desktop (offline; scripts/download_models.py re-fetches)
requirements.txt  runtime deps (opencv-python, numpy<2, mediapipe>=0.10.14)
scripts/          download_models.py (model fetch + checksum)
tests/            pytest: geometry + decide() priority + asset manifest/parity
Launch Gesture Meme.command / .bat   desktop launchers
Launch Web Meme.bat                  web launcher (local :8901)
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
node --check app.js
python scripts/download_models.py --check-only
```

CI (`.github/workflows/ci.yml`) runs pytest + ruff + `node --check` on Python 3.11–3.13 × Ubuntu/Windows, plus non-blocking `pip-audit` and Dependabot.

## Android app (Capacitor wrapper, 11 browser gestures)

The browser build (`index.html` + `app.js` + `memes/`) is wrapped with
[Capacitor](https://capacitorjs.com) as `com.meowmeowcatcam.app`.
Needs internet on first launch (MediaPipe code/models load from CDN).

- GitHub Actions (`.github/workflows/android.yml`) builds a sideloadable
  **debug APK** on every push to `main`/`master`, PR, or manual dispatch.
  Download it from the run's `app-debug` artifact and install with
  `adb install app-debug.apk` (enable "Install unknown apps" first).
- Local build: needs Node 20+ and JDK 17 (Gradle 8.2.1 can't run on JDK 21):

```bash
npm ci
npm run build:android   # copies web assets to www/ and runs `cap sync`
cd android && ./gradlew assembleDebug   # APK at app/build/outputs/apk/debug/
```

`scripts/copy-www.js` is the source of truth for what goes into `www/`;
never edit `www/` or `android/app/src/main/assets/` by hand.
Camera permission (`android.permission.CAMERA`) is declared in
`android/app/src/main/AndroidManifest.xml`.

Uninstall: delete `.venv/`; nothing else is installed (no autostart, no background processes).
