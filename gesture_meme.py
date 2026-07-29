"""
Webcam gesture -> meme detector (desktop version).

Opens two windows, side by side like the OBS/streamer setups:
  - "Camera": your webcam feed with hand landmarks drawn on top
  - "Meme": the cat meme matching whatever gesture you're making

Gestures:
  rockstar / shaka  -> memes/cat.jpg
  default (no hand) -> memes/pokercat.jpg
  one finger up     -> memes/profcat.jpg, memes/professorcat.jpg
  fist / punch      -> memes/punchcat.jpg
  shhh              -> memes/shhcat.jpg
  two fingers together (both hands, tips touching) -> memes/uwucat.jpg, memes/uwucatt.jpg

Press q or ESC to quit.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes"

GESTURE_MEMES = {
    "rockstar": ["cat.jpg"],
    "default": ["pokercat.jpg"],
    "oneFingerUp": ["profcat.jpg", "professorcat.jpg"],
    "fist": ["punchcat.jpg"],
    "shhh": ["shhcat.jpg"],
    "twoFingersTogether": ["uwucat.jpg", "uwucatt.jpg"],
}

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- geometry helpers (ported from the JS version) -----------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


class GestureState:
    def __init__(self):
        self.last_face = None  # (mouth_center, face_width, t)

    def update_face(self, face_result):
        now = time.time() * 1000
        if face_result.face_landmarks:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(p3(f[234]), p3(f[454]))
            self.last_face = (mouth_center, face_width, now)

    def decide(self, hand_result):
        if not hand_result.hand_landmarks:
            return "default"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        if len(hands) == 2 and is_pointing(hands[0]) and is_pointing(hands[1]):
            avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
            tip_gap = dist(hands[0]["indexTip"], hands[1]["indexTip"]) / avg_scale
            if tip_gap < 1.4:
                return "twoFingersTogether"

        h = hands[0]

        if h["curledCount"] == 4:
            return "fist"

        if h["thumbOut"] and h["pinkyUp"] and not h["indexUp"] and not h["middleUp"] and not h["ringUp"]:
            return "rockstar"

        if h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]:
            now = time.time() * 1000
            if self.last_face and now - self.last_face[2] < FACE_STALE_MS:
                mouth_center, face_width, _ = self.last_face
                d = dist(h["indexTip"], mouth_center) / face_width
                if d < 0.55:
                    return "shhh"
            return "oneFingerUp"

        return "default"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )
    )

    memes = load_memes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                current_meme = random.choice(memes[gesture])

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)

            meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
