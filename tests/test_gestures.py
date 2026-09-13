"""decide() priority tests. Geometry is bypassed by patching classify_hand
with canned dicts; hand_result/face state are minimal stubs."""
from unittest.mock import patch

import numpy as np

import gesture_meme as gm


class Lm:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class HandResult:
    def __init__(self, hands):
        self.hand_landmarks = hands  # [] = no hands; else list of landmark-lists


def H(curled=0, indexUp=False, middleUp=False, ringUp=False, pinkyUp=False,
      thumbOut=False, palm=(9.0, 9.0, 0.0), tip=(9.0, 9.0, 0.0), scale=0.1):
    return {"indexUp": indexUp, "middleUp": middleUp, "ringUp": ringUp,
            "pinkyUp": pinkyUp, "thumbOut": thumbOut, "curledCount": curled,
            "handScale": scale, "indexTip": np.array(tip, dtype=float),
            "wrist": np.zeros(3), "palmCenter": np.array(palm, dtype=float)}


def fresh_state(jaw=0.0, eye_wide=0.0, yaw=0.0, pitch=0.0, age_ms=100, seen=True, mono_now=1_000_000.0):
    st = gm.GestureState()
    st.flow_history = []
    st.last_face = (np.array([0.5, 0.5, 0.0]), 0.2, 0.05, yaw, mono_now - age_ms)
    st.last_jaw_open_debug = jaw
    st.last_eye_wide_debug = eye_wide
    st.last_pitch_debug = pitch
    st.face_seen_this_frame = seen
    return st, mono_now


def decide_with(st, canned_hands, mono_now=1_000_000.0):
    hr = HandResult(hands=[[Lm()] for _ in canned_hands])  # length only; classify patched
    with patch.object(gm, "classify_hand", side_effect=canned_hands):
        return st.decide(hr, now=mono_now)


def test_fist_priority_over_everything_single_hand():
    st, now = fresh_state()
    assert decide_with(st, [H(curled=4)], now) == "fist"


def test_rockstar_shape():
    st, now = fresh_state()
    assert decide_with(st, [H(curled=2, pinkyUp=True, thumbOut=True)], now) == "rockstar"


def test_pointing_far_from_face_is_oneFingerUp_near_is_shhh():
    st, now = fresh_state()  # mouth at (0.5,0.5), width 0.2; shhh radius 0.55*width=0.11
    far = H(curled=3, indexUp=True, palm=(9, 9, 0), tip=(9, 9, 0))
    assert decide_with(st, [far], now) == "oneFingerUp"
    near = H(curled=3, indexUp=True, palm=(9, 9, 0), tip=(0.5, 0.52, 0))
    assert decide_with(st, [near], now) == "shhh"


def test_twoFingersTogether_tip_gap_boundary():
    st, now = fresh_state()
    close = [H(curled=3, indexUp=True, tip=(0.1, 0.1, 0), scale=0.1),
             H(curled=3, indexUp=True, tip=(0.12, 0.1, 0), scale=0.1)]  # gap 0.2*scale
    assert decide_with(st, close, now) == "twoFingersTogether"
    apart = [H(curled=3, indexUp=True, tip=(0.1, 0.1, 0), scale=0.1),
             H(curled=3, indexUp=True, tip=(0.9, 0.9, 0), scale=0.1)]  # gap >> 1.4
    assert decide_with(st, apart, now) != "twoFingersTogether"


def test_crashOutCat_requires_both_fists_regression():
    # Palms at 1.0*face_width from mouth: inside nearFace radius 2.2 but
    # outside handCoverFace radius 0.7 (face seen), y below head-top so
    # twoHandsOnHead does not fire. Open hands must NOT read as crashOut.
    st, now = fresh_state()
    mouth = np.array([0.5, 0.5, 0.0])
    p1, p2 = tuple(mouth + np.array([0.2, 0.0, 0])), tuple(mouth + np.array([-0.2, 0.0, 0]))
    fists = [H(curled=4, palm=p1), H(curled=4, palm=p2)]
    assert decide_with(st, fists, now) == "crashOutCat"
    opens = [H(curled=0, palm=p1), H(curled=0, palm=p2)]
    assert decide_with(st, opens, now) != "crashOutCat"  # old app.js returned crashOut here


def test_mouthOpenCat_needs_hand_plus_mouth():
    st, now = fresh_state(jaw=gm.MOUTH_OPEN_JAW_THRESHOLD + 0.1)
    assert decide_with(st, [H(curled=4)], now) == "mouthOpenCat"  # any hand shape counts
    st2, now2 = fresh_state(jaw=gm.MOUTH_OPEN_JAW_THRESHOLD + 0.1)
    assert st2.decide(HandResult([]), now=now2) != "mouthOpenCat"  # no hands


def test_huhCat_needs_mouth_plus_wide_eyes_no_hands_and_excludes_mouthOpenCat():
    st, now = fresh_state(jaw=gm.HUH_JAW_THRESHOLD + 0.05, eye_wide=gm.EYE_WIDE_THRESHOLD + 0.05)
    assert st.decide(HandResult([]), now=now) == "huhCat"
    st2, now2 = fresh_state(jaw=gm.HUH_JAW_THRESHOLD + 0.05, eye_wide=0.0)
    assert st2.decide(HandResult([]), now=now2) != "huhCat"
    st3, now3 = fresh_state(jaw=1.0, eye_wide=1.0)
    assert decide_with(st3, [H(curled=0)], now3) != "huhCat"


def test_sideEye_yaw_strict_threshold_boundary():
    st, now = fresh_state(yaw=gm.SIDE_EYE_YAW_DEG)
    assert st.decide(HandResult([]), now=now) == "default"  # == thr, strict >
    st2, now2 = fresh_state(yaw=gm.SIDE_EYE_YAW_DEG + 0.1)
    assert st2.decide(HandResult([]), now=now2) == "sideEyeCat"
    st3, now3 = fresh_state(yaw=-(gm.SIDE_EYE_YAW_DEG + 1))
    assert st3.decide(HandResult([]), now=now3) == "sideEyeCat"
    st4, now4 = fresh_state(yaw=30.0, age_ms=gm.FACE_STALE_MS + 100)
    assert st4.decide(HandResult([]), now=now4) == "default"


def test_default_fallback_no_face_no_hands():
    st = gm.GestureState()  # last_face None
    assert st.decide(HandResult([]), now=1_000_000.0) == "default"


def test_spin_beats_hands_and_face():
    st, now = fresh_state(jaw=1.0)
    st.flow_history = [(now - i * 100, gm.SPIN_MAG_THRESHOLD + 1.0) for i in range(20)]
    assert decide_with(st, [H(curled=4)], now) == "spinCat"
