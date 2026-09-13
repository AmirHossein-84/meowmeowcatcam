import math

import numpy as np
import pytest

import gesture_meme as gm


def test_angle_deg_orthogonal():
    assert gm.angle_deg(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(90.0)


def test_angle_deg_parallel():
    assert gm.angle_deg(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) == pytest.approx(0.0)


def test_angle_deg_degenerate_zero_vector_returns_180():
    # gesture_meme.py guards norm < 1e-9, matches JS app.js
    assert gm.angle_deg(np.zeros(3), np.array([1.0, 0, 0])) == 180.0
    assert gm.angle_deg(np.array([1.0, 0, 0]), np.zeros(3)) == 180.0


def test_dist_3d():
    assert gm.dist(np.array([0.0, 0, 0]), np.array([1.0, 2, 2])) == pytest.approx(3.0)


def _pts(mcp, pip, tip):
    pts = [np.zeros(3) for _ in range(21)]
    pts[5], pts[6], pts[8] = (np.asarray(p, dtype=float) for p in (mcp, pip, tip))
    return pts


def test_finger_extended_straight():
    pts = _pts((0, 0, 0), (0, 1, 0), (0, 2, 0))  # angle 0 deg
    assert gm.finger_extended(pts, 5, 6, 8) is True


def test_finger_extended_curled():
    pts = _pts((0, 0, 0), (0, 1, 0), (0, 0, 0))  # folds back -> 180 deg
    assert gm.finger_extended(pts, 5, 6, 8) is False


def test_yaw_identity_is_zero():
    assert gm.yaw_from_transform_matrix(np.eye(4)) == pytest.approx(0.0)


def test_yaw_plus20deg_about_Y():
    # Ry(+20deg) = [[c,0,s],[0,1,0],[-s,0,c]]; yaw = atan2(-r20, sy) = +20.
    # Sign matches JS yawFromTransformMatrix. Validate left-vs-right on HUD.
    t = math.radians(20.0)
    c, s = math.cos(t), math.sin(t)
    m = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    assert gm.yaw_from_transform_matrix(m) == pytest.approx(20.0, abs=1e-6)


def test_pitch_identity_is_zero():
    assert gm.pitch_from_transform_matrix(np.eye(4)) == pytest.approx(0.0)


def test_pitch_plus15deg_about_X():
    t = math.radians(15.0)
    c, s = math.cos(t), math.sin(t)
    m = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    assert gm.pitch_from_transform_matrix(m) == pytest.approx(15.0, abs=1e-6)


def test_blendshape_scores_empty_when_no_face():
    class FR:
        face_blendshapes: list = []  # noqa: RUF012 - test stub only

    assert gm.blendshape_scores(FR()) == {}


def test_wink_score_gated_by_threshold():
    assert gm.wink_score({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.4}) == 0.0  # max < 0.5
    assert gm.wink_score({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == pytest.approx(0.8)
    assert gm.wink_score({}) == 0.0


def test_eye_wide_score_takes_max_and_defaults_zero():
    assert gm.eye_wide_score({"eyeWideLeft": 0.2, "eyeWideRight": 0.7}) == pytest.approx(0.7)
    assert gm.eye_wide_score({}) == 0.0


def test_fit_to_height_guards_degenerate():
    assert gm.fit_to_height(None, 100).shape[:2] == (1, 1)
    assert gm.fit_to_height(np.zeros((0, 10, 3), dtype=np.uint8), 100).shape[:2] == (1, 1)


def test_now_ms_monotonic():
    a, b = gm.now_ms(), gm.now_ms()
    assert b >= a
