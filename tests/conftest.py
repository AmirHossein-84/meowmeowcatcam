"""Headless stubs so `import gesture_meme` works without cv2/mediapipe installed."""
import pathlib
import sys
import types
from unittest.mock import MagicMock


def _stub_cv2():
    if "cv2" in sys.modules:
        return
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2GRAY = 6
    cv2.COLOR_BGR2RGB = 4
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 16
    cv2.CAP_PROP_POS_FRAMES = 1

    def __getattr__(name, _m=cv2):
        mock = MagicMock(name=f"cv2.{name}")
        setattr(_m, name, mock)
        return mock

    cv2.__getattr__ = __getattr__
    sys.modules["cv2"] = cv2


def _stub_mediapipe():
    if "mediapipe" in sys.modules:
        return
    mp = types.ModuleType("mediapipe")
    mp.Image = object
    mp.ImageFormat = types.SimpleNamespace(SRGB=1)

    tasks = types.ModuleType("mediapipe.tasks")
    py = types.ModuleType("mediapipe.tasks.python")
    vision = types.ModuleType("mediapipe.tasks.python.vision")

    py.BaseOptions = object
    for name in (
        "FaceLandmarker",
        "FaceLandmarkerOptions",
        "HandLandmarker",
        "HandLandmarkerOptions",
        "RunningMode",
    ):
        setattr(vision, name, object)

    py.vision = vision
    tasks.python = py
    mp.tasks = tasks
    sys.modules["mediapipe"] = mp
    sys.modules["mediapipe.tasks"] = tasks
    sys.modules["mediapipe.tasks.python"] = py
    sys.modules["mediapipe.tasks.python.vision"] = vision


root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

_stub_cv2()
_stub_mediapipe()
