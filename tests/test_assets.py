import re
from pathlib import Path

import gesture_meme as gm

ROOT = Path(gm.__file__).parent
MEMES = ROOT / "memes"

EXPECTED = {
    "rockstar": ["cat.jpg"],
    "default": ["pokercat.jpg"],
    "oneFingerUp": ["profcat.jpg", "professorcat.jpg"],
    "fist": ["punchcat.jpg"],
    "shhh": ["shhcat.jpg"],
    "twoFingersTogether": ["uwucat.jpg", "uwucatt.jpg", "fingers-together-muehehe.jpg"],
    "handCoverFace": ["hand-cover-face.jpg"],
    "crashOutCat": ["crashout-cat.jpg"],
    "twoHandsOnHead": ["two-hands-on-head.jpg"],
    "handStretchedOut": ["hand-stretched-out-palm-up.jpg"],
    "sideEyeCat": ["side-eye-cat.jpg"],
    "sideEyeDownCat": ["side-eye-down.png"],
    "mouthOpenCat": ["laugh-and-point.jpg"],
    "huhCat": ["huh.png"],
    "danceCat": ["two-palms-up.mov"],
    "spinCat": ["spin-cat.mov"],
}
ALLOWED_MOV = {"spin-cat.mov", "two-palms-up.mov"}
DOCUMENTED_UNUSED = {"iunno-cat.jpg"}  # kept intentionally, no gesture maps to it yet


def test_manifest_matches_expected_kebab_names():
    assert gm.GESTURE_MEMES == EXPECTED


def test_every_manifest_file_exists():
    missing = [f for files in gm.GESTURE_MEMES.values() for f in files
               if not (MEMES / f).is_file()]
    assert not missing, f"missing meme files: {missing}"


def test_no_space_laden_filenames_remain():
    spaced = [p.name for p in MEMES.iterdir() if " " in p.name]
    assert not spaced, f"space-laden filenames remain: {spaced}"


def test_no_orphan_mov_except_allowed():
    movs = {p.name for p in MEMES.iterdir() if p.suffix.lower() == ".mov"}
    assert movs <= ALLOWED_MOV, f"orphan .mov: {movs - ALLOWED_MOV}"


def test_no_orphan_images_except_documented():
    referenced = {f for files in gm.GESTURE_MEMES.values() for f in files}
    on_disk = {p.name for p in MEMES.iterdir() if p.is_file()}
    assert on_disk - referenced <= DOCUMENTED_UNUSED | ALLOWED_MOV, \
        f"unexpected orphans: {on_disk - referenced - DOCUMENTED_UNUSED - ALLOWED_MOV}"


def test_video_gestures_map_to_mov():
    for g in gm.VIDEO_GESTURES:
        assert gm.GESTURE_MEMES[g][0].endswith(".mov"), g


def test_js_keys_subset_of_python_keys():
    text = (ROOT / "app.js").read_text(encoding="utf-8")
    m = re.search(r"const GESTURE_MEMES = \{(.*?)\};", text, re.DOTALL)
    assert m, "GESTURE_MEMES block not found in app.js"
    js_keys = set(re.findall(r"^\s*(\w+)\s*:", m.group(1), re.MULTILINE))
    # Browser build is a scoped subset: no spin/dance/huh/mouth-open/sideEyeDown
    assert js_keys <= set(gm.GESTURE_MEMES), f"JS-only keys: {js_keys - set(gm.GESTURE_MEMES)}"


def test_gestures_json_keys_match_python():
    import json

    data = json.loads((ROOT / "gestures.json").read_text(encoding="utf-8"))
    json_ids = {g["id"] for g in data["gestures"]}
    assert json_ids == set(gm.GESTURE_MEMES), f"gestures.json drift: {json_ids ^ set(gm.GESTURE_MEMES)}"
    for g in data["gestures"]:
        assert gm.GESTURE_MEMES[g["id"]] == g["memes"], g["id"]
