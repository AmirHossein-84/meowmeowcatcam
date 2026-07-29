import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping -----------------------------------------------------
// Each gesture maps to one or more meme images. When a gesture has more
// than one image, one is picked at random each time the gesture is newly
// (re)triggered, so repeated gestures don't always show the same frame.
const GESTURE_MEMES = {
  rockstar: ["memes/cat.jpg"],
  default: ["memes/pokercat.jpg"],
  oneFingerUp: ["memes/profcat.jpg", "memes/professorcat.jpg"],
  fist: ["memes/punchcat.jpg"],
  shhh: ["memes/shhcat.jpg"],
  twoFingersTogether: ["memes/uwucat.jpg", "memes/uwucatt.jpg"],
};

// how many consecutive frames a gesture must hold before we switch to it
const STABLE_FRAMES_REQUIRED = 5;
// if no hand / no gesture is seen for this long, fall back to default
const DEFAULT_FALLBACK_MS = 600;
// how long we trust a stale face box after the face detector loses the face
// (e.g. hand covering the mouth during a shush)
const FACE_STALE_MS = 1200;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, t }

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 3D-aware geometry helpers -----------------------------------------
// Using z (depth) as well as x/y makes these tests far more robust to hand
// rotation, foreshortening, and motion blur than a plain 2D/wrist-distance
// check would be.
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// a finger is "extended" if its two segments (mcp->pip, pip->tip) point in
// roughly the same direction; "curled" if it folds back sharply.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; // wrist -> middle mcp

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  // thumb + pinky spread apart from each other = shaka/rock-on shape.
  // tucked thumb sits close to the pinky-side of the palm; an abducted
  // thumb sticks straight out and this distance grows a lot.
  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return { indexUp, middleUp, ringUp, pinkyUp, thumbOut, curledCount, handScale, indexTip: lm[8] };
}

function updateFace(faceResult) {
  const now = performance.now();
  if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(f[234], f[454]);
    lastFace = { mouthCenter, faceWidth, t: now };
  }
}

// a hand is "pointing" if only the index finger is extended (thumb can be
// either way) - the shape both hands make in the finger-tips-touching pose.
function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function decideGesture(handResult) {
  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  // two fingers pointing together: BOTH hands present, each pointing with
  // just the index finger, fingertips close to each other in the frame.
  if (hands.length === 2 && isPointing(hands[0]) && isPointing(hands[1])) {
    const avgScale = (hands[0].handScale + hands[1].handScale) / 2;
    const tipGap = dist(hands[0].indexTip, hands[1].indexTip) / avgScale;
    if (tipGap < 1.4) {
      return "twoFingersTogether";
    }
  }

  const h = hands[0];

  // 1. fist / punch: everything curled
  if (h.curledCount === 4) {
    return "fist";
  }

  // 2. rockstar / shaka: thumb + pinky out, index/middle/ring curled
  if (h.thumbOut && h.pinkyUp && !h.indexUp && !h.middleUp && !h.ringUp) {
    return "rockstar";
  }

  // 3/4. index-only shapes: shhh (near mouth) vs one-finger-up (away from face)
  if (h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp) {
    const now = performance.now();
    if (lastFace && now - lastFace.t < FACE_STALE_MS) {
      const d = dist(h.indexTip, lastFace.mouthCenter) / lastFace.faceWidth;
      if (d < 0.55) {
        return "shhh";
      }
    }
    return "oneFingerUp";
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    // debounce: require a gesture to be seen for several consecutive
    // frames before we commit to it, to avoid flicker between frames
    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }
  }
  requestAnimationFrame(loop);
}

init().catch((err) => console.error(err));
