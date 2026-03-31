"""
Webcam capture + MediaPipe Hand Landmarker running on a background thread.

Publishes smooth hand velocity (not gesture states) to the render thread.
Hand position is filtered with exponential moving average before computing
velocity — this kills jitter without adding the lag of dead zones.
"""
import threading
import time
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# MediaPipe landmark indices
WRIST       = 0
THUMB_TIP   = 4
INDEX_MCP   = 5;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_TIP = 12
RING_MCP    = 13; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_TIP  = 20

# Smoothing factor for exponential moving average (0 = no smoothing, 1 = frozen)
_EMA_ALPHA = 0.45


class HandTracker:
    """
    Background thread: reads webcam frames, runs MediaPipe, publishes
    smooth velocity signals for camera control and building interaction.

    Gesture dict:
        mode       : 'idle' | 'move'
        pan_vel    : (vx, vz)  smooth velocity from single open hand
        orbit_vel  : float     horizontal rotation velocity
        zoom_vel   : float     zoom velocity from two-hand pinch
        grab       : bool      True when fist closed (building grab)
        grab_uv    : (u,v)|None  palm centre when grabbing
        point_uv   : (u,v)|None  index fingertip screen coords when pointing
        hands      : int       number of hands visible
    """

    MODEL_PATH = "hand_landmarker.task"

    def __init__(self, camera_index: int = 0, flip: bool = True):
        self._flip = flip
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._gesture = _idle_gesture()

        # Smoothed hand positions (EMA-filtered)
        self._smooth1: np.ndarray | None = None  # [x, y] normalised
        self._smooth2: np.ndarray | None = None
        self._prev_smooth1: np.ndarray | None = None
        self._prev_smooth2: np.ndarray | None = None
        self._prev_dist: float = 0.0  # for two-hand zoom

        # Frames since hand was last seen (for quick idle transition)
        self._idle_frames = 0

        self._running = True
        self._thread = threading.Thread(target=self._run,
                                        args=(camera_index,), daemon=True)
        self._thread.start()

    def get_state(self):
        """Returns (bgr_frame_or_None, gesture_dict) — safe to call from any thread."""
        with self._lock:
            frame   = self._frame.copy() if self._frame is not None else None
            gesture = dict(self._gesture)
        return frame, gesture

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    #  Background thread
    # ------------------------------------------------------------------

    def _run(self, camera_index: int):
        base_opts = mp_python.BaseOptions(model_asset_path=self.MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(opts)

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ts_ms = 0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            if self._flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms += 33
            result = landmarker.detect_for_video(mp_img, ts_ms)

            gesture = self._process(result)

            with self._lock:
                self._frame   = frame
                self._gesture = gesture

        cap.release()
        landmarker.close()

    # ------------------------------------------------------------------
    #  Core processing: smooth positions → velocities (no state machine)
    # ------------------------------------------------------------------

    def _process(self, result) -> dict:
        hands = result.hand_landmarks
        n = len(hands)

        if n == 0:
            self._idle_frames += 1
            if self._idle_frames > 3:
                # Reset smoothing so next hand appearance starts fresh
                self._smooth1 = None
                self._smooth2 = None
                self._prev_smooth1 = None
                self._prev_smooth2 = None
                self._prev_dist = 0.0
            gesture = _idle_gesture()
            gesture['landmarks'] = []
            return gesture

        self._idle_frames = 0

        if n >= 2:
            gesture = self._process_two_hands(hands[0], hands[1], hands)
        else:
            # --- Single hand ---
            lms = hands[0]
            raw = np.array(_palm_centre(lms), dtype=np.float64)

            # EMA filter
            if self._smooth1 is None:
                self._smooth1 = raw.copy()
                self._prev_smooth1 = raw.copy()
                self._smooth2 = None
                self._prev_smooth2 = None
                gesture = _idle_gesture()
                gesture['landmarks'] = hands
                return gesture  # first frame: no velocity yet

            self._prev_smooth1 = self._smooth1.copy()
            self._smooth1 = self._smooth1 * _EMA_ALPHA + raw * (1 - _EMA_ALPHA)

            # Velocity = smoothed position delta
            vel = self._smooth1 - self._prev_smooth1

            grab_conf = _grab_confidence(lms)
            pinch = _is_pinching(lms)

            if _is_pointing(lms):
                tip = lms[INDEX_TIP]
                point_dir = _pointing_direction(lms)
                gesture = {
                    'mode': 'move',
                    'pan_vel': (0.0, 0.0),
                    'orbit_vel': 0.0,
                    'zoom_vel': 0.0,
                    'clench': grab_conf,
                    'grab': False,
                    'grab_uv': None,
                    'point_uv': (tip.x, tip.y),
                    'point_dir': point_dir,
                    'hands': 1,
                    'landmarks': hands,
                }
            elif pinch or grab_conf > 0.55:
                palm = _palm_centre(lms)
                gesture = {
                    'mode': 'move',
                    'pan_vel': (float(vel[0]), float(vel[1])),
                    'orbit_vel': 0.0,
                    'zoom_vel': 0.0,
                    'clench': grab_conf,
                    'grab': True,
                    'grab_uv': palm,
                    'point_uv': None,
                    'point_dir': None,
                    'hands': 1,
                    'landmarks': hands,
                }
            else:
                # Open hand → pan camera (hand movement → camera movement)
                gesture = {
                    'mode': 'move',
                    'pan_vel': (float(vel[0]), float(vel[1])),
                    'orbit_vel': 0.0,
                    'zoom_vel': 0.0,
                    'clench': grab_conf,
                    'grab': False,
                    'grab_uv': None,
                    'point_uv': None,
                    'point_dir': None,
                    'hands': 1,
                    'landmarks': hands,
                }

        return gesture

    def _process_two_hands(self, lms1, lms2, hands) -> dict:
        raw1 = np.array(_palm_centre(lms1), dtype=np.float64)
        raw2 = np.array(_palm_centre(lms2), dtype=np.float64)

        # EMA filter both hands
        if self._smooth1 is None:
            self._smooth1 = raw1.copy()
            self._smooth2 = raw2.copy()
            self._prev_smooth1 = raw1.copy()
            self._prev_smooth2 = raw2.copy()
            self._prev_dist = float(np.linalg.norm(raw2 - raw1))
            gesture = _idle_gesture()
            gesture['landmarks'] = hands
            return gesture

        if self._smooth2 is None:
            self._smooth2 = raw2.copy()
            self._prev_smooth2 = raw2.copy()
            self._prev_dist = float(np.linalg.norm(raw2 - raw1))
            gesture = _idle_gesture()
            gesture['landmarks'] = hands
            return gesture

        self._prev_smooth1 = self._smooth1.copy()
        self._prev_smooth2 = self._smooth2.copy()
        self._smooth1 = self._smooth1 * _EMA_ALPHA + raw1 * (1 - _EMA_ALPHA)
        self._smooth2 = self._smooth2 * _EMA_ALPHA + raw2 * (1 - _EMA_ALPHA)

        # Per-hand gesture states for mixed two-hand mode
        point1 = _is_pointing(lms1)
        point2 = _is_pointing(lms2)
        clench1 = _grab_confidence(lms1)
        clench2 = _grab_confidence(lms2)

        # Two-hand mode: one hand points for movement, other hand clenches for zoom
        if point1 or point2:
            if point1 and not point2:
                # Hand 1 pointing, hand 2 clenching for zoom
                pointer = lms1
                zoom_clench = clench2  # Use hand 2 clench for zoom (0=open/zoom out, 1=closed/zoom in)
            elif point2 and not point1:
                # Hand 2 pointing, hand 1 clenching for zoom
                pointer = lms2
                zoom_clench = clench1  # Use hand 1 clench for zoom
            else:
                # Both pointing: use hand 1 for direction
                pointer = lms1
                zoom_clench = max(clench2, clench1)

            tip = pointer[INDEX_TIP]
            return {
                'mode': 'move',
                'pan_vel': (0.0, 0.0),
                'orbit_vel': 0.0,
                'zoom_vel': 0.0,
                'clench': zoom_clench,
                'grab': False,
                'grab_uv': None,
                'point_uv': (tip.x, tip.y),
                'point_dir': _pointing_direction(pointer),
                'hands': 2,
                'landmarks': hands,
            }

        # Two-hand distance → zoom/orbit (fallback old behavior)
        dist = float(np.linalg.norm(self._smooth2 - self._smooth1))
        zoom_vel = 0.0
        if self._prev_dist > 0.01:
            # Positive = hands moving apart = zoom in (distance decreases)
            zoom_vel = (self._prev_dist - dist) / self._prev_dist
        self._prev_dist = dist

        # Midpoint velocity → orbit (horizontal component only)
        mid = (self._smooth1 + self._smooth2) / 2
        prev_mid = (self._prev_smooth1 + self._prev_smooth2) / 2
        mid_vel = mid - prev_mid
        orbit_vel = float(mid_vel[0])  # horizontal sweep → orbit

        return {
            'mode': 'move',
            'pan_vel': (0.0, 0.0),
            'orbit_vel': orbit_vel,
            'zoom_vel': zoom_vel,
            'grab': False,
            'grab_uv': None,
            'point_uv': None,
            'point_dir': None,
            'hands': 2,
            'landmarks': hands,
        }


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _idle_gesture():
    return {
        'mode': 'idle',
        'pan_vel': (0.0, 0.0),
        'orbit_vel': 0.0,
        'zoom_vel': 0.0,
        'grab': False,
        'grab_uv': None,
        'point_uv': None,
        'hands': 0,
        'landmarks': [],
    }


def _palm_centre(lms):
    xs = [lms[i].x for i in (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    ys = [lms[i].y for i in (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    return (sum(xs) / 4, sum(ys) / 4)


def _grab_confidence(lms) -> float:
    """1.0 = fully closed fist, 0.0 = open hand."""
    wrist_x, wrist_y = lms[WRIST].x, lms[WRIST].y
    mid_x,   mid_y   = lms[MIDDLE_MCP].x, lms[MIDDLE_MCP].y
    cdx = wrist_x - mid_x
    cdy = wrist_y - mid_y
    clen = math.hypot(cdx, cdy) or 1e-6
    cdx /= clen; cdy /= clen

    def curl(mcp, tip):
        dx = lms[tip].x - lms[mcp].x
        dy = lms[tip].y - lms[mcp].y
        return dx * cdx + dy * cdy

    scores = [
        curl(INDEX_MCP,  INDEX_TIP),
        curl(MIDDLE_MCP, MIDDLE_TIP),
        curl(RING_MCP,   RING_TIP),
        curl(PINKY_MCP,  PINKY_TIP),
    ]
    avg = sum(scores) / len(scores)
    return float(np.clip((avg - 0.01) / 0.10, 0.0, 1.0))


def _is_pinching(lms) -> bool:
    """True when thumb and index fingertips are close together."""
    dx = lms[INDEX_TIP].x - lms[THUMB_TIP].x
    dy = lms[INDEX_TIP].y - lms[THUMB_TIP].y
    dz = lms[INDEX_TIP].z - lms[THUMB_TIP].z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    palm_w = math.hypot(
        lms[INDEX_MCP].x - lms[PINKY_MCP].x,
        lms[INDEX_MCP].y - lms[PINKY_MCP].y
    )
    palm_w = max(palm_w, 1e-3)
    return dist < 0.20 * palm_w


def _pinch_strength(lms) -> float:
    """0.0 = open hand, 1.0 = fully pinched thumb-index."""
    dx = lms[INDEX_TIP].x - lms[THUMB_TIP].x
    dy = lms[INDEX_TIP].y - lms[THUMB_TIP].y
    dz = lms[INDEX_TIP].z - lms[THUMB_TIP].z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    palm_w = math.hypot(
        lms[INDEX_MCP].x - lms[PINKY_MCP].x,
        lms[INDEX_MCP].y - lms[PINKY_MCP].y
    )
    palm_w = max(palm_w, 1e-3)

    # 0.20*palm_w = pinched threshold, 0.40*pal_w = open reference
    norm = (0.40 * palm_w - dist) / (0.20 * palm_w)
    return float(np.clip(norm, 0.0, 1.0))


def _pointing_direction(lms):
    """Map index finger pose to (forward, right, elevation) deltas."""
    wrist = lms[WRIST]
    tip = lms[INDEX_TIP]
    dx = tip.x - wrist.x    # right-left in screen
    dy = tip.y - wrist.y    # down-up in screen (y increases downward)
    dz = tip.z - wrist.z    # negative when towards camera

    forward = np.clip(-dz * 12.0, -1.0, 1.0)
    right   = np.clip(dx * 12.0, -1.0, 1.0)

    elev = 0.0
    if dy > 0.02:
        elev = -np.clip((dy - 0.02) * 20.0, 0.0, 1.0)  # finger down lowers elevation
    elif dy < -0.02:
        elev = np.clip((-dy - 0.02) * 20.0, 0.0, 1.0)  # finger up raises elevation

    return (forward, right, elev)


def _is_pointing(lms) -> bool:
    """True when only the index finger is clearly extended and hand is open."""
    # Index finger must be significantly extended
    index_ext = lms[INDEX_TIP].y < lms[INDEX_MCP].y - 0.08
    
    # All other fingers must be clearly curled
    middle_curl = lms[MIDDLE_TIP].y > lms[MIDDLE_MCP].y + 0.02
    ring_curl   = lms[RING_TIP].y   > lms[RING_MCP].y + 0.02
    pinky_curl  = lms[PINKY_TIP].y  > lms[PINKY_MCP].y + 0.02
    
    # Thumb should not be pressing against other fingers (not a fist)
    thumb_not_closed = lms[THUMB_TIP].x > lms[WRIST].x - 0.15
    
    return index_ext and middle_curl and ring_curl and pinky_curl and thumb_not_closed
