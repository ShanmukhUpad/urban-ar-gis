"""
Urban AR GIS Viewer
===================
Gesture-controlled 3D urban viewer with live webcam background.

Usage:
    # Synthetic city (no data needed):
    python main.py

    # Real city from OpenStreetMap:
    python main.py --place "Times Square, New York"
    python main.py --lat 40.758 --lon -73.985 --radius 500

    # Local GeoJSON file:
    python main.py --geojson data/my_city.geojson --lat 40.758 --lon -73.985

Gestures:
    Finger pointing toward screen -> move forward
    Finger pointing left/right     -> move left/right
    Finger pointing down           -> lower camera elevation
    Open hand (move)               -> pan camera
    Closed fist (move)             -> grab and drag building
    Pinch (thumb+index)            -> grab and drag building
    Two hands                      -> pinch in/out to zoom
    As hand closes to fist -> zoom in; open hand -> zoom out (full map)
    Index extended                 -> point to inspect building (info shown top-left)
    R                              -> Reset camera
    +/-                            -> Adjust sun angle (time of day)
    Q / Esc                        -> Quit
"""
import sys
import math
import time
import argparse
import threading
import numpy as np
import moderngl_window as mglw
from moderngl_window.context.base import BaseKeys

from src.camera      import OrbitCamera
from src.hand_tracker import HandTracker
from src.gis_loader  import fetch_osm, load_geojson, generate_sample_city
from src.renderer    import Renderer


# ----------------------------------------------------------------
#  Parse CLI
# ----------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Urban AR GIS Viewer")
    p.add_argument("--city",    type=str,   help="Pre-fetched city key (e.g. nyc, chicago)")
    p.add_argument("--place",   type=str,   help="OSM place name (live fetch)")
    p.add_argument("--lat",     type=float, help="Centre latitude")
    p.add_argument("--lon",     type=float, help="Centre longitude")
    p.add_argument("--radius",  type=int,   default=500, help="Fetch radius (metres)")
    p.add_argument("--geojson", type=str,   help="Local GeoJSON file")
    p.add_argument("--no-cam",  action="store_true", help="Disable webcam")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # strip our flags so mglw doesn't choke
    return args


# ----------------------------------------------------------------
#  Named places → lat/lon (simple lookup; use --lat/--lon for others)
# ----------------------------------------------------------------
PLACES = {
    "times square, new york": (40.7580, -73.9855),
    "canary wharf, london":   (51.5054, -0.0235),
    "shibuya, tokyo":         (35.6581, 139.7017),
    "la defense, paris":      (48.8919,  2.2385),
    "downtown chicago":       (41.8827, -87.6233),
    "downtown toronto":       (43.6484, -79.3820),
    "cbd singapore":          (1.2840,  103.8510),
}


# ----------------------------------------------------------------
#  Load a GeoJSON file saved by fetch_city.py
# ----------------------------------------------------------------
def _load_cached_city(city_key: str):
    import json, os
    path = os.path.join("data", f"{city_key}.geojson")
    if not os.path.exists(path):
        sys.exit(
            f"No cached data for '{city_key}'.\n"
            f"Run first:  python fetch_city.py {city_key}"
        )
    with open(path) as f:
        gj = json.load(f)
    centre = gj.get("_centre", {})
    lat = centre.get("lat", 0.0)
    lon = centre.get("lon", 0.0)
    print(f"Loaded cached city '{city_key}' (centre {lat:.4f}, {lon:.4f})")
    return load_geojson(path, lat, lon)


# ----------------------------------------------------------------
#  Application
# ----------------------------------------------------------------
class UrbanGIS(mglw.WindowConfig):
    gl_version    = (3, 3)
    title         = "Urban AR GIS"
    window_size   = (1280, 720)
    resizable     = True

    # Injected before mglw.run_window_config()
    _args = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = UrbanGIS._args

        # ---- Load GIS data (blocking, runs before first frame) ----
        print("Loading city data …")
        if args.city:
            scene = _load_cached_city(args.city)
        elif args.geojson and args.lat and args.lon:
            scene = load_geojson(args.geojson, args.lat, args.lon)
        elif args.place:
            key = args.place.lower()
            if key in PLACES:
                lat, lon = PLACES[key]
            elif args.lat and args.lon:
                lat, lon = args.lat, args.lon
            else:
                print(f"Unknown place '{args.place}', using synthetic city.")
                scene = generate_sample_city()
                lat, lon = 0.0, 0.0
                args.place = None
            if args.place:
                scene = fetch_osm(lat, lon, args.radius)
        elif args.lat and args.lon:
            scene = fetch_osm(args.lat, args.lon, args.radius)
        else:
            print("No data source specified — generating synthetic city.")
            scene = generate_sample_city()

        # ---- Renderer -----------------------------------------
        self.renderer = Renderer(self.ctx, self.wnd.size)
        self.renderer.upload_scene(scene)

        # ---- Camera — 70-ft-tall person perspective ---------------
        extent = getattr(scene, 'extent_m', 500.0)
        self.camera = OrbitCamera(distance=extent * 0.5)
        self.camera.elevation = 25.0   # low angle, looking across
        self._scene = scene  # keep reference for export

        # Set spawn point — convert lat/lon to local world-space XZ
        SPAWN_LAT =  40.10725
        SPAWN_LON = -88.22807
        clat, clon = scene.center_latlon
        cos_lat = math.cos(math.radians(clat))
        spawn_x = (SPAWN_LON - clon) * cos_lat * 111_320.0
        spawn_z = -(SPAWN_LAT - clat) * 110_540.0
        self.camera.target = np.array([spawn_x, 0.0, spawn_z], dtype=np.float32)

        # ---- Hand tracker -------------------------------------
        self.hand_tracker = None
        if not (args.no_cam if args else False):
            try:
                self.hand_tracker = HandTracker()
                print("Hand tracker running (webcam background enabled).")
            except Exception as e:
                print(f"Could not start hand tracker: {e}")

        # Sun angle
        self._sun_angle = 45.0   # degrees above horizon

        self._keys_held = set()
        self._last_time = time.time()
        print("Ready. " + _controls_hint())

    # ---- Render loop ----------------------------------------

    def on_render(self, time_: float, frame_time: float):
        dt = min(frame_time, 0.1)  # cap at 100ms to avoid big jumps

        # Get webcam + gesture
        cam_frame, gesture = (None, None)
        if self.hand_tracker:
            cam_frame, gesture = self.hand_tracker.get_state()

        # Update camera
        self.camera.apply_gesture(gesture, dt)
        self.camera.apply_keyboard(self._keys_held, dt)

        # Update sun direction
        a = math.radians(self._sun_angle)
        self.renderer.sun_dir = np.array([0.5, math.sin(a), 0.3], np.float32)
        norm = np.linalg.norm(self.renderer.sun_dir)
        if norm > 0:
            self.renderer.sun_dir /= norm

        self.renderer.draw(cam_frame, self.camera, gesture or {})

    # ---- Events ---------------------------------------------

    def on_resize(self, width: int, height: int):
        self.renderer.resize(width, height)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        # Map movement keys to string identifiers for the camera
        _KEY_MAP = {
            keys.W: 'W', keys.A: 'A', keys.S: 'S', keys.D: 'D',
            keys.LEFT: 'LEFT', keys.RIGHT: 'RIGHT',
            keys.UP: 'UP', keys.DOWN: 'DOWN',
        }
        if key in _KEY_MAP:
            if action == keys.ACTION_PRESS:
                self._keys_held.add(_KEY_MAP[key])
            elif action == keys.ACTION_RELEASE:
                self._keys_held.discard(_KEY_MAP[key])
            return

        if action != keys.ACTION_PRESS:
            return
        if key in (keys.Q, keys.ESCAPE):
            self.wnd.close()
        elif key == keys.R:
            extent = self.camera.distance
            self.camera = OrbitCamera(distance=extent)
            self.camera.elevation = 25.0
            self._keys_held.clear()
            print("Camera reset.")
        elif key == keys.E:
            self._export_layout()
        elif key == keys.EQUAL or key == getattr(keys, 'NUM_ADD', None):
            self._sun_angle = min(self._sun_angle + 5, 90)
            print(f"Sun elevation: {self._sun_angle:.0f}")
        elif key == keys.MINUS or key == getattr(keys, 'NUM_SUBTRACT', None):
            self._sun_angle = max(self._sun_angle - 5, 5)
            print(f"Sun elevation: {self._sun_angle:.0f}")

    def _export_layout(self):
        """Export the current building layout as GeoJSON (reflecting any moves)."""
        import json, os
        aabbs = self.renderer.building_aabbs
        if not aabbs:
            print("No buildings to export.")
            return

        clat, clon = self._scene.center_latlon
        cos_lat = math.cos(math.radians(clat))

        features = []
        for aabb in aabbs:
            # Convert AABB back to lat/lon
            cx = (aabb['min_x'] + aabb['max_x']) / 2
            cz = (aabb['min_z'] + aabb['max_z']) / 2
            hw = (aabb['max_x'] - aabb['min_x']) / 2
            hz = (aabb['max_z'] - aabb['min_z']) / 2

            def to_lonlat(x, z):
                lon = clon + x / (cos_lat * 111_320.0)
                lat = clat - z / 110_540.0
                return [lon, lat]

            coords = [
                to_lonlat(cx - hw, cz - hz),
                to_lonlat(cx + hw, cz - hz),
                to_lonlat(cx + hw, cz + hz),
                to_lonlat(cx - hw, cz + hz),
                to_lonlat(cx - hw, cz - hz),
            ]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "height": round(aabb['height'], 1),
                    "name": aabb.get('name', ''),
                },
            })

        gj = {
            "type": "FeatureCollection",
            "features": features,
            "_centre": {"lat": clat, "lon": clon},
        }

        out_path = os.path.join("data", "modified_layout.geojson")
        os.makedirs("data", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(gj, f, indent=2)
        print(f"Exported {len(features)} buildings -> {out_path}")

    def close(self):
        if self.hand_tracker:
            self.hand_tracker.stop()


# ----------------------------------------------------------------
#  Entry point
# ----------------------------------------------------------------

def _controls_hint():
    return ("WASD=move  Arrows=orbit  "
            "Open hand=pan  Fist=grab building  Two-hands=zoom  "
            "Index=inspect  E=export  R=reset  +/-=sun  Q=quit")


if __name__ == "__main__":
    args = parse_args()
    UrbanGIS._args = args
    mglw.run_window_config(UrbanGIS)
