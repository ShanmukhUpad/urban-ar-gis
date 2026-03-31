"""
Photorealistic Raster Map Interactor using PyVista, MediaPipe, and OpenCV.

This script creates a 3D scene with a satellite image textured plane representing
the UIUC campus. Hand tracking with MediaPipe allows interaction with the map
using finger pointing gestures, with a dynamic sphere cursor following the index finger.
"""

import pyvista as pv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import cv2
import numpy as np
import threading
import time

# Import GIS loader for realistic buildings
from src.gis_loader import load_geojson, generate_sample_city


class RasterMapInteractor:
    """
    Interactive 3D raster map using hand tracking for cursor control.
    """

    def __init__(self, satellite_image_path='uiuc_satellite.png'):
        """
        Initialize the raster map interactor.

        Args:
            satellite_image_path (str): Path to the satellite image file
        """
        # Load GIS scene with realistic buildings
        print("Loading UIUC city data with realistic buildings...")
        try:
            # Try to load UIUC data
            scene = load_geojson('data/uiuc.geojson', 40.10725, -88.22807)
            print("Loaded UIUC GeoJSON data")
        except Exception as e:
            print(f"Could not load UIUC data: {e}")
            print("Generating sample city instead...")
            scene = generate_sample_city()

        # Initialize PyVista plotter
        self.plotter = pv.Plotter(window_size=(1200, 800))

        # Create plane with satellite texture
        self.plane = pv.Plane(i_size=1000, j_size=1000, i_resolution=1, j_resolution=1)
        try:
            self.texture = pv.read_texture(satellite_image_path)
        except FileNotFoundError:
            print(f"Warning: {satellite_image_path} not found. Using default texture.")
            self.texture = None

        # Add the textured plane to the scene
        if self.texture:
            self.plane_actor = self.plotter.add_mesh(self.plane, texture=self.texture)
        else:
            self.plane_actor = self.plotter.add_mesh(self.plane, color='lightblue')

        # Add realistic 3D buildings
        self._add_buildings_to_scene(scene)

        # Add cursor sphere (initially hidden)
        self.cursor = pv.Sphere(radius=5, center=(0, 0, 10))
        self.cursor_actor = self.plotter.add_mesh(self.cursor, color='red', opacity=0.8)

        # Set up camera for top-down view
        self.plotter.camera_position = 'xy'
        self.plotter.camera.zoom(1.2)

        # Initialize MediaPipe hand tracking
        base_opts = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.hands = mp_vision.HandLandmarker.create_from_options(opts)

        # Initialize OpenCV webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Set webcam resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Threading variables
        self.finger_pos = None
        self.running = False
        self.tracking_thread = None

        # Store plane bounds for coordinate mapping
        self.plane_bounds = self.plane.bounds

    def _add_buildings_to_scene(self, scene):
        """Add realistic 3D buildings from UrbanScene to PyVista plotter."""
        if len(scene.building_verts) == 0:
            print("No building data found in scene")
            return

        # Extract vertices, normals, and colors from the scene data
        # building_verts format: [x, y, z, nx, ny, nz, r, g, b] per vertex
        vertices = scene.building_verts[:, :3]  # x, y, z positions
        colors = scene.building_verts[:, 6:9]   # r, g, b colors

        # Create PyVista mesh from vertices and faces
        # building_idx contains triangle indices in groups of 3
        faces = []
        for i in range(0, len(scene.building_idx), 3):
            if i + 2 < len(scene.building_idx):
                i1, i2, i3 = scene.building_idx[i:i+3]
                faces.extend([3, int(i1), int(i2), int(i3)])

        if faces:
            # Create mesh with vertices and faces
            building_mesh = pv.PolyData(vertices, faces)

            # Add colors to the mesh
            colors_uint8 = (colors * 255).astype(np.uint8)
            building_mesh.point_data['colors'] = colors_uint8

            # Add the building mesh to the plotter
            self.plotter.add_mesh(building_mesh, rgb=True, show_edges=False)

            print(f"Added {len(scene.building_aabbs)} realistic buildings to the scene")
        else:
            print("No valid building faces found")

        # Optionally add roads if available
        if len(scene.road_verts) > 0:
            road_vertices = scene.road_verts[:, :3]
            road_colors = scene.road_verts[:, 6:9]

            # Roads are typically lines, not triangles
            # Create lines from road vertices
            lines = []
            for i in range(0, len(scene.road_idx), 2):
                if i + 1 < len(scene.road_idx):
                    lines.extend([2, scene.road_idx[i], scene.road_idx[i+1]])

            if lines:
                road_mesh = pv.PolyData(road_vertices, lines=lines)
                road_colors_uint8 = (road_colors * 255).astype(np.uint8)
                road_mesh.point_data['colors'] = road_colors_uint8
                self.plotter.add_mesh(road_mesh, rgb=True, line_width=3)
                print("Added roads to the scene")

    def start_tracking(self):
        """Start the hand tracking thread."""
        self.running = True
        self.tracking_thread = threading.Thread(target=self._track_hands, daemon=True)
        self.tracking_thread.start()

        # Add render callback for updating cursor
        self.plotter.add_on_render_callback(self._update_cursor)

    def _track_hands(self):
        """Background thread for hand tracking."""
        ts_ms = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process frame with MediaPipe
            ts_ms += 33  # Approximate 30 FPS
            results = self.hands.detect_for_video(mp_image, ts_ms)

            # Extract index finger tip position
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Get index finger tip (landmark 8)
                    tip = hand_landmarks[8]
                    # Store normalized coordinates (0.0 to 1.0)
                    self.finger_pos = (tip.x, tip.y)
                    break  # Use first detected hand
            else:
                self.finger_pos = None

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

    def _update_cursor(self, *args):
        """Update cursor position based on finger tracking."""
        if self.finger_pos and self.running:
            x_norm, y_norm = self.finger_pos

            # Map normalized coordinates to 3D world coordinates on the plane
            # Assuming plane is in XY plane at z=0
            x_world = self.plane_bounds[0] + x_norm * (self.plane_bounds[1] - self.plane_bounds[0])
            y_world = self.plane_bounds[2] + y_norm * (self.plane_bounds[3] - self.plane_bounds[2])
            z_world = 0.1  # Slightly above plane to avoid z-fighting

            # Update cursor position
            self.cursor.translate([x_world, y_world, z_world], inplace=True)

            # Ensure cursor is visible
            if self.cursor_actor.GetVisibility() == 0:
                self.cursor_actor.VisibilityOn()

        elif self.cursor_actor.GetVisibility() == 1:
            # Hide cursor when no finger detected
            self.cursor_actor.VisibilityOff()

    def run(self):
        """Run the interactive application."""
        self.start_tracking()

        # Show the plotter (blocking call)
        self.plotter.show()

        # Cleanup when plotter is closed
        self.stop()

    def stop(self):
        """Stop tracking and cleanup resources."""
        self.running = False
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)

        if self.cap.isOpened():
            self.cap.release()

        self.hands.close()


def main():
    """Main entry point."""
    try:
        app = RasterMapInteractor()
        app.run()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()