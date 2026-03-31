# Urban AR GIS

An interactive hand-tracked 3D urban visualization system that combines GeoJSON city data with satellite imagery and realistic building models. Control the scene with hand gestures detected via webcam.

## Features

✨ **Hand Gesture Control**
- **Left hand**: Open/close to zoom (progressive zoom based on clench)
- **Right hand**: Point to pan and move around the scene
- **Grab gesture**: Pick up and move individual buildings

📍 **Photorealistic Rendering**
- Satellite base map (2816×3072 resolution)
- 3D realistic buildings from GeoJSON data
- Procedural windows with day/night lighting
- Dynamic atmospheric haze and distance fade

🎯 **Interactive Features**
- Point at buildings to inspect metadata (height, dimensions, density stats)
- Grab buildings with closed fist to relocate them
- Real-time hand landmark visualization for debugging
- Input status HUD showing gesture mode and hand count

🌍 **City Support**
- UIUC (University of Illinois) with 973 buildings
- NYC ready (expandable to any GeoJSON source)
- Automatic satellite tile loading

## Requirements

- **Python 3.10+**
- **Webcam** (for hand tracking)
- **GPU** (recommended for smooth 3D rendering)

### Dependencies

```
moderngl>=0.14.0
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
shapely>=2.0.0
pyproj>=3.6.0
pyvista>=0.43.0
mediapipe>=0.10.0
imageio>=2.33.0
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/ShanmukhUpad/urban-ar-gis.git
cd urban-ar-gis
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Hand Landmark Model
The `hand_landmarker.task` file is included in the repo. If missing, download from:
```
https://storage.googleapis.com/mediapipe-tasks/python/vision/latest/hand_landmarker.task
```

## Usage

### Basic Launch
```bash
python main.py --city uiuc
```

### With NYC Data
```bash
python main.py --city nyc
```

### Alternative: Raster Map Viewer
PyVista-based standalone satellite map:
```bash
python raster_map_interactor.py
```

## Hand Gestures

| Gesture | Action | Hand |
|---------|--------|------|
| **Open hand** | Zoom out | Left |
| **Closed fist** | Zoom in | Left |
| **Pointing** (index extended) | Pan/orbit | Right |
| **Closed fist + point area** | Grab building | Both |
| Release fist | Drop building | Both |

### Pointing Detection
- Index finger extended >0.08
- Other fingers curled (<0.02 distance from palm)
- Thumb below hand (positioned away from finger tips)

## Project Structure

```
urban-ar-gis/
├── main.py                    # Main entry point with ModernGL renderer
├── raster_map_interactor.py  # PyVista-based satellite map viewer
├── requirements.txt           # Python dependencies
├── hand_landmarker.task      # MediaPipe hand detection model
├── .gitignore                # Git exclusion rules
├── src/
│   ├── __init__.py
│   ├── camera.py             # Orbit camera with projection matrices
│   ├── hand_tracker.py       # MediaPipe integration + gesture recognition
│   ├── gis_loader.py         # GeoJSON parsing + mesh generation
│   ├── renderer.py           # ModernGL shaders + drawing
│   └── uiuc.py              # UIUC-specific utilities
└── data/
    ├── uiuc.geojson         # UIUC buildings + roads (973 buildings)
    ├── uiuc_tiles.json      # Satellite map bounds metadata
    ├── uiuc_tiles.png       # Satellite imagery (2816×3072)
    ├── nyc.geojson          # NYC buildings data
    └── README.md            # Data source documentation
```

## Configuration

### Basemap Changes

Replace the satellite imagery:

1. Swap out `data/uiuc_tiles.png` with your satellite image (PNG format)
2. Update `data/uiuc_tiles.json` with new map bounds:
   ```json
   {
     "min_x": -400,
     "max_x": 5800,
     "min_z": -3200,
     "max_z": 3200
   }
   ```
3. Run: `python main.py --city uiuc`

### Shader Appearance

Adjust basemap brightness in [src/renderer.py](src/renderer.py#L85):
```glsl
mapCol = pow(mapCol, vec3(0.92)) * 1.05;  // gamma & brightness multiplier
```

- Lower gamma → brighter (e.g., 0.85)
- Higher multiplier → brighter (e.g., 1.2)

### Hand Gesture Thresholds

Fine-tune detection sensitivity in [src/hand_tracker.py](src/hand_tracker.py):
```python
INDEX_EXTENSION_THRESHOLD = 0.08    # How much finger must extend
CURL_THRESHOLD = 0.02               # Finger curl margin
THUMB_POS_CHECK = -0.15             # Thumb position validation
```

## Rendering Pipeline

### Background
- Webcam feed with hand landmark overlays (red circles, green skeleton)

### Ground Plane (1x)
- Satellite texture with conditional blending
- Procedural fallback (asphalt + grid) outside mapped region
- Distance-based atmospheric haze fade

### Buildings (3D Mesh)
- Blinn-Phong lighting with procedural window grid
- Window multiplier changes with day/night (ambient level)
- Selection highlight (golden outline when pointed/grabbed)
- Per-building vertex animation (grab-and-move)

### HUD Overlays
- Top-left: Building inspector (height, dimensions, density)
- Top-right: Input status (mode, hand count, clench value)
- Both with semi-transparent backgrounds

## Performance Notes

- **GPU**: Intel Arc Graphics (tested), NVIDIA/AMD recommended
- **Hand Tracking**: ~30 FPS (background thread)
- **Rendering**: 60 FPS target @ 1920×1080
- **Memory**: ~500MB (venv + data)

## Troubleshooting

### Webcam not opening
```bash
# Test webcam access
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Hand tracking not detecting
- Ensure good lighting and clear hand visibility
- Check MediaPipe model is present (`hand_landmarker.task`)
- Verify webcam permissions

### Slow rendering
- Lower window resolution
- Check GPU usage with system monitor
- Disable hand landmark drawing (comment `_draw_hand_landmarks()`)

### Satellite map not showing
- Verify `data/uiuc_tiles.png` exists
- Check `data/uiuc_tiles.json` has valid bounds
- Examine console output for "Map tiles loaded" message

## Development

### Running with Debugging
```bash
# Enable hand landmark visualization and input status HUD
python main.py --city uiuc
# These are on by default; comment in renderer.py to disable
```

### Testing Gesture Detection
```bash
python -c "
from src.hand_tracker import HandTracker
tracker = HandTracker()
# Will print gesture state to console
"
```

## Architecture Overview

```
Webcam Input
    ↓
HandTracker (background thread)
    → MediaPipe HandLandmarker
    → Gesture Recognition (_is_pointing, _is_clenched)
    → Camera Control (pan, zoom, orbit)
    ↓
Camera (OrbitCamera)
    → View/Projection matrices
    ↓
Renderer (ModernGL)
    → Background (webcam + landmarks)
    → Ground (satellite + procedural)
    → Buildings & Roads (mesh)
    → HUD overlays
    ↓
Display @ 60 FPS
```

## Future Enhancements

- [ ] Multi-hand independent control (both hands pointing)
- [ ] Building measurement tool (tap to measure distance)
- [ ] Time-of-day simulation with dynamic lighting
- [ ] GeoJSON feature export from repositioned buildings
- [ ] Voice commands integration
- [ ] VR headset support

## License

[Add your license here]

## Contributors

Created by ShanmukhUpad

## Support

For issues or feature requests, open an issue on GitHub.
