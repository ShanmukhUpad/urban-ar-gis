# Raster Map Interactor

A photorealistic 3D raster map application that uses MediaPipe hand tracking for interactive cursor control.

## Features

- **3D Scene**: PyVista-based 3D visualization with satellite imagery
- **Hand Tracking**: Real-time index finger tip tracking using MediaPipe
- **Interactive Cursor**: Dynamic red sphere that follows your finger movements
- **High-Resolution Maps**: Support for satellite images as textures

## Requirements

- Python 3.8+
- Webcam
- Satellite image file (`uiuc_satellite.jpg`)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your satellite image as `uiuc_satellite.jpg` in the project directory

## Usage

Run the application:
```bash
python raster_map_interactor.py
```

## Controls

- **Point with index finger**: Move the red cursor sphere on the map
- **Close application**: Close the PyVista window

## How it Works

1. **Hand Tracking**: MediaPipe detects hand landmarks from webcam feed
2. **Coordinate Mapping**: Index finger tip coordinates (0.0-1.0) are mapped to 3D world coordinates on the plane
3. **Real-time Updates**: Cursor position updates at ~30 FPS using PyVista's timer callbacks
4. **Multithreading**: Hand tracking runs in background thread for smooth performance

## Architecture

- `RasterMapInteractor` class: Main application controller
- Separate threads for hand tracking and 3D rendering
- PyVista timer callbacks for efficient updates
- OpenCV for webcam capture, MediaPipe for hand detection