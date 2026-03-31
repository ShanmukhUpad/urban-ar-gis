"""
GIS data loading: OSM Overpass API or local GeoJSON → GPU-ready geometry.

Coordinate system (Y-up, metres from scene centre):
    x = East
    y = Building height (Up)
    z = South  (North is -z)
"""
import json
import math
import time
import requests
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
try:
    from mapbox_earcut import triangulate_float32 as earcut
    HAS_EARCUT = True
except ImportError:
    HAS_EARCUT = False

from pyproj import Transformer


# ------------------------------------------------------------------
#  Building colour palette  (linear RGB, ~gamma-corrected in shader)
# ------------------------------------------------------------------
BUILDING_COLORS = {
    'residential':  np.array([0.95, 0.88, 0.76], np.float32),
    'apartments':   np.array([0.90, 0.80, 0.70], np.float32),
    'commercial':   np.array([0.65, 0.75, 0.92], np.float32),
    'retail':       np.array([0.85, 0.72, 0.88], np.float32),
    'office':       np.array([0.60, 0.72, 0.88], np.float32),
    'industrial':   np.array([0.72, 0.65, 0.55], np.float32),
    'hotel':        np.array([0.72, 0.88, 0.78], np.float32),
    'public':       np.array([0.88, 0.78, 0.62], np.float32),
    'school':       np.array([0.88, 0.78, 0.62], np.float32),
    'church':       np.array([0.80, 0.75, 0.70], np.float32),
    'parking':      np.array([0.60, 0.60, 0.60], np.float32),
    'default':      np.array([0.80, 0.80, 0.82], np.float32),
}

ROAD_COLORS = {
    'motorway':    np.array([0.90, 0.60, 0.20], np.float32),
    'trunk':       np.array([0.85, 0.65, 0.25], np.float32),
    'primary':     np.array([0.85, 0.78, 0.35], np.float32),
    'secondary':   np.array([0.78, 0.80, 0.40], np.float32),
    'tertiary':    np.array([0.65, 0.65, 0.65], np.float32),
    'residential': np.array([0.55, 0.55, 0.55], np.float32),
    'default':     np.array([0.45, 0.45, 0.45], np.float32),
}

ROAD_WIDTHS = {
    'motorway': 12.0, 'trunk': 10.0, 'primary': 8.0,
    'secondary': 7.0, 'tertiary': 5.0, 'residential': 4.0,
    'default': 3.5,
}

AREA_COLORS = {
    'park':        np.array([0.30, 0.62, 0.25], np.float32),
    'grass':       np.array([0.38, 0.68, 0.30], np.float32),
    'forest':      np.array([0.18, 0.48, 0.15], np.float32),
    'garden':      np.array([0.35, 0.60, 0.28], np.float32),
    'meadow':      np.array([0.50, 0.72, 0.35], np.float32),
    'water':       np.array([0.18, 0.42, 0.72], np.float32),
    'riverbank':   np.array([0.20, 0.45, 0.70], np.float32),
    'parking':     np.array([0.42, 0.42, 0.44], np.float32),
    'pitch':       np.array([0.32, 0.58, 0.28], np.float32),
    'playground':  np.array([0.60, 0.52, 0.35], np.float32),
    'recreation':  np.array([0.42, 0.62, 0.35], np.float32),
    'farmland':    np.array([0.62, 0.70, 0.38], np.float32),
    'cemetery':    np.array([0.35, 0.50, 0.30], np.float32),
    'default':     np.array([0.45, 0.55, 0.35], np.float32),
}

LEVELS_TO_HEIGHT = 3.5   # metres per floor
DEFAULT_HEIGHT   = 12.0  # fallback if no height tag


class UrbanScene:
    """Holds numpy arrays ready to upload to GPU."""
    def __init__(self):
        self.building_verts:  np.ndarray = np.empty((0, 9), np.float32)  # pos3+nrm3+col3
        self.building_idx:    np.ndarray = np.empty((0,),   np.uint32)
        self.road_verts:      np.ndarray = np.empty((0, 9), np.float32)  # pos3+nrm3+col3
        self.road_idx:        np.ndarray = np.empty((0,),   np.uint32)
        self.area_verts:      np.ndarray = np.empty((0, 9), np.float32)  # pos3+nrm3+col3
        self.area_idx:        np.ndarray = np.empty((0,),   np.uint32)
        self.building_aabbs:  list       = []   # [{min_x,max_x,min_z,max_z, height, tags}]
        self.center_latlon:   tuple      = (0.0, 0.0)
        self.extent_m:        float      = 500.0
        # Map tile ground texture (loaded from fetch_city output)
        self.map_image:       np.ndarray | None = None   # RGB uint8 array
        self.map_bounds:      dict | None       = None   # {min_x, max_x, min_z, max_z}


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------

def fetch_osm(lat: float, lon: float, radius: int = 600) -> UrbanScene:
    """Download real city data from OSM Overpass API."""
    print(f"Fetching OSM data for ({lat:.4f}, {lon:.4f}) radius={radius}m …")
    query = f"""
    [out:json][timeout:60];
    (
      way["building"](around:{radius},{lat},{lon});
      way["highway"]["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]
         (around:{radius},{lat},{lon});
      way["leisure"~"^(park|garden|pitch|playground)$"](around:{radius},{lat},{lon});
      way["landuse"~"^(grass|forest|meadow|farmland|cemetery|recreation_ground)$"](around:{radius},{lat},{lon});
      way["natural"="water"](around:{radius},{lat},{lon});
      way["amenity"="parking"](around:{radius},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    resp = requests.post("https://overpass-api.de/api/interpreter",
                         data={"data": query}, timeout=60)
    resp.raise_for_status()
    return _parse_osm_json(resp.json(), lat, lon, radius)


def load_geojson(path: str, center_lat: float, center_lon: float) -> UrbanScene:
    """Load a local GeoJSON file (buildings + roads in one FeatureCollection)."""
    import os
    from PIL import Image

    with open(path) as f:
        gj = json.load(f)
    scene = _parse_geojson(gj, center_lat, center_lon)

    # Try to load companion map tiles
    base = path.rsplit('.', 1)[0]  # strip .geojson
    tiles_img_path  = base + '_tiles.png'
    tiles_meta_path = base + '_tiles.json'
    if os.path.exists(tiles_img_path) and os.path.exists(tiles_meta_path):
        with open(tiles_meta_path) as f:
            scene.map_bounds = json.load(f)
        img = Image.open(tiles_img_path).convert('RGB')
        scene.map_image = np.array(img, dtype=np.uint8)
        print(f"  Map tiles loaded: {img.size[0]}x{img.size[1]}")

    return scene


def generate_sample_city() -> UrbanScene:
    """Synthetic grid city — no data needed, great for testing."""
    rng = np.random.default_rng(42)
    buildings_raw = []
    roads_raw     = []

    block = 100.0   # metres between block centroids
    gap   = 20.0    # road gap between blocks

    for bx in range(-3, 4):
        for bz in range(-3, 4):
            ox = bx * block
            oz = bz * block
            bldg_types = ['residential', 'commercial', 'office', 'retail', 'apartments']
            num = rng.integers(2, 5)
            for _ in range(num):
                w  = rng.uniform(18, 38)
                d  = rng.uniform(18, 38)
                px = ox + rng.uniform(-30, 30)
                pz = oz + rng.uniform(-30, 30)
                h  = rng.choice([8, 12, 18, 30, 50, 80, 120, 200],
                                p=[0.15, 0.20, 0.20, 0.15, 0.12, 0.08, 0.06, 0.04])
                btype = rng.choice(bldg_types)
                coords = [(px, pz), (px+w, pz), (px+w, pz+d), (px, pz+d)]
                buildings_raw.append({'coords': coords, 'height': float(h), 'type': btype})

    # Streets
    for i in range(-4, 5):
        x = i * block
        roads_raw.append({'coords': [(x, -400), (x, 400)], 'type': 'secondary'})
        roads_raw.append({'coords': [(-400, x), (400, x)], 'type': 'secondary'})

    # Add some parks to the sample city
    areas_raw = [
        {'coords': [(-280, -280), (-200, -280), (-200, -200), (-280, -200)], 'type': 'park'},
        {'coords': [(200, 200), (280, 200), (280, 280), (200, 280)], 'type': 'grass'},
        {'coords': [(-50, 250), (50, 250), (50, 300), (-50, 300)], 'type': 'water'},
    ]
    return _build_scene(buildings_raw, roads_raw, areas_raw,
                        center_latlon=(0.0, 0.0), extent=700.0)


# ------------------------------------------------------------------
#  OSM JSON parser
# ------------------------------------------------------------------

def _parse_osm_json(data: dict, center_lat: float, center_lon: float,
                    radius: int) -> UrbanScene:
    nodes = {el['id']: (el['lon'], el['lat'])
             for el in data['elements'] if el['type'] == 'node'}
    proj  = _make_projector(center_lat, center_lon)

    buildings_raw = []
    roads_raw     = []
    areas_raw     = []

    for el in data['elements']:
        if el['type'] != 'way':
            continue
        tags  = el.get('tags', {})
        nd_ids = el.get('nodes', [])
        coords = [proj(nodes[n]) for n in nd_ids if n in nodes]
        if len(coords) < 2:
            continue

        if 'building' in tags:
            h = _parse_height(tags)
            btype = (tags.get('building:use') or tags.get('building') or 'default').lower()
            if btype == 'yes':
                btype = 'default'
            buildings_raw.append({'coords': coords, 'height': h, 'type': btype,
                                   'name': tags.get('name', '')})

        elif 'highway' in tags:
            rtype = tags.get('highway', 'default')
            roads_raw.append({'coords': coords, 'type': rtype})

        else:
            atype = _classify_area(tags)
            if atype and len(coords) >= 3:
                areas_raw.append({'coords': coords, 'type': atype})

    print(f"  {len(buildings_raw)} buildings, {len(roads_raw)} roads, {len(areas_raw)} areas")
    return _build_scene(buildings_raw, roads_raw, areas_raw,
                        center_latlon=(center_lat, center_lon),
                        extent=float(radius) * 1.1)


def _classify_area(tags: dict) -> str | None:
    """Return area type string from OSM tags, or None if not a recognized area."""
    leisure = tags.get('leisure', '')
    landuse = tags.get('landuse', '')
    natural = tags.get('natural', '')
    amenity = tags.get('amenity', '')

    if natural == 'water':
        return 'water'
    if leisure in ('park', 'garden'):
        return leisure
    if leisure == 'pitch':
        return 'pitch'
    if leisure == 'playground':
        return 'playground'
    if landuse in ('grass', 'meadow', 'village_green'):
        return 'grass'
    if landuse in ('forest', 'wood'):
        return 'forest'
    if landuse == 'farmland':
        return 'farmland'
    if landuse == 'cemetery':
        return 'cemetery'
    if landuse == 'recreation_ground':
        return 'recreation'
    if amenity == 'parking':
        return 'parking'
    return None


def _parse_height(tags: dict) -> float:
    if 'height' in tags:
        try: return float(tags['height'].replace('m','').strip())
        except ValueError: pass
    if 'building:levels' in tags:
        try: return float(tags['building:levels']) * LEVELS_TO_HEIGHT
        except ValueError: pass
    return DEFAULT_HEIGHT


def _make_projector(center_lat, center_lon):
    """Returns a function (lon, lat) → (local_x, local_z) in metres."""
    cos_lat = math.cos(math.radians(center_lat))
    def proj(lonlat):
        dx = (lonlat[0] - center_lon) * cos_lat * 111_320.0   # east
        dz = (lonlat[1] - center_lat)           * 110_540.0   # north → -z
        return (dx, -dz)
    return proj


# ------------------------------------------------------------------
#  GeoJSON parser
# ------------------------------------------------------------------

def _parse_geojson(gj: dict, clat: float, clon: float) -> UrbanScene:
    proj = _make_projector(clat, clon)
    buildings_raw = []
    roads_raw     = []
    areas_raw     = []

    for feat in gj.get('features', []):
        geom  = feat.get('geometry', {})
        props = feat.get('properties', {}) or {}
        gtype = geom.get('type', '')

        if gtype == 'Polygon':
            coords = [proj(c) for c in geom['coordinates'][0]]
            # Check if it's a building or an area feature
            if props.get('building'):
                btype = (props.get('building') or 'default').lower()
                if btype == 'yes':
                    btype = 'default'
                h = float(props.get('height') or props.get('building:height') or DEFAULT_HEIGHT)
                buildings_raw.append({'coords': coords, 'height': h, 'type': btype,
                                       'name': props.get('name', '')})
            else:
                atype = _classify_area(props) or props.get('type', '')
                if atype and atype in AREA_COLORS:
                    areas_raw.append({'coords': coords, 'type': atype})
        elif gtype == 'LineString':
            coords = [proj(c) for c in geom['coordinates']]
            roads_raw.append({'coords': coords, 'type': props.get('highway', 'default')})

    return _build_scene(buildings_raw, roads_raw, areas_raw, (clat, clon), 500.0)


# ------------------------------------------------------------------
#  Geometry builder
# ------------------------------------------------------------------

def _build_scene(buildings_raw, roads_raw, areas_raw=None,
                  center_latlon=(0, 0), extent=500.0) -> UrbanScene:
    if areas_raw is None:
        areas_raw = []
    scene = UrbanScene()
    scene.center_latlon = center_latlon
    scene.extent_m      = extent

    b_verts_list, b_idx_list, aabbs = [], [], []
    r_verts_list, r_idx_list        = [], []
    a_verts_list, a_idx_list        = [], []

    v_off = 0
    for i, b in enumerate(buildings_raw):
        coords = b['coords']
        h      = b['height']
        color  = _building_color(b['type'], h, i)
        v, idx, aabb = _extrude_building(coords, h, color, v_off, b.get('name', ''))
        if v is not None:
            b_verts_list.append(v)
            b_idx_list.append(idx)
            aabbs.append(aabb)
            v_off += len(v)

    r_off = 0
    for r in roads_raw:
        coords = r['coords']
        rtype  = r['type']
        color  = ROAD_COLORS.get(rtype, ROAD_COLORS['default'])
        width  = ROAD_WIDTHS.get(rtype, ROAD_WIDTHS['default'])
        v, idx = _road_strip(coords, color, width, r_off)
        if v is not None:
            r_verts_list.append(v)
            r_idx_list.append(idx)
            r_off += len(v)

    a_off = 0
    for area in areas_raw:
        coords = area['coords']
        color  = AREA_COLORS.get(area['type'], AREA_COLORS['default'])
        v, idx = _flat_polygon(coords, color, a_off)
        if v is not None:
            a_verts_list.append(v)
            a_idx_list.append(idx)
            a_off += len(v)

    if b_verts_list:
        scene.building_verts = np.concatenate(b_verts_list, axis=0).astype(np.float32)
        scene.building_idx   = np.concatenate(b_idx_list,   axis=0).astype(np.uint32)
    if r_verts_list:
        scene.road_verts = np.concatenate(r_verts_list, axis=0).astype(np.float32)
        scene.road_idx   = np.concatenate(r_idx_list,   axis=0).astype(np.uint32)
    if a_verts_list:
        scene.area_verts = np.concatenate(a_verts_list, axis=0).astype(np.float32)
        scene.area_idx   = np.concatenate(a_idx_list,   axis=0).astype(np.uint32)

    scene.building_aabbs = aabbs
    print(f"Scene: {len(scene.building_verts)} building verts, "
          f"{len(scene.road_verts)} road verts, {len(scene.area_verts)} area verts")
    return scene


def _building_color(btype: str, height: float, index: int) -> np.ndarray:
    """Generate a varied building color based on type, height, and index."""
    base = BUILDING_COLORS.get(btype, BUILDING_COLORS['default']).copy()

    # Hash-based hue/saturation variation so identical-type buildings look different
    h = ((index * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF  # Knuth hash → [0,1]
    shift = np.array([
        (h - 0.5) * 0.12,
        ((h * 1.618) % 1.0 - 0.5) * 0.08,
        ((h * 2.718) % 1.0 - 0.5) * 0.10,
    ], np.float32)
    base += shift

    # Taller buildings get a cooler, more glass-like tint
    if height > 40:
        glass_mix = min((height - 40) / 160.0, 0.45)
        glass = np.array([0.55, 0.68, 0.82], np.float32)
        base = base * (1 - glass_mix) + glass * glass_mix
    elif height < 8:
        # Short buildings: warmer, more residential feel
        warm = np.array([0.92, 0.85, 0.72], np.float32)
        base = base * 0.7 + warm * 0.3

    return np.clip(base, 0.0, 1.0)


# ------------------------------------------------------------------
#  Building extrusion: polygon footprint → 3D solid
# ------------------------------------------------------------------

def _extrude_building(coords, height, color, v_off, name=''):
    """
    coords : [(x, z), ...]  local metres
    height : float metres
    Returns (vertices (N,9), indices (M,), aabb_dict) or (None, None, None).
    """
    if len(coords) < 3:
        return None, None, None

    # Remove duplicate last vertex (closed ring)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None, None, None

    xs  = [c[0] for c in coords]
    zs  = [c[1] for c in coords]
    n   = len(coords)

    verts = []   # list of [x, y, z, nx, ny, nz, r, g, b]
    idxs  = []

    # --- Walls ---
    for i in range(n):
        x0, z0 = coords[i]
        x1, z1 = coords[(i + 1) % n]
        # Outward normal (XZ plane)
        ex = x1 - x0;  ez = z1 - z0
        l  = math.hypot(ex, ez) or 1e-6
        nx =  ez / l;  nz = -ex / l;  ny = 0.0

        base0 = [x0, 0.0, z0, nx, ny, nz, *color]
        base1 = [x1, 0.0, z1, nx, ny, nz, *color]
        top0  = [x0, height, z0, nx, ny, nz, *color]
        top1  = [x1, height, z1, nx, ny, nz, *color]

        vi = v_off + len(verts)
        verts.extend([base0, base1, top1, top0])
        idxs.extend([vi, vi+1, vi+2,  vi, vi+2, vi+3])

    # --- Roof ---
    roof_color = np.clip(color * 0.85, 0, 1)
    roof_verts = [[coords[i][0], height, coords[i][1], 0, 1, 0, *roof_color]
                  for i in range(n)]
    vi_roof = v_off + len(verts)

    # Triangulate roof polygon
    # mapbox-earcut >= 2.0 API: triangulate_float32(vertices (n,2), holes (m,))
    poly2d = np.array(coords, dtype=np.float32)  # shape (n, 2)
    if HAS_EARCUT:
        tri_idx = earcut(poly2d, np.array([len(poly2d)], dtype=np.uint32))
        if len(tri_idx) >= 3:
            verts.extend(roof_verts)
            for k in range(0, len(tri_idx), 3):
                a, b, c = tri_idx[k], tri_idx[k+1], tri_idx[k+2]
                idxs.extend([vi_roof + a, vi_roof + b, vi_roof + c])
    else:
        # Fan triangulation fallback (works for convex polygons)
        verts.extend(roof_verts)
        for i in range(1, n - 1):
            idxs.extend([vi_roof, vi_roof + i, vi_roof + i + 1])

    aabb = {
        'min_x': min(xs), 'max_x': max(xs),
        'min_z': min(zs), 'max_z': max(zs),
        'height': height,
        'name': name,
    }
    return np.array(verts, np.float32), np.array(idxs, np.uint32), aabb


# ------------------------------------------------------------------
#  Road quad strips
# ------------------------------------------------------------------

def _road_strip(coords, color, width, v_off):
    if len(coords) < 2:
        return None, None

    hw = width / 2.0
    verts = []
    idxs  = []

    for i in range(len(coords) - 1):
        x0, z0 = coords[i]
        x1, z1 = coords[i + 1]
        ex = x1 - x0;  ez = z1 - z0
        l  = math.hypot(ex, ez) or 1e-6
        # Perpendicular offset
        px = -ez / l * hw;  pz = ex / l * hw

        vi = v_off + len(verts)
        verts.extend([
            [x0 - px, 0.02, z0 - pz, 0, 1, 0, *color],   # pos + up-normal + color
            [x0 + px, 0.02, z0 + pz, 0, 1, 0, *color],
            [x1 + px, 0.02, z1 + pz, 0, 1, 0, *color],
            [x1 - px, 0.02, z1 - pz, 0, 1, 0, *color],
        ])
        idxs.extend([vi, vi+1, vi+2,  vi, vi+2, vi+3])

    return np.array(verts, np.float32), np.array(idxs, np.uint32)


# ------------------------------------------------------------------
#  Flat area polygons (parks, water, etc.)
# ------------------------------------------------------------------

def _flat_polygon(coords, color, v_off):
    """Triangulate a polygon at ground level (y=0.03) with a flat color."""
    if len(coords) < 3:
        return None, None

    # Remove duplicate closing vertex
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None, None

    n = len(coords)
    # Vertices: pos(3) + normal(3) + color(3), normal is straight up
    verts = [[c[0], 0.03, c[1], 0.0, 1.0, 0.0, *color] for c in coords]

    if HAS_EARCUT:
        poly2d = np.array(coords, dtype=np.float32)
        tri_idx = earcut(poly2d, np.array([len(poly2d)], dtype=np.uint32))
        if len(tri_idx) < 3:
            return None, None
        idxs = [v_off + int(i) for i in tri_idx]
    else:
        idxs = []
        for i in range(1, n - 1):
            idxs.extend([v_off, v_off + i, v_off + i + 1])

    return np.array(verts, np.float32), np.array(idxs, np.uint32)
