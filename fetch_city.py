"""
fetch_city.py — Download high-quality 3D city data and cache it to data/

Usage:
    python fetch_city.py nyc              # Manhattan (LiDAR roof heights)
    python fetch_city.py chicago          # Chicago Loop
    python fetch_city.py "london"         # OSM fallback
    python fetch_city.py --lat 51.50 --lon -0.12 --radius 500 --name myarea

After fetching, load with:
    python main.py --city nyc
    python main.py --city chicago

Data sources:
    NYC      → NYC Open Data (DoITT Building Footprints, LiDAR heights in feet)
    Chicago  → Chicago Data Portal (Building Footprints)
    Others   → OpenStreetMap Overpass API
"""
import os
import sys
import json
import math
import argparse
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
#  Known city definitions
# -----------------------------------------------------------------------
CITIES = {
    "nyc": {
        "name": "New York City — Midtown",
        "lat": 40.7549, "lon": -73.9840, "radius": 800,
        "source": "nyc_opendata",
    },
    "chicago": {
        "name": "Chicago — The Loop",
        "lat": 41.8827, "lon": -87.6344, "radius": 700,
        "source": "chicago_opendata",
    },
    "london": {
        "name": "London — City of London",
        "lat": 51.5130, "lon": -0.0900, "radius": 600,
        "source": "osm",
    },
    "paris": {
        "name": "Paris — La Défense",
        "lat": 48.8919, "lon":  2.2385, "radius": 500,
        "source": "osm",
    },
    "tokyo": {
        "name": "Tokyo — Shinjuku",
        "lat": 35.6896, "lon": 139.6917, "radius": 500,
        "source": "osm",
    },
    "singapore": {
        "name": "Singapore CBD",
        "lat":  1.2840, "lon": 103.8510, "radius": 600,
        "source": "osm",
    },
    "sydney": {
        "name": "Sydney CBD",
        "lat": -33.8688, "lon": 151.2093, "radius": 600,
        "source": "osm",
    },
    "toronto": {
        "name": "Toronto Downtown",
        "lat": 43.6484, "lon": -79.3820, "radius": 600,
        "source": "osm",
    },
    "uiuc": {
        "name": "UIUC — University of Illinois Urbana-Champaign",
        "lat": 40.10725, "lon": -88.22807, "radius": 1400,
        "source": "osm_bbox",
        "bbox": (40.09803694382813, -88.23870788701966,
                 40.11646487843522, -88.21743009052165),
    },
}


# -----------------------------------------------------------------------
#  NYC Open Data  (DoITT Building Footprints, real LiDAR heights)
# -----------------------------------------------------------------------
def fetch_nyc(lat, lon, radius_m):
    """
    Fetch from NYC Open Data — Building Footprints with real LiDAR heights.
    Falls back to OSM if the API is unavailable.
    """
    d_lat = radius_m / 111_540.0
    d_lon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    min_lat, max_lat = lat - d_lat, lat + d_lat
    min_lon, max_lon = lon - d_lon, lon + d_lon
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    # Try two known dataset IDs — NYC occasionally migrates datasets
    candidates = [
        "https://data.cityofnewyork.us/resource/nqwf-w8eh.geojson",
        "https://data.cityofnewyork.us/resource/qb5r-6dgf.geojson",
    ]
    raw = None
    for url in candidates:
        try:
            print(f"  Trying {url.split('/resource/')[1].split('.')[0]} …")
            params = {
                "$where": f"within_box(the_geom, {min_lat}, {min_lon}, {max_lat}, {max_lon})",
                "$limit": 8000,
                "$select": "the_geom,heightroof,cnstrct_yr",
            }
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                raw = resp.json()
                break
            print(f"  Got {resp.status_code}, trying next …")
        except requests.RequestException as e:
            print(f"  Request failed: {e}")

    if raw is None:
        print("  NYC Open Data unavailable — falling back to OSM.")
        return fetch_osm_raw(lat, lon, radius_m)

    features = []
    for feat in raw.get("features", []):
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        try:
            height_ft = float(props.get("heightroof") or 0)
        except (TypeError, ValueError):
            height_ft = 30.0
        height_m = max(height_ft * 0.3048, 3.5)
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "height": round(height_m, 1),
                "building": "yes",
                "year": props.get("cnstrct_yr"),
            },
        })

    print(f"  {len(features)} buildings fetched")
    if len(features) == 0:
        print("  Empty result — falling back to OSM.")
        return fetch_osm_raw(lat, lon, radius_m)
    return {"type": "FeatureCollection", "features": features}


# -----------------------------------------------------------------------
#  Chicago Data Portal
# -----------------------------------------------------------------------
def fetch_chicago(lat, lon, radius_m):
    d_lat = radius_m / 111_540.0
    d_lon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    min_lat, max_lat = lat - d_lat, lat + d_lat
    min_lon, max_lon = lon - d_lon, lon + d_lon

    url = "https://data.cityofchicago.org/resource/hz9b-7nh8.geojson"
    params = {
        "$where": f"within_box(the_geom, {min_lat}, {min_lon}, {max_lat}, {max_lon})",
        "$limit": 6000,
        "$select": "the_geom,stories,year_built,bldg_name",
    }
    print("  Querying Chicago Data Portal …")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    raw = resp.json()

    features = []
    for feat in raw.get("features", []):
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        if not geom:
            continue
        try:
            stories = int(props.get("stories") or 3)
        except (TypeError, ValueError):
            stories = 3
        height_m = max(stories * 3.5, 3.5)

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "height": round(height_m, 1),
                "building": "yes",
                "name": props.get("bldg_name", ""),
                "year": props.get("year_built"),
            },
        })

    print(f"  {len(features)} buildings fetched")
    return {"type": "FeatureCollection", "features": features}


# -----------------------------------------------------------------------
#  OSM Overpass fallback
# -----------------------------------------------------------------------
def fetch_osm_raw(lat, lon, radius_m):
    print("  Querying OSM Overpass API …")
    query = f"""
    [out:json][timeout:60];
    (
      way["building"](around:{radius_m},{lat},{lon});
      way["highway"]["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]
         (around:{radius_m},{lat},{lon});
      way["leisure"~"^(park|garden|pitch|playground)$"](around:{radius_m},{lat},{lon});
      way["landuse"~"^(grass|forest|meadow|farmland|cemetery|recreation_ground)$"](around:{radius_m},{lat},{lon});
      way["natural"="water"](around:{radius_m},{lat},{lon});
      way["amenity"="parking"](around:{radius_m},{lat},{lon});
    );
    out body; >; out skel qt;
    """
    resp = requests.post("https://overpass-api.de/api/interpreter",
                         data={"data": query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    nodes = {el["id"]: (el["lon"], el["lat"])
             for el in data["elements"] if el["type"] == "node"}

    LEVELS_TO_M = 3.5
    features = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        tags   = el.get("tags", {})
        coords = [(nodes[n][0], nodes[n][1]) for n in el.get("nodes", [])
                  if n in nodes]
        if len(coords) < 3:
            continue

        if "building" in tags:
            h = 12.0
            if "height" in tags:
                try: h = float(tags["height"].replace("m", "").strip())
                except ValueError: pass
            elif "building:levels" in tags:
                try: h = float(tags["building:levels"]) * LEVELS_TO_M
                except ValueError: pass

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "height": round(h, 1),
                    "building": tags.get("building", "yes"),
                    "name": tags.get("name", ""),
                },
            })
        elif "highway" in tags:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"highway": tags.get("highway", "residential")},
            })
        else:
            # Area features: parks, water, landuse
            area_type = None
            if tags.get("natural") == "water":
                area_type = "water"
            elif tags.get("leisure") in ("park", "garden", "pitch", "playground"):
                area_type = tags["leisure"]
            elif tags.get("landuse") in ("grass", "forest", "meadow", "farmland",
                                          "cemetery", "recreation_ground"):
                area_type = tags["landuse"]
            elif tags.get("amenity") == "parking":
                area_type = "parking"

            if area_type:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"type": area_type},
                })

    print(f"  {len(features)} features fetched")
    return {"type": "FeatureCollection", "features": features}


# -----------------------------------------------------------------------
#  OSM Overpass — bounding box query (more precise than radius)
# -----------------------------------------------------------------------
def fetch_osm_bbox(south, west, north, east):
    """Fetch OSM data within an exact lat/lon bounding box."""
    print(f"  Querying OSM Overpass API (bbox) ...")
    bbox = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:90][bbox:{bbox}];
    (
      way["building"];
      way["highway"]["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|footway|path|cycleway|service)$"];
      way["leisure"~"^(park|garden|pitch|playground)$"];
      way["landuse"~"^(grass|forest|meadow|farmland|cemetery|recreation_ground)$"];
      way["natural"="water"];
      way["amenity"="parking"];
    );
    out body; >; out skel qt;
    """
    resp = requests.post("https://overpass-api.de/api/interpreter",
                         data={"data": query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    nodes = {el["id"]: (el["lon"], el["lat"])
             for el in data["elements"] if el["type"] == "node"}

    LEVELS_TO_M = 3.5
    features = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        tags   = el.get("tags", {})
        coords = [(nodes[n][0], nodes[n][1]) for n in el.get("nodes", [])
                  if n in nodes]
        if len(coords) < 2:
            continue

        if "building" in tags:
            h = 12.0
            if "height" in tags:
                try: h = float(tags["height"].replace("m", "").strip())
                except ValueError: pass
            elif "building:levels" in tags:
                try: h = float(tags["building:levels"]) * LEVELS_TO_M
                except ValueError: pass
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "height": round(h, 1),
                    "building": tags.get("building", "yes"),
                    "name": tags.get("name", ""),
                },
            })
        elif "highway" in tags:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"highway": tags.get("highway", "residential")},
            })
        else:
            area_type = None
            if tags.get("natural") == "water":
                area_type = "water"
            elif tags.get("leisure") in ("park", "garden", "pitch", "playground"):
                area_type = tags["leisure"]
            elif tags.get("landuse") in ("grass", "forest", "meadow", "farmland",
                                          "cemetery", "recreation_ground"):
                area_type = tags["landuse"]
            elif tags.get("amenity") == "parking":
                area_type = "parking"
            if area_type:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"type": area_type},
                })

    print(f"  {len(features)} features fetched")
    return {"type": "FeatureCollection", "features": features}


# -----------------------------------------------------------------------
#  Map tile downloader (OSM raster tiles → stitched ground texture)
# -----------------------------------------------------------------------

def _lon_to_tile(lon, zoom):
    return int((lon + 180.0) / 360.0 * (1 << zoom))

def _lat_to_tile(lat, zoom):
    lat_r = math.radians(lat)
    return int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * (1 << zoom))

def _tile_to_lon(x, zoom):
    return x / (1 << zoom) * 360.0 - 180.0

def _tile_to_lat(y, zoom):
    n = math.pi - 2.0 * math.pi * y / (1 << zoom)
    return math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))

def download_map_tiles(lat, lon, radius_m, name, bbox=None):
    """Download OSM map tiles covering the area and stitch into one image.
    If bbox=(south,west,north,east) is given, use it directly instead of radius.
    """
    from PIL import Image
    import io

    zoom = 17  # good detail for campus/city scale (~1.2m/pixel)
    TILE_SIZE = 256

    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
    else:
        # Compute lat/lon bounds from centre + radius
        d_lat = radius_m / 111_540.0
        d_lon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
        min_lat, max_lat = lat - d_lat, lat + d_lat
        min_lon, max_lon = lon - d_lon, lon + d_lon

    # Add padding (1 extra tile each side)
    tx_min = _lon_to_tile(min_lon, zoom) - 1
    tx_max = _lon_to_tile(max_lon, zoom) + 1
    ty_min = _lat_to_tile(max_lat, zoom) - 1  # note: y is flipped
    ty_max = _lat_to_tile(min_lat, zoom) + 1

    nx = tx_max - tx_min + 1
    ny = ty_max - ty_min + 1
    print(f"  Downloading {nx}x{ny} = {nx*ny} map tiles at zoom {zoom} ...")

    stitched = Image.new('RGB', (nx * TILE_SIZE, ny * TILE_SIZE))
    headers = {"User-Agent": "urban-ar-gis-viewer/1.0"}

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    tile_img = Image.open(io.BytesIO(resp.content))
                    px = (tx - tx_min) * TILE_SIZE
                    py = (ty - ty_min) * TILE_SIZE
                    stitched.paste(tile_img, (px, py))
                else:
                    print(f"    Tile {tx},{ty} -> HTTP {resp.status_code}")
            except requests.RequestException as e:
                print(f"    Tile {tx},{ty} failed: {e}")

    # Compute world-space bounds of the stitched image
    # Top-left corner of top-left tile → bottom-right corner of bottom-right tile
    bounds_lon_min = _tile_to_lon(tx_min, zoom)
    bounds_lon_max = _tile_to_lon(tx_max + 1, zoom)
    bounds_lat_max = _tile_to_lat(ty_min, zoom)      # top
    bounds_lat_min = _tile_to_lat(ty_max + 1, zoom)  # bottom

    # Convert to local metres (same projection as gis_loader)
    cos_lat = math.cos(math.radians(lat))
    world_min_x = (bounds_lon_min - lon) * cos_lat * 111_320.0
    world_max_x = (bounds_lon_max - lon) * cos_lat * 111_320.0
    world_min_z = -(bounds_lat_max - lat) * 110_540.0  # north is -z
    world_max_z = -(bounds_lat_min - lat) * 110_540.0

    # Save image
    img_path = os.path.join(DATA_DIR, f"{name}_tiles.png")
    stitched.save(img_path, "PNG")

    # Save bounds metadata
    meta = {
        "min_x": world_min_x, "max_x": world_max_x,
        "min_z": world_min_z, "max_z": world_max_z,
        "zoom": zoom, "tiles": f"{nx}x{ny}",
    }
    meta_path = os.path.join(DATA_DIR, f"{name}_tiles.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"  Map tiles saved -> {img_path} ({stitched.size[0]}x{stitched.size[1]})")
    return img_path, meta


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("city", nargs="?", help="City key (nyc, chicago, london …)")
    p.add_argument("--lat",    type=float)
    p.add_argument("--lon",    type=float)
    p.add_argument("--radius", type=int, default=600)
    p.add_argument("--name",   type=str, help="Output filename (without .geojson)")
    args = p.parse_args()

    if args.city:
        key = args.city.lower().strip()
        if key not in CITIES:
            print(f"Unknown city '{key}'. Known cities: {', '.join(CITIES)}")
            print("Trying OSM fallback with that name …")
            # Try as a place name via OSM geocoder
            geo = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": args.city, "format": "json", "limit": 1},
                headers={"User-Agent": "urban-ar-gis-viewer/1.0"},
                timeout=15,
            ).json()
            if not geo:
                sys.exit(f"Could not geocode '{args.city}'")
            lat = float(geo[0]["lat"])
            lon = float(geo[0]["lon"])
            radius = args.radius
            name   = key.replace(" ", "_")
            source = "osm"
            bbox   = None
        else:
            cfg    = CITIES[key]
            lat    = cfg["lat"]
            lon    = cfg["lon"]
            radius = args.radius or cfg["radius"]
            name   = key
            source = cfg["source"]
            bbox   = cfg.get("bbox")
            print(f"Fetching: {cfg['name']}")
    elif args.lat and args.lon:
        lat    = args.lat
        lon    = args.lon
        radius = args.radius
        name   = args.name or f"custom_{lat:.3f}_{lon:.3f}"
        source = "osm"
        bbox   = None
    else:
        p.print_help()
        sys.exit(1)

    print(f"Centre: ({lat:.4f}, {lon:.4f})  radius={radius}m  source={source}")

    if source == "nyc_opendata":
        gj = fetch_nyc(lat, lon, radius)
    elif source == "chicago_opendata":
        gj = fetch_chicago(lat, lon, radius)
    elif source == "osm_bbox" and bbox:
        south, west, north, east = bbox
        gj = fetch_osm_bbox(south, west, north, east)
    else:
        gj = fetch_osm_raw(lat, lon, radius)

    # Embed centre coordinates so main.py can read them
    gj["_centre"] = {"lat": lat, "lon": lon}

    out_path = os.path.join(DATA_DIR, f"{name}.geojson")
    with open(out_path, "w") as f:
        json.dump(gj, f)
    print(f"\nSaved -> {out_path}")

    # Download map tiles for ground texture
    print("\nDownloading map tiles for ground texture ...")
    try:
        download_map_tiles(lat, lon, radius, name, bbox=bbox)
    except Exception as e:
        print(f"  Map tile download failed: {e} (ground will use procedural texture)")

    print(f"Run:   python main.py --city {name}")


if __name__ == "__main__":
    main()
