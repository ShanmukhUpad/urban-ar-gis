# Data

Place GeoJSON files here for local loading.

Expected format: FeatureCollection with Polygon features (buildings) and
LineString features (roads).  Property names:

| Property         | Type   | Notes                              |
|------------------|--------|------------------------------------|
| `building`       | string | building type (residential, office…) |
| `height`         | number | metres                             |
| `building:levels`| number | floors (multiplied by 3.5m)        |
| `highway`        | string | road class (primary, secondary…)   |

Export from QGIS: Layer → Export → Save Features As → GeoJSON, CRS = EPSG:4326.

For OSM data, use Overpass Turbo (overpass-turbo.eu) or the --place / --lat/--lon
flags which fetch directly at runtime.
