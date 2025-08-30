import argparse
import logging
import numpy as np
import cv2
import torch
import rasterio
import csv
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import polygon2mask
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pyproj


def mask2poly(mask, tolerance=1):
    """Convert binary mask to polygons."""
    if not isinstance(mask, np.ndarray):
        logging.error(f"mask must be numpy.ndarray, got {type(mask)}")
        return None, False
    if mask.dtype == bool:
        arr = mask.astype(np.int64)
    elif np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.floating):
        arr = mask.astype(np.int64)
    else:
        logging.error("Unsupported dtype for mask")
        return None, False

    if not set(np.unique(arr)).issubset({0, 1}):
        logging.error("Mask must be binary (0/1 only)")
        return None, False

    arr = arr.astype(np.float64)
    contours = find_contours(arr, 0.5, fully_connected="low", positive_orientation="low")
    polygons = [approximate_polygon(c, tolerance) for c in contours]

    return polygons, True


def extract_polygons(img, area_threshold=900):
    """Extract polygons from binary image with area filtering."""
    ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    polygons = []
    for contour in contours:
        coords = [[pt[0][0], pt[0][1]] for pt in contour]
        polygon = np.array(coords)
        mask = polygon2mask(img.shape, polygon)
        if torch.count_nonzero(torch.from_numpy(mask)).item() >= area_threshold:
            polygons.append(polygon.tolist())
    return polygons


def geocode_polygons(polygons, ref_tif, transformer, rate_limiter):
    """Convert polygon pixel coords to lat/lon and reverse geocode."""
    results = []
    with rasterio.open(ref_tif) as map_layer:
        for poly in polygons:
            # Take first vertex as representative point
            px, py = poly[0]
            x, y = map_layer.xy(py, px)  # Note: rasterio expects (row, col)
            lon, lat = transformer.transform(x, y)
            location = rate_limiter((lat, lon), language="en")
            if location:
                results.append(location.raw)
    return results


def save_to_csv(data_list, output_csv):
    """Save reverse geocode results to CSV."""
    if not data_list:
        print("⚠️ No data to save.")
        return
    header = data_list[0].keys()
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)
    print(f"✅ CSV file saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract polygons from mask, georeference, and reverse geocode.")
    parser.add_argument("-i", "--input", required=True, help="Input mask image (PNG/TIFF)")
    parser.add_argument("-r", "--reference", required=True, help="Reference GeoTIFF for georeferencing")
    parser.add_argument("-o", "--output", default="results.csv", help="Output CSV file")
    parser.add_argument("--epsg_in", default="32643", help="Input projection EPSG (default: 32643)")
    parser.add_argument("--epsg_out", default="4326", help="Output projection EPSG (default: 4326)")
    parser.add_argument("--area", type=int, default=900, help="Minimum area threshold for polygons")
    args = parser.parse_args()

    # Load image (grayscale if RGB)
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded image {args.input}, shape={img.shape}")

    # Extract polygons
    polygons = extract_polygons(img, args.area)
    print(f"Found {len(polygons)} polygons above threshold {args.area}")

    # Setup geocoder + transformer
    geolocator = Nominatim(user_agent="binary_change_detector")
    rate_limiter = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    transformer = pyproj.Transformer.from_crs(f"epsg:{args.epsg_in}", f"epsg:{args.epsg_out}")

    # Geocode polygons
    data_list = geocode_polygons(polygons, args.reference, transformer, rate_limiter)

    # Save results
    save_to_csv(data_list, args.output)


if __name__ == "__main__":
    main()

