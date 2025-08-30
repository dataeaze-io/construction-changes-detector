import os
import cv2
import argparse
import numpy as np
import torch
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import polygon2mask


def mask2poly(mask, tolerance=1):
    """Convert binary mask to polygons using skimage."""
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be numpy.ndarray")

    if mask.dtype == bool:
        arr = mask.astype(np.int64)
    else:
        arr = mask.astype(np.int64)

    if not set(np.unique(arr).tolist()).issubset({0, 1}):
        raise ValueError("Mask must be binary (0/1 only)")

    arr = arr.astype(np.float64)
    contours = find_contours(arr, 0.5, fully_connected="low", positive_orientation="low")
    polygons = [approximate_polygon(c, tolerance) for c in contours]
    return polygons


def filter_polygons_cv2(img, th):
    """Detect polygons using cv2 contours, filter by area, return rectified mask."""
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    maskr = np.zeros_like(img, dtype=np.uint8)
    total, kept = 0, 0

    for cnt in contours:
        coords = [[p[0][1], p[0][0]] for p in cnt]  # (x,y) -> (row,col)
        polygon = np.array(coords)
        maskt = torch.from_numpy(polygon2mask(img.shape, polygon))

        total += 1
        if torch.count_nonzero(maskt).item() >= th:
            kept += 1
            maskr += (polygon2mask(img.shape, polygon).astype(np.uint8) * 255)

    return maskr, total, kept


def filter_polygons_skimage(img, th):
    """Detect polygons using skimage find_contours, filter by area, return rectified mask."""
    mask = img.astype(bool)
    polygons = mask2poly(mask)

    maskr = np.zeros_like(img, dtype=np.uint8)
    total, kept = 0, 0

    for polygon in polygons:
        polygon = np.array(polygon)
        maskt = torch.from_numpy(polygon2mask(img.shape, polygon))

        total += 1
        if torch.count_nonzero(maskt).item() >= th:
            kept += 1
            maskr += (polygon2mask(img.shape, polygon).astype(np.uint8) * 255)

    return maskr, total, kept


def process_dataset(input_dir, output_dir, th, mode="cv2", ext=".png"):
    """Process all masks in dataset, save rectified ones."""
    os.makedirs(output_dir, exist_ok=True)

    total_all, kept_all = 0, 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(ext):
            continue

        fpath = os.path.join(input_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Skipping {fname}, could not read")
            continue

        if mode == "cv2":
            maskr, total, kept = filter_polygons_cv2(img, th)
        else:
            maskr, total, kept = filter_polygons_skimage(img, th)

        total_all += total
        kept_all += kept

        outpath = os.path.join(output_dir, fname)
        cv2.imwrite(outpath, maskr)

    print(f"✅ Finished. Polygons found: {total_all}, kept: {kept_all}, dropped: {total_all - kept_all}")


def main():
    parser = argparse.ArgumentParser(description="Polygon detection and rectification with area threshold")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory containing masks")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for rectified masks")
    parser.add_argument("-t", "--th", type=int, default=900, help="Area threshold in pixels")
    parser.add_argument("--mode", choices=["cv2", "skimage"], default="cv2",
                        help="Polygon detection backend (cv2 contours or skimage find_contours)")
    parser.add_argument("--ext", default=".png", help="File extension filter (default: .png)")

    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_dir, args.th, mode=args.mode, ext=args.ext)


if __name__ == "__main__":
    main()

