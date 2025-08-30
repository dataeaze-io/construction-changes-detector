import cv2
import numpy as np
import argparse
import os


def mse(a, b):
    """Compute Mean Squared Error between two arrays."""
    return np.mean((a - b) ** 2)


def find_best_shift(img1, img2, start_x=5000, start_y=5000, wsize=6000, shift_range=20):
    """
    Find best alignment (shift) between two images using MSE minimization.

    Args:
        img1: Reference image (grayscale).
        img2: Target image (grayscale).
        start_x, start_y: Starting coordinates for the window.
        wsize: Window size for patch comparison.
        shift_range: Maximum pixels to shift in both directions.

    Returns:
        (best_dx, best_dy, min_error)
    """
    patch_ref = img1[start_y:start_y+wsize, start_x:start_x+wsize]
    min_error = float("inf")
    best_shift = (0, 0)

    for dx in range(-shift_range, shift_range + 1):
        for dy in range(-shift_range, shift_range + 1):
            try:
                patch_target = img2[start_y+dy:start_y+dy+wsize,
                                    start_x+dx:start_x+dx+wsize]
                if patch_target.shape != patch_ref.shape:
                    continue
                error = mse(patch_ref, patch_target)
                if error < min_error:
                    min_error = error
                    best_shift = (dx, dy)
                    print(f"Better shift found: dx={dx}, dy={dy}, MSE={error}")
            except Exception:
                continue

    return best_shift, min_error


def apply_shift(img, dx, dy):
    """Apply pixel shift to an image."""
    rows, cols = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def main():
    parser = argparse.ArgumentParser(description="Image registration via pixel shift MSE minimization")
    parser.add_argument("-r", "--reference", required=True, help="Path to reference image (grayscale)")
    parser.add_argument("-t", "--target", required=True, help="Path to target image (grayscale)")
    parser.add_argument("-o", "--output", required=True, help="Path to save the aligned image")
    parser.add_argument("--start_x", type=int, default=5000, help="X coordinate for patch start")
    parser.add_argument("--start_y", type=int, default=5000, help="Y coordinate for patch start")
    parser.add_argument("--wsize", type=int, default=6000, help="Window size")
    parser.add_argument("--shift", type=int, default=20, help="Shift range (+/-)")
    args = parser.parse_args()

    # Load images
    img1 = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one or both images")

    print(f"Reference shape: {img1.shape}, Target shape: {img2.shape}")

    # Find best shift
    (dx, dy), error = find_best_shift(img1, img2,
                                      start_x=args.start_x,
                                      start_y=args.start_y,
                                      wsize=args.wsize,
                                      shift_range=args.shift)

    print(f"\nBest shift: dx={dx}, dy={dy}, with MSE={error}")

    # Apply shift and save
    aligned = apply_shift(img2, dx, dy)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, aligned)
    print(f"Aligned image saved at: {args.output}")


if __name__ == "__main__":
    main()

