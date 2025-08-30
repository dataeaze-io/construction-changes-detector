import cv2
import numpy as np
import argparse
import os

def convert_tif_to_png(input_path, output_path, is_mask=False, scale_to_8bit=True):
    """Convert TIFF to PNG. Handles both masks and images."""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    print(f"Input: {input_path}, Shape: {img.shape}, Dtype: {img.dtype}")

    if is_mask:
        # Ensure binary mask (0/255)
        img[img > 253] = 0
        img[img > 0] = 255
        img = img.astype('uint8')
    else:
        # Convert image data to 8-bit if needed
        if scale_to_8bit and img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype('uint8')

    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIFF to PNG (supports masks & images)")
    parser.add_argument("--input", required=True, help="Input .tif file")
    parser.add_argument("--output", required=False, help="Output .png file")
    parser.add_argument("--mask", action="store_true", help="If set, process as mask (binary)")
    args = parser.parse_args()

    # Auto-generate output name if not provided
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + ".png"

    convert_tif_to_png(args.input, args.output, is_mask=args.mask)

