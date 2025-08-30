import argparse
import cv2
from skimage.exposure import match_histograms
from skimage.filters import unsharp_mask


def process_image(image_path, reference_path, output_prefix, save_hist=True, unsharp_params=None):
    """Apply histogram matching and unsharp masking with different params."""

    # Load images
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if reference is None or image is None:
        raise FileNotFoundError("❌ Could not load input or reference image.")

    print(f"Input image shape: {image.shape}, Reference shape: {reference.shape}")

    # Histogram matching
    matched = match_histograms(image, reference)
    if save_hist:
        hist_out = f"{output_prefix}_histmatched.png"
        cv2.imwrite(hist_out, matched)
        print(f"✅ Saved histogram matched image → {hist_out}")

    # Apply unsharp masks with given parameter sets
    if unsharp_params:
        for idx, (radius, amount) in enumerate(unsharp_params, start=1):
            result = unsharp_mask(matched, radius=radius, amount=amount)
            # Convert back to uint8 (skimage returns float in [0,1])
            result = (result * 255).clip(0, 255).astype("uint8")
            out_path = f"{output_prefix}_um{idx}.png"
            cv2.imwrite(out_path, result)
            print(f"✅ Saved unsharp mask result (radius={radius}, amount={amount}) → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Histogram matching + unsharp masking.")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-r", "--reference", required=True, help="Reference image path")
    parser.add_argument("-o", "--output", required=True, help="Output prefix (without extension)")
    parser.add_argument("--skip-hist", action="store_true", help="Skip saving histogram matched image")
    parser.add_argument("--unsharp", nargs="+", type=float, default=[1, 1, 5, 2, 20, 1],
                        help="Unsharp params as radius,amount pairs (e.g., --unsharp 1 1 5 2 20 1)")
    args = parser.parse_args()

    # Parse unsharp params into list of (radius, amount)
    if len(args.unsharp) % 2 != 0:
        raise ValueError("Unsharp parameters must be in pairs: radius amount ...")

    unsharp_params = [(args.unsharp[i], args.unsharp[i + 1]) for i in range(0, len(args.unsharp), 2)]

    process_image(args.input, args.reference, args.output,
                  save_hist=not args.skip_hist,
                  unsharp_params=unsharp_params)


if __name__ == "__main__":
    main()

