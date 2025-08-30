import cv2
import argparse
import os


def histogram_equalization(input_path: str, output_path: str):
    """
    Perform histogram equalization on a grayscale image.
    
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the equalized image.
    """
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Apply histogram equalization
    equ = cv2.equalizeHist(img)

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, equ)
    print(f"Equalized image saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Histogram Equalization using OpenCV")
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save the equalized image"
    )
    args = parser.parse_args()

    histogram_equalization(args.input, args.output)


if __name__ == "__main__":
    main()

