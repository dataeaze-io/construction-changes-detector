import cv2
import os
import argparse


def split_images(before_path, after_path, label_path, output_dir, patch_size=256, shift=0):
    """
    Split large images into smaller patches.

    Args:
        before_path (str): Path to "before" image.
        after_path (str): Path to "after" image.
        label_path (str): Path to label/mask image.
        output_dir (str): Output directory where subfolders A, B, label will be created.
        patch_size (int): Patch size (default=256).
        shift (int): Optional horizontal shift for overlapped crops (default=0).
    """
    # Read images
    before = cv2.imread(before_path, cv2.IMREAD_COLOR)
    after = cv2.imread(after_path, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    if before is None or after is None or label is None:
        raise FileNotFoundError("One or more input images not found.")

    print(f"Input shapes -> before: {before.shape}, after: {after.shape}, label: {label.shape}")

    # Create output dirs
    for sub in ["A", "B", "label"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    h, w = before.shape[:2]
    hr, wr = h // patch_size, w // patch_size

    print(f"Splitting into {hr} x {wr} patches -> Total = {hr * wr}")

    count = 0
    for i in range(hr):
        for j in range(wr):
            x0, y0 = j * patch_size + shift, i * patch_size
            x1, y1 = x0 + patch_size, y0 + patch_size

            if x1 > w or y1 > h:
                continue  # skip incomplete tiles

            n1 = before[y0:y1, x0:x1]
            n2 = after[y0:y1, x0:x1]
            n3 = label[y0:y1, x0:x1]

            cv2.imwrite(os.path.join(output_dir, "A", f"w{count}.png"), n1)
            cv2.imwrite(os.path.join(output_dir, "B", f"w{count}.png"), n2)
            cv2.imwrite(os.path.join(output_dir, "label", f"w{count}.png"), n3)
            count += 1

    print(f"Done! Saved {count} patches to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large before/after/label images into patches")
    parser.add_argument("--before", required=True, help="Path to before image (p1.png)")
    parser.add_argument("--after", required=True, help="Path to after image (p2mse.png)")
    parser.add_argument("--label", required=True, help="Path to label image (mask.png)")
    parser.add_argument("--output_dir", required=True, help="Output directory to save patches")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size (default=256)")
    parser.add_argument("--shift", type=int, default=0, help="Optional horizontal shift (default=0)")
    args = parser.parse_args()

    split_images(args.before, args.after, args.label, args.output_dir, args.patch_size, args.shift)

