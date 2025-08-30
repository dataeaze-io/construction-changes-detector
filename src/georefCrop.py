import argparse
import random
import rasterio
from rasterio.windows import Window


def crop_tiff(input_tif, output_tif, xsize, ysize, xoff=None, yoff=None, randomize=False):
    """Crop a GeoTIFF image to given size and offsets."""
    with rasterio.open(input_tif) as src:
        # Ensure crop fits inside image
        xmin, xmax = 0, src.width - xsize
        ymin, ymax = 0, src.height - ysize

        if randomize:
            xoff = random.randint(xmin, xmax)
            yoff = random.randint(ymin, ymax)
        else:
            # Default to top-left if offsets not provided
            xoff = 0 if xoff is None else min(max(xmin, xoff), xmax)
            yoff = 0 if yoff is None else min(max(ymin, yoff), ymax)

        window = Window(xoff, yoff, xsize, ysize)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": ysize,
            "width": xsize,
            "transform": transform,
        })

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(src.read(window=window))

        print(f"✅ Cropped {input_tif} → {output_tif}")
        print(f"   Window offset: ({xoff}, {yoff}), size: ({xsize}, {ysize})")


def main():
    parser = argparse.ArgumentParser(description="Crop a GeoTIFF image.")
    parser.add_argument("-i", "--input", required=True, help="Input GeoTIFF file")
    parser.add_argument("-o", "--output", required=True, help="Output cropped GeoTIFF file")
    parser.add_argument("--xsize", type=int, required=True, help="Crop width in pixels")
    parser.add_argument("--ysize", type=int, required=True, help="Crop height in pixels")
    parser.add_argument("--xoff", type=int, default=None, help="X offset in pixels (ignored if randomize=True)")
    parser.add_argument("--yoff", type=int, default=None, help="Y offset in pixels (ignored if randomize=True)")
    parser.add_argument("--randomize", action="store_true", help="Randomize crop position")
    args = parser.parse_args()

    crop_tiff(args.input, args.output, args.xsize, args.ysize, args.xoff, args.yoff, args.randomize)


if __name__ == "__main__":
    main()

