import argparse
import rasterio as rio
from rasterio.enums import Resampling
import numpy as np
import os


def png_to_geotiff(png_path, reference_tif, output_tif, dtype="uint16"):
    """
    Convert PNG to GeoTIFF by copying georeferencing info from a reference .tif

    Args:
        png_path (str): Input PNG image path
        reference_tif (str): Reference GeoTIFF with CRS + transform
        output_tif (str): Output GeoTIFF file path
        dtype (str): Output data type (default: uint16)
    """
    # Open PNG
    with rio.open(png_path) as src:
        img = src.read([1])  # read first band only
        img = img.astype(dtype)
        print(f"PNG loaded: {png_path}, shape={img.shape}, dtype={img.dtype}")

    # Reference GeoTIFF
    with rio.open(reference_tif) as ref:
        print(f"Reference CRS: {ref.crs}")
        print(f"Reference transform: {ref.transform}")

        # Ensure output folder exists
        os.makedirs(os.path.dirname(output_tif), exist_ok=True)

        # Write output GeoTIFF
        with rio.open(
            output_tif,
            "w",
            driver="GTiff",
            count=1,
            height=img.shape[1],
            width=img.shape[2],
            dtype=img.dtype,
            crs=ref.crs,
            transform=ref.transform,
        ) as dst:
            dst.write(img)

    print(f"✅ GeoTIFF written: {output_tif}")


def resample_to_reference(input_tif, reference_tif, output_tif, resampling=Resampling.nearest):
    """
    Resample a GeoTIFF so it aligns with reference (pixels overlap).
    Useful when dataset grids differ.
    """
    with rio.open(reference_tif) as ref:
        with rio.open(input_tif) as src:
            data = src.read(
                out_shape=(
                    src.count,
                    ref.height,
                    ref.width
                ),
                resampling=resampling
            )
            transform = ref.transform
            print(f"Resampled shape: {data.shape}")

            with rio.open(
                output_tif,
                "w",
                driver="GTiff",
                count=data.shape[0],
                height=data.shape[1],
                width=data.shape[2],
                dtype=data.dtype,
                crs=ref.crs,
                transform=transform,
            ) as dst:
                dst.write(data)

    print(f"✅ Resampled GeoTIFF written: {output_tif}")


def main():
    parser = argparse.ArgumentParser(description="Convert PNG to GeoTIFF using reference TIFF georeferencing.")
    parser.add_argument("-p", "--png", required=True, help="Input PNG file")
    parser.add_argument("-r", "--reference", required=True, help="Reference GeoTIFF file (with CRS/transform)")
    parser.add_argument("-o", "--output", required=True, help="Output GeoTIFF file")
    parser.add_argument("--dtype", default="uint16", help="Output datatype (default: uint16)")
    parser.add_argument("--resample", action="store_true", help="Also resample PNG GeoTIFF to match reference grid")
    args = parser.parse_args()

    png_to_geotiff(args.png, args.reference, args.output, args.dtype)

    if args.resample:
        resample_path = args.output.replace(".tif", "_resampled.tif")
        resample_to_reference(args.output, args.reference, resample_path)


if __name__ == "__main__":
    main()

