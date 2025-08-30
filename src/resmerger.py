import cv2
import numpy as np
import argparse
import os
import time

def merge_tiles(input_dir, output_path, rows, cols, tile_size=256, delay=0):
    """
    Merge tiles into one large image.
    
    Args:
        input_dir (str): Directory containing tile images (named sequentially 0per.png, 1per.png, ...).
        output_path (str): Path to save merged image.
        rows (int): Number of tile rows.
        cols (int): Number of tile columns.
        tile_size (int): Size of each square tile (default=256).
        delay (float): Optional delay every 10 rows (default=0 sec).
    """
    # Preallocate large canvas (single channel, since you only use [:,:,0])
    big_h, big_w = rows * tile_size, cols * tile_size
    merged = np.zeros((big_h, big_w, 1), dtype=np.uint8)

    count = 0
    for i in range(rows):
        if delay > 0 and i % 10 == 0:
            time.sleep(delay)
        for j in range(cols):
            tile_path = os.path.join(input_dir, f"{count}per.png")
            tile = cv2.imread(tile_path, cv2.IMREAD_COLOR)
            if tile is None:
                raise FileNotFoundError(f"Missing tile: {tile_path}")
            merged[i*tile_size:(i+1)*tile_size,
                   j*tile_size:(j+1)*tile_size, 0] = tile[:, :, 0]
            count += 1

    cv2.imwrite(output_path, merged)
    print(f"Merged {count} tiles into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge tiled model outputs into a single large image")
    parser.add_argument("--input_dir", required=True, help="Directory with tile images (e.g. 0per.png, 1per.png...)")
    parser.add_argument("--output", required=True, help="Path to save merged image")
    parser.add_argument("--rows", type=int, required=True, help="Number of tile rows")
    parser.add_argument("--cols", type=int, required=True, help="Number of tile columns")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size (default=256)")
    parser.add_argument("--delay", type=float, default=0, help="Optional delay every 10 rows (sec)")
    args = parser.parse_args()

    merge_tiles(args.input_dir, args.output, args.rows, args.cols, args.tile_size, args.delay)

