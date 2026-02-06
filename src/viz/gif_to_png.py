#!/usr/bin/env python3
"""
Script to convert a GIF file into its constituent PNG frames.
"""

import argparse
from pathlib import Path
from PIL import Image


def gif_to_png(gif_path, output_dir=None, prefix="frame"):
    """
    Convert a GIF file into individual PNG frames.
    
    Args:
        gif_path: Path to the input GIF file
        output_dir: Directory to save PNG frames (default: same as GIF location)
        prefix: Prefix for output PNG files (default: "frame")
    """
    gif_path = Path(gif_path)
    
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF file not found: {gif_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = gif_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the GIF
    with Image.open(gif_path) as gif:
        print(f"Processing GIF: {gif_path}")
        print(f"Output directory: {output_dir}")
        
        frame_count = 0
        try:
            while True:
                # Save current frame as PNG
                frame_filename = output_dir / f"{prefix}_{frame_count:03d}.png"
                gif.save(frame_filename, "PNG")
                print(f"Saved frame {frame_count}: {frame_filename}")
                
                frame_count += 1
                gif.seek(gif.tell() + 1)  # Move to next frame
                
        except EOFError:
            # End of GIF reached
            pass
    
    print(f"\nSuccessfully extracted {frame_count} frames from {gif_path.name}")
    return frame_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GIF file into individual PNG frames"
    )
    parser.add_argument(
        "gif_path",
        type=str,
        help="Path to the input GIF file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNG frames (default: same as GIF location)"
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="frame",
        help="Prefix for output PNG filenames (default: 'frame')"
    )
    
    args = parser.parse_args()
    
    try:
        gif_to_png(args.gif_path, args.output_dir, args.prefix)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
