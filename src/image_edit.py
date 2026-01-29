#!/usr/bin/env python3
"""
Image editing utility for filling transparent backgrounds and resizing images.

# Place image on 1024x768 canvas, scaled to 50% of canvas dimensions
python src/misc/image_edit.py input.png output.png --canvas-width 1024 --canvas-height 768 --scale 0.5

# Place image on 800x800 canvas, image covers 70% of the canvas
python src/misc/image_edit.py input.png output.png --canvas-width 800 --canvas-height 800 --scale 0.7

# Scale original image to 150% and fill transparent areas
python src/misc/image_edit.py input.png output.png --scale 1.5

# Black background, HD canvas, small image at 30% scale
python src/misc/image_edit.py input.png output.png --bg-color 0,0,0 --canvas-width 1920 --canvas-height 1080 --scale 0.3

# Place image in corner instead of centered
python src/misc/image_edit.py input.png output.png --canvas-width 1000 --canvas-height 1000 --scale 0.6 --no-center

"""

import argparse
from PIL import Image
import os


def fill_transparent_background(image_path, output_path, bg_color=(255, 255, 255), 
                                canvas_width=None, canvas_height=None, 
                                scale=1.0, maintain_aspect=True, center=True):
    """
    Fill transparent background of a PNG image and place it on a canvas.
    
    Args:
        image_path: Path to input PNG image with transparency
        output_path: Path to save the output image
        bg_color: Background color as RGB tuple (default: white)
        canvas_width: Output canvas width (default: original image width)
        canvas_height: Output canvas height (default: original image height)
        scale: Scale factor for the base image relative to canvas (default: 1.0)
        maintain_aspect: If True, maintain aspect ratio when scaling (default: True)
        center: If True, center the image on the canvas (default: True)
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    original_width, original_height = img.size
    
    # Determine canvas size
    if canvas_width is None and canvas_height is None:
        # No canvas size specified, use scaled image size
        canvas_width = int(original_width * scale)
        canvas_height = int(original_height * scale)
    elif canvas_width is None:
        canvas_width = canvas_height if maintain_aspect else original_width
    elif canvas_height is None:
        canvas_height = canvas_width if maintain_aspect else original_height
    
    # Calculate scaled image size relative to canvas
    if scale != 1.0:
        # Scale the image relative to the canvas size
        target_width = int(canvas_width * scale)
        target_height = int(canvas_height * scale)
        
        if maintain_aspect:
            # Maintain aspect ratio - fit within target dimensions
            aspect_ratio = original_width / original_height
            if target_width / aspect_ratio <= target_height:
                # Width is the limiting factor
                scaled_width = target_width
                scaled_height = int(target_width / aspect_ratio)
            else:
                # Height is the limiting factor
                scaled_height = target_height
                scaled_width = int(target_height * aspect_ratio)
        else:
            scaled_width = target_width
            scaled_height = target_height
        
        # Resize using high-quality resampling
        img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        print(f"Scaled image from {original_width}x{original_height} to {scaled_width}x{scaled_height}")
        print(f"Scale factor: {scale:.2f} ({scale*100:.1f}% of canvas)")
    else:
        scaled_width, scaled_height = original_width, original_height
    
    # Create canvas with background color
    canvas = Image.new('RGBA', (canvas_width, canvas_height), bg_color + (255,))
    
    # Calculate position to paste the image
    if center:
        x_offset = (canvas_width - scaled_width) // 2
        y_offset = (canvas_height - scaled_height) // 2
    else:
        x_offset = 0
        y_offset = 0
    
    # Paste the image onto the canvas
    canvas.paste(img, (x_offset, y_offset), img)
    
    # Convert to RGB (removes alpha channel)
    result = canvas.convert('RGB')
    
    # Save the result
    result.save(output_path)
    print(f"Saved image to: {output_path}")
    print(f"Canvas size: {canvas_width}x{canvas_height}")
    print(f"Image size: {scaled_width}x{scaled_height}")
    print(f"Image coverage: {(scaled_width*scaled_height)/(canvas_width*canvas_height)*100:.1f}% of canvas")
    print(f"Background color: RGB{bg_color}")
    print(f"Position: {'centered' if center else f'({x_offset}, {y_offset})'}")


def parse_color(color_string):
    """Parse color string in format 'R,G,B' to tuple."""
    try:
        r, g, b = map(int, color_string.split(','))
        if not all(0 <= c <= 255 for c in [r, g, b]):
            raise ValueError("Color values must be between 0 and 255")
        return (r, g, b)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid color format. Use 'R,G,B' (e.g., '255,255,255'): {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Fill transparent background of PNG images and place on a canvas with scaling.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', help='Input PNG image with transparency')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--bg-color', type=parse_color, default=(255, 255, 255),
                        help='Background color in R,G,B format (default: 255,255,255 for white)')
    parser.add_argument('--canvas-width', type=int, 
                        help='Output canvas width in pixels (default: original image width * scale)')
    parser.add_argument('--canvas-height', type=int, 
                        help='Output canvas height in pixels (default: original image height * scale)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for image relative to canvas (default: 1.0). '
                             'E.g., 0.5 = image covers 50%% of canvas, 2.0 = image is 2x canvas size')
    parser.add_argument('--no-aspect', action='store_false', dest='maintain_aspect',
                        help='Do not maintain aspect ratio when scaling')
    parser.add_argument('--no-center', action='store_false', dest='center',
                        help='Do not center the image on canvas (place at top-left instead)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # Check if input is a PNG
    if not args.input.lower().endswith('.png'):
        print("Warning: Input file may not be a PNG image")
    
    try:
        fill_transparent_background(
            args.input,
            args.output,
            bg_color=args.bg_color,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            scale=args.scale,
            maintain_aspect=args.maintain_aspect,
            center=args.center
        )
        print("Success!")
        return 0
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
