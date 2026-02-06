"""
Example usage of gsam2_terminator module with both termination modes.

This demonstrates how to use:
1. Mask Size Comparison Mode: Compares segmentation mask sizes
2. SSIM Mode: Compares structural similarity of segmented objects

Both modes use Grounded SAM2 to first segment the target object,
then apply different comparison metrics.
"""

from pathlib import Path
from gsam2_terminator import TerminationHandler

# Example 1: Using mask size comparison (default)
# This requires Grounded SAM2 models to be loaded
def example_mask_size_mode():
    print("=" * 70)
    print("EXAMPLE 1: Mask Size Comparison Mode")
    print("=" * 70)
    
    handler = TerminationHandler(
        target_image_path="/path/to/target/image.png",
        mode='mask_size',  # Compare segmentation mask sizes
        text_prompt="cube.",  # What object to detect
        similarity_threshold=0.15,  # 15% difference allowed
        box_threshold=0.35,
        text_threshold=0.25,
        enabled=True
    )
    
    # In your main loop:
    for i in range(100):
        current_image_path = f"/path/to/current/image_{i}.png"
        
        # Check termination condition
        if handler.check_termination(current_image_path, iteration=i):
            print("Terminated based on mask size similarity!")
            break
    
    # Get statistics
    stats = handler.get_stats()
    print(f"Handler stats: {stats}")


# Example 2: Using SSIM mode on segmented objects
# Compares structural similarity of the segmented object regions
def example_ssim_mode():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: SSIM (Structural Similarity) Mode")
    print("=" * 70)
    
    handler = TerminationHandler(
        target_image_path="/path/to/target/image.png",
        mode='ssim',  # Compare using Structural Similarity Index
        ssim_threshold=0.85,  # SSIM threshold (0-1, higher = more similar)
        enabled=True
    )
    
    # In your main loop:
    for i in range(100):
        current_image_path = f"/path/to/current/image_{i}.png"
        
        # Check termination condition
        if handler.check_termination(current_image_path, iteration=i):
            print("Terminated based on SSIM similarity!")
            break
    
    # Get statistics
    stats = handler.get_stats()
    print(f"Handler stats: {stats}")


# Example 3: Disabling termination (for debugging or testing)
def example_disabled_mode():
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Disabled Mode")
    print("=" * 70)
    
    handler = TerminationHandler(
        target_image_path="/path/to/target/image.png",
        enabled=False  # Termination checking disabled
    )
    
    # check_termination will always return False
    for i in range(100):
        current_image_path = f"/path/to/current/image_{i}.png"
        
        if handler.check_termination(current_image_path, iteration=i):
            print("Will never reach here")
            break


# Example 4: Using in main_sim.py context
def example_main_sim_usage():
    """
    Example of how to use in main_sim.py
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Usage in main_sim.py")
    print("=" * 70)
    
    # Option A: Mask size mode (original implementation)
    handler_mask = TerminationHandler(
        target_image_path="/home/khw/IBVS-interbotix/assets/resized_cat.png",
        mode='mask_size',
        text_prompt="cube.",
        similarity_threshold=0.15,
        enabled=True
    )
    
    # Option B: SSIM mode (faster, no GPU needed)
    handler_ssim = TerminationHandler(
        target_image_path="/home/khw/IBVS-interbotix/assets/resized_cat.png",
        mode='ssim',
        ssim_threshold=0.85,
        enabled=True
    )
    
    print("\nIn your main loop, replace:")
    print("    if termination_handler.check_termination(save_impath, iteration=i):")
    print("        break")
    print("\nThe handler automatically uses the correct mode!")


if __name__ == "__main__":
    print("This is an example file showing different usage modes.")
    print("Uncomment the examples below to run them:\n")
    
    # example_mask_size_mode()
    # example_ssim_mode()
    # example_disabled_mode()
    example_main_sim_usage()
    
    print("\n" + "=" * 70)
    print("COMPARISON: When to use each mode")
    print("=" * 70)
    print("""
    Mask Size Mode:
    ✓ Simple metric: just counts pixels in segmentation mask
    ✓ Good for scale/distance-based termination
    ✓ Less sensitive to object appearance/pose changes
    ✓ Fast comparison (after segmentation)
    ✗ Doesn't consider object structure or appearance
    ✗ Can be fooled by partial occlusions
    
    SSIM Mode (on segmented objects):
    ✓ Considers structural similarity of the object itself
    ✓ Better for pose-based termination (object looks similar)
    ✓ Captures appearance, texture, and structure
    ✓ More robust to partial occlusions
    ✗ More sensitive to viewpoint changes
    ✗ Slightly slower comparison (SSIM computation)
    
    Both modes:
    • Require Grounded SAM2 models (GPU recommended)
    • Focus comparison on target object only (not background)
    • Require model downloads (~1GB)
    
    Recommendation:
    - Use Mask Size when you care about distance/scale to object
    - Use SSIM when you care about viewing the object from correct pose/angle
    - For IBVS: Mask Size is usually better (scale indicates correct distance)
    """)
