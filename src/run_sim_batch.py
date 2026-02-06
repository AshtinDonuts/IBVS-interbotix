#!/usr/bin/env python3
"""
Batch runner script to run multiple simulations with different poses.

This script reads a list of poses (6 numbers each) and runs the simulation
for each pose sequentially.

Usage:
    # Run with poses defined in this script
    python src/run_sim_batch.py
    
    # Run with poses from a file
    python src/run_sim_batch.py poses.txt
    
File format (one pose per line, 6 numbers: x y z roll pitch yaw):
    0 0 1.0 0 0 0
    1.0 0.5 2.0 0.1 0.2 0.3
    -0.5 0 1.5 0 0.1 0
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


# Default poses to run if no file is provided
DEFAULT_POSES = [
    # Format: [x, y, z, roll, pitch, yaw]
    [0, 0, 1.0, 0, 0, 0],           # Default pose
    [1.0, 0, 1.0, 0, 0, 0],         # Offset in x
    [0, 0, 1.5, 0, 0, 0],           # Higher z
    [0, 0, 1.0, 0.1, 0, 0],         # Small roll
    [0.5, 0, 1.2, 0, 0.1, 0],       # Combined offset
    [-0.5, 0, 1.0, 0, 0, 0.1],      # Negative x with yaw
]


def load_poses_from_file(filepath: str) -> List[List[float]]:
    """Load poses from a text file."""
    poses = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse the line
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) != 6:
                        print(f"Warning: Line {line_num} has {len(values)} values (expected 6), skipping")
                        continue
                    poses.append(values)
                except ValueError as e:
                    print(f"Warning: Line {line_num} has invalid numbers, skipping: {e}")
                    continue
        
        return poses
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def run_simulation(pose: List[float], pose_num: int, total_poses: int) -> bool:
    """
    Run a single simulation with the given pose.
    
    Returns True if successful, False otherwise.
    """
    x, y, z, roll, pitch, yaw = pose
    
    # Get the script directory
    script_dir = Path(__file__).parent
    main_script = script_dir / "main_sim_lg_gam_chaumette.py" ##
    
    if not main_script.exists():
        print(f"Error: Could not find {main_script}")
        return False
    
    # Build the command
    cmd = [
        sys.executable,
        str(main_script),
        "--robot-pos", str(x), str(y), str(z),
        "--robot-orn", str(roll), str(pitch), str(yaw)
    ]
    
    # Print header
    print("\n" + "=" * 70)
    print(f"SIMULATION {pose_num}/{total_poses}")
    print("=" * 70)
    print(f"Position:    [{x}, {y}, {z}]")
    print(f"Orientation: [{roll}, {pitch}, {yaw}] (radians)")
    print("=" * 70 + "\n")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Simulation {pose_num}/{total_poses} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Simulation {pose_num}/{total_poses} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n✗ Batch interrupted by user")
        raise


def main():
    # Determine which poses to run
    if len(sys.argv) > 1:
        # Load poses from file
        poses_file = sys.argv[1]
        print(f"Loading poses from: {poses_file}")
        poses = load_poses_from_file(poses_file)
        if not poses:
            print("Error: No valid poses found in file")
            sys.exit(1)
        print(f"Loaded {len(poses)} poses from file\n")
    else:
        # Use default poses
        poses = DEFAULT_POSES
        print(f"Using {len(poses)} default poses\n")
    
    # Print summary
    print("=" * 70)
    print("BATCH SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total simulations: {len(poses)}")
    print("\nPoses:")
    for i, pose in enumerate(poses, 1):
        x, y, z, roll, pitch, yaw = pose
        print(f"  {i}. pos=[{x:6.2f}, {y:6.2f}, {z:6.2f}]  orn=[{roll:6.3f}, {pitch:6.3f}, {yaw:6.3f}]")
    print("=" * 70)
    
    input("\nPress Enter to start batch simulation (Ctrl+C to cancel)...")
    
    # Run simulations
    results = []
    try:
        for i, pose in enumerate(poses, 1):
            success = run_simulation(pose, i, len(poses))
            results.append((i, pose, success))
    except KeyboardInterrupt:
        print("\n\nBatch simulation interrupted by user")
        sys.exit(130)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("BATCH SIMULATION COMPLETE")
    print("=" * 70)
    
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total:      {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    
    if failed > 0:
        print("\nFailed simulations:")
        for i, pose, success in results:
            if not success:
                x, y, z, roll, pitch, yaw = pose
                print(f"  {i}. pos=[{x:6.2f}, {y:6.2f}, {z:6.2f}]  orn=[{roll:6.3f}, {pitch:6.3f}, {yaw:6.3f}]")
    
    print("=" * 70 + "\n")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
