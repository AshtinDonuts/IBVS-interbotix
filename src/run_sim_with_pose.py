#!/usr/bin/env python3
"""
Convenience script to run main_sim_lg_gam.py with a pose specified as 6 numbers.

Usage:
    python src/run_sim_with_pose.py 1.0 0.5 2.0 0.1 0.2 0.3
    
Where the 6 numbers are:
    [x, y, z, roll, pitch, yaw]
"""

import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) != 7:
        print("Error: Expected exactly 6 numbers (x, y, z, roll, pitch, yaw)")
        print("\nUsage:")
        print("    python src/run_sim_with_pose.py <x> <y> <z> <roll> <pitch> <yaw>")
        print("\nExample:")
        print("    python src/run_sim_with_pose.py 1.0 0.5 2.0 0.1 0.2 0.3")
        print("\nOr use default values:")
        print("    python src/run_sim_with_pose.py 0 0 1.0 0 0 0")
        sys.exit(1)
    
    # Parse the 6 numbers
    try:
        x, y, z, roll, pitch, yaw = [float(arg) for arg in sys.argv[1:7]]
    except ValueError as e:
        print(f"Error: All arguments must be numbers. {e}")
        sys.exit(1)
    
    # Get the script directory
    script_dir = Path(__file__).parent
    main_script = script_dir / "main_sim_lg_gam.py"
    
    if not main_script.exists():
        print(f"Error: Could not find {main_script}")
        sys.exit(1)
    
    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(main_script),
        "--robot-pos", str(x), str(y), str(z),
        "--robot-orn", str(roll), str(pitch), str(yaw)
    ]
    
    # Print what we're running
    print(f"Running simulation with:")
    print(f"  Position:    [{x}, {y}, {z}]")
    print(f"  Orientation: [{roll}, {pitch}, {yaw}] (radians)")
    print(f"\nCommand: {' '.join(cmd)}\n")
    print("=" * 60)
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Simulation failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
