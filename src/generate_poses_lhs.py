#!/usr/bin/env python3
"""
Generate robot poses using Latin Hypercube Sampling (LHS).

This script generates a set of robot poses that provide good coverage of the
parameter space. LHS ensures better distribution than random sampling by
dividing each parameter's range into equal intervals and sampling once from each.

Usage:
    # Generate 200 poses with default bounds
    python src/generate_poses_lhs.py -n 200 -o poses_lhs_200.txt
    
    # Generate with custom bounds
    python src/generate_poses_lhs.py -n 100 --x-range -1 1 --z-range 0.5 2.0
    
    # See all options
    python src/generate_poses_lhs.py --help
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.stats import qmc
from datetime import datetime


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate robot poses using Latin Hypercube Sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Number of samples
    parser.add_argument('-n', '--num-samples', type=int, default=200,
                        help='Number of poses to generate')
    
    # Output file
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: poses_lhs_N_TIMESTAMP.txt)')
    
    # Position bounds
    parser.add_argument('--x-range', type=float, nargs=2, default=[-1.0, 1.0],
                        metavar=('MIN', 'MAX'),
                        help='Range for x position (meters)')
    parser.add_argument('--y-range', type=float, nargs=2, default=[0.0, 0.0],
                        metavar=('MIN', 'MAX'),
                        help='Range for y position (meters). Default is constant at 0.0')
    parser.add_argument('--z-range', type=float, nargs=2, default=[0.5, 2.0],
                        metavar=('MIN', 'MAX'),
                        help='Range for z position (meters)')
    
    # Orientation bounds (in radians)
    parser.add_argument('--roll-range', type=float, nargs=2, default=[-0.3, 0.3],
                        metavar=('MIN', 'MAX'),
                        help='Range for roll orientation (radians)')
    parser.add_argument('--pitch-range', type=float, nargs=2, default=[-0.3, 0.3],
                        metavar=('MIN', 'MAX'),
                        help='Range for pitch orientation (radians)')
    parser.add_argument('--yaw-range', type=float, nargs=2, default=[-0.5, 0.5],
                        metavar=('MIN', 'MAX'),
                        help='Range for yaw orientation (radians)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Scrambling
    parser.add_argument('--no-scramble', action='store_true',
                        help='Disable scrambling (scrambling improves randomness)')
    
    return parser.parse_args()


def generate_lhs_poses(
    n_samples: int,
    x_range: tuple,
    y_range: tuple,
    z_range: tuple,
    roll_range: tuple,
    pitch_range: tuple,
    yaw_range: tuple,
    seed: int = None,
    scramble: bool = True
) -> np.ndarray:
    """
    Generate robot poses using Latin Hypercube Sampling.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    x_range, y_range, z_range : tuple
        (min, max) bounds for position coordinates
    roll_range, pitch_range, yaw_range : tuple
        (min, max) bounds for orientation angles (radians)
    seed : int, optional
        Random seed for reproducibility
    scramble : bool
        Whether to scramble the samples (improves randomness)
    
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 6) with columns [x, y, z, roll, pitch, yaw]
    """
    # Define all parameter ranges
    all_ranges = [x_range, y_range, z_range, roll_range, pitch_range, yaw_range]
    param_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    
    # Identify which parameters are constant (min == max) and which vary
    varying_indices = []
    varying_bounds_l = []
    varying_bounds_u = []
    constant_indices = []
    constant_values = []
    
    for i, (param_range, name) in enumerate(zip(all_ranges, param_names)):
        if param_range[0] == param_range[1]:
            # Constant parameter
            constant_indices.append(i)
            constant_values.append(param_range[0])
        else:
            # Varying parameter
            varying_indices.append(i)
            varying_bounds_l.append(param_range[0])
            varying_bounds_u.append(param_range[1])
    
    # Number of varying dimensions
    n_varying = len(varying_indices)
    
    # Initialize output array
    samples_scaled = np.zeros((n_samples, 6))
    
    if n_varying > 0:
        # Create Latin Hypercube Sampler for varying dimensions only
        # Note: scramble parameter may not be available in older scipy versions
        try:
            sampler = qmc.LatinHypercube(d=n_varying, seed=seed, scramble=scramble)
        except TypeError:
            # Fall back to version without scramble parameter
            sampler = qmc.LatinHypercube(d=n_varying, seed=seed)
        
        # Generate samples in [0, 1]^n_varying
        samples_unit = sampler.random(n=n_samples)
        
        # Scale samples to actual parameter ranges
        l_bounds = np.array(varying_bounds_l)
        u_bounds = np.array(varying_bounds_u)
        samples_varying_scaled = qmc.scale(samples_unit, l_bounds, u_bounds)
        
        # Place varying samples in the correct columns
        for i, idx in enumerate(varying_indices):
            samples_scaled[:, idx] = samples_varying_scaled[:, i]
    
    # Fill in constant values
    for idx, val in zip(constant_indices, constant_values):
        samples_scaled[:, idx] = val
    
    return samples_scaled


def save_poses(poses: np.ndarray, output_path: str, args: argparse.Namespace):
    """
    Save poses to a text file with metadata header.
    
    Parameters
    ----------
    poses : np.ndarray
        Array of poses (n_samples x 6)
    output_path : str
        Path to output file
    args : argparse.Namespace
        Command-line arguments for metadata
    """
    with open(output_path, 'w') as f:
        # Write header with metadata
        f.write("# Robot poses generated using Latin Hypercube Sampling\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Number of samples: {len(poses)}\n")
        f.write(f"# Random seed: {args.seed if args.seed is not None else 'None (random)'}\n")
        f.write(f"# Scrambled: {not args.no_scramble}\n")
        f.write("#\n")
        f.write("# Parameter ranges:\n")
        f.write(f"#   x:     [{args.x_range[0]:7.3f}, {args.x_range[1]:7.3f}] meters\n")
        f.write(f"#   y:     [{args.y_range[0]:7.3f}, {args.y_range[1]:7.3f}] meters\n")
        f.write(f"#   z:     [{args.z_range[0]:7.3f}, {args.z_range[1]:7.3f}] meters\n")
        f.write(f"#   roll:  [{args.roll_range[0]:7.3f}, {args.roll_range[1]:7.3f}] radians ({np.degrees(args.roll_range[0]):6.2f}°, {np.degrees(args.roll_range[1]):6.2f}°)\n")
        f.write(f"#   pitch: [{args.pitch_range[0]:7.3f}, {args.pitch_range[1]:7.3f}] radians ({np.degrees(args.pitch_range[0]):6.2f}°, {np.degrees(args.pitch_range[1]):6.2f}°)\n")
        f.write(f"#   yaw:   [{args.yaw_range[0]:7.3f}, {args.yaw_range[1]:7.3f}] radians ({np.degrees(args.yaw_range[0]):6.2f}°, {np.degrees(args.yaw_range[1]):6.2f}°)\n")
        f.write("#\n")
        f.write("# Format: x y z roll pitch yaw\n")
        f.write("#\n")
        
        # Write poses
        for pose in poses:
            x, y, z, roll, pitch, yaw = pose
            f.write(f"{x:8.5f} {y:8.5f} {z:8.5f} {roll:8.5f} {pitch:8.5f} {yaw:8.5f}\n")


def print_statistics(poses: np.ndarray):
    """Print statistics about the generated poses."""
    labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    units = ['m', 'm', 'm', 'rad', 'rad', 'rad']
    
    print("\n" + "="*70)
    print("GENERATED POSES STATISTICS")
    print("="*70)
    print(f"{'Parameter':<10} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Unit':>6}")
    print("-"*70)
    
    for i, (label, unit) in enumerate(zip(labels, units)):
        col = poses[:, i]
        print(f"{label:<10} {col.min():10.5f} {col.max():10.5f} "
              f"{col.mean():10.5f} {col.std():10.5f} {unit:>6}")
    
    print("="*70)


def visualize_distribution(poses: np.ndarray, output_dir: Path):
    """
    Create visualization plots of the pose distribution.
    
    Parameters
    ----------
    poses : np.ndarray
        Array of poses (n_samples x 6)
    output_dir : Path
        Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        labels = ['x (m)', 'y (m)', 'z (m)', 'roll (rad)', 'pitch (rad)', 'yaw (rad)']
        
        # 1. Histograms for each dimension
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Latin Hypercube Sampling - Univariate Distributions', fontsize=14)
        
        for i, (ax, label) in enumerate(zip(axes.flat, labels)):
            ax.hist(poses[:, i], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{label} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = output_dir / 'lhs_distributions.png'
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved histogram plot: {hist_path}")
        plt.close()
        
        # 2. 2D scatter plots for position (x, y, z)
        fig = plt.figure(figsize=(15, 5))
        
        # x-y plane
        ax1 = fig.add_subplot(131)
        ax1.scatter(poses[:, 0], poses[:, 1], alpha=0.5, s=20)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Position: X-Y Plane')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # x-z plane
        ax2 = fig.add_subplot(132)
        ax2.scatter(poses[:, 0], poses[:, 2], alpha=0.5, s=20)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('z (m)')
        ax2.set_title('Position: X-Z Plane')
        ax2.grid(True, alpha=0.3)
        
        # y-z plane
        ax3 = fig.add_subplot(133)
        ax3.scatter(poses[:, 1], poses[:, 2], alpha=0.5, s=20)
        ax3.set_xlabel('y (m)')
        ax3.set_ylabel('z (m)')
        ax3.set_title('Position: Y-Z Plane')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = output_dir / 'lhs_position_scatter.png'
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved scatter plot: {scatter_path}")
        plt.close()
        
        # 3. 2D scatter plots for orientation
        fig = plt.figure(figsize=(15, 5))
        
        # roll-pitch
        ax1 = fig.add_subplot(131)
        ax1.scatter(poses[:, 3], poses[:, 4], alpha=0.5, s=20)
        ax1.set_xlabel('roll (rad)')
        ax1.set_ylabel('pitch (rad)')
        ax1.set_title('Orientation: Roll-Pitch')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # roll-yaw
        ax2 = fig.add_subplot(132)
        ax2.scatter(poses[:, 3], poses[:, 5], alpha=0.5, s=20)
        ax2.set_xlabel('roll (rad)')
        ax2.set_ylabel('yaw (rad)')
        ax2.set_title('Orientation: Roll-Yaw')
        ax2.grid(True, alpha=0.3)
        
        # pitch-yaw
        ax3 = fig.add_subplot(133)
        ax3.scatter(poses[:, 4], poses[:, 5], alpha=0.5, s=20)
        ax3.set_xlabel('pitch (rad)')
        ax3.set_ylabel('yaw (rad)')
        ax3.set_title('Orientation: Pitch-Yaw')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        orient_path = output_dir / 'lhs_orientation_scatter.png'
        plt.savefig(orient_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved orientation plot: {orient_path}")
        plt.close()
        
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization plots")
    except Exception as e:
        print(f"\nWarning: Could not create visualization plots: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate arguments
    if args.num_samples <= 0:
        print("Error: Number of samples must be positive")
        return 1
    
    # Determine output path
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"poses_lhs_{args.num_samples}_{timestamp}.txt"
    else:
        output_path = args.output
    
    output_path = Path(output_path)
    
    # Print configuration
    print("="*70)
    print("LATIN HYPERCUBE SAMPLING - POSE GENERATOR")
    print("="*70)
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed if args.seed is not None else 'None (random)'}")
    print(f"Scrambling: {not args.no_scramble}")
    print(f"\nParameter ranges:")
    print(f"  Position:")
    print(f"    x: [{args.x_range[0]:7.3f}, {args.x_range[1]:7.3f}] meters")
    print(f"    y: [{args.y_range[0]:7.3f}, {args.y_range[1]:7.3f}] meters")
    print(f"    z: [{args.z_range[0]:7.3f}, {args.z_range[1]:7.3f}] meters")
    print(f"  Orientation:")
    print(f"    roll:  [{args.roll_range[0]:7.3f}, {args.roll_range[1]:7.3f}] radians "
          f"({np.degrees(args.roll_range[0]):6.2f}°, {np.degrees(args.roll_range[1]):6.2f}°)")
    print(f"    pitch: [{args.pitch_range[0]:7.3f}, {args.pitch_range[1]:7.3f}] radians "
          f"({np.degrees(args.pitch_range[0]):6.2f}°, {np.degrees(args.pitch_range[1]):6.2f}°)")
    print(f"    yaw:   [{args.yaw_range[0]:7.3f}, {args.yaw_range[1]:7.3f}] radians "
          f"({np.degrees(args.yaw_range[0]):6.2f}°, {np.degrees(args.yaw_range[1]):6.2f}°)")
    print("="*70)
    
    # Generate poses
    print(f"\nGenerating {args.num_samples} poses using Latin Hypercube Sampling...")
    poses = generate_lhs_poses(
        n_samples=args.num_samples,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
        roll_range=tuple(args.roll_range),
        pitch_range=tuple(args.pitch_range),
        yaw_range=tuple(args.yaw_range),
        seed=args.seed,
        scramble=not args.no_scramble
    )
    print("✓ Poses generated successfully")
    
    # Save poses
    print(f"\nSaving poses to: {output_path}")
    save_poses(poses, str(output_path), args)
    print("✓ Poses saved successfully")
    
    # Print statistics
    print_statistics(poses)
    
    # Generate visualizations
    print("\nGenerating visualization plots...")
    viz_dir = output_path.parent / f"{output_path.stem}_viz"
    visualize_distribution(poses, viz_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"Number of poses: {args.num_samples}")
    print(f"\nTo run simulations with these poses:")
    print(f"  python src/run_sim_batch.py {output_path}")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
