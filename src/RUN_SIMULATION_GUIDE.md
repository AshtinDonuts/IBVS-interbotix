# Simulation Runner Scripts Guide

This guide explains how to use the new scripts for running the IBVS simulation with custom robot poses.

## Overview

Four scripts are available for running simulations:

1. **`main_sim_lg_gam.py`** - The main simulation script (now with command-line args)
2. **`run_sim_with_pose.py`** - Convenience wrapper for running a single simulation
3. **`run_sim_batch.py`** - Batch runner for multiple simulations with different poses
4. **`generate_poses_lhs.py`** - Generate poses using Latin Hypercube Sampling for better parameter space coverage

## 1. Running a Single Simulation

### Method A: Using main_sim_lg_gam.py directly

```bash
# Use default pose (0, 0, 1.0) with no rotation
python src/main_sim_lg_gam.py

# Specify custom position
python src/main_sim_lg_gam.py --robot-pos 1.0 0.5 2.0

# Specify custom position and orientation
python src/main_sim_lg_gam.py --robot-pos 1.0 0.5 2.0 --robot-orn 0.1 0.2 0.3

# Get help
python src/main_sim_lg_gam.py --help
```

**Arguments:**
- `--robot-pos X Y Z`: Initial robot position in meters (default: 0 0 1.0)
- `--robot-orn ROLL PITCH YAW`: Initial robot orientation in radians (default: 0 0 0)

### Method B: Using run_sim_with_pose.py (simplified)

This script takes exactly 6 numbers in order: x, y, z, roll, pitch, yaw

```bash
# Run with 6 numbers: x y z roll pitch yaw
python src/run_sim_with_pose.py 1.0 0.5 2.0 0.1 0.2 0.3

# Run with default pose
python src/run_sim_with_pose.py 0 0 1.0 0 0 0
```

This automatically constructs the proper command-line arguments and provides clear output showing what pose is being used.

## 2. Running Multiple Simulations (Batch Mode)

### Method A: Using default poses

The batch runner has built-in default poses for testing:

```bash
python src/run_sim_batch.py
```

This will run through 6 pre-defined test poses.

### Method B: Using a custom poses file

Create a text file with one pose per line (6 numbers: x y z roll pitch yaw):

```bash
python src/run_sim_batch.py path/to/poses.txt
```

**Example poses file format (`example_poses.txt`):**

```
# Comments start with #
# Format: x y z roll pitch yaw

# Default pose
0 0 1.0 0 0 0

# Offset in x
1.0 0 1.0 0 0 0

# Higher altitude
0 0 1.5 0 0 0

# With rotation
0 0 1.0 0.1 0.1 0
```

An example file is provided at `src/example_poses.txt`.

### Batch Runner Features

- Shows a summary of all poses before starting
- Runs simulations sequentially
- Reports success/failure for each simulation
- Provides final summary with statistics
- Can be interrupted with Ctrl+C

## 3. Generating Poses with Latin Hypercube Sampling

### What is Latin Hypercube Sampling (LHS)?

Latin Hypercube Sampling is a statistical method for generating near-random samples that provide better coverage of the parameter space than pure random sampling. It's particularly useful for:

- **Parameter sweeps**: Systematically exploring different configurations
- **Statistical analysis**: Getting representative samples with fewer runs
- **Experimental design**: Ensuring even coverage across all parameter ranges

### Basic Usage

Generate 200 poses with default parameter ranges:

```bash
python src/generate_poses_lhs.py -n 200 -o poses_lhs_200.txt
```

### Custom Parameter Ranges

Specify custom bounds for each parameter:

```bash
python src/generate_poses_lhs.py \
    -n 100 \
    --x-range -1.5 1.5 \
    --y-range -1.0 1.0 \
    --z-range 0.8 2.5 \
    --roll-range -0.5 0.5 \
    --pitch-range -0.5 0.5 \
    --yaw-range -0.785 0.785 \
    -o poses_custom.txt
```

### Reproducibility

Use a fixed random seed for reproducible results:

```bash
python src/generate_poses_lhs.py -n 200 --seed 42 -o poses_reproducible.txt
```

### Output Features

The script generates:

1. **Poses file** (`.txt`) - Ready to use with `run_sim_batch.py`
2. **Statistics** - Min, max, mean, std for each parameter
3. **Visualizations** (if matplotlib available):
   - Histograms showing distribution of each parameter
   - 2D scatter plots for position (x-y, x-z, y-z planes)
   - 2D scatter plots for orientation (roll-pitch, roll-yaw, pitch-yaw)

### Default Ranges

- **Position (meters)**:
  - x: [-1.0, 1.0] (left/right)
  - y: [0.0, 0.0] (forward/backward - constant at 0)
  - z: [0.5, 2.0] (height)

- **Orientation (radians)**:
  - roll: [-0.3, 0.3] (≈ -17° to +17°)
  - pitch: [-0.3, 0.3] (≈ -17° to +17°)
  - yaw: [-0.5, 0.5] (≈ -29° to +29°)

### Complete Workflow Example

```bash
# 1. Generate 200 poses with LHS
python src/generate_poses_lhs.py -n 200 --seed 42 -o my_experiment.txt

# 2. Review the statistics and visualizations
# Check the output statistics and plots in my_experiment_viz/

# 3. Run all simulations in batch
python src/run_sim_batch.py my_experiment.txt
```

## 4. Understanding Pose Parameters

### Position (x, y, z)
- **x**: Left/right position (meters)
- **y**: Forward/backward position (meters)  
- **z**: Up/down position (meters)
- Default: `[0, 0, 1.0]` (1 meter above ground, centered)

### Orientation (roll, pitch, yaw)
- **roll**: Rotation around x-axis (radians)
- **pitch**: Rotation around y-axis (radians)
- **yaw**: Rotation around z-axis (radians)
- Default: `[0, 0, 0]` (no rotation, looking straight ahead)
- Note: π radians = 180 degrees, so 0.1 radians ≈ 5.7 degrees

## 5. Examples

### Example 1: Test different starting heights
```bash
python src/run_sim_with_pose.py 0 0 0.8 0 0 0   # Low
python src/run_sim_with_pose.py 0 0 1.0 0 0 0   # Default
python src/run_sim_with_pose.py 0 0 1.5 0 0 0   # High
```

### Example 2: Test different orientations
```bash
python src/run_sim_with_pose.py 0 0 1.0 0.1 0 0     # Tilted roll
python src/run_sim_with_pose.py 0 0 1.0 0 0.1 0     # Tilted pitch
python src/run_sim_with_pose.py 0 0 1.0 0 0 0.1     # Rotated yaw
```

### Example 3: Run a batch of experiments
```bash
# Create a custom poses file
cat > my_experiments.txt << EOF
0 0 1.0 0 0 0
0.5 0 1.0 0 0 0
1.0 0 1.0 0 0 0
0 0 1.5 0 0 0
EOF

# Run the batch
python src/run_sim_batch.py my_experiments.txt
```

### Example 4: Generate systematic test poses
```bash
# Create poses at different heights
for z in 0.8 1.0 1.2 1.4 1.6; do
    echo "0 0 $z 0 0 0"
done > height_test.txt

python src/run_sim_batch.py height_test.txt
```

### Example 5: Full LHS experimental workflow
```bash
# Generate 200 poses with Latin Hypercube Sampling
python src/generate_poses_lhs.py \
    -n 200 \
    --seed 42 \
    --x-range -0.8 0.8 \
    --z-range 0.8 1.8 \
    -o experiments/lhs_study_200.txt

# Review the generated visualizations
ls experiments/lhs_study_200_viz/

# Run all 200 simulations in batch
python src/run_sim_batch.py experiments/lhs_study_200.txt

# Analyze results from the logs directory
ls src/logs/
```

### Example 6: Narrow parameter ranges for focused study
```bash
# Generate poses with small variations around the default
python src/generate_poses_lhs.py \
    -n 50 \
    --x-range -0.2 0.2 \
    --y-range -0.1 0.1 \
    --z-range 0.9 1.1 \
    --roll-range -0.1 0.1 \
    --pitch-range -0.1 0.1 \
    --yaw-range -0.1 0.1 \
    --seed 123 \
    -o near_default_50.txt

python src/run_sim_batch.py near_default_50.txt
```

## 6. Output and Logs

All simulations save their data to `src/logs/` with auto-incrementing filenames:
- `ibvs_sim_001.csv`
- `ibvs_sim_002.csv`
- etc.

Each log file contains:
- Iteration-by-iteration metrics
- Robot pose throughout the simulation
- Feature matching statistics
- Error values

## 7. Tips

1. **Start simple**: Begin with the default pose to ensure everything works
2. **Small changes**: Make small adjustments to pose parameters to understand their effect
3. **Check logs**: Review the CSV logs to analyze convergence behavior
4. **Batch testing**: Use batch mode for systematic parameter sweeps
5. **Use LHS for experiments**: For parameter studies, use Latin Hypercube Sampling instead of grid search for better efficiency
6. **Reproducibility**: Always use `--seed` when generating poses for reproducible experiments
7. **Interruption**: All scripts handle Ctrl+C gracefully for safe interruption

## 8. Troubleshooting

**Problem**: Script says "No valid poses found"
- **Solution**: Check your poses file format (6 numbers per line, space-separated)

**Problem**: Simulation fails immediately
- **Solution**: Check if the robot starts inside an obstacle or too far from the scene

**Problem**: Batch runner skips some lines
- **Solution**: Check the warnings - lines might have wrong number of values or invalid numbers

## 9. See Also

- Main simulation: `src/main_sim_lg_gam.py`
- Single pose runner: `src/run_sim_with_pose.py`
- Batch runner: `src/run_sim_batch.py`
- LHS pose generator: `src/generate_poses_lhs.py`
- Example poses: `src/example_poses.txt`
- Project README: `../README.md`
