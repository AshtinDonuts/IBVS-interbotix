# IBVS Simulation - Quick Reference Card

## Single Simulation

```bash
# Run with default pose
python src/main_sim_lg_gam.py

# Run with custom pose
python src/main_sim_lg_gam.py --robot-pos 1.0 0.5 2.0 --robot-orn 0.1 0.2 0.3

# Simplified wrapper (6 numbers: x y z roll pitch yaw)
python src/run_sim_with_pose.py 1.0 0.5 2.0 0.1 0.2 0.3
```

## Batch Simulations

```bash
# Run with default poses
python src/run_sim_batch.py

# Run with custom poses file
python src/run_sim_batch.py poses.txt
```

## Generate Poses (Latin Hypercube Sampling)

```bash
# Generate 200 poses with defaults
python src/generate_poses_lhs.py -n 200 -o poses_lhs_200.txt

# Generate with custom ranges and fixed seed
python src/generate_poses_lhs.py \
    -n 200 \
    --x-range -1.5 1.5 \
    --z-range 0.8 2.0 \
    --seed 42 \
    -o my_poses.txt

# All options
python src/generate_poses_lhs.py --help
```

## Complete Workflow

```bash
# 1. Generate poses
python src/generate_poses_lhs.py -n 200 --seed 42 -o experiment.txt

# 2. Run batch simulations
python src/run_sim_batch.py experiment.txt

# 3. Analyze logs
ls src/logs/
```

## File Formats

### Poses File Format
```
# Comments start with #
# Format: x y z roll pitch yaw
0 0 1.0 0 0 0
1.0 0.5 1.2 0.1 0 0.05
```

## Default Parameter Ranges (LHS Generator)

| Parameter | Range | Unit | Notes |
|-----------|-------|------|-------|
| x | [-1.0, 1.0] | meters | Left/right |
| y | [0.0, 0.0] | meters | Forward/backward (constant) |
| z | [0.5, 2.0] | meters | Height |
| roll | [-0.3, 0.3] | radians | ≈ ±17° |
| pitch | [-0.3, 0.3] | radians | ≈ ±17° |
| yaw | [-0.5, 0.5] | radians | ≈ ±29° |

## Quick Examples

```bash
# Test different heights
python src/run_sim_with_pose.py 0 0 0.8 0 0 0   # Low
python src/run_sim_with_pose.py 0 0 1.5 0 0 0   # High

# Test rotations
python src/run_sim_with_pose.py 0 0 1.0 0.1 0 0     # Roll
python src/run_sim_with_pose.py 0 0 1.0 0 0.1 0     # Pitch
python src/run_sim_with_pose.py 0 0 1.0 0 0 0.1     # Yaw

# Generate and run 50 experiments
python src/generate_poses_lhs.py -n 50 -o test.txt --seed 42
python src/run_sim_batch.py test.txt
```

## Output Locations

- **Logs**: `src/logs/ibvs_sim_XXX.csv`
- **Images**: `src/img/rgbimage_X.png`
- **Visualizations**: `<poses_file>_viz/`

## Tips

- Use `--seed` for reproducible LHS generation
- Start with small batch sizes (n=10-20) for testing
- Check visualizations to verify parameter coverage
- Use Ctrl+C to safely interrupt any script
- Review CSV logs for convergence analysis

## For More Details

See `src/RUN_SIMULATION_GUIDE.md` for comprehensive documentation.
