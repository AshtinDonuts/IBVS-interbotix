"""
Simple forward motion simulation with LightGlue feature matching.

The robot moves forward in a straight line and captures images at each step.
At each step, LightGlue is used to match features between the current frame
and the target image, and the number of matches is logged.
"""

import sys
from time import sleep
from typing import Tuple, List, Optional
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import cv2
import csv
import torch

from image import convert_img_to_arr, get_image_config
from superpoint_utils import match_superpoints


# Configuration
MAX_ITERATIONS = 40
OUTPUT_DIR = Path('/home/khw/IBVS-interbotix/src/straight_line_img')
TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/resized_lego.png')

# Forward velocity (in camera frame, positive y is forward)
FORWARD_VELOCITY = 2.5  # units per timestep
DT = 0.05  # timestep

# LightGlue Configuration
ENABLE_LIGHTGLUE = True  # Set to True to enable LightGlue feature matching
MATCH_THRESHOLD = 0.1  # LightGlue match confidence threshold


class DataLogger:
    """
    Logger for recording simulation metrics at each timestep.
    """
    
    def __init__(self, log_dir: Path, experiment_name: str = None):
        """
        Initialize the data logger.
        
        Parameters
        ----------
        log_dir : Path
            Directory where log files will be saved.
        experiment_name : str, optional
            Name for this experiment run. If None, uses default name.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name
        if experiment_name is None:
            experiment_name = "straight_line"
        
        # Find next available file number
        existing_files = list(self.log_dir.glob(f"{experiment_name}_*.csv"))
        if existing_files:
            # Extract numbers from existing files
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.stem.split('_')[-1])
                    numbers.append(num)
                except ValueError:
                    continue
            next_num = max(numbers) + 1 if numbers else 1
        else:
            next_num = 1
        
        # Create CSV file
        self.log_file = self.log_dir / f"{experiment_name}_{next_num:03d}.csv"
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow([
            'iteration',
            'robot_pos_x',
            'robot_pos_y',
            'robot_pos_z',
            'robot_orn_x',
            'robot_orn_y',
            'robot_orn_z',
            'distance_traveled',
            'lightglue_enabled',
            'num_matches',
            'image_path'
        ])
        self.csv_file.flush()
        
        print(f"Data logger initialized. Log file: {self.log_file}")
    
    def log_iteration(
        self,
        iteration: int,
        robot_pos: List[float],
        robot_orn: List[float],
        distance_traveled: float,
        image_path: str,
        lightglue_enabled: bool = False,
        num_matches: Optional[int] = None
    ):
        """
        Log data for a single iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        robot_pos : List[float]
            Current robot position [x, y, z].
        robot_orn : List[float]
            Current robot orientation [roll, pitch, yaw].
        distance_traveled : float
            Total distance traveled from start.
        image_path : str
            Path to the saved image for this iteration.
        lightglue_enabled : bool
            Whether LightGlue matching is enabled.
        num_matches : int, optional
            Number of feature matches found by LightGlue.
        """
        # Write row
        self.csv_writer.writerow([
            iteration,
            f"{robot_pos[0]:.6f}",
            f"{robot_pos[1]:.6f}",
            f"{robot_pos[2]:.6f}",
            f"{robot_orn[0]:.6f}",
            f"{robot_orn[1]:.6f}",
            f"{robot_orn[2]:.6f}",
            f"{distance_traveled:.6f}",
            lightglue_enabled,
            num_matches if num_matches is not None else "",
            image_path
        ])
        self.csv_file.flush()
    
    def close(self):
        """Close the log file."""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print(f"Data log saved to: {self.log_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def init_pybullet() -> int:
    """Initialize the pybullet scene."""
    pclient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # Turn off gravity
    return pclient


def set_aruco_marker_texture(obstacle_id: int) -> None:
    """Set the target texture on the obstacle."""
    texture_id = p.loadTexture(str(TARGET_PATH))
    p.changeVisualShape(obstacle_id, -1, textureUniqueId=texture_id)


def init_scene(robot_pos: List[float]) -> Tuple[int, List[int]]:
    """
    Initialize the scene with plane and cubes.
    
    Returns
    -------
    plane_id : int
        The pybullet id of the plane.
    obstacles : list[int]
        List of pybullet ids for all obstacle cubes.
    """
    plane_id = p.loadURDF("plane.urdf")
    
    obstacles = []
    base = robot_pos
    base_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    # Offset to prevent physics issues due to clipping
    new_offsetx = 0.75
    new_offsetz = 0.75
    
    # Build a wall of cubes
    for z_offset in [0, -1, 1]:  # z is up/down
        for y_offset in [6.0]:  # y is forward-backward
            for x_offset in [0, -1, 1]:  # x is left-right
                body_id = p.loadURDF(
                    "cube_small.urdf",
                    [
                        # base[0] + x_offset + new_offsetx,
                        # base[1] + y_offset,
                        # base[2] + z_offset + new_offsetz,
                        base[0] + x_offset,
                        base[1] + y_offset,
                        base[2] + z_offset,
                    ],
                    base_orn,
                    globalScaling=20,
                )
                obstacles.append(body_id)
    
    # Texture the first cube (goal)
    if obstacles:
        goal_obs_id = obstacles[0]
        set_aruco_marker_texture(goal_obs_id)
    
    return plane_id, obstacles


def get_robot_rotation_matrix(robot_orientation: List[float]) -> np.ndarray:
    """Get the robot rotation matrix from orientation."""
    robot_rotation_matrix = p.getMatrixFromQuaternion(
        p.getQuaternionFromEuler(robot_orientation)
    )
    return np.array(robot_rotation_matrix).reshape(3, 3)


def get_view_matrix(
    init_camera_vector: Tuple[int, int, int],
    init_up_vector: Tuple[int, int, int],
    robot_pos: List[float],
    robot_rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Get the view matrix for the camera."""
    camera_vector = np.dot(robot_rotation_matrix, init_camera_vector)
    up_vector = np.dot(robot_rotation_matrix, init_up_vector)
    return p.computeViewMatrix(robot_pos, robot_pos + camera_vector, up_vector)


def get_projection_matrix() -> np.ndarray:
    """Get the projection matrix for the camera."""
    image_conf = get_image_config()
    return p.computeProjectionMatrixFOV(
        image_conf["fov"],
        image_conf["aspect"],
        image_conf["near_val"],
        image_conf["far_val"],
    )


def capture_camera_image(
    robot_pos: List[float], robot_rotation_matrix: np.ndarray
) -> Tuple:
    """
    Capture an image from the camera.
    
    Returns
    -------
    img_details : tuple
        (width, height, rgbImg, depthImg, segImg)
    """
    image_conf = get_image_config()
    
    # Initial camera vectors
    init_camera_vector = (0, 1, 0)  # y axis
    init_up_vector = (0, 0, 1)  # z axis
    
    # Calculate view and projection matrices
    view_matrix = get_view_matrix(
        init_camera_vector,
        init_up_vector,
        robot_pos,
        robot_rotation_matrix,
    )
    projection_matrix = get_projection_matrix()
    
    # Capture the image
    img_details = p.getCameraImage(
        image_conf["width"],
        image_conf["height"],
        view_matrix,
        projection_matrix,
    )
    return img_details


def update_position_forward(
    robot_pos: List[float],
    robot_rotation_matrix: np.ndarray,
    forward_velocity: float,
    dt: float
) -> List[float]:
    """
    Update robot position by moving forward in the camera frame.
    
    Parameters
    ----------
    robot_pos : List[float]
        Current robot position [x, y, z]
    robot_rotation_matrix : np.ndarray
        3x3 rotation matrix
    forward_velocity : float
        Forward velocity in camera frame
    dt : float
        Timestep
        
    Returns
    -------
    new_pos : List[float]
        Updated robot position
    """
    # Camera frame: forward is +y direction
    camera_forward = np.array([0, 1, 0])
    
    # Transform to world frame
    world_forward = robot_rotation_matrix @ camera_forward
    
    # Update position
    displacement = world_forward * forward_velocity * dt
    new_pos = [
        robot_pos[0] + displacement[0],
        robot_pos[1] + displacement[1],
        robot_pos[2] + displacement[2]
    ]
    
    return new_pos


def main() -> None:
    """Main execution loop."""
    # Initialize PyBullet
    _ = init_pybullet()
    img_conf = get_image_config()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize robot position and orientation
    robot_pos = [0, 0, 1.75]  # [x, y, z]
    robot_orientation = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]
    initial_pos = robot_pos.copy()  # Store for distance calculation
    
    # Set up scene
    _, _ = init_scene(robot_pos)
    
    # Initialize data logger
    log_dir = Path(__file__).parent / 'logs'
    data_logger = DataLogger(log_dir, experiment_name='straight_line')
    
    # Verify target image exists
    if ENABLE_LIGHTGLUE:
        if not TARGET_PATH.exists():
            print(f"ERROR: Target image not found at {TARGET_PATH}")
            sys.exit(1)
        print("\n" + "="*60)
        print("LightGlue Feature Matching Enabled")
        print("="*60)
        print(f"Target image: {TARGET_PATH}")
        print(f"Match threshold: {MATCH_THRESHOLD}")
        print("="*60 + "\n")
    
    print(f"Starting straight line motion simulation")
    print(f"Initial position: {robot_pos}")
    print(f"Initial orientation: {robot_orientation}")
    print(f"Forward velocity: {FORWARD_VELOCITY}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    sleep(1)  # Let the scene load
    
    for i in range(MAX_ITERATIONS):
        p.stepSimulation()
        
        # Get rotation matrix
        robot_rot_matrix = get_robot_rotation_matrix(robot_orientation)
        
        # Capture image
        img = capture_camera_image(robot_pos, robot_rot_matrix)
        
        # Extract RGB image
        rgba = img[2]  # rgbaImg : (h x w x 4)
        rgba_arr = convert_img_to_arr(
            rgba, int(img_conf["height"]), int(img_conf["width"])
        )
        rgb_img_arr = rgba_arr[:, :, :3]  # Remove alpha channel
        
        # Save image
        save_path = OUTPUT_DIR / f'frame_{i:03d}.png'
        cv2.imwrite(str(save_path), cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR))
        
        # Calculate distance traveled
        distance_traveled = np.linalg.norm(np.array(robot_pos) - np.array(initial_pos))
        
        # Initialize LightGlue variables
        num_matches = None
        
        # Match features with LightGlue if enabled
        if ENABLE_LIGHTGLUE:
            try:
                # Match current frame with target image
                m_kpts0, m_kpts1 = match_superpoints(TARGET_PATH, save_path)
                
                # Count number of matches
                if isinstance(m_kpts0, torch.Tensor):
                    num_matches = m_kpts0.shape[0]
                elif isinstance(m_kpts0, np.ndarray):
                    num_matches = m_kpts0.shape[0]
                else:
                    num_matches = 0
                
                print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                      f"matches={num_matches}, dist={distance_traveled:.3f}")
            except Exception as e:
                print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                      f"matches=ERROR ({e})")
                num_matches = 0
        else:
            # Just print position if LightGlue is disabled
            print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
        
        # Log iteration data
        data_logger.log_iteration(
            iteration=i,
            robot_pos=robot_pos,
            robot_orn=robot_orientation,
            distance_traveled=distance_traveled,
            image_path=str(save_path),
            lightglue_enabled=ENABLE_LIGHTGLUE,
            num_matches=num_matches
        )
        
        # Update position (move forward)
        robot_pos = update_position_forward(
            robot_pos, robot_rot_matrix, FORWARD_VELOCITY, DT
        )
        
        sleep(0.01)  # Small delay for visualization
    
    # Close data logger
    data_logger.close()
    
    print(f"\n{'='*60}")
    print(f"Simulation complete!")
    print(f"{'='*60}")
    print(f"Final position: {robot_pos}")
    print(f"Final distance traveled: {distance_traveled:.3f} units")
    print(f"Total frames: {MAX_ITERATIONS}")
    print(f"Images saved to: {OUTPUT_DIR}")
    if ENABLE_LIGHTGLUE:
        print(f"\nLightGlue Feature Matching:")
        print(f"  Target image: {TARGET_PATH}")
        print(f"  Match threshold: {MATCH_THRESHOLD}")
        print(f"  Matched features per frame logged to CSV")
    print(f"{'='*60}\n")
    
    p.disconnect()
    sys.exit(0)


if __name__ == "__main__":
    main()
