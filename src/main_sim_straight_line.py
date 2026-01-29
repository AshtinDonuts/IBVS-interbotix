"""
Simple forward motion simulation.

The robot moves forward in a straight line and captures images at each step.
No IBVS control or feature matching - just basic forward motion.
"""

import sys
from time import sleep
from typing import Tuple, List
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import cv2

from image import convert_img_to_arr, get_image_config


# Configuration
MAX_ITERATIONS = 40
OUTPUT_DIR = Path('/home/khw/IBVS-interbotix/src/straight_line_img')
TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/resized_cat.png')

# Forward velocity (in camera frame, positive y is forward)
FORWARD_VELOCITY = 1.0  # units per timestep
DT = 0.05  # timestep


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
    
    # Offset for cube placement
    new_offsetx = 0.75
    new_offsetz = 0.75
    
    # Build a wall of cubes
    for z_offset in [0, -1, 1]:  # z is up/down
        for y_offset in [2.5]:  # y is forward-backward
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
    
    # Set up scene
    _, _ = init_scene(robot_pos)
    
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
        
        print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}] saved to {save_path.name}")
        
        # Update position (move forward)
        robot_pos = update_position_forward(
            robot_pos, robot_rot_matrix, FORWARD_VELOCITY, DT
        )
        
        sleep(0.01)  # Small delay for visualization
    
    print(f"\nSimulation complete!")
    print(f"Final position: {robot_pos}")
    print(f"Images saved to: {OUTPUT_DIR}")
    
    p.disconnect()
    sys.exit(0)


if __name__ == "__main__":
    main()
