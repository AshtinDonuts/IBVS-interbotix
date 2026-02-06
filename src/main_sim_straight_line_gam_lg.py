"""
Simple forward motion simulation with GSAM2 + LightGlue matching.

The robot moves forward in a straight line and captures images at each step.
Uses Grounded SAM2 to segment the target object in both the target and current images,
then performs LightGlue feature matching on the appearance-preserving masked RGB images.
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


# Uasge Configurations
MAX_ITERATIONS = 40
GSAM2_TEXT_PROMPT = "lego."
TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/resized_lego.png')

OUTPUT_DIR = Path('/home/khw/IBVS-interbotix/src/straight_line_img') 
FORWARD_VELOCITY = 2.5  # units per timestep
DT = 0.05

# GSAM2 Configuration
ENABLE_GSAM2 = True  # Set to True to enable GSAM2 segmentation

GSAM2_SIMILARITY_THRESHOLD = 0.15  # Report if similarity is within this threshold (15%)
GSAM2_BOX_THRESHOLD = 0.35  # Detection confidence
GSAM2_TEXT_THRESHOLD = 0.25  # Text matching confidence

# LightGlue Configuration
ENABLE_LIGHTGLUE = True  # Set to True to enable LightGlue feature matching on masked images


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
            'gsam2_enabled',
            'gsam2_mask_size',
            'gsam2_similarity',
            'gsam2_detected',
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
        gsam2_enabled: bool = False,
        gsam2_mask_size: Optional[float] = None,
        gsam2_similarity: Optional[float] = None,
        gsam2_detected: Optional[bool] = None,
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
        gsam2_enabled : bool
            Whether GSAM2 is enabled.
        gsam2_mask_size : float, optional
            Current mask size from GSAM2.
        gsam2_similarity : float, optional
            Similarity score from GSAM2.
        gsam2_detected : bool, optional
            Whether object was detected by GSAM2.
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
            gsam2_enabled,
            f"{gsam2_mask_size:.0f}" if gsam2_mask_size is not None else "",
            f"{gsam2_similarity:.6f}" if gsam2_similarity is not None else "",
            gsam2_detected if gsam2_detected is not None else "",
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


def get_masked_rgb_image(terminator, image_path: str) -> Optional[np.ndarray]:
    """
    Get appearance-preserving masked RGB image using GSAM2.
    
    Parameters
    ----------
    terminator : SegmentationTerminator
        The GSAM2 terminator instance
    image_path : str
        Path to the image file
        
    Returns
    -------
    np.ndarray or None
        Masked RGB image with background zeroed out. Returns None if no object detected.
    """
    from PIL import Image
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Set image for SAM2
    terminator.sam2_predictor.set_image(image_np)
    
    # Run Grounding DINO detection
    inputs = terminator.processor(images=image, text=terminator.text_prompt, return_tensors="pt").to(terminator.device)
    
    with torch.no_grad():
        outputs = terminator.grounding_model(**inputs)
    
    results = terminator.processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=terminator.box_threshold,
        text_threshold=terminator.text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    # Check if any objects were detected
    if len(results) == 0 or len(results[0]["boxes"]) == 0:
        print(f"Warning: No objects detected in {image_path}")
        return None
    
    # Get bounding boxes for SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()
    
    # Get segmentation masks
    masks, scores, logits = terminator.sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Convert shape to (n, H, W) if needed
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # Combine all masks (logical OR)
    combined_mask = np.any(masks, axis=0)
    
    # Apply mask to RGB image (preserve appearance)
    masked_rgb = image_np.copy()
    masked_rgb[~combined_mask] = 0  # Set background to black
    
    return masked_rgb


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
    
    # Use local variables to track enabled state (avoid scope issues with module-level constants)
    gsam2_enabled = ENABLE_GSAM2
    lightglue_enabled = ENABLE_LIGHTGLUE
    
    # Initialize GSAM2 segmentation if enabled
    gsam2_terminator = None
    target_masked_rgb_path = None
    
    if gsam2_enabled:
        from gsam2_terminator import SegmentationTerminator
        print("\n" + "="*60)
        print("Initializing Grounded SAM2 for segmentation...")
        print("="*60 + "\n")
        gsam2_terminator = SegmentationTerminator(
            text_prompt=GSAM2_TEXT_PROMPT,
            similarity_threshold=GSAM2_SIMILARITY_THRESHOLD,
            box_threshold=GSAM2_BOX_THRESHOLD,
            text_threshold=GSAM2_TEXT_THRESHOLD
        )
        
        # Set target mask size for tracking
        gsam2_terminator.set_target_mask_size(str(TARGET_PATH))
        
        # Create masked target image for LightGlue
        if lightglue_enabled:
            print(f"\nCreating appearance-preserving masked target image...")
            target_masked_rgb = get_masked_rgb_image(gsam2_terminator, str(TARGET_PATH))
            if target_masked_rgb is not None:
                # Save masked target for LightGlue
                target_masked_rgb_path = OUTPUT_DIR / 'target_masked.png'
                cv2.imwrite(str(target_masked_rgb_path), cv2.cvtColor(target_masked_rgb, cv2.COLOR_RGB2BGR))
                print(f"Masked target image saved to: {target_masked_rgb_path}")
            else:
                print("WARNING: Could not create masked target image - LightGlue will be disabled")
                lightglue_enabled = False
        
        print(f"\nGSAM2 enabled - will segment '{GSAM2_TEXT_PROMPT}' objects")
        print(f"Similarity threshold: {GSAM2_SIMILARITY_THRESHOLD:.1%}\n")
    
    # Verify LightGlue setup
    if lightglue_enabled:
        if not gsam2_enabled:
            print("WARNING: LightGlue requires GSAM2 to be enabled. Disabling LightGlue.")
            lightglue_enabled = False
        elif not TARGET_PATH.exists():
            print(f"ERROR: Target image not found at {TARGET_PATH}")
            sys.exit(1)
        else:
            print("\n" + "="*60)
            print("LightGlue Feature Matching Enabled")
            print("="*60)
            print(f"Target image: {TARGET_PATH}")
            print(f"Masked target: {target_masked_rgb_path}")
            print("Will match features on appearance-preserving masked images")
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
        
        # Initialize GSAM2 and LightGlue variables
        gsam2_mask_size = None
        gsam2_similarity = None
        gsam2_detected = None
        num_matches = None
        
        # Process with GSAM2 and LightGlue if enabled
        current_masked_rgb_path = None
        if gsam2_enabled and gsam2_terminator is not None:
            try:
                # Get current mask size
                gsam2_mask_size = gsam2_terminator.get_mask_size(str(save_path))
                gsam2_detected = gsam2_mask_size > 0
                
                # Calculate similarity if object detected
                if gsam2_detected and gsam2_terminator.target_mask_size is not None:
                    target_size = gsam2_terminator.target_mask_size
                    if gsam2_mask_size > 0 and target_size > 0:
                        gsam2_similarity = min(gsam2_mask_size, target_size) / max(gsam2_mask_size, target_size)
                
                # Create masked current image for LightGlue
                if lightglue_enabled and gsam2_detected:
                    current_masked_rgb = get_masked_rgb_image(gsam2_terminator, str(save_path))
                    if current_masked_rgb is not None:
                        # Save masked current image
                        current_masked_rgb_path = OUTPUT_DIR / f'frame_{i:03d}_masked.png'
                        cv2.imwrite(str(current_masked_rgb_path), cv2.cvtColor(current_masked_rgb, cv2.COLOR_RGB2BGR))
                        
                        # Match features with LightGlue on masked images
                        try:
                            m_kpts0, m_kpts1 = match_superpoints(target_masked_rgb_path, current_masked_rgb_path)
                            
                            # Count number of matches
                            if isinstance(m_kpts0, torch.Tensor):
                                num_matches = m_kpts0.shape[0]
                            elif isinstance(m_kpts0, np.ndarray):
                                num_matches = m_kpts0.shape[0]
                            else:
                                num_matches = 0
                            
                            sim_str = f"{gsam2_similarity:.2%}" if gsam2_similarity is not None else "N/A"
                            print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                                  f"mask={gsam2_mask_size:.0f}, sim={sim_str}, "
                                  f"matches={num_matches}")
                        except Exception as e:
                            sim_str = f"{gsam2_similarity:.2%}" if gsam2_similarity is not None else "N/A"
                            print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                                  f"mask={gsam2_mask_size:.0f}, sim={sim_str}, "
                                  f"matches=ERROR ({e})")
                            num_matches = 0
                    else:
                        sim_str = f"{gsam2_similarity:.2%}" if gsam2_similarity is not None else "N/A"
                        print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                              f"mask={gsam2_mask_size:.0f}, sim={sim_str}, "
                              f"matches=N/A (no mask)")
                else:
                    # GSAM2 only (no LightGlue)
                    sim_str = f"{gsam2_similarity:.2%}" if gsam2_similarity is not None else "N/A"
                    print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                          f"mask={gsam2_mask_size:.0f}, sim={sim_str}")
                
            except Exception as e:
                print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}], "
                      f"ERROR: {e}")
        else:
            # Just print position if GSAM2 is disabled
            print(f"Frame {i:03d}: pos=[{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
        
        # Log iteration data
        data_logger.log_iteration(
            iteration=i,
            robot_pos=robot_pos,
            robot_orn=robot_orientation,
            distance_traveled=distance_traveled,
            image_path=str(save_path),
            gsam2_enabled=gsam2_enabled,
            gsam2_mask_size=gsam2_mask_size,
            gsam2_similarity=gsam2_similarity,
            gsam2_detected=gsam2_detected,
            lightglue_enabled=lightglue_enabled,
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
    
    if gsam2_enabled and gsam2_terminator is not None:
        print(f"\nGSAM2 Statistics:")
        print(f"  Target mask size: {gsam2_terminator.target_mask_size:.0f} pixels")
        print(f"  Similarity threshold: {gsam2_terminator.similarity_threshold:.1%}")
        print(f"  Text prompt: '{gsam2_terminator.text_prompt}'")
    
    if lightglue_enabled:
        print(f"\nLightGlue Feature Matching:")
        print(f"  Target masked image: {target_masked_rgb_path}")
        print(f"  Matched features on appearance-preserving masked images")
        print(f"  Match counts logged to CSV")
    
    print(f"{'='*60}\n")
    
    p.disconnect()
    sys.exit(0)


if __name__ == "__main__":
    main()
