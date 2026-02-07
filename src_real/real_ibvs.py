#!/usr/bin/env python3

# Copyright 2024 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Real-world IBVS implementation using LightGlue, GAM, and Chaumette control.

This is a real-world version of main_sim_lg_gam_chaumette.py that:
- Uses RealSense camera for live RGB and depth feed
- Uses Interbotix robot for control
- Implements Chaumette IBVS control with real depth
- Uses appearance-preserving masking for LightGlue matching
- Uses Grounded SAM2 for termination checking


    conda env : ibvs2

    To get started, open a terminal and type:

    ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=aloha_vx300s

    Then change to this directory and type:

    python3 real_ibvs.py

"""

import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2
import yaml
import csv
import torch
from time import sleep
from PIL import Image

# Add the src directory to Python path for imports
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from image import convert_img_to_arr, save_image, get_image_config
from superpoint_utils import match_superpoints
import motion_utils
from motion import LAMBDA
from gsam2_terminator import TerminationHandler, SegmentationTerminator

# ROS 2 and Interbotix imports
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# RealSense imports
import pyrealsense2 as rs

# Configuration
K = 20  # Number of keypoints to sample
ENABLE_APPEARANCE_MASK = True  # Enable appearance-preserving mask segmentation before LightGlue matching
GSAM2_TEXT_PROMPT_FOR_MASK = "green ball."  # Text prompt for GSAM2 segmentation


class DataLogger:
    """
    Logger for recording real-world experiment metrics at each timestep.
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
            experiment_name = "real_ibvs_experiment"
        
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
            'error_magnitude',
            'num_matched_keypoints',
            'gam_similarity',
            'ssim_score',
            'robot_pos_x',
            'robot_pos_y',
            'robot_pos_z',
            'robot_orn_roll',
            'robot_orn_pitch',
            'robot_orn_yaw',
            'velocity_x',
            'velocity_y',
            'velocity_z',
            'velocity_rx',
            'velocity_ry',
            'velocity_rz',
            'image_path'
        ])
        self.csv_file.flush()
        
        print(f"Data logger initialized. Log file: {self.log_file}")
    
    def log_iteration(
        self,
        iteration: int,
        error_magnitude: float,
        robot_pos: List[float],
        robot_orn: List[float],
        num_matched_keypoints: int,
        velocity: np.ndarray,
        image_path: str,
        gam_similarity: float = 0.0,
        ssim_score: float = 0.0
    ):
        """
        Log data for a single iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        error_magnitude : float
            MSE error magnitude.
        robot_pos : List[float]
            Current robot position [x, y, z].
        robot_orn : List[float]
            Current robot orientation [roll, pitch, yaw].
        num_matched_keypoints : int
            Number of matched keypoints between live and target.
        velocity : np.ndarray
            Computed velocity vector (6,).
        image_path : str
            Path to the saved image for this iteration.
        gam_similarity : float
            GAM (mask-based) similarity score.
        ssim_score : float
            SSIM similarity score.
        """
        # Write row
        self.csv_writer.writerow([
            iteration,
            f"{error_magnitude:.6f}",
            num_matched_keypoints,
            f"{gam_similarity:.6f}",
            f"{ssim_score:.6f}",
            f"{robot_pos[0]:.6f}",
            f"{robot_pos[1]:.6f}",
            f"{robot_pos[2]:.6f}",
            f"{robot_orn[0]:.6f}",
            f"{robot_orn[1]:.6f}",
            f"{robot_orn[2]:.6f}",
            f"{velocity[0]:.6f}",
            f"{velocity[1]:.6f}",
            f"{velocity[2]:.6f}",
            f"{velocity[3]:.6f}",
            f"{velocity[4]:.6f}",
            f"{velocity[5]:.6f}",
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


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / 'src' / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


# def get_initial_pose() -> np.ndarray:
#     """Returns a well known valid pose for the robot arm"""
#     T_sd = np.identity(4)
#     T_sd[0, 3] = 0.2
#     T_sd[1, 3] = 0
#     T_sd[2, 3] = 0.25
#     return T_sd

def get_initial_joint_position() -> np.ndarray:
    """Returns a well known valid pose for the robot arm"""
    T_sd = np.identity(4)
    T_sd[0, 3] = 0.2
    T_sd[1, 3] = 0
    T_sd[2, 3] = 0.25
    return T_sd


def test_bound(ee_pose) -> bool:
    """Check if robot end-effector is within safe bounds"""
    x_bound = 0.4
    if ee_pose[0, 3] > x_bound:
        print('Bound reached!\nTerminating...')
        return False
    else:
        return True


def get_masked_rgb_image(segmentation_terminator, image_path: str) -> Optional[np.ndarray]:
    """
    Get appearance-preserving masked RGB image using GSAM2.
    
    This function segments the target object in the image and returns a masked
    RGB image where the background is zeroed out, but the object's appearance
    (RGB values) is preserved. This allows feature matchers like LightGlue to
    focus on the object of interest while maintaining visual features.
    
    Parameters
    ----------
    segmentation_terminator : SegmentationTerminator
        The GSAM2 terminator instance with loaded models
    image_path : str
        Path to the image file to be masked
        
    Returns
    -------
    np.ndarray or None
        Masked RGB image (H x W x 3) with background zeroed out. 
        Returns None if no object detected.
    """
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    image_np = np.array(image)
    
    # Set image for SAM2
    segmentation_terminator.sam2_predictor.set_image(image_np)
    
    # Run Grounding DINO detection
    inputs = segmentation_terminator.processor(
        images=image, 
        text=segmentation_terminator.text_prompt, 
        return_tensors="pt"
    ).to(segmentation_terminator.device)
    
    with torch.no_grad():
        outputs = segmentation_terminator.grounding_model(**inputs)
    
    results = segmentation_terminator.processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=segmentation_terminator.box_threshold,
        text_threshold=segmentation_terminator.text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    # Check if any objects were detected
    if len(results) == 0 or len(results[0]["boxes"]) == 0:
        print(f"Warning: No objects detected in image")
        return None
    
    # Get bounding boxes for SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()
    
    # Get segmentation masks
    masks, scores, logits = segmentation_terminator.sam2_predictor.predict(
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


def convert_realsense_depth_to_meters(depth_frame: rs.depth_frame, depth_image: np.ndarray) -> np.ndarray:
    """
    Convert RealSense depth frame to depth in meters.
    
    Parameters
    ----------
    depth_frame : rs.depth_frame
        RealSense depth frame
    depth_image : np.ndarray
        Depth image array from RealSense
        
    Returns
    -------
    np.ndarray
        Depth in meters (H, W)
    """
    # RealSense depth is already in millimeters, convert to meters
    depth_meters = depth_image.astype(np.float32) / 1000.0
    return depth_meters


def get_ibvs_velocity_realsense(
    src_kpts: np.ndarray,
    tgt_kpts: np.ndarray,
    depth_image: np.ndarray,
    img_config: dict,
    lambda_gain: float = 1.0
) -> np.ndarray:
    """
    Computes camera velocity using Chaumette's classical IBVS control law with RealSense depth.
    v = -lambda * L_s^+ * (s - s*)
    
    Args:
        src_kpts: Current image keypoints (N, 2) in pixels
        tgt_kpts: Target image keypoints (N, 2) in pixels
        depth_image: Depth image from RealSense in meters (H, W)
        img_config: Image configuration dictionary
        lambda_gain: Control gain
        
    Returns:
        velocity: 6D velocity vector [vx, vy, vz, wx, wy, wz]
    """
    # Get camera intrinsics
    width = img_config['width']
    height = img_config['height']
    fov_rad = np.radians(img_config['fov'])
    # Assuming square pixels and principal point at center
    fx = fy = (width / 2.0) / np.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    
    # Interaction matrix and error vector
    L_s_list = []
    error_list = []
    
    for i in range(len(src_kpts)):
        u, v = src_kpts[i]
        u_star, v_star = tgt_kpts[i]
        
        # Convert to normalized coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy
        
        x_star = (u_star - cx) / fx
        y_star = (v_star - cy) / fy
        
        # Get depth at current point
        # Clamp coordinates to image bounds
        u_idx = int(np.clip(u, 0, width - 1))
        v_idx = int(np.clip(v, 0, height - 1))
        
        # RealSense depth is (height, width) and already in meters
        Z = depth_image[v_idx, u_idx]
        
        # Skip if depth is invalid (zero or too far)
        if Z <= 0 or Z > 10.0:  # Skip invalid or very far depths
            continue
        
        # Construct interaction matrix for this point (2x6)
        # L_s = [[-1/Z, 0, x/Z, xy, -(1+x^2), y],
        #        [0, -1/Z, y/Z, 1+y^2, -xy, -x]]
        
        L_point = np.array([
            [-1/Z, 0, x/Z, x*y, -(1+x**2), y],
            [0, -1/Z, y/Z, 1+y**2, -x*y, -x]
        ])
        
        L_s_list.append(L_point)
        
        # Error: s - s*
        error_list.append([x - x_star, y - y_star])
    
    if len(L_s_list) == 0:
        # No valid depth points, return zero velocity
        return np.zeros(6)
    
    # Stack interaction matrices: (2N, 6)
    L_s = np.vstack(L_s_list)
    
    # Stack error vectors: (2N, 1)
    error = np.array(error_list).flatten()
    
    # Compute pseudo-inverse
    L_s_pinv = np.linalg.pinv(L_s)
    
    # Control law: v = -lambda * L_s^+ * e
    vel = -lambda_gain * (L_s_pinv @ error)
    
    return vel


def robot_velocity_control(bot, velocity: np.ndarray, config: dict, dt: float):
    """
    Moves interbotix arm by cartesian control based on input velocity.
    
    Parameters
    ----------
    bot : InterbotixManipulatorXS
        Robot controller instance
    velocity : np.ndarray
        6D velocity vector [vx, vy, vz, wx, wy, wz] in camera frame
        Camera frame: x right, y down, z forward
    config : dict
        Configuration dictionary with scale factors
    dt : float
        Time step for control
    """
    # Transform from camera frame to robot frame
    # Camera frame: x right, y down, z forward
    # Robot frame: x right, y forward, z up
    # Transformation: 
    #   Robot x = Camera x (right)
    #   Robot y = Camera z (forward)
    #   Robot z = -Camera y (up, since camera y is down)
    
    # Extract velocities and transform
    # Note: Following existing real.py convention: dely, delx, delz = velocity[0], velocity[1], velocity[2]
    # But we need to map camera frame correctly
    dely = velocity[2] * dt  # Camera z (forward) -> Robot y (forward)
    delx = velocity[0] * dt  # Camera x (right) -> Robot x (right)
    delz = -velocity[1] * dt  # Camera y (down) -> Robot z (up, negated)
    
    # Flip y direction for interbotix convention (matching existing real.py)
    dely = -1 * dely
    
    # Apply scaling from config
    delx *= config.get('x_scale', 1.0)
    dely *= config.get('y_scale', 1.0)
    delz *= config.get('z_scale', 1.0)
    
    # Angular velocity control
    # Mapping:
    # Roll (around Robot X) <-> wx (around Camera X)
    # Pitch (around Robot Y) <-> wz (around Camera Z) (flipped due to Y flip)
    # Yaw (around Robot Z) <-> -wy (around Camera Y) (flipped due to Z-Y mapping)
    
    del_roll = velocity[3] * dt
    del_pitch = -velocity[5] * dt
    del_yaw = -velocity[4] * dt
    
    # Apply scaling (default to 1.0 if not in config)
    del_roll *= config.get('roll_scale', 1.0)
    del_pitch *= config.get('pitch_scale', 1.0)
    del_yaw *= config.get('yaw_scale', 1.0)
    
    print(f"Moving robot with velocities:")
    print(f"  x: {delx:6.3f}")
    print(f"  y: {dely:6.3f}")
    print(f"  z: {delz:6.3f}")
    print(f"  roll: {del_roll:6.3f}")
    print(f"  pitch: {del_pitch:6.3f}")
    print(f"  yaw: {del_yaw:6.3f}")
    
    # Move by cartesian control
    bot.arm.set_ee_cartesian_trajectory(
        x=delx, y=dely, z=delz,
        roll=del_roll, pitch=del_pitch, yaw=del_yaw
    )


def main():
    """
    Real-world IBVS deployment for Interbotix robot with RealSense camera.
    """
    
    # Load configuration
    config = load_config()
    
    # Load initial config vals
    dt = config.get('dt', 0.1)
    K = config.get('K', K)
    max_iterations = config.get('max_iterations', 200)
    target_image_path = Path(config.get('target_image_path', '/home/khw/IBVS-interbotix/assets/frame_031.png'))
    
    # Create output directories
    save_dir = Path(config.get('save_dir', 'real_output'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize robot
    print("Initializing robot...")
    bot = InterbotixManipulatorXS(
        robot_model='aloha_vx300s',
        group_name='arm',
        gripper_name='gripper',
        require_gravity_torques=False,
    )
    
    robot_startup()
    
    # T_startpose = get_initial_pose()
    # bot.arm.set_ee_pose_matrix(T_startpose)

    initial_joint_position = get_initial_joint_position()
    bot.arm.set_ee_pose_matrix(initial_joint_position)

    # Allow robot to reach pose
    sleep(2)
    
    # Camera stream configs and start stream
    pipeline = rs.pipeline()
    rs_config = rs.config()
    
    # Enable both color and depth streams
    img_conf = get_image_config()
    width = int(img_conf['width'])
    height = int(img_conf['height'])
    
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    
    try:
        pipeline.start(rs_config)
        print('Camera started successfully')
        
        # Warm up camera
        for _ in range(10):
            pipeline.wait_for_frames()
    except Exception as e:
        print(f"Failed to start camera: {e}")
        robot_shutdown()
        return
    
    # Load target image
    if not target_image_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_image_path}")
    
    target_img = Image.open(target_image_path).convert("RGB")
    print(f"Loaded target image: {target_image_path}")
    
    # Initialize termination handler (using Grounded SAM2)
    termination_handler = TerminationHandler(
        target_image_path=str(target_image_path),
        text_prompt=config.get('text_prompt', 'cat.'),
        similarity_threshold=0.15,  # 15% difference threshold
        box_threshold=config.get('box_threshold', 0.35),
        text_threshold=config.get('text_threshold', 0.25),
        enabled=True  # Set to False to disable segmentation termination
    )
    
    # Initialize appearance-preserving mask segmentation for LightGlue matching
    mask_segmentation_terminator = None
    target_masked_rgb_path = None
    appearance_mask_enabled = ENABLE_APPEARANCE_MASK
    
    if appearance_mask_enabled:
        print("\n" + "="*60)
        print("Initializing Appearance-Preserving Mask Segmentation")
        print("="*60)
        print("Creating segmentation masks for LightGlue feature matching")
        print(f"Text prompt: '{GSAM2_TEXT_PROMPT_FOR_MASK}'")
        print("="*60 + "\n")
        
        # Create a separate segmentation terminator for masking
        mask_segmentation_terminator = SegmentationTerminator(
            text_prompt=GSAM2_TEXT_PROMPT_FOR_MASK,
            similarity_threshold=0.15,
            box_threshold=config.get('box_threshold', 0.35),
            text_threshold=config.get('text_threshold', 0.25)
        )
        
        # Create masked target image for LightGlue
        print(f"Creating appearance-preserving masked target image...")
        target_masked_rgb = get_masked_rgb_image(mask_segmentation_terminator, str(target_image_path))
        if target_masked_rgb is not None:
            # Save masked target for LightGlue
            target_masked_rgb_path = save_dir / 'target_masked.png'
            cv2.imwrite(str(target_masked_rgb_path), cv2.cvtColor(target_masked_rgb, cv2.COLOR_RGB2BGR))
            print(f"✓ Masked target image saved to: {target_masked_rgb_path}")
            print(f"✓ Will perform LightGlue matching on masked images\n")
        else:
            print("WARNING: Could not create masked target image - will use original images")
            appearance_mask_enabled = False
    
    # Initialize data logger
    log_dir = Path(__file__).parent.parent / 'src' / 'logs'
    data_logger = DataLogger(log_dir, experiment_name='ibvs_real')
    
    itr = 0
    MIN_ERROR = float("inf")
    
    try:
        while test_bound(bot.arm.get_ee_pose()) and itr < max_iterations:
            itr += 1
            print(f"\n{'='*60}")
            print(f"Iteration {itr}")
            print(f"{'='*60}")
            
            try:
                # Capture current frame
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    print("Warning: Missing color or depth frame, skipping iteration")
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Convert depth to meters
                depth_meters = convert_realsense_depth_to_meters(depth_frame, depth_image)
                
                # Resize image to target_image size if needed
                target_height, target_width = target_img.size[1], target_img.size[0]
                if color_image.shape[:2] != (target_height, target_width):
                    color_image = cv2.resize(color_image, (target_width, target_height))
                    depth_meters = cv2.resize(depth_meters, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                
                # Convert OpenCV BGR image to RGB
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Save current RGB frame
                frame_path = save_dir / f"frame_{itr:03d}.png"
                cv2.imwrite(str(frame_path), color_image)
                
                # Prepare images for feature matching
                # If appearance-preserving masking is enabled, use masked images; otherwise use original
                current_match_path = str(frame_path)
                target_match_path = str(target_image_path)
                
                if appearance_mask_enabled and mask_segmentation_terminator is not None and target_masked_rgb_path is not None:
                    # Create masked version of current image
                    current_masked_rgb = get_masked_rgb_image(mask_segmentation_terminator, frame_path)
                    if current_masked_rgb is not None:
                        # Save masked current image
                        current_masked_path = save_dir / f"frame_{itr:03d}_masked.png"
                        cv2.imwrite(str(current_masked_path), cv2.cvtColor(current_masked_rgb, cv2.COLOR_RGB2BGR))
                        
                        # Use masked images for matching
                        current_match_path = str(current_masked_path)
                        target_match_path = str(target_masked_rgb_path)
                        
                        print(f"[Iteration {itr}] Using appearance-preserving masked images for LightGlue matching")
                    else:
                        print(f"[Iteration {itr}] Warning: Could not create mask, using original images")
                
                # Perform feature matching (on masked or original images)
                try:
                    src_kpts, tgt_kpts = match_superpoints(
                        current_match_path, target_match_path
                    )
                    
                    if len(src_kpts) == 0:
                        print("Warning: No Superpoints detected. Skipping iteration.")
                        continue
                    
                    assert len(src_kpts) == len(tgt_kpts), "Error from match_superpoints()"
                    
                except Exception as e:
                    print(f"Error matching keypoints: {e}")
                    continue
                
                # Only use a subset of SuperPoints
                try:
                    K_sample_src, K_sample_tgt = motion_utils.sample_points(src_kpts, tgt_kpts, K)
                except:
                    K_sample_src, K_sample_tgt = src_kpts, tgt_kpts
                
                # Calculate error
                error_vec = motion_utils.get_error_vec_Ksample(src_kpts, tgt_kpts)
                mse_error = motion_utils.get_error_mse(error_vec)
                
                # Update minimum error
                if mse_error < MIN_ERROR:
                    MIN_ERROR = mse_error
                
                print(f"MSE Error: {mse_error:.6f} (min: {MIN_ERROR:.6f})")
                
                # Save image with error printed
                if itr % 2 == 0:
                    save_image(mse_error, itr, color_image_rgb, MIN_ERROR)
                
                # Compute GAM similarity scores for logging
                metrics = termination_handler.compute_metrics(str(frame_path))
                gam_similarity = metrics["mask_similarity"]
                ssim_score = metrics["ssim_score"]
                
                # Compute velocity using Chaumette IBVS control with real depth
                vel = get_ibvs_velocity_realsense(
                    src_kpts=K_sample_src,
                    tgt_kpts=K_sample_tgt,
                    depth_image=depth_meters,
                    img_config=img_conf,
                    lambda_gain=LAMBDA
                )
                print(f"IBVS Vel: {vel}")
                
                # Scale velocity
                scaled_velocity = motion_utils.scale_velocity(vel, config)
                
                # Get current robot state for logging
                ee_pose = bot.arm.get_ee_pose()
                robot_pos = [ee_pose[0, 3], ee_pose[1, 3], ee_pose[2, 3]]
                rotation_matrix = ee_pose[:3, :3]
                robot_orn_rodrigues = cv2.Rodrigues(rotation_matrix)[0].flatten()
                robot_orn = np.degrees(robot_orn_rodrigues).tolist()
                
                # Log iteration data
                data_logger.log_iteration(
                    iteration=itr,
                    error_magnitude=mse_error,
                    robot_pos=robot_pos,
                    robot_orn=robot_orn,
                    num_matched_keypoints=len(src_kpts),
                    velocity=scaled_velocity,
                    image_path=str(frame_path),
                    gam_similarity=gam_similarity,
                    ssim_score=ssim_score
                )
                
                # Execute robot motion
                robot_velocity_control(bot, scaled_velocity, config, dt)
                
                # Early termination check
                # Only perform similarity checking when sufficiently close
                # (We can use error magnitude as a proxy for distance)
                if mse_error < 100.0:  # Threshold for checking termination
                    print(f"Error ({mse_error:.3f}) below threshold. Checking termination condition...")
                    if termination_handler.check_termination(str(frame_path), iteration=itr):
                        print("Termination condition met!")
                        break
                
                sleep(dt)
                
            except Exception as e:
                print(f"Error during iteration: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Save final image before stopping
        print("\nCapturing final image...")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(str(save_dir / "final_image.png"), color_image)
    
    finally:
        # Cleanup
        pipeline.stop()
        print('Camera stopped successfully')
        
        data_logger.close()
        
        # Move robot to sleep pose
        print('##  Trajectory completed. Entering sleep pose.  ##')
        bot.arm.go_to_sleep_pose()
        
        robot_shutdown()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {itr}")
    print(f"Appearance-preserving masking: {'ENABLED' if appearance_mask_enabled else 'DISABLED'}")
    if appearance_mask_enabled and target_masked_rgb_path is not None:
        print(f"Masked target image: {target_masked_rgb_path}")
        print(f"Segmentation prompt: '{GSAM2_TEXT_PROMPT_FOR_MASK}'")
    print(f"Data logged to: {data_logger.log_file}")
    print(f"{'='*60}\n")
    
    return


if __name__ == '__main__':
    import pdb
    pdb.set_trace = lambda: 1  # COMMENT OUT if DEBUG, otherwise UNCOMMENT.
    
    main()
