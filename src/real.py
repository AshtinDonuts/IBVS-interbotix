"""
IBVS implementation for real-world deployment with Interbotix Viper X300s robot.

This combines:
- Robot control from OLD_main_real.py (Interbotix + RealSense)
- Improved structure from main_sim.py (DataLogger, termination handler)
- Feature-based visual servoing using LightGlue/SuperPoint
- Grounded SAM 2 for object segmentation

Requires:
* Interbotix module dependencies
* ROS 2
* RealSense camera
* Grounded SAM 2 for segmentation
"""

import shutil
from typing import List, Tuple, Optional
import cv2
import yaml
import os
import sys
from time import sleep
from pathlib import Path
import numpy as np
import pyrealsense2 as rs
import csv
import torch

# Import local modules
from image import convert_img_to_arr, save_image, get_image_config
from superpoint_utils import match_superpoints
import motion_utils
from motion import LAMBDA, get_error_mse
from gsam2_terminator import SegmentationTerminator

# Add interbotix-xs path
sys.path.append('/home/khw/interbotix_ws/src/interbotix_xs_modules')
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from PIL import Image


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
            experiment_name = "real_experiment"
        
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
        image_path: str
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
        """
        # Write row
        self.csv_writer.writerow([
            iteration,
            f"{error_magnitude:.6f}",
            num_matched_keypoints,
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_initial_pose() -> np.ndarray:
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


class SAM2Segmentor:
    """
    Wrapper around Grounded SAM 2 for real-time object segmentation.
    Supports video tracking by keeping the predictor state between frames.
    """
    
    def __init__(self, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        """
        Initialize the SAM2 segmentor.
        
        Parameters
        ----------
        text_prompt : str
            Object detection prompt (e.g., "object.", "cube.")
        box_threshold : float
            Confidence threshold for bounding box detection
        text_threshold : float
            Confidence threshold for text grounding
        """
        print("Initializing Grounded SAM 2 for object segmentation...")
        self.segmentor = SegmentationTerminator(
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            similarity_threshold=0.15  # Not used for segmentation, only for termination
        )
        self.text_prompt = text_prompt
        print("Grounded SAM 2 initialized successfully!")
    
    def get_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Get binary mask for the target object in an image.
        
        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image
            
        Returns
        -------
        PIL.Image.Image or None
            Binary mask as PIL Image (mode 'L'), or None if no object detected
        """
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Set image for SAM2
        self.segmentor.sam2_predictor.set_image(image_np)
        
        # Run Grounding DINO detection
        inputs = self.segmentor.processor(
            images=image, 
            text=self.text_prompt, 
            return_tensors="pt"
        ).to(self.segmentor.device)
        
        with torch.no_grad():
            outputs = self.segmentor.grounding_model(**inputs)
        
        results = self.segmentor.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.segmentor.box_threshold,
            text_threshold=self.segmentor.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        # Check if any objects were detected
        if len(results) == 0 or len(results[0]["boxes"]) == 0:
            print(f"Warning: No objects detected with prompt '{self.text_prompt}'")
            return None
        
        # Get bounding boxes for SAM2
        input_boxes = results[0]["boxes"].cpu().numpy()
        
        # Get segmentation masks
        masks, scores, logits = self.segmentor.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Convert shape to (n, H, W) if needed
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Use the first (highest confidence) mask
        mask = masks[0].astype(np.uint8) * 255
        
        # Convert to PIL Image
        mask_pil = Image.fromarray(mask, mode='L')
        
        return mask_pil
    
    def crop_masked_region(self, image: Image.Image, mask: Image.Image, expand_margin: int = 0) -> np.ndarray:
        """
        Crop the masked region from the image.
        
        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image
        mask : PIL.Image.Image
            Binary mask
        expand_margin : int
            Margin to expand the bounding box (default: 0)
            
        Returns
        -------
        np.ndarray
            Cropped RGB image as numpy array
        """
        # Convert to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Find bounding box of mask
        coords = np.argwhere(mask_np > 0)
        if len(coords) == 0:
            # No mask pixels, return original image
            return image_np
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Expand margin
        h, w = mask_np.shape
        y_min = max(0, y_min - expand_margin)
        x_min = max(0, x_min - expand_margin)
        y_max = min(h, y_max + expand_margin)
        x_max = min(w, x_max + expand_margin)
        
        # Apply mask (set background to black)
        masked_image = image_np.copy()
        mask_bool = mask_np > 0
        if len(image_np.shape) == 3:
            # RGB image
            for c in range(3):
                masked_image[:, :, c][~mask_bool] = 0
        else:
            # Grayscale
            masked_image[~mask_bool] = 0
        
        # Crop to bounding box
        cropped = masked_image[y_min:y_max, x_min:x_max]
        
        return cropped


def get_valid_color_frame(pipeline):
    """Wait for a valid color frame from the camera pipeline."""
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            return color_frame


def robot_linear_control(bot, velocity: np.ndarray, config: dict, dt: float):
    """
    Moves interbotix arm by cartesian control based on input velocity.
    
    Parameters
    ----------
    bot : InterbotixManipulatorXS
        Robot controller instance
    velocity : np.ndarray
        6D velocity vector [vx, vy, vz, wx, wy, wz]
    config : dict
        Configuration dictionary with scale factors
    dt : float
        Time step for control
    """
    # Extract velocities (already scaled)
    # Note: x, y are switched from pybullet to interbotix convention
    dely, delx, delz = velocity[0], velocity[1], velocity[2]
    
    # Flip y direction for interbotix convention
    dely = -1 * dely
    
    # Apply additional scaling from config
    delx *= config['x_scale']
    dely *= config['y_scale']
    delz *= config['z_scale']
    
    print(f"Moving robot with velocities:")
    print(f"  x: {delx:6.3f}")
    print(f"  y: {dely:6.3f}")
    print(f"  z: {delz:6.3f}")
    
    # TODO: Add angular velocity control if needed
    # For now, we only control linear motion
    
    # Move by cartesian control
    bot.arm.set_ee_cartesian_trajectory(x=delx, y=dely, z=delz)


def main():
    """
    Real-world IBVS deployment for Interbotix robot with RealSense camera.
    """
    
    # Load configuration from YAML file
    config = load_config(Path(__file__).parent / 'config.yaml')
    
    # Load initial config vals
    dt = config['dt']
    K = config['K']
    
    # Clear and recreate directories
    shutil.rmtree(config.get('save_dir', 'real_output'), ignore_errors=True)
    shutil.rmtree(config.get('historical_rgb_path', 'real_historical'), ignore_errors=True)
    save_dir = Path(config.get('save_dir', 'real_output'))
    historical_rgb_path = Path(config.get('historical_rgb_path', 'real_historical'))
    save_dir.mkdir(parents=True, exist_ok=True)
    historical_rgb_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize robot
    print("Initializing robot...")
    bot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
    
    # Move to initial pose
    # bot.arm.go_to_home_pose()
    T_startpose = get_initial_pose()
    bot.arm.set_ee_pose_matrix(T_startpose)
    sleep(1)  # Allow robot to reach pose
    
    
    # Camera stream configs and start stream
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(rs_config)
        print('Camera started successfully')
    except Exception as e:
        print(f"Failed to start camera: {e}")
        return
    
    # Load target image after verifying camera stream success
    target_img = Image.open(config['target_image_path'])
    if target_img is None:
        raise FileNotFoundError("## Could not load target image. Check if file exists. ##")
    
    # Initialize SAM2 segmentor
    # Get text prompt from config or use default
    text_prompt = config.get('text_prompt', 'object.')
    print(f"Initializing SAM2 with text prompt: '{text_prompt}'")
    
    sam2_segmentor = SAM2Segmentor(
        text_prompt=text_prompt,
        box_threshold=config.get('box_threshold', 0.35),
        text_threshold=config.get('text_threshold', 0.25)
    )
    
    # Process target image with SAM2
    print("Processing target image with Grounded SAM 2...")
    target_mask = sam2_segmentor.get_mask(target_img)
    
    if target_mask is None:
        raise ValueError(f"Could not detect object in target image with prompt '{text_prompt}'")
    
    seg_target_rgb_np = sam2_segmentor.crop_masked_region(target_img, target_mask, expand_margin=2)
    seg_target_rgb = Image.fromarray(seg_target_rgb_np)
    
    # Save target mask and segmented target RGB
    target_mask.save(save_dir / "target_mask.png")
    seg_target_rgb.save(save_dir / "target_seg.png")
    
    # Initialize data logger
    log_dir = Path(__file__).parent / 'logs'
    data_logger = DataLogger(log_dir, experiment_name='ibvs_real')
    
    itr = 0
    
    try:
        while test_bound(bot.arm.get_ee_pose()) and itr < config['max_iterations']:
            itr += 1
            print(f"\n{'='*60}")
            print(f"Iteration {itr}")
            print(f"{'='*60}")
            
            try:
                # Capture current frame
                color_frame = get_valid_color_frame(pipeline)
                color_image = np.asanyarray(color_frame.get_data())
                
                # Resize image to target_image size
                target_height, target_width = target_img.size
                color_image = cv2.resize(color_image, (target_width, target_height))
                
                # Convert OpenCV BGR image to PIL RGB image
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image_pil = Image.fromarray(color_image)
                
                # Save current RGB frame
                frame_path = save_dir / f"frame_{itr:03d}.png"
                color_image_pil.save(frame_path)
                color_image_pil.save(historical_rgb_path / f"frame_{itr:03d}.png")
                
                # Segment current frame with SAM2
                print('##  Segmenting frame with Grounded SAM 2...  ##')
                curr_mask = sam2_segmentor.get_mask(color_image_pil)
                
                if curr_mask is None:
                    print(f"Warning: Could not detect object in frame {itr}")
                    # Fall back to constant forward velocity
                    velocity = np.zeros(6)
                    velocity[1] = config['constant_forward_vel']
                    
                    # Log iteration with error
                    ee_pose = bot.arm.get_ee_pose()
                    robot_pos = [ee_pose[0, 3], ee_pose[1, 3], ee_pose[2, 3]]
                    robot_orn = [0, 0, 0]  # Placeholder
                    
                    data_logger.log_iteration(
                        iteration=itr,
                        error_magnitude=float('inf'),
                        robot_pos=robot_pos,
                        robot_orn=robot_orn,
                        num_matched_keypoints=0,
                        velocity=velocity,
                        image_path=str(frame_path)
                    )
                    
                    # Execute motion
                    robot_linear_control(bot, velocity, config, dt)
                    sleep(dt)
                    continue
                
                # Save mask
                mask_path = save_dir / f"mask_{itr:03d}.png"
                curr_mask.save(mask_path)
                
                # Overlay mask with RGB to get segmented region
                seg_src_rgb_np = sam2_segmentor.crop_masked_region(color_image_pil, curr_mask, expand_margin=2)
                seg_src_rgb = Image.fromarray(seg_src_rgb_np)
                
                # Save segmented RGB
                seg_path = save_dir / f"seg_{itr:03d}.png"
                seg_src_rgb.save(seg_path)
                
                # Extract & match SuperPoints
                try:
                    m_src_kpts, m_tgt_kpts = match_superpoints(
                        seg_path,
                        save_dir / "target_seg.png"
                    )
                    assert len(m_src_kpts) > 0, "Number of keypoints is 0"
                    assert len(m_src_kpts) == len(m_tgt_kpts), "Matched keypoint lengths not equal"
                    
                    print(f"Matched {len(m_src_kpts)} keypoints")
                    
                except Exception as e:
                    print(f"## Error matching keypoints: {e}")
                    # Fall back to constant forward velocity
                    velocity = np.zeros(6)
                    velocity[1] = config['constant_forward_vel']
                    
                    # Log iteration with error
                    ee_pose = bot.arm.get_ee_pose()
                    robot_pos = [ee_pose[0, 3], ee_pose[1, 3], ee_pose[2, 3]]
                    robot_orn = [0, 0, 0]  # Placeholder
                    
                    data_logger.log_iteration(
                        iteration=itr,
                        error_magnitude=float('inf'),
                        robot_pos=robot_pos,
                        robot_orn=robot_orn,
                        num_matched_keypoints=0,
                        velocity=velocity,
                        image_path=str(frame_path)
                    )
                    
                    # Execute motion
                    robot_linear_control(bot, velocity, config, dt)
                    sleep(dt)
                    continue
                
                # Only use a subset of SuperPoints if too many
                try:
                    K_sample_src, K_sample_tgt = motion_utils.sample_points(m_src_kpts, m_tgt_kpts, K)
                except:
                    K_sample_src, K_sample_tgt = m_src_kpts, m_tgt_kpts
                
                # Calculate error
                error_vec = motion_utils.get_error_vec_Ksample(m_src_kpts, m_tgt_kpts)
                mse_error = get_error_mse(error_vec)
                
                print(f"MSE Error: {mse_error:.6f}")
                
                # Compute velocity using motion utilities
                # Note: We don't have target position in real world, so we use image-based control only
                vel = np.zeros(6)
                
                # Add perpendicular velocity (left-right, up-down correction)
                vel = motion_utils.update_perpendicular_velocity(vel, K_sample_src, K_sample_tgt)
                
                # Add constant forward velocity (since we don't have depth/position info)
                vel[1] += config.get('constant_forward_vel', 2.5)
                
                # Add angular velocity (rotation correction)
                vel = motion_utils.update_angular_velocity(
                    vel, K_sample_src, K_sample_tgt, lambda_gain=LAMBDA
                )
                
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
                    num_matched_keypoints=len(m_src_kpts),
                    velocity=scaled_velocity,
                    image_path=str(frame_path)
                )
                
                # Execute robot motion
                robot_linear_control(bot, scaled_velocity, config, dt)
                
                # Optional: Check termination condition using mask size similarity
                # This reuses the SAM2 segmentor to compare mask sizes
                # Uncomment if you want automatic termination based on mask similarity
                # try:
                #     current_mask_size = np.sum(np.array(curr_mask) > 0)
                #     target_mask_size = np.sum(np.array(target_mask) > 0)
                #     similarity = min(current_mask_size, target_mask_size) / max(current_mask_size, target_mask_size)
                #     print(f"Mask similarity: {similarity:.2%}")
                #     if similarity >= 0.85:  # 85% similarity threshold
                #         print("Mask similarity threshold reached!")
                #         break
                # except:
                #     pass
                
                sleep(dt)
                
            except Exception as e:
                print(f"Error during iteration: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Save final image before stopping
        print("\nCapturing final image...")
        color_frame = get_valid_color_frame(pipeline)
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
    
    return


if __name__ == "__main__":
    import pdb
    pdb.set_trace = lambda: 1  # COMMENT OUT if DEBUG, otherwise UNCOMMENT.
    
    main()
