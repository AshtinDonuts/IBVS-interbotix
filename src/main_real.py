"""
IBVS module designed to work with the ALOHA Viper X300s.

Requires:
* Interbotix module dependencies
* ROS 2

TODO:
Replacing HuggingFace GAM with GAM2 as stand-in

"""
# from pdb import set_trace

# load all modules from main_new
import shutil
from typing import List, Tuple
import cv2
import yaml

# TODO: Refactor script to import regardless of location
# =========
import os, sys
from time import sleep
from pathlib import Path
import numpy as np
import pyrealsense2 as rs

# import scripts
from image import convert_img_to_arr, save_image, get_image_config
from servo_new import match_superpoints
from motion_new import get_linear_vel

# Add interbotix-xs path
sys.path.append('/home/khw/interbotix_ws/src/interbotix_xs_modules')  ##
from interbotix_xs_modules.arm import InterbotixManipulatorXS

# Add image utils. TODO: bring im_uitils to pwd
sys.path.append('/home/khw/Documents/Proj2')
from im_utils import crop_masked_region

# import HFGAM
sys.path.append('/home/khw/Documents/Proj2')
from HFGAM import process_frame

from PIL import Image

# Import Cutie wrapper
sys.path.append('/home/khw/Documents/Proj2/Cutie')
from cutie_wrapper import generate_mask_images as cutie_gen_mask_images


# ==========


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# TODO
def init_robot_transform_matrix():
    return np.zeros([4,4], dtype=float)  ## placeholder

def GAM(input_image):
    assert isinstance(input_image, Image.Image)
    processed_image : Image.Image = process_frame(input_image)
    return processed_image

def Cutie(image_path, mask_path)-> List[Image.Image]:
    """
    Image_path is the parent directory containing historical images"""
    assert isinstance(image_path, str), "image_path must be a string"
    assert isinstance(mask_path, str), "mask_path must be a string"
    cutie_masks_pil = cutie_gen_mask_images(image_path, mask_path, return_masks=True)
    return cutie_masks_pil

def get_initial_pose() -> np.ndarray:
    """ Returns a well known valid pose for the robot arm"""
    T_sd = np.identity(4)
    T_sd[0,3] = 0.2
    T_sd[1,3] = 0
    T_sd[2,3] = 0.25
    return T_sd

# to get pose, run:
# > bot.arm.get_ee_pose()


def test_bound(ee_pose)->bool:
    x_bound = 0.4    ##
    if ee_pose[0,3] > x_bound:
        print('Bound reached!\n Terminating...')
        return False
    else:
        return True

def get_valid_color_frame(pipeline):
    """Wait for a valid color frame from the camera pipeline."""
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            return color_frame
            
def main():
    """
    Final stage servoing for real world Interbotix deployment for ICRA 25 Project.
    """
    
    # Load configuration from YAML file
    config = load_config(Path(__file__).parent / 'config.yaml')
    
    # load initial config vals
    dt = config['dt']
    K = config['K']
    # Clear and recreate directories
    shutil.rmtree(config['save_dir'], ignore_errors=True)
    shutil.rmtree(config['historical_rgb_path'], ignore_errors=True)
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['historical_rgb_path']).mkdir(parents=True, exist_ok=True)
    
    # bot.arm.go_to_home_pose()
    T_startpose = get_initial_pose()
    bot.arm.set_ee_pose_matrix(T_startpose)
    

    # intialize robot orientation frame
    robot_transform_matrix = init_robot_transform_matrix()
    robot_pos = robot_transform_matrix[:3, 3]  # Last column contains xyz translation
    rotation_matrix = robot_transform_matrix[:3, :3]    
    robot_orientation = np.degrees(cv2.Rodrigues(rotation_matrix)[0].flatten())

    # Initialize frame mask obj
    curr_mask = None

    # camera stream configs and start stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(config)
        print('Camera started successfully')
    except Exception as e:
        print(f"Failed to start camera: {e}")


    # load target image after verifying camera stream success
    target_img = Image.open(config['target_image_path'])
    if target_img is None:
        raise FileNotFoundError("## Could not load target image. Check if file exists. ##")
    target_mask = GAM(target_img)
    seg_target_rgb_np = crop_masked_region(target_img, target_mask)
    seg_target_rgb = Image.fromarray(seg_target_rgb_np)
    # Save target mask and segmented target RGB
    target_mask.save(Path(config['save_dir']) / "target_mask.png")
    seg_target_rgb.save(Path(config['save_dir']) / "target_seg.png")
    
    itr = 0

    try:

        while test_bound(bot.arm.get_ee_pose()) and itr < config['max_iterations']:

            itr += 1

            try:

                color_frame = get_valid_color_frame(pipeline)
                color_image = np.asanyarray(color_frame.get_data())

                # resize image to target_image size
                target_height, target_width = target_img.size
                color_image = cv2.resize(color_image, (target_width, target_height))

                # Convert OpenCV BGR image to PIL RGB image
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image = Image.fromarray(color_image)

                # Save current RGB frame (already a PIL image)
                color_image.save(Path(config['save_dir']) / f"frame_{itr}.png")
                color_image.save(Path(config['historical_rgb_path']) / f"frame_{itr}.png")

                # if we don't have previous mask frame, use GAM. Else use Cutie
                if not curr_mask:
                    print('##  No mask currently. Using GAM. ##')
                    curr_mask = GAM(color_image)
                else:
                    print('##  Mask exists. Using Cutie.  ##')
                    cutie_masks = Cutie(image_path=config['historical_rgb_path'],
                                         mask_path=str(Path(config['save_dir']) / f"mask_1.png"))  ## TODO: 1. Modify to maintain Cutie instance once initialized, 2. elimiante redundant file saving/loading
                    assert isinstance(cutie_masks, List)
                    curr_mask = cutie_masks[-1]
                    assert isinstance(curr_mask, Image.Image)
                # Reduce queue size
                # if len(historical_masks) > config['historical_mask_queue']:
                #     historical_masks = historical_masks[-config['historical_mask_queue']:]

                # Save mask (already a PIL image)
                curr_mask.save(Path(config['save_dir']) / f"mask_{itr}.png")

                ## Overlay mask with RGB
                seg_src_rgb_np = crop_masked_region(color_image, curr_mask, 2)
                # Convert masked RGB to PIL image
                seg_src_rgb = Image.fromarray(seg_src_rgb_np)

                # Save segmented RGB (already a PIL image)
                seg_src_rgb.save(Path(config['save_dir']) / f"seg_{itr}.png")

                # set_trace()

                # Extract & return matched SuperPoints
                try:
                    m_src_kpts, m_tgt_kpts = match_superpoints(
                        # seg_src_rgb_np, seg_target_rgb_np
                        Path(config['save_dir']) / f"seg_{itr}.png",
                        Path(config['save_dir']) / "target_seg.png"
                    )
                    assert len(m_src_kpts) > 0, "Number of keypoints is 0"
                    assert len(m_src_kpts) == len(m_tgt_kpts), " Matched keypoint lengths not equal between target / source "
                except Exception as e:
                    print(f" ## Error matching keypoints: {e}")
                    velocity = np.zeros(6)
                    velocity[1] = config['constant_forward_vel'] # Move forward with constant velocity
                    continue

                # print('length of number of source kpts :', len(m_src_kpts))

                # # TODO: Plot to visualize dense matching
            
                velocity = get_linear_vel(m_src_kpts, m_tgt_kpts)

                def robot_linear_control(velocity):
                    """Moves interbotix arm by cartesian control based on input velocity"""


                    dely, delx, delz = velocity[0], velocity[1], velocity[2]  # x, y are switched from pybullet

                    dely = -1 * dely

                    delx *= config['x_scale']
                    dely *= config['y_scale'] 
                    delz *= config['z_scale']

                    print(f"Moving robot with velocities:")
                    print(f"  x: {delx:6.3f}")
                    print(f"  y: {dely:6.3f}") 
                    print(f"  z: {delz:6.3f}")

                    # move forward by cartesian control
                    bot.arm.set_ee_cartesian_trajectory(x=delx, y=dely, z=delz)

                robot_linear_control(velocity)

                from hydra.core.global_hydra import GlobalHydra
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()


            except Exception as e:
                print(f"Error during camera or mask processing: {e}")
                # Optionally: pipeline.stop(), cleanup, or break
                break
        
        # Save final image before stopping
        color_frame = get_valid_color_frame(pipeline)
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(str(Path(config['save_dir']) / "final_image.png"), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    
    finally:
        pipeline.stop()
        print('Camera stopped successfully')

    return


def sleep():
    print('##  Trajectory completed. Entering sleep pose.  ## ')
    bot.arm.go_to_sleep_pose()


if __name__ == "__main__":
    
    #init bot
    bot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
    main()
    sleep()
    

