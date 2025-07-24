"""
main IBVS module
"""

# load all modules from main_new
from typing import List, Tuple
import cv2
import config_file as cf

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
from main_new import update_pos_and_orn

# Add interbotix-xs path
sys.path.append('/home/khw/interbotix_ws/src/interbotix_xs_modules')  ##
from interbotix_xs_modules.arm import InterbotixManipulatorXS

# Add image utils. TODO: bring im_uitils to pwd
sys.path.append('/home/khw/Documents/Proj2')
from im_utils import crop_masked_region

# ==========


# TODO
def init_robot_transform_matrix():
    return np.zeros([4,4], dtype=float)  ## placeholder

def GAM():
    return NotImplemented

def Cutie():
    return NotImplemented

def get_kpts():
    return NotImplemented

def main():
    """
    Implement main function for interbotix arms
    """
    # load config file values
    dt = cf.dt
    K = cf.K
    
    # init bot
    bot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
    bot.arm.go_to_home_pose()
    
    # load target
    target_img = cv2.imread(cf.TARGET_IMPATH)
    if target_img is None:
        raise FileNotFoundError("## Could not load target image. Check if file exists. ##")
    # get mask, segmented image, and feat-kpts
    target_mask = GAM(target_img)
    masked_tgt_rgb = crop_masked_region(target_img, target_mask)

    # intialize robot orientation frame
    robot_transform_matrix = init_robot_transform_matrix()
    robot_pos = robot_transform_matrix[:3, 3]  # Last column contains xyz translation
    rotation_matrix = robot_transform_matrix[:3, :3]    
    robot_orientation = np.degrees(cv2.Rodrigues(rotation_matrix)[0].flatten())

    # setup camera stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Initialize frame mask obj
    curr_mask = None

    while True:

        pose = bot.arm.get_ee_pose()

        try:
            pipeline.start(config)
            print('Camera started successfully')

            # get real-life rgb image
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # if frame available, move to next step
                if color_frame:
                    break

            # convert img to array
            color_image = np.asanyarray(color_frame.get_data())

            # resize image to target_image size
            target_height, target_width = target_img.shape[:2]
            color_image = cv2.resize(color_image, (target_width, target_height))

            # if we don't have initial frame, use GAM. Else use Cutie
            if not curr_mask:
                curr_mask = GAM(color_image)
            else:
                curr_mask = Cutie(color_image)  ## TODO: see if Cutie requires warm-start

            # Overlay mask with RGB
            masked_src_rgb = crop_masked_region(color_image, curr_mask)

            # Extract + return matched SuperPoints
            try:
                m_src_kpts, m_tgt_kpts = match_superpoints(
                    masked_src_rgb, masked_tgt_rgb
                )
                assert len(m_src_kpts) > 0, "Number of keypoints is 0"
                assert len(m_src_kpts) == len(m_tgt_kpts), " Matched keypoint lengths not equal between target / source "
            except Exception as e:
                print(f" ## Error matching keypoints: {e}")
                velocity = np.zeros(6)
                velocity[1] = cf.CONSTANT_FORWARD_VEL # Move forward with constant velocity
                continue
        
            velocity = get_linear_vel(m_src_kpts, m_tgt_kpts)

            def robot_linear_control(velocity):
                """Moves interbotix arm by cartesian control based on input velocity"""

                delx, dely, delz = velocity

                delx *= cf.x_scale
                dely *= cf.y_scale 
                delz *= cf.z_scale

                # move forward by cartesian control
                bot.arm.set_ee_cartesian_trajectory(x=delx, y=dely, z=delz)

            robot_linear_control(velocity)


        except Exception as e:
            print(f'Error: {e}')
            return None
        
        finally:
            pipeline.stop()
            print('Camera stopped')
    
    return


if __name__ == "__main__":
    main()
    # simple_forward()