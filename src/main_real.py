"""
main IBVS module
"""

# TODO: Refactor script to import regardless of location
# =========
import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
# ==========


import pdb

import sys
from time import sleep
from typing import Tuple, List

import pybullet as p
import pybullet_data

import numpy as np
from pathlib import Path

from image import convert_img_to_arr, save_image, get_image_config
from servo import get_marker_corners
from motion import get_error_mag, get_error_vec, get_velocity

from servo_new import match_superpoints
from motion_new import get_error_mag, get_error_vec_K, sample_points, get_velocity_K_points, get_linear_vel

import cv2


MAX_ITERATIONS = 70
TARGET_PATH=Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000050_crop_padded.png')
K = 20


def init_pybullet() -> int:
    """
    initialises the pybullet scene
    """
    pclient = p.connect(p.GUI)  # p.GUI/p.DIRECT 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # Turn off gravity
    # p.setRealTimeSimulation(True)

    return pclient

def set_aruco_marker_texture(obstacle_id: int) -> None:
    """
    sets the aruco marker texture on the obstacle
    """
    texture_id = p.loadTexture(str(TARGET_PATH))
    p.changeVisualShape(obstacle_id, -1, textureUniqueId=texture_id)


# TODO: change scene target block offset values
def init_scene(robot_pos: list[float]) -> Tuple[int, List[int]]:
    """
    initialises the scene and returns created objects
    """

    plane_id = p.loadURDF("plane.urdf")
    # loading obstacles, with the main cube at the first index
    base = robot_pos  # for now
    base_orn = p.getQuaternionFromEuler([0, 0, 0])
    obstacles = []

    # new offset for testing
    new_offsetx = 0
    new_offsetx = 0.75
    new_offsetz = 0
    new_offsetz = 0.75

    ## builds a wall of cubes
    for z_offset in [0, 1]:   # z is up/down
        for y_offset in [2.5]:  # 7.5    # y is forward-backward
            for x_offset in [0, -1, 1]:  # x is left-right
                obstacles.append(
                    p.loadURDF(
                        "cube_small.urdf",
                        [
                            base[0] + x_offset + new_offsetx,
                            base[1] + y_offset,
                            base[2] + z_offset + new_offsetz,
                        ],
                        base_orn,
                        globalScaling=20,  # Remove or adjust as needed
                    )
                )

    # texture the first cube (set as goal)
    goal_obs_id = obstacles[0]
    set_aruco_marker_texture(goal_obs_id)

    return plane_id, obstacles


def get_robot_rotation_matrix(robot_orientation: List[float]) -> np.ndarray:
    """
    gets the robot rotation matrix on the basis of the orientation
    """
    robot_rotation_matrix = p.getMatrixFromQuaternion(
        p.getQuaternionFromEuler(robot_orientation)
    )
    return np.array(object=robot_rotation_matrix).reshape(3, 3)


def get_view_matrix(
    init_camera_vector: Tuple[int, int, int],
    init_up_vector: Tuple[int, int, int],
    robot_pos: List[float],
    robot_rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    get the view matrix for the camera
    """
    camera_vector = np.dot(robot_rotation_matrix, init_camera_vector)
    up_vector = np.dot(robot_rotation_matrix, init_up_vector)
    return p.computeViewMatrix(robot_pos, robot_pos + camera_vector, up_vector)


def get_projection_matrix() -> np.ndarray:
    """
    get the projection matrix for the camera
    """
    image_conf = get_image_config()
    return p.computeProjectionMatrixFOV(
        image_conf["fov"],
        image_conf["aspect"],
        image_conf["near_val"],
        image_conf["far_val"],
    )


def get_transformation_matrix(
    robot_pos: List[float], robot_rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    create the transformation matrix to convert from image (world) coordinates to local
        camera coordinates
    """
    # creating the transformation matrix

    # translate the camera to the robot position
    t_c = np.zeros((4, 4))
    for i, val in enumerate(robot_pos):
        t_c[i][3] = -val
        t_c[i][i] = 1
    t_c[3][3] = 1

    # rotate the camera to the robot orientation
    r_i = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            r_i[i][j] = robot_rotation_matrix[i][j]
    r_i[3][3] = 1

    transform = np.matmul(r_i, t_c)

    return transform


def capture_camera_image(
    robot_pos: List[float], robot_rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    captures the image from the camera on the basis of the robot position and rotation matrix
    """
    image_conf = get_image_config()

    # robot rotation matrix
    # initial camera vectors
    init_camera_vector = (0, 1, 0)  # y axis
    init_up_vector = (0, 0, 1)  # z axis

    # calculating the view matrix
    view_matrix = get_view_matrix(
        init_camera_vector,
        init_up_vector,
        robot_pos,
        robot_rotation_matrix,
    )
    # calculating the projection matrix
    projection_matrix = get_projection_matrix()


    # capturing the image
    # -------------------
    # p.getCameraImage
    # width (int) – Width of the rendered image.
    # height (int) – Height of the rendered image.
    # rgbImg (list or np.array) – Color image data in RGBA format (depending on settings).
    # depthImg (list or np.array) – Depth image data. The values are typically normalized between 0 and 1, unless a custom projection is used.
    # segImg (list or np.array) – Segmentation mask. Contains object unique IDs

    img_details = p.getCameraImage(
        image_conf["width"],
        image_conf["height"],
        view_matrix,
        projection_matrix,
    )
    return img_details  # (width, height, rgbImg, depthImg, segImg)

def update_pos_and_orn(
    transform: np.ndarray,
    velocity: np.ndarray,
    robot_pos: List[float],
    robot_orn: List[float],
    dt: float,
) -> Tuple[List[float], List[float]]:
    """
    returns the updated position and orientation of the robot
    uses homogenous coordinates
    """
    # del_pos = np.matmul(transform, [*velocity[:3], 1])
    # for i in range(3):
    #     robot_pos[i] += (del_pos[i] / del_pos[-1]) * dt

    robot_pos[1] += 2.5 * dt

    del_orn = np.matmul(transform, [*velocity[3:], 1])
    for i in range(3):
        robot_orn[i] += (del_orn[i] / del_orn[-1]) * dt

    return robot_pos, robot_orn


MIN_ERROR = float("inf")
ERROR_GROWTH_LIMIT = 0.90  # 5%


def update_error(error_mag: float, i: int | None = None) -> None:
    """
    updates the global min error and determines when to break
    """

    global MIN_ERROR

    if i is not None:
        print(f"{i}:", end="")
    if error_mag < MIN_ERROR:
        MIN_ERROR = error_mag
        print(f"new min error: {MIN_ERROR}")
    else:
        if error_mag > MIN_ERROR:
            print(f"increase from min: {error_mag - MIN_ERROR}")
        else:
            print(f"error remained same: {error_mag}")

    if error_mag > (1 + ERROR_GROWTH_LIMIT) * MIN_ERROR:
        # indicator of some divergence
        print("DONE: error growth limit reached")
        p.disconnect()
        sys.exit(0)

def simple_forward() -> None:
    """Move robot forward OR rotation"""
    _ = init_pybullet()
    img_conf = get_image_config()
    dt: float = 0.05  # 0.0001

    # initialise the robot position and orientation (arbitrary)
    robot_pos = [0, 0, 1.0]  # [0, 0, 1]   # only z matters as target offsets from cam
    robot_orientation = [0, 0, 0]
    # robot_orientation = [0, 0, 0 - np.pi / 10]

    _, _ = init_scene(robot_pos)
    sleep(2)  # scene load buffer

    # wipe all files in dist_img directory
    import os, shutil
    if os.path.exists('dist_img'):
        shutil.rmtree('dist_img')
    os.makedirs('dist_img')

    for i in range(30):
        p.stepSimulation()
        robot_rot_matrix = get_robot_rotation_matrix(robot_orientation)  # what frame?

        ## img : (width, height, rgbImg, depthImg, segImg)
        img = capture_camera_image(robot_pos, robot_rot_matrix)

        rgba = img[2]  # Img[2]: (h x w x 4)
        rgba_arr = convert_img_to_arr(
            rgba, int(img_conf["height"]), int(img_conf["width"])  # Img[2]: (h x w x 4)
        )
        rgb_img_arr = rgba_arr[:, :, :3]  # remove alpha channel [..,4] -> [..,3]

        # TODO : modify to get image from pybullet
        save_impath = f'dist_img/distance_image_{i}.png'
        cv2.imwrite(save_impath, cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR))

        # pdb.set_trace()

        # remove batch dim [1, 500, 800, 3] -> [500, 800, 3]
        if rgb_img_arr.ndim == 4:
            rgb_img_arr = rgb_img_arr[0]

        src_kpts, tgt_kpts = match_superpoints(
            save_impath, TARGET_PATH
        )
        assert len(src_kpts) == len(tgt_kpts), "bug in match-superpoints()"
        # pdb.set_trace()

        speed = 1
        # robot_pos = [robot_pos[0], robot_pos[1] + speed * dt, robot_pos[2]]
        robot_orientation[2] += np.pi / 50

        sleep(0.01)  # arbitrary sleep to let the changes take place

        # log the number of source keypoints detected
        num_src_kpts = len(src_kpts) if src_kpts is not None else 0
        with open('dist_img/src_kpts_count.txt', 'a') as f:
            f.write(f"Iteration {i}: {num_src_kpts} matching keypoints detected, robot position: {robot_pos}\n")

def main() -> None:
    """
    the main flow
    """
    _ = init_pybullet()
    img_conf = get_image_config()
    dt: float = 0.005

    # initialise the robot position and orientation (arbitrary)
    robot_pos = [0, 0, 1.0]    # [x, y, z]
    robot_orientation = [0, 0, 0]

    _, _ = init_scene(robot_pos)

    sleep(1)  # arbitrary sleep to let the scene load
    
    for i in range(MAX_ITERATIONS):
        p.stepSimulation()
        robot_rot_matrix = get_robot_rotation_matrix(robot_orientation)  # what frame?

        ## img : (width, height, rgbaImg, depthImg, segImg)
        img = capture_camera_image(robot_pos, robot_rot_matrix)

        rgba = img[2]  # Img[2]: (h x w x 4)
        rgba_arr = convert_img_to_arr(
            rgba, int(img_conf["height"]), int(img_conf["width"])  # Img[2]: (h x w x 4)
        )
        rgb_img_arr = rgba_arr[:, :, :3]  # remove alpha channel [..,4] -> [..,3]

        # save_impath = f'dist_img/distance_image_{i}.png'
        save_impath = f'dist_img/_temp_image.png'
        cv2.imwrite(save_impath, cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR))

        src_kpts, tgt_kpts = match_superpoints(
            save_impath, TARGET_PATH
        )
        assert len(src_kpts) == len(tgt_kpts), "bug in match-superpoints()"

        # print(f"number of SuperPoints detected: {len(src_kpts)} vs {len(tgt_kpts)}")

        # if no SuperPoints detected, skip iter and keep rotating
        if src_kpts is None or tgt_kpts is None or len(src_kpts) < K:
            print("no SuperPoints detected, rotating")
            robot_orientation[2] += np.pi / 50
            save_image(None, i, rgb_img_arr, MIN_ERROR)
            continue

        K_sample_src, K_sample_tgt = sample_points(src_kpts, tgt_kpts, K)

        error = get_error_mag(get_error_vec_K(K_sample_src, K_sample_tgt))

        # save image with error printed on it.
        save_image(error, i, rgb_img_arr, MIN_ERROR)

        ## turn this on
        if 0:
            update_error(error, i=i)

        # get the velocity, transform vector, and update position and orientation
        velocity = get_linear_vel(K_sample_src, K_sample_tgt)  ##
        # velocity = get_velocity_K_points(K_sample_src, K_sample_tgt, depth_buffer=img[3])

        transform = get_transformation_matrix(robot_pos, robot_rot_matrix)

        robot_pos, robot_orientation = update_pos_and_orn(
            transform, velocity, robot_pos, robot_orientation, dt
        )

        sleep(0.01)  # arbitrary sleep to let the changes take place

    p.disconnect()
    sys.exit(0)

def main_real():
    """
    Implement main function for interbotix arms
    """
    
    # Jul 24 - 21:30
    # initialize target image
    
    # intialize robot orientation frame
    # do
    #   get real-life rgb image
    #   resize real-life rgb image
    #   get the robot rotation
    #   run Cutie/GAM to extract mask
    #   run LightGlue to match curr-im vs tgt-im (q: sampling?)
    #   if not-in-contact:
    #       move forward by constant velocity
    #   velocity <- get_linear_vel(src-kpts, tgt-kpts)
    #   transform <- get_transform-matrix()
    #   compute newpos, neworn = update-pos-and-orn(transform, velocity, robot-pos, robot-orn, dt)
    #
    
    pass

if __name__ == "__main__":
    main()
    # simple_forward()