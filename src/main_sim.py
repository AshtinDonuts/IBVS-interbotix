"""
PVM-enhanced IBVS implementation
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import sys
from time import sleep
from typing import Tuple, List, Dict
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import yaml
import cv2

from image import convert_img_to_arr, save_image, get_image_config
from superpoint_utils import match_superpoints
import motion_utils


#  Simple config
MAX_ITERATIONS = 70
TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/cat.png')
K = 20

#  Error exit condition config
MIN_ERROR = float("inf")
ERROR_GROWTH_LIMIT = 0.90

# Global scene manager instance for accessing object locations
SCENE_MANAGER = None


class SceneManager:
    """
    Helper class to manage the simulation scene and query object locations.
    """

    def __init__(self, robot_pos: list[float]) -> None:
        self.robot_pos = robot_pos
        self.plane_id: int | None = None
        self.obstacles: List[int] = []
        # Mapping from logical object name to pybullet body unique ID
        self.objects: Dict[str, int] = {}
        self.goal_id: int | None = None

    def build_default_scene(self) -> Tuple[int, List[int]]:
        """
        Build the default scene: plane + wall of cubes with a goal cube.

        Returns
        -------
        plane_id : int
            The pybullet id of the plane.
        obstacles : list[int]
            List of pybullet ids for all obstacle cubes (first is the goal).
        """
        plane_id = p.loadURDF("plane.urdf")
        self.plane_id = plane_id

        # loading obstacles, with the main cube at the first index
        base = self.robot_pos  # for now
        base_orn = p.getQuaternionFromEuler([0, 0, 0])

        # new offset for testing
        new_offsetx = 0.75
        new_offsetz = 0.75

        # builds a wall of cubes
        for z_offset in [0, 1]:   # z is up/down
            for y_offset in [2.5]:  # y is forward-backward
                for x_offset in [0, -1, 1]:  # x is left-right
                    body_id = p.loadURDF(
                        "cube_small.urdf",
                        [
                            base[0] + x_offset + new_offsetx,
                            base[1] + y_offset,
                            base[2] + z_offset + new_offsetz,
                        ],
                        base_orn,
                        globalScaling=20,  # Remove or adjust as needed
                    )
                    self.obstacles.append(body_id)

                    # give each obstacle a deterministic name based on its offsets
                    name = f"cube_x{x_offset}_y{y_offset}_z{z_offset}"
                    self.objects[name] = body_id

        # texture the first cube (set as goal)
        if self.obstacles:
            goal_obs_id = self.obstacles[0]
            self.goal_id = goal_obs_id
            set_aruco_marker_texture(goal_obs_id)
            self.objects["goal"] = goal_obs_id

        return plane_id, self.obstacles

    def get_object_position(self, name: str) -> np.ndarray:
        """
        Get the world position of an object by its logical name.

        Parameters
        ----------
        name : str
            Logical name used when creating the object (e.g. "goal",
            "cube_x0_y2.5_z0").

        Returns
        -------
        np.ndarray
            3D position (x, y, z) in world coordinates.
        """
        if name not in self.objects:
            raise KeyError(f"Object '{name}' not found in scene.")

        body_id = self.objects[name]
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return np.array(pos)

    def get_goal_position(self) -> np.ndarray:
        """
        Convenience method to get the goal cube position.
        """
        if self.goal_id is None:
            raise RuntimeError("Goal object has not been created yet.")
        pos, _ = p.getBasePositionAndOrientation(self.goal_id)
        return np.array(pos)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def init_scene(robot_pos: list[float]) -> Tuple[int, List[int]]:
    """
    Initialise the scene and return created objects.

    This function now uses a global `SceneManager` instance so that
    object locations can be queried later in the program.

    Returns
    -------
    plane_id : int
        The pybullet id of the plane.
    obstacles : list[int]
        List of pybullet ids for all obstacle cubes (first is the goal).
    """
    global SCENE_MANAGER
    SCENE_MANAGER = SceneManager(robot_pos)
    plane_id, obstacles = SCENE_MANAGER.build_default_scene()
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

def convert_to_transformation_matrix(
    robot_pos: List[float], robot_rotation_matrix: np.ndarray) -> np.ndarray:
    """
    create the transformation matrix to convert from image (world) coordinates to local
        camera coordinates
    """
    # creating the translation matrix (SE(3))
    t_c = np.zeros((4, 4))
    for i, val in enumerate(robot_pos):
        t_c[i][3] = -val
        t_c[i][i] = 1
    t_c[3][3] = 1

    # creating the rotational matrix (SE(3))
    r_i = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            r_i[i][j] = robot_rotation_matrix[i][j]
    r_i[3][3] = 1
    
    # multiply the translation and rotation together to create a single transformation matrix (SE(3))
    transform = np.matmul(r_i, t_c)

    return transform

def capture_camera_image(
    robot_pos: List[float], robot_rotation_matrix: np.ndarray) -> np.ndarray:
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
    """
    # First convert into homogeneous coords
    # Then transform into new coordinate frame
    del_pos = np.matmul(transform, [*velocity[:3], 1])
    
    # convert back into Cartesian coords
    for i in range(3):
        robot_pos[i] += (del_pos[i] / del_pos[-1]) * dt

    # constant forward velocity
    robot_pos[1] += 0.0 * dt

    # del_orn = np.matmul(transform, [*velocity[3:], 1])
    # for i in range(3):
    #     robot_orn[i] += (del_orn[i] / del_orn[-1]) * dt

    return robot_pos, robot_orn

def update_error(error_mag: float, iteration: int | None = None) -> None:
    """
    updates the global min error and determines when to break
    """

    global MIN_ERROR

    if iteration is not None:
        print(f"{iteration}:", end="")
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

def OLD_simple_forward() -> None:
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

        rgba = img[2]  # Img[2]: (H x W x 4)
        rgba_arr = convert_img_to_arr(
            rgba, int(img_conf["height"]), int(img_conf["width"])  # Img[2]: (H x W x 4)
        )
        rgb_img_arr = rgba_arr[:, :, :3]  # remove alpha channel (..,4) -> (..,3)

        # TODO : Directly extract image from pybullet
        save_impath = f'dist_img/distance_image_{i}.png'
        cv2.imwrite(save_impath, cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR))

        # TODO: Remove redundant reshape
        # remove batch dim [1, 500, 800, 3] -> [500, 800, 3]
        # rgb_img_arr = np.squeeze(rgb_img_arr, axis=0)
        
        assert rgb_img_arr.ndim == 3 and rgb_img_arr.shape[2] == 3, \
            f"Expected 3D RGB image, got shape {rgb_img_arr.shape}"

        src_kpts, tgt_kpts = match_superpoints(
            save_impath, TARGET_PATH
        )
        assert len(src_kpts) == len(tgt_kpts), "bug in match-superpoints()"
        # pdb.set_trace()

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
    # Load configuration
    config = load_config(Path(__file__).parent / 'config.yaml')
    
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

        img = capture_camera_image(robot_pos, robot_rot_matrix)   # (width, height, rgbaImg, depthImg, segImg)

        rgba = img[2]       # rgbaImg : (h x w x 4)
        rgba_arr = convert_img_to_arr(
            rgba, int(img_conf["height"]), int(img_conf["width"])
        )
        rgb_img_arr = rgba_arr[:, :, :3]  # remove alpha channel [..,4] -> [..,3]

        save_impath = f'/home/khw/IBVS-interbotix/src/dist_img/dist_img_{i}.png'
        cv2.imwrite(save_impath, cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR))

        # import pdb; pdb.set_trace()

        src_kpts, tgt_kpts = match_superpoints(
            save_impath, TARGET_PATH
        )

        assert len(src_kpts) == 0, "No Superpoints detected. It is recommended to reconfigure the experiment space."
        assert len(src_kpts) == len(tgt_kpts), "Error from match_superpoints()"

        print(f"number of SuperPoints detected: {len(src_kpts)} vs {len(tgt_kpts)}")

        # Only use a subset of SuperPoints
        K_sample_src, K_sample_tgt = motion_utils.sample_points(src_kpts, tgt_kpts, K)

        error = motion_utils.get_error_mse(motion_utils.get_error_vec_K(K_sample_src, K_sample_tgt))

        # Save image with error printed on it
        save_image(error, i, rgb_img_arr, MIN_ERROR)

        # Uncomment for early exit conditioned on MSE divergence
        update_error(error, iteration=i)

        # init new velocity vector for this iteration
        unit_velocity = np.zeros(6)

        #  Sets the velocity, transform vector, and update position and orientation
        #  This is a custom controller
        #  TODO: refactor into a controller class
        unit_velocity = motion_utils.update_perpendicular_velocity(unit_velocity, K_sample_src, K_sample_tgt)
        unit_velocity = motion_utils.update_forward_velocity(unit_velocity)
        unit_velocity = motion_utils.update_angular_velocity(unit_velocity)

        scaled_velocity = motion_utils.scale_velocity(unit_velocity, config)
        transform = convert_to_transformation_matrix(robot_pos, robot_rot_matrix)
        robot_pos, robot_orientation = update_pos_and_orn(
            transform, scaled_velocity, robot_pos, robot_orientation, dt
        )

        sleep(0.01)  # sleep to let the changes take place

    p.disconnect()
    sys.exit(0)


if __name__ == "__main__":
    main()
    # OLD_simple_forward()