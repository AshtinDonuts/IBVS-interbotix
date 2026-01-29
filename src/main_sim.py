"""
PVM-enhanced IBVS implementation.

This uses LightGlue for general feature matching.

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
import csv

from image import convert_img_to_arr, save_image, get_image_config
from superpoint_utils import match_superpoints
import motion_utils
from motion import LAMBDA
from gsam2_terminator import TerminationHandler


#  Simple config
MAX_ITERATIONS = 200
# TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/cat.png')
TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/resized_cat.png')
# TARGET_PATH = Path('/home/khw/IBVS-interbotix/assets/aruco.png')
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
        for z_offset in [0, -1, 1]:   # z is up/down
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
            experiment_name = "experiment"
        
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
            'distance_to_target',
            'oracle_orientation_diff',  # norm of orientation diff against [0, 0, 0], which is the correct target pose.
            'num_matched_keypoints',
            'robot_pos_x',
            'robot_pos_y',
            'robot_pos_z',
            'robot_orn_x',
            'robot_orn_y',
            'robot_orn_z',
            'target_pos_x',
            'target_pos_y',
            'target_pos_z',
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
        target_pos: np.ndarray,
        num_matched_keypoints: int,
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
        target_pos : np.ndarray
            Target position [x, y, z].
        num_matched_keypoints : int
            Number of matched keypoints between live and target.
        image_path : str
            Path to the saved image for this iteration.
        """
        # Calculate distance to target
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(robot_pos))
        
        # Calculate oracle orientation difference
        orientation_diff = np.linalg.norm(robot_orn)  # Magnitude of orientation angles
        
        # Write row
        self.csv_writer.writerow([
            iteration,
            f"{error_magnitude:.6f}",
            f"{distance_to_target:.6f}",
            f"{orientation_diff:.6f}",
            num_matched_keypoints,
            f"{robot_pos[0]:.6f}",
            f"{robot_pos[1]:.6f}",
            f"{robot_pos[2]:.6f}",
            f"{robot_orn[0]:.6f}",
            f"{robot_orn[1]:.6f}",
            f"{robot_orn[2]:.6f}",
            f"{target_pos[0]:.6f}",
            f"{target_pos[1]:.6f}",
            f"{target_pos[2]:.6f}",
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
    # Then back to Cartesian coords

    del_pos = np.matmul(transform, [*velocity[:3], 1])
    for i in range(3):
        robot_pos[i] += (del_pos[i] / del_pos[-1]) * dt

    # Angular velocity should be transformed using only the rotation part
    # Extract rotation matrix from transform (top-left 3x3)
    R_transform = transform[:3, :3]
    # Transform angular velocity: omega_world = R @ omega_camera
    omega_world = R_transform @ velocity[3:6]
    for i in range(3):
        robot_orn[i] += omega_world[i] * dt

    return robot_pos, robot_orn

def update_error(error_mag: float, iteration: int | None = None) -> None:
    """
    updates the global min error
    removed the break
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
    

## ============ Driving code ============== ##

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
    robot_orientation = [0.25, 0.25, 0.25]
    robot_orientation = [0.2, 0.1, 0.0]

    # set up scene
    _, _ = init_scene(robot_pos)

    # retrieve static scene information
    target_pos = SCENE_MANAGER.get_goal_position()

    # for simulating no depth sensing / real-world
    # in-sim, we can easily obtain the current distance to the object
    # With real-world depth sensing, we will also have to segment out the goal object.
    initial_dist_obj_goal = np.linalg.norm(target_pos - robot_pos)

    import os
    os.makedirs('/home/khw/IBVS-interbotix/src/dist_img', exist_ok=True)

    # Initialize data logger
    log_dir = Path(__file__).parent / 'logs'
    data_logger = DataLogger(log_dir, experiment_name='ibvs_sim')

    # Initialize termination handler (using Grounded SAM2)
    termination_handler = TerminationHandler(
        target_image_path=str(TARGET_PATH),
        text_prompt="cube.",  # Adjust based on your target object
        similarity_threshold=0.15,  # 15% difference threshold
        box_threshold=0.35,
        text_threshold=0.25,
        enabled=True  # Set to False to disable segmentation termination
    )

    sleep(1)  # arbitrary sleep to let the scene load
    
    for i in range(100):

        p.stepSimulation()
        robot_rot_matrix = get_robot_rotation_matrix(robot_orientation) 

        img = capture_camera_image(robot_pos, robot_rot_matrix)   # (width, height, rgbaImg, depthImg, segImg)

        rgba = img[2]       # rgbaImg : (h x w x 4)
        rgba_arr = convert_img_to_arr(
            rgba, int(img_conf["height"]), int(img_conf["width"])
        )
        rgb_img_arr = rgba_arr[:, :, :3]  # remove alpha channel [..,4] -> [..,3]

        save_impath = f'/home/khw/IBVS-interbotix/src/dist_img/dist_img_{i}.png'
        cv2.imwrite(save_impath, cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGB2BGR))

        src_kpts, tgt_kpts = match_superpoints(
            save_impath, TARGET_PATH
        )

        assert len(src_kpts) > 0, "No Superpoints detected. It is recommended to reconfigure the experiment space."
        assert len(src_kpts) == len(tgt_kpts), "Error from match_superpoints()"

        # Only use a subset of SuperPoints
        # TODO: change to allow accessing SuperPoints configuration directly
        try:
            K_sample_src, K_sample_tgt = motion_utils.sample_points(src_kpts, tgt_kpts, K)
        except:
            K_sample_src, K_sample_tgt = src_kpts, tgt_kpts

        error_vec = motion_utils.get_error_vec_Ksample(src_kpts, tgt_kpts)
        mse_error = motion_utils.get_error_mse(error_vec)

        # Save image with error printed
        save_image(mse_error, i, rgb_img_arr, MIN_ERROR)

        # Log iteration data
        data_logger.log_iteration(
            iteration=i,
            error_magnitude=mse_error,
            robot_pos=robot_pos,
            robot_orn=robot_orientation,
            target_pos=target_pos,
            num_matched_keypoints=len(src_kpts),
            image_path=save_impath
        )

        # UNcomment code for early exit conditioned on MSE divergence
        # update_error(mse_error, iteration=i)

        #  Sets the velocity, transform vector, and update position and orientation
        #  This is a custom controller
        #  TODO: refactor into a controller class

        vel = np.zeros(6)
        vel = motion_utils.update_forward_velocity(vel, config, curr_robot_pos=robot_pos, goal_pos=target_pos)
        vel = motion_utils.update_perpendicular_velocity(vel, K_sample_src, K_sample_tgt)
        # vel = motion_utils.update_forward_velocity(vel, config,  d0 = initial_dist_obj_goal, i=i, dt=dt)     # Option 2:  setting a forward velocity based on d0
        vel = motion_utils.update_angular_velocity(vel, K_sample_src, K_sample_tgt, lambda_gain=LAMBDA)
        print(vel[3:6])

        # Scale each velocity dimension for easier use
        scaled_velocity = motion_utils.scale_velocity(vel, config)
        transform = convert_to_transformation_matrix(robot_pos, robot_rot_matrix)
        robot_pos, robot_orientation = update_pos_and_orn(
            transform, scaled_velocity, robot_pos, robot_orientation, dt
        )

        sleep(0.01)  # sleep to let the changes take place

        # Check termination condition using segmentation mask similarity
        if termination_handler.check_termination(save_impath, iteration=i):
            break


    # Close data logger
    data_logger.close()
    
    p.disconnect()
    sys.exit(0)


if __name__ == "__main__":

    import pdb
    pdb.set_trace = lambda : 1  # COMMENT OUT if DEBUG, otherwise UNCOMMENT.

    main()
    # OLD_simple_forward()
