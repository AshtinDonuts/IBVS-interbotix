"""
servoing module 
--------

Supplementary update:
Jun 24 : Integrate SuperPoint feature detection

"""

import numpy as np
import cv2
from typing import List, Union, Tuple

# ---- New functions BEGIN ----
def get_SuperPoints(img_arr : np.ndarray) -> Tuple[Union[List , None], List]:
    """
    Detects SuperPoints in the given image
    """
    import sys
    sys.path.append('../../LightGlue')

    from lightglue import SuperPoint

    # Initialize SuperPoint feature extractor
    extractor = SuperPoint(max_num_keypoints=30) # 50
    
    # Convert image to grayscale and float32
    if len(img_arr.shape) == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_arr
    gray = gray.astype(np.float32) / 255.0
    
    # Extract features
    feats = extractor.extract(gray)
    keypoints = feats['keypoints'].cpu().numpy()
    
    # Format output to match get_markers() interface
    # Each keypoint becomes a "marker corner" with 1 point
    marker_corners = []
    marker_ids = []
    
    if len(keypoints) > 0:
        # Convert each keypoint to the expected format: (1,1,2) array
        marker_corners = [np.array([kp]).reshape(1,1,2) for kp in keypoints]
        marker_ids = np.arange(len(keypoints)).reshape(-1,1)
        
    return marker_corners, marker_ids


# ---- New functions END ----

# NOTE: this implementation of visual servoing uses Aruco markers

def get_markers(img_arr: np.ndarray) -> Tuple[Union[List, None], List]:
    """
    gets the corner markers in the given image
    """
    # detect the aruco marker in the image
    marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    param_markers = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(marker_dict, param_markers)
    gray_frame = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray_frame)

    return marker_corners, marker_ids


def mark_corners(img_arr: np.ndarray, points: List[List[int]]) -> np.ndarray:
    """
    marks circles on the given points
    """
    for point in points:
        cv2.circle(img_arr, tuple(point), 5, (255, 255, 255), 3)

    return img_arr


def get_marker_corners(img_arr: np.ndarray) -> Union[List[List[float]], None]:
    """
    gets the corners of the marker in the given image
    """
    marker_corners, marker_ids = get_markers(img_arr)
    if not marker_corners:
        return None

    corners, ids = marker_corners[0], marker_ids[0]
    cv2.polylines(
        img_arr, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
    )

    corners = corners.reshape(4, 2)
    corners = corners.astype(int)

    top_left = list(corners[0].ravel())
    top_right = list(corners[1].ravel())
    bottom_right = list(corners[2].ravel())
    bottom_left = list(corners[3].ravel())

    # we have to ensure the points always follow a fixed order for accurate error calc
    points = sorted(
        [top_left, top_right, bottom_left, bottom_right],
        key=lambda point: (point[0], point[1]),
    )

    return points
