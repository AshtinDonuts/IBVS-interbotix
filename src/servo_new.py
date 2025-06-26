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

# Global configuration for SuperPoint
SUPERPOINT_CONFIG = {
    'max_num_keypoints': 30,
    'keypoint_threshold': 0.005,
    'remove_borders': 4
}

def configure_superpoint(max_keypoints: int = 30, keypoint_threshold: float = 0.005, 
                        remove_borders: int = 4) -> None:
    """
    Configure SuperPoint parameters globally
    
    Args:
        max_keypoints: Maximum number of keypoints to detect
        keypoint_threshold: Minimum confidence threshold for keypoints
        remove_borders: Border pixels to ignore
    """
    global SUPERPOINT_CONFIG
    SUPERPOINT_CONFIG.update({
        'max_num_keypoints': max_keypoints,
        'keypoint_threshold': keypoint_threshold,
        'remove_borders': remove_borders
    })
    
    # Reset extractor to use new config
    if hasattr(get_SuperPoints, '_extractor'):
        delattr(get_SuperPoints, '_extractor')

def get_SuperPoints(img_arr : np.ndarray) -> Tuple[Union[List , None], List]:
    """
    Detects SuperPoints in the given image
    """
    import sys
    sys.path.append('../../LightGlue')  ## fix this

    from lightglue import SuperPoint
    import torch

    # Initialize SuperPoint feature extractor (create once and reuse)
    if not hasattr(get_SuperPoints, '_extractor'):
        get_SuperPoints._extractor = SuperPoint(
            max_num_keypoints=SUPERPOINT_CONFIG['max_num_keypoints'],
            keypoint_threshold=SUPERPOINT_CONFIG['keypoint_threshold'],
            remove_borders=SUPERPOINT_CONFIG['remove_borders']
        )
    
    extractor = get_SuperPoints._extractor
    
    # Convert image to grayscale and float32
    if img_arr is None:
        return None, []
    
    if len(img_arr.shape) == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_arr
    
    # Ensure image is in the correct format and convert to PyTorch tensor
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0
    
    # Convert numpy array to PyTorch tensor
    gray_tensor = torch.from_numpy(gray).unsqueeze(0)  # Add batch dimension
    
    # Extract features
    try:
        feats = extractor.extract(gray_tensor)
        keypoints = feats['keypoints'].cpu().numpy()
    except Exception as e:
        print(f"Error extracting SuperPoint features: {e}")
        return None, []
    
    # Format output to match get_markers() interface
    # Each keypoint becomes a "marker corner" with 1 point
    marker_corners = []
    marker_ids = []
    
    # Debug information
    print(f"Debug: keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'No shape'}")  ## (1, 30, 2)
    print(f"Debug: keypoints type: {type(keypoints)}")
    print(f"Debug: number of keypoints: {len(keypoints)}")
    if len(keypoints) > 0:
        print(f"Debug: first keypoint: {keypoints[0]}")
    
    # Reshape keypoint to expected format: (1, 30, 2) -> (30, 1, 2)
    keypoints = np.reshape(keypoints, (30, 1, 2))

    if len(keypoints) > 0:
        # marker_corners : List[np.array(1,1,2)] with len 30
        marker_corners = [np.array([kp]).reshape(1,1,2) for kp in keypoints]
        marker_ids = np.arange(len(keypoints)).reshape(-1,1)

    # reshape into np.array(1, N, 2)
    marker_corners = np.array(marker_corners).reshape(1, -1, 2)

    return marker_corners, marker_ids


def DO_NOT_USE_match_superpoints(img1: np.ndarray, img2: np.ndarray) -> Tuple[List, List, List]:
    """
    Match SuperPoint features between two images
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Tuple of (matched_kpts1, matched_kpts2, match_scores)
    """
    try:
        from lightglue import LightGlue, SuperPoint
        
        # Extract features from both images
        kpts1, ids1 = get_SuperPoints(img1)
        kpts2, ids2 = get_SuperPoints(img2)
        
        if not kpts1 or not kpts2:
            return [], [], []
        
        # Convert to LightGlue format
        feats1 = {'keypoints': kpts1[0].reshape(-1, 2), 'descriptors': None}
        feats2 = {'keypoints': kpts2[0].reshape(-1, 2), 'descriptors': None}
        
        # Initialize matcher
        matcher = LightGlue(features='superpoint')
        
        # Match features
        matches = matcher.match(feats1, feats2)
        
        # Extract matched keypoints
        matched_kpts1 = feats1['keypoints'][matches['matches'][:, 0]]
        matched_kpts2 = feats2['keypoints'][matches['matches'][:, 1]]
        scores = matches['scores']
        
        return matched_kpts1, matched_kpts2, scores
        
    except ImportError:
        print("LightGlue not available for feature matching")
        return [], [], []
    except Exception as e:
        print(f"Error in feature matching: {e}")
        return [], [], []


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

    return marker_corners[0], marker_ids[0]


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

    marker_corners, _ = get_markers(img_arr)
    if not marker_corners:
        return None
    corners, _ = marker_corners, _
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
