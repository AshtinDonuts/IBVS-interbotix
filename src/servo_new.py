"""
servoing module 
--------

Supplementary update:
Jun 24 : Integrate SuperPoint feature detection

"""

import numpy as np
import cv2
from typing import List, Union, Tuple
import torch
from pathlib import Path

from servo import *   # methods from original script
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

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
    
    # Debug information
    # print(f"Debug: keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'No shape'}")  ## (1, 30, 2)
    # print(f"Debug: keypoints type: {type(keypoints)}")
    # print(f"Debug: number of keypoints: {len(keypoints)}")
    # if len(keypoints) > 0:
    #     print(f"Debug: first keypoint: {keypoints[0]}")
    
    # TODO : fix reshape transformation redundancies.
    # Reshape keypoint to expected format: (1, n, 2) -> (n, 1, 2), n = min(n, 30)
    keypoints = np.reshape(keypoints, (-1, 1, 2))

    if len(keypoints) > 0:
        # marker_corners : List[np.array(1,1,2)] : (n,)
        marker_corners = [np.array([kp]).reshape(1,1,2) for kp in keypoints]

    # reshape into np.array(1, N, 2)
    marker_corners = np.array(marker_corners).reshape(1, -1, 2)
    
    return marker_corners, None


def match_superpoints(im0_path: Union[Path, str], im1_path: Union[Path, str]) -> Tuple[List, List, List]:
    """
    Match SuperPoint features between two images
    Args:
        Path object or str object path of either the two input images
    Returns:
        Tuple of (matched_kpts1, matched_kpts2, match_scores)
    """
    image0 = load_image(im0_path)
    image1 = load_image(im1_path)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    
    # matched subset of kpts
    # It is implicit same rows in m_kpts0, m_kpts1 are matches.
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return m_kpts0, m_kpts1

def _compute_distances(m_kpts0: np.ndarray, m_kpts1: np.ndarray):
    raise NotImplemented

# TODO
def compute_motion_vector(m_kpts0, m_mpts1):
    matched_distances = _compute_distances(...)
    
    def _compute_mean(dist):
        raise NotImplemented
    
    avg_motion_vec = _compute_mean(matched_distances)
    
    return avg_motion_vec

##
#--------------
### Remaining are test functions

def test():

    print('Running servo_new.py')

    from pathlib import Path

    # Get target / ref image pair
    target_path=Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000050_crop.png')

    import cv2
    target_img = cv2.imread(str(target_path))

    # Verify images were loaded successfully
    assert target_img is not None, f"Failed to load target image from {target_path}"

    ret, _ = get_SuperPoints(target_img)
    print(f'{ret.shape}')

if __name__ == '__main__':
    test()