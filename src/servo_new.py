"""
servoing module 
--------

Supplementary update:
Jun 24 : Integrate SuperPoint feature detection

"""
##
import pdb
#

import numpy as np
import cv2
from typing import List, Union, Tuple
import torch
from pathlib import Path

from servo import *   # methods from original script
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, resize_image
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

def _configure_superpoint(max_keypoints: int = 30, keypoint_threshold: float = 0.005, 
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

def _get_SuperPoints(img_arr : np.ndarray) -> Tuple[Union[List , None], List]:
    """
    Detects SuperPoints in image and returns ALL Points.
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
    
    # print(f"Debug: keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'No shape'}")  ## (1, 30, 2)
    # print(f"Debug: keypoints type: {type(keypoints)}")
    # print(f"Debug: number of keypoints: {len(keypoints)}")
    
    # TODO : REFACTOR reshape transformation redundancies.
    # Reshape keypoint to expected format: (1, n, 2) -> (n, 1, 2), n = min(n, 30)
    keypoints = np.reshape(keypoints, (-1, 1, 2))

    if len(keypoints) > 0:
        # marker_corners : List[np.array(1,1,2)] : (n,)
        _keypoints = [np.array([kp]).reshape(1,1,2) for kp in keypoints]

    # reshape into np.array(1, N, 2)
    _keypoints = np.array(_keypoints).reshape(1, -1, 2)
    
    return _keypoints, None


def match_superpoints(im0: Union[Path, np.ndarray], im1: Union[Path, np.ndarray]) -> Tuple[List, List]:
    """
    Match SuperPoint features between two images
    Args:
        Path or Nd.array of either the two input images
    Returns:
        Tuple of (matched_kpts1, matched_kpts2, match_scores)
    """
    
    def load_im(im: Union[Path, np.ndarray], resize_HW: Union[Tuple[int, int] | None] = None) -> torch.Tensor:
        """ Load image from Path or convert numpy array to tensor """

        if isinstance(im, np.ndarray):
            assert im.ndim == 3, "...Expected image to be a 3D array (H, W, C)."
            # convert from [H x W x C] to [C x H x W]
            if resize_HW is not None:
                im, scale = resize_image(im, resize_HW)
                im = im.transpose(2, 0, 1)  # HxWxC to CxHxW
            else:
                raise ValueError(f"Invalid image shape: {im.shape}. Expected 3D array (H, W, C).")
            im  = torch.from_numpy(im).float()
            assert isinstance(im, torch.Tensor), f"Expected torch.Tensor, got {type(im)}"

            return im

        elif isinstance(im, Path) or isinstance(im, str):
            im_path = str(im)
            if im_path.endswith('.jpg') or im_path.endswith('.png'):
                return load_image(im_path, resize_HW)
            else:
                raise ValueError(f"Unsupported image format: {im_path}")
        else:
            raise ValueError(f"Unsupported input type: {type(im)}")

    # Load images
    image0, image1 = load_im(im0, (480, 640)), load_im(im1, (480, 640))

    # for debug
    # print(f'image0: {image0.shape}, image1: {image1.shape}')
    # print(f'image0: {image0.dtype}, image1: {image1.dtype}')

    # TODO: bug - feats0 length mismatch w ipynb
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    # pdb.set_trace()

    with open('dist_img/src_kpts_count.txt', 'a') as f:
        f.write(f"Number of features in feats0: {feats0['keypoints'].shape}\n")
        f.write(f"Number of features in feats1: {feats1['keypoints'].shape}\n")

    # pdb.set_trace()

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    
    # matched subset of kpts
    # It is implicit same rows in m_kpts0, m_kpts1 are matches.
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return m_kpts0, m_kpts1

def _compute_motion_vector(m_kpts0: Union[np.ndarray, torch.Tensor], m_kpts1: Union[np.ndarray, torch.Tensor], normalize=True):

    def _compute_kp_dist(m_kpts0, m_kpts1):
        """ compute average distance between matached kpts in 1D array """
        # Convert to numpy array if tensor
        if isinstance(m_kpts0, torch.Tensor):
            m_kpts0 = m_kpts0.cpu().numpy()
        if isinstance(m_kpts1, torch.Tensor):
            m_kpts1 = m_kpts1.cpu().numpy()
            
        # Compute differences between matched keypoints
        diffs = m_kpts1 - m_kpts0
        
        # Return mean of absolute differences for this component
        return np.mean(np.abs(diffs), axis=0)
    
    vecx, vecy = _compute_kp_dist(m_kpts0, m_kpts1)
    
    if normalize:
        norm_coef = np.sqrt(vecx**2 + vecy**2)
        unit_vecx = vecx / norm_coef
        unit_vecy = vecy / norm_coef
    
    return (unit_vecx, unit_vecy)

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

    # DO NOT USE get_Superpoints()
    # Use match_superpoints() instead.
    raise 
    ret, _ = get_SuperPoints(target_img)
    print(f'{ret.shape}')

if __name__ == '__main__':
    test()