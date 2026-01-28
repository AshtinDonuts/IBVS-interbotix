"""
Example usage of SuperPoint integration with visual servoing
"""

import cv2
import numpy as np
from src.servo_new import get_SuperPoints, configure_superpoint, match_superpoints

def main():
    # Configure SuperPoint parameters
    configure_superpoint(max_keypoints=50, keypoint_threshold=0.01)
    
    # Load an image
    impath = '/home/khw/IBVS-interbotix/assets/crop1a.png'
    img = cv2.imread(impath)
    if img is None:
        print("Could not load image")
        return
    
    # Detect SuperPoint features
    marker_corners, marker_ids = get_SuperPoints(img)
    
    if marker_corners is not None:
        # marker_corners has shape (1, N, 2) where N is the number of keypoints
        num_keypoints = marker_corners.shape[1]
        print(f"Detected {num_keypoints} SuperPoint features")
        
        # Reshape to (N, 2) for easier iteration
        keypoints = marker_corners.reshape(-1, 2)
        
        # Visualize keypoints
        for i, kpt in enumerate(keypoints):
            cv2.circle(img, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)
            cv2.putText(img, str(i), (int(kpt[0])+5, int(kpt[1])-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Show result
    cv2.imshow('SuperPoint Features', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
