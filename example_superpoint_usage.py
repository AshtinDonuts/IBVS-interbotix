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
    img = cv2.imread('path_to_your_image.jpg')
    if img is None:
        print("Could not load image")
        return
    
    # Detect SuperPoint features
    marker_corners, marker_ids = get_SuperPoints(img)
    
    if marker_corners:
        print(f"Detected {len(marker_corners)} SuperPoint features")
        
        # Visualize keypoints
        for i, corner in enumerate(marker_corners):
            kpt = corner.reshape(-1, 2)[0]
            cv2.circle(img, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)
            cv2.putText(img, str(i), (int(kpt[0])+5, int(kpt[1])-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Show result
    cv2.imshow('SuperPoint Features', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 