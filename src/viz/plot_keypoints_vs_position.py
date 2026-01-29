"""
Consumes the text file that counts the source keypoints based on position from the target \
    and generates a plot

Requires a file with this format:

Number of features in feats0: torch.Size([1, 612, 2])
Number of features in feats1: torch.Size([1, 293, 2])
Iteration 0: 1 matching keypoints detected, robot position: [0, 0, 1.0]
Number of features in feats0: torch.Size([1, 549, 2])
Number of features in feats1: torch.Size([1, 293, 2])
Iteration 1: 0 matching keypoints detected, robot position: [0, 0, 1.0]
...

TODO: Make the code less hacky

"""

import matplotlib.pyplot as plt
import re
import numpy as np

def parse_keypoints_data(filename):
    """Parse the keypoints data file and extract robot positions and keypoint counts."""
    robot_positions = []
    keypoint_counts = []
    
    with open(filename, 'r') as file:
        for line in file:
            # Look for lines with iteration data
            if line.startswith('Iteration'):
                # Extract keypoint count
                kpt_match = re.search(r'(\d+) matching keypoints detected', line)
                if kpt_match:
                    keypoint_count = int(kpt_match.group(1))
                    keypoint_counts.append(keypoint_count)
                
                # Extract robot position
                pos_match = re.search(r'robot position: \[([\d., ]+)\]', line)
                if pos_match:
                    position_str = pos_match.group(1)
                    # Parse the position values
                    position_values = [float(x.strip()) for x in position_str.split(',')]
                    # Extract Y position (index 1)
                    y_position = position_values[1]
                    robot_positions.append(y_position)
    
    return robot_positions, keypoint_counts

def create_plot(robot_positions, keypoint_counts, output_filename='keypoints_vs_rotation.png'):
    """Create a plot of keypoint counts vs robot Y position."""
    plt.figure(figsize=(12, 8))

    # use_offset = True ##
    # # use robot position relative to y-offset
    # if use_offset:
    #     robot_positions = 2 - np.array(robot_positions)
    #     robot_positions = list(robot_positions)

    robot_positions = [-np.pi/10 + i*(np.pi/50) for i in range(len(keypoint_counts))]

    plt.plot(robot_positions, keypoint_counts, 'bo-', linewidth=2, markersize=6, label='Matching Keypoints')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Customize plot
    plt.xlabel('Robot Rot Position', fontsize=14)
    plt.ylabel('Number of Matching Keypoints', fontsize=14)
    plt.title('Matching Keypoints vs Robot Rot Position', fontsize=16, fontweight='bold')
    
    # Add statistics
    max_kpts = max(keypoint_counts)
    max_kpts_pos = robot_positions[keypoint_counts.index(max_kpts)]
    plt.annotate(f'Max: {max_kpts} keypoints\nat Y={max_kpts_pos:.2f}', 
                xy=(max_kpts_pos, max_kpts), 
                xytext=(max_kpts_pos + 0.1, max_kpts - 20),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    plt.show()
    
    return plt

def print_statistics(robot_positions, keypoint_counts):
    """Print some statistics about the data."""
    print("\n=== STATISTICS ===")
    print(f"Total iterations: {len(robot_positions)}")
    print(f"Y position range: {min(robot_positions):.3f} to {max(robot_positions):.3f}")
    print(f"Keypoint count range: {min(keypoint_counts)} to {max(keypoint_counts)}")
    print(f"Average keypoints: {np.mean(keypoint_counts):.1f}")
    print(f"Max keypoints: {max(keypoint_counts)} at Y={robot_positions[keypoint_counts.index(max(keypoint_counts))]:.3f}")
    print(f"Min keypoints: {min(keypoint_counts)} at Y={robot_positions[keypoint_counts.index(min(keypoint_counts))]:.3f}")

def main():

    filename = '/home/khw/Documents/6dpose/Image-Based-Visual-Servoing/src/dist_img/src_kpts_count.txt'
    robot_positions, keypoint_counts = parse_keypoints_data(filename)
    
    create_plot(robot_positions, keypoint_counts)

if __name__ == "__main__":
    main() 