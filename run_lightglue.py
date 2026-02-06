#!/usr/bin/env python3
"""
Script to run LightGlue feature matching on two images.

Usage:
    python run_lightglue.py <image0_path> <image1_path> [--feature-type <type>] [--visualize]
    
Example:
    python run_lightglue.py image1.jpg image2.jpg --feature-type superpoint --visualize
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Import LightGlue components
from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
from lightglue.utils import load_image, rbd
from lightglue import viz2d


def main():
    parser = argparse.ArgumentParser(
        description="Run LightGlue feature matching on two images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image0",
        type=str,
        help="Path to the first image",
    )
    parser.add_argument(
        "image1",
        type=str,
        help="Path to the second image",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="superpoint",
        choices=["superpoint", "disk", "aliked", "sift"],
        help="Feature extractor type (default: superpoint)",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Maximum number of keypoints to extract (default: 2048)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the matches using matplotlib",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps). Auto-detected if not specified",
    )

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Check if image files exist
    image0_path = Path(args.image0)
    image1_path = Path(args.image1)

    if not image0_path.exists():
        print(f"Error: Image not found: {image0_path}", file=sys.stderr)
        sys.exit(1)

    if not image1_path.exists():
        print(f"Error: Image not found: {image1_path}", file=sys.stderr)
        sys.exit(1)

    # Load feature extractor based on type
    print(f"Loading {args.feature_type} extractor...")
    if args.feature_type == "superpoint":
        extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    elif args.feature_type == "disk":
        extractor = DISK(max_num_keypoints=args.max_keypoints).eval().to(device)
    elif args.feature_type == "aliked":
        extractor = ALIKED(max_num_keypoints=args.max_keypoints).eval().to(device)
    elif args.feature_type == "sift":
        extractor = SIFT(max_num_keypoints=args.max_keypoints).eval().to(device)

    # Load LightGlue matcher
    print(f"Loading LightGlue matcher for {args.feature_type}...")
    matcher = LightGlue(features=args.feature_type).eval().to(device)

    # Load images
    print(f"Loading images...")
    print(f"  Image 0: {image0_path}")
    print(f"  Image 1: {image1_path}")

    try:
        image0 = load_image(image0_path).to(device)
        image1 = load_image(image1_path).to(device)
    except Exception as e:
        print(f"Error loading images: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

        # Match features
        print("Matching features...")
        matches01 = matcher({"image0": feats0, "image1": feats1})

        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Extract matched keypoints
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01["matches"]
    match_confidence = matches01.get("matching_scores", None)

    num_matches = len(matches)
    num_keypoints0 = len(kpts0)
    num_keypoints1 = len(kpts1)

    print("\n" + "=" * 50)
    print("Matching Results:")
    print("=" * 50)
    print(f"Keypoints in image 0: {num_keypoints0}")
    print(f"Keypoints in image 1: {num_keypoints1}")
    print(f"Number of matches: {num_matches}")
    if "stop" in matches01:
        print(f"Stopped after {matches01['stop']} layers (out of {matcher.conf.n_layers})")

    if num_matches > 0:
        m_kpts0 = kpts0[matches[..., 0]]
        m_kpts1 = kpts1[matches[..., 1]]
        print(f"\nMatched keypoints shape: {m_kpts0.shape}")

        if match_confidence is not None:
            print(f"Average match confidence: {match_confidence.mean().item():.4f}")

    # Visualize if requested
    if args.visualize:
        print("\nVisualizing matches...")
        try:
            axes = viz2d.plot_images([image0, image1])
            if num_matches > 0:
                viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
                if "stop" in matches01:
                    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
            plt.show()
        except Exception as e:
            print(f"Warning: Could not visualize matches: {e}", file=sys.stderr)
            print("You may need to install matplotlib or run in an environment with display support")

    return 0


if __name__ == "__main__":
    sys.exit(main())
