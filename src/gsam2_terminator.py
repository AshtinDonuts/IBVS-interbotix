"""
Grounded SAM2 Termination Module

This module provides segmentation-based termination conditions for IBVS.
Uses Grounded SAM2 to segment target objects, then supports two comparison modes:

1. Mask Size Mode: Compare the number of pixels in segmentation masks
2. SSIM Mode: Compare structural similarity of the segmented object regions

Both modes first segment the object using Grounded SAM2, ensuring comparisons
focus on the target object rather than the entire scene.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Literal
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2

# Add Grounded-SAM-2 to path
GSAM2_DIR = Path(__file__).resolve().parent.parent / "lib" / "Grounded-SAM-2"
if str(GSAM2_DIR) not in sys.path:
    sys.path.insert(0, str(GSAM2_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class SegmentationTerminator:
    """
    Uses Grounded SAM2 to compare segmentation masks between current and target images.
    Terminates when mask sizes are similar, indicating proper object positioning.
    """
    
    def __init__(
        self,
        text_prompt: str = "cube.",
        sam2_checkpoint: str = None,
        sam2_model_config: str = None,
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        device: str = None,
        similarity_threshold: float = 0.15,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ):
        """
        Initialize the segmentation terminator with Grounded SAM2 models.
        
        Parameters
        ----------
        text_prompt : str
            Text prompt for object detection (e.g., "cube." or "cat.").
            Must be lowercase and end with a dot.
        sam2_checkpoint : str, optional
            Path to SAM2 checkpoint. Defaults to sam2.1_hiera_large.pt
        sam2_model_config : str, optional
            Path to SAM2 model config. Defaults to sam2.1_hiera_l.yaml
        grounding_model : str
            HuggingFace model ID for Grounding DINO
        device : str, optional
            Device to run on ('cuda' or 'cpu'). Auto-detects if None.
        similarity_threshold : float
            Relative difference threshold for mask size similarity (0.15 = 15%)
        box_threshold : float
            Confidence threshold for bounding box detection
        text_threshold : float
            Confidence threshold for text grounding
        """
        self.text_prompt = text_prompt
        self.similarity_threshold = similarity_threshold
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing SegmentationTerminator on {self.device}...")
        
        # Set default paths
        if sam2_checkpoint is None:
            sam2_checkpoint = str(GSAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt")
        if sam2_model_config is None:
            # Hydra expects relative path from sam2 directory, not absolute path
            sam2_model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # Enable autocast for efficiency
        if self.device == "cuda":
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Build SAM2 model
        # Need to change to GSAM2_DIR for Hydra to find configs
        original_dir = os.getcwd()
        try:
            os.chdir(str(GSAM2_DIR))
            print(f"Loading SAM2 from {sam2_checkpoint}...")
            print(f"Using config: {sam2_model_config}")
            sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        finally:
            os.chdir(original_dir)
        
        # Build Grounding DINO model
        print(f"Loading Grounding DINO: {grounding_model}...")
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model
        ).to(self.device)
        
        # Cache for target mask size and target image
        self.target_mask_size = None
        self.target_image = None  # For SSIM comparison
        
        print("SegmentationTerminator initialized successfully!")
    
    def get_mask_size(self, image_path: str) -> float:
        """
        Get the total mask size (number of pixels) for the detected object in an image.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
            
        Returns
        -------
        float
            Total number of pixels in the segmentation mask. Returns 0 if no object detected.
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            # Assume it's already a PIL Image
            image = image_path
        
        # Set image for SAM2
        self.sam2_predictor.set_image(np.array(image))
        
        # Run Grounding DINO detection
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        # Check if any objects were detected
        if len(results) == 0 or len(results[0]["boxes"]) == 0:
            print("Warning: No objects detected in image")
            return 0.0
        
        # Get bounding boxes for SAM2
        input_boxes = results[0]["boxes"].cpu().numpy()
        
        # Get segmentation masks
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Convert shape to (n, H, W) if needed
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Calculate total mask size (sum of all detected object masks)
        total_mask_size = np.sum(masks)
        
        return float(total_mask_size)
    
    def set_target_mask_size(self, target_image_path: str):
        """
        Compute and cache the mask size for the target image.
        
        Parameters
        ----------
        target_image_path : str
            Path to the target image
        """
        print(f"Computing target mask size from {target_image_path}...")
        self.target_mask_size = self.get_mask_size(target_image_path)
        print(f"Target mask size: {self.target_mask_size:.0f} pixels")
    
    def get_segmented_object(self, image_path: str) -> np.ndarray:
        """
        Get the segmented object region from an image using SAM2.
        Returns the cropped, masked object region.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
            
        Returns
        -------
        np.ndarray
            Cropped and masked object region (grayscale)
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
        else:
            image = image_path
            image_np = np.array(image)
        
        # Set image for SAM2
        self.sam2_predictor.set_image(image_np)
        
        # Run Grounding DINO detection
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        # Check if any objects were detected
        if len(results) == 0 or len(results[0]["boxes"]) == 0:
            print("Warning: No objects detected in image")
            return None
        
        # Get bounding boxes for SAM2
        input_boxes = results[0]["boxes"].cpu().numpy()
        
        # Get segmentation masks
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Convert shape to (n, H, W) if needed
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Use the first (highest confidence) mask
        mask = masks[0]
        box = input_boxes[0].astype(int)
        
        # Convert image to grayscale
        if len(image_np.shape) == 3:
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_np
        
        # Apply mask to image (ensure mask is boolean for the ~ operator)
        mask_bool = mask.astype(bool)
        masked_image = image_gray.copy()
        masked_image[~mask_bool] = 0  # Set background to black
        
        # Crop to bounding box
        x1, y1, x2, y2 = box
        cropped_object = masked_image[y1:y2, x1:x2]
        
        return cropped_object
    
    def set_target_image(self, target_image_path: str):
        """
        Segment and cache the target object for SSIM comparison.
        
        Parameters
        ----------
        target_image_path : str
            Path to the target image
        """
        print(f"Segmenting target object from {target_image_path}...")
        self.target_image = self.get_segmented_object(target_image_path)
        if self.target_image is not None:
            print(f"Target object segmented: {self.target_image.shape}")
        else:
            raise ValueError(f"Could not segment object from {target_image_path}")
    
    def compute_ssim(self, current_image_path: str) -> float:
        """
        Compute SSIM between current segmented object and target segmented object.
        
        Parameters
        ----------
        current_image_path : str
            Path to the current image
            
        Returns
        -------
        float
            SSIM score between 0 and 1 (1 = identical objects)
        """
        if self.target_image is None:
            raise RuntimeError("Target image not set. Call set_target_image() first.")
        
        # Segment current image
        current_object = self.get_segmented_object(current_image_path)
        
        if current_object is None:
            print("Warning: Could not segment object from current image")
            return 0.0
        
        # Resize current object to match target size for comparison
        if current_object.shape != self.target_image.shape:
            current_object = cv2.resize(current_object, 
                                       (self.target_image.shape[1], self.target_image.shape[0]))
        
        # Compute SSIM on segmented objects
        ssim_score = ssim(self.target_image, current_object, data_range=255)
        
        return float(ssim_score)
    
    def should_terminate(self, current_image_path: str) -> Tuple[bool, float, float]:
        """
        Check if current image mask size is similar enough to target to terminate.
        
        Parameters
        ----------
        current_image_path : str
            Path to the current camera image
            
        Returns
        -------
        should_terminate : bool
            True if mask sizes are similar enough to terminate
        current_mask_size : float
            Mask size of current image
        similarity : float
            Relative difference between current and target mask sizes
        """
        if self.target_mask_size is None:
            raise RuntimeError("Target mask size not set. Call set_target_mask_size() first.")
        
        current_mask_size = self.get_mask_size(current_image_path)
        
        # Handle case where no object detected
        if current_mask_size == 0 or self.target_mask_size == 0:
            return False, current_mask_size, float('inf')
        
        # Calculate relative difference
        relative_diff = abs(current_mask_size - self.target_mask_size) / self.target_mask_size
        
        # Check if similar enough
        terminate = relative_diff < self.similarity_threshold
        
        return terminate, current_mask_size, relative_diff
    
    def should_terminate_ssim(self, current_image_path: str, ssim_threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Check if current image is similar enough to target using SSIM to terminate.
        
        Parameters
        ----------
        current_image_path : str
            Path to the current camera image
        ssim_threshold : float
            SSIM threshold for termination (default: 0.85, range: 0-1)
            
        Returns
        -------
        should_terminate : bool
            True if SSIM is high enough to terminate
        ssim_score : float
            SSIM score between 0 and 1
        """
        ssim_score = self.compute_ssim(current_image_path)
        terminate = ssim_score >= ssim_threshold
        return terminate, ssim_score


class TerminationHandler:
    """
    High-level handler for managing segmentation-based termination.
    Provides a clean interface for the main simulation loop.
    
    Supports two termination modes:
    - 'mask_size': Compare segmentation mask sizes between target and current
    - 'ssim': Compare segmented objects using Structural Similarity Index
    
    Both modes use Grounded SAM2 to first segment the target object,
    then apply different comparison metrics.
    """
    
    def __init__(
        self,
        target_image_path: str,
        mode: Literal['mask_size', 'ssim'] = 'mask_size',
        text_prompt: str = "cube.",
        similarity_threshold: float = 0.15,
        ssim_threshold: float = 0.85,
        enabled: bool = True,
        **kwargs
    ):
        """
        Initialize the termination handler.
        
        Parameters
        ----------
        target_image_path : str
            Path to the target image
        mode : {'mask_size', 'ssim'}
            Termination mode:
            - 'mask_size': Compare segmentation mask sizes (default)
            - 'ssim': Compare segmented objects using SSIM
        text_prompt : str
            Object detection prompt (e.g., "cube.", "cat.") - used in both modes
        similarity_threshold : float
            Threshold for mask size similarity (default: 0.15 = 15%) - only used in 'mask_size' mode
        ssim_threshold : float
            Threshold for SSIM on segmented objects (default: 0.85) - only used in 'ssim' mode
        enabled : bool
            Whether termination checking is enabled
        **kwargs
            Additional arguments passed to SegmentationTerminator
        
        Note
        ----
        Both modes require Grounded SAM2 models to segment the target object first.
        The difference is in how the segmented objects are compared:
        - mask_size: Compares number of pixels in masks
        - ssim: Compares structural similarity of segmented object regions
        """
        self.enabled = enabled
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.ssim_threshold = ssim_threshold
        self.terminator = None
        
        if self.enabled:
            print("\n" + "="*60)
            
            if mode == 'mask_size':
                print("Initializing Grounded SAM2 for mask-based termination...")
                print("="*60 + "\n")
                
                self.terminator = SegmentationTerminator(
                    text_prompt=text_prompt,
                    similarity_threshold=similarity_threshold,
                    **kwargs
                )
                
                # Set target mask size
                self.terminator.set_target_mask_size(target_image_path)
                
            elif mode == 'ssim':
                print("Initializing SSIM-based termination on segmented objects...")
                print(f"SSIM threshold: {ssim_threshold}")
                print("Note: SSIM will be computed on segmented object regions, not whole images")
                print("="*60 + "\n")
                
                # SSIM mode also uses SAM2 to segment objects first
                self.terminator = SegmentationTerminator(
                    text_prompt=text_prompt,
                    similarity_threshold=similarity_threshold,
                    **kwargs
                )
                
                # Set target segmented object for SSIM comparison
                self.terminator.set_target_image(target_image_path)
            
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'mask_size' or 'ssim'.")
            
            print("\n" + "="*60 + "\n")
        else:
            print("Termination checking disabled.")
    
    def check_termination(self, current_image_path: str, iteration: int = None) -> bool:
        """
        Check if termination condition is met.
        
        Parameters
        ----------
        current_image_path : str
            Path to current camera image
        iteration : int, optional
            Current iteration number (for logging)
            
        Returns
        -------
        bool
            True if should terminate, False otherwise
        """
        if not self.enabled or self.terminator is None:
            return False
        
        iter_str = f"[Iter {iteration}] " if iteration is not None else ""
        
        if self.mode == 'mask_size':
            should_terminate, current_mask_size, similarity = self.terminator.should_terminate(
                current_image_path
            )
            
            # Print status
            print(f"{iter_str}Mask size: current={current_mask_size:.0f}, "
                  f"target={self.terminator.target_mask_size:.0f}, "
                  f"diff={similarity:.2%}")
            
            if should_terminate:
                self._print_termination_message_mask(
                    similarity=similarity,
                    current_mask_size=current_mask_size,
                    iteration=iteration
                )
        
        elif self.mode == 'ssim':
            should_terminate, ssim_score = self.terminator.should_terminate_ssim(
                current_image_path, 
                ssim_threshold=self.ssim_threshold
            )
            
            # Print status
            print(f"{iter_str}SSIM: {ssim_score:.4f} (threshold: {self.ssim_threshold:.4f})")
            
            if should_terminate:
                self._print_termination_message_ssim(
                    ssim_score=ssim_score,
                    iteration=iteration
                )
        
        return should_terminate
    
    def _print_termination_message_mask(
        self,
        similarity: float,
        current_mask_size: float,
        iteration: int = None
    ):
        """Print formatted termination message for mask size mode."""
        print("\n" + "="*60)
        print("TERMINATION CONDITION MET! (Mask Size)")
        print(f"Mask size similarity achieved: {similarity:.2%} < "
              f"{self.terminator.similarity_threshold:.2%}")
        print(f"Current mask size: {current_mask_size:.0f} pixels")
        print(f"Target mask size: {self.terminator.target_mask_size:.0f} pixels")
        if iteration is not None:
            print(f"Converged in {iteration + 1} iterations")
        print("="*60 + "\n")
    
    def _print_termination_message_ssim(
        self,
        ssim_score: float,
        iteration: int = None
    ):
        """Print formatted termination message for SSIM mode."""
        print("\n" + "="*60)
        print("TERMINATION CONDITION MET! (SSIM)")
        print(f"SSIM threshold achieved: {ssim_score:.4f} >= {self.ssim_threshold:.4f}")
        print(f"Structural similarity: {ssim_score:.2%}")
        if iteration is not None:
            print(f"Converged in {iteration + 1} iterations")
        print("="*60 + "\n")
    
    def get_stats(self) -> dict:
        """
        Get current statistics about the terminator.
        
        Returns
        -------
        dict
            Dictionary with terminator statistics
        """
        if not self.enabled or self.terminator is None:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "mode": self.mode,
        }
        
        if self.mode == 'mask_size':
            stats.update({
                "target_mask_size": self.terminator.target_mask_size,
                "similarity_threshold": self.terminator.similarity_threshold,
                "text_prompt": self.terminator.text_prompt,
            })
        elif self.mode == 'ssim':
            stats.update({
                "ssim_threshold": self.ssim_threshold,
                "target_image_shape": self.terminator.target_image.shape if self.terminator.target_image is not None else None,
            })
        
        return stats
