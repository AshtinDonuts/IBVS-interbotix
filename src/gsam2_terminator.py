"""
Grounded SAM2 Termination Module

This module provides segmentation-based termination condition for IBVS.
Uses Grounded SAM2 to compare object mask sizes between current and target images.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from PIL import Image

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
        
        # Set default paths relative to Grounded-SAM-2 directory
        if sam2_checkpoint is None:
            sam2_checkpoint = str(GSAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt")
        if sam2_model_config is None:
            sam2_model_config = str(GSAM2_DIR / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml")
        
        # Enable autocast for efficiency
        if self.device == "cuda":
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Build SAM2 model
        print(f"Loading SAM2 from {sam2_checkpoint}...")
        sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # Build Grounding DINO model
        print(f"Loading Grounding DINO: {grounding_model}...")
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model
        ).to(self.device)
        
        # Cache for target mask size
        self.target_mask_size = None
        
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


class TerminationHandler:
    """
    High-level handler for managing segmentation-based termination.
    Provides a clean interface for the main simulation loop.
    """
    
    def __init__(
        self,
        target_image_path: str,
        text_prompt: str = "cube.",
        similarity_threshold: float = 0.15,
        enabled: bool = True,
        **kwargs
    ):
        """
        Initialize the termination handler.
        
        Parameters
        ----------
        target_image_path : str
            Path to the target image
        text_prompt : str
            Object detection prompt (e.g., "cube.", "cat.")
        similarity_threshold : float
            Threshold for mask size similarity (default: 0.15 = 15%)
        enabled : bool
            Whether termination checking is enabled
        **kwargs
            Additional arguments passed to SegmentationTerminator
        """
        self.enabled = enabled
        self.terminator = None
        
        if self.enabled:
            print("\n" + "="*60)
            print("Initializing Grounded SAM2 for termination condition...")
            print("="*60 + "\n")
            
            self.terminator = SegmentationTerminator(
                text_prompt=text_prompt,
                similarity_threshold=similarity_threshold,
                **kwargs
            )
            
            # Set target mask size
            self.terminator.set_target_mask_size(target_image_path)
            print("\n" + "="*60 + "\n")
        else:
            print("Segmentation termination disabled.")
    
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
        
        should_terminate, current_mask_size, similarity = self.terminator.should_terminate(
            current_image_path
        )
        
        # Print status
        iter_str = f"[Iter {iteration}] " if iteration is not None else ""
        print(f"{iter_str}Mask size: current={current_mask_size:.0f}, "
              f"target={self.terminator.target_mask_size:.0f}, "
              f"diff={similarity:.2%}")
        
        if should_terminate:
            self._print_termination_message(
                similarity=similarity,
                current_mask_size=current_mask_size,
                iteration=iteration
            )
        
        return should_terminate
    
    def _print_termination_message(
        self,
        similarity: float,
        current_mask_size: float,
        iteration: int = None
    ):
        """Print formatted termination message."""
        print("\n" + "="*60)
        print("TERMINATION CONDITION MET!")
        print(f"Mask size similarity achieved: {similarity:.2%} < "
              f"{self.terminator.similarity_threshold:.2%}")
        print(f"Current mask size: {current_mask_size:.0f} pixels")
        print(f"Target mask size: {self.terminator.target_mask_size:.0f} pixels")
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
        
        return {
            "enabled": True,
            "target_mask_size": self.terminator.target_mask_size,
            "similarity_threshold": self.terminator.similarity_threshold,
            "text_prompt": self.terminator.text_prompt,
        }
