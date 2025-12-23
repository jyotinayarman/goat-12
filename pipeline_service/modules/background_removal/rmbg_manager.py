from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening

from config import Settings
from logger_config import logger


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService with enhanced features.
        """
        self.settings = settings

        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding
        self.mask_threshold = self.settings.background_mask_threshold

        # Enhanced settings for improved preprocessing
        self.use_multi_threshold = True  # Use multiple thresholds for better mask quality
        self.use_adaptive_padding = True  # Use adaptive padding based on object shape
        self.min_mask_threshold = 0.7  # Lower threshold for better edge coverage
        self.max_mask_threshold = 0.9  # Higher threshold for core mask

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # Set model
        self.model: AutoModelForImageSegmentation | None = None

        # Set transform - preserve aspect ratio for better quality
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size, antialias=True), 
                transforms.ToTensor(),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.background_removal_model_id} model...")

        # Load model
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.background_removal_model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.success(f"{self.settings.background_removal_model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.background_removal_model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.background_removal_model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.background_removal_model_id} model not initialized.")

    def _refine_mask(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Refine mask using morphological operations and smoothing.
        
        Args:
            mask: Binary mask array (0-1 range)
            threshold: Threshold for binary operations
            
        Returns:
            Refined mask
        """
        # Convert to binary
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Remove small holes (fill interior holes)
        binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3)))
        
        # Remove small noise (isolated pixels)
        binary_mask = binary_opening(binary_mask, structure=np.ones((3, 3)))
        
        # Smooth edges using Gaussian blur
        smoothed = ndimage.gaussian_filter(binary_mask.astype(np.float32), sigma=1.0)
        
        # Apply edge feathering for smooth transitions
        # Create distance transform for feathering
        distance = ndimage.distance_transform_edt(binary_mask)
        # Feather edges within 3 pixels
        feather_radius = 3
        feathered = np.clip(distance / feather_radius, 0, 1)
        # Combine with smoothed mask
        refined_mask = smoothed * 0.7 + feathered * 0.3
        
        return np.clip(refined_mask, 0, 1)

    def _calculate_adaptive_threshold(self, mask: torch.Tensor) -> float:
        """
        Calculate adaptive threshold based on mask statistics.
        
        Args:
            mask: Raw mask tensor (0-1 range)
            
        Returns:
            Adaptive threshold value
        """
        mask_np = mask.cpu().numpy()
        
        # Calculate Otsu-like threshold
        hist, bins = np.histogram(mask_np.flatten(), bins=256, range=(0, 1))
        hist = hist.astype(np.float32)
        
        # Find threshold that maximizes between-class variance
        total = mask_np.size
        sum_total = np.sum(hist * bins[:-1])
        
        sum_bg = 0
        w_bg = 0
        max_var = 0
        best_threshold = self.mask_threshold
        
        for i in range(256):
            w_bg += hist[i]
            if w_bg == 0:
                continue
            w_fg = total - w_bg
            if w_fg == 0:
                break
            
            sum_bg += hist[i] * bins[i]
            mean_bg = sum_bg / w_bg
            mean_fg = (sum_total - sum_bg) / w_fg
            
            # Between-class variance
            var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
            
            if var_between > max_var:
                max_var = var_between
                best_threshold = bins[i]
        
        # Use adaptive threshold but don't go too low (minimum 0.5)
        adaptive_threshold = max(0.5, min(best_threshold, 0.9))
        
        # Blend with configured threshold for stability
        final_threshold = self.mask_threshold * 0.6 + adaptive_threshold * 0.4
        
        return float(final_threshold)

    def _validate_mask_quality(self, mask: torch.Tensor, threshold: float) -> tuple[bool, dict]:
        """
        Validate mask quality and return metrics.
        
        Args:
            mask: Mask tensor
            threshold: Threshold used
            
        Returns:
            Tuple of (is_valid, metrics_dict)
        """
        mask_np = mask.cpu().numpy()
        binary_mask = (mask_np > threshold).astype(np.uint8)
        
        # Calculate metrics
        foreground_ratio = np.sum(binary_mask) / binary_mask.size
        mask_coverage = float(np.mean(mask_np))
        
        # Check for edge cases
        is_valid = True
        issues = []
        
        # Check if mask is too small (less than 1% of image)
        if foreground_ratio < 0.01:
            is_valid = False
            issues.append("mask_too_small")
        
        # Check if mask is too large (more than 95% of image - probably failed)
        if foreground_ratio > 0.95:
            is_valid = False
            issues.append("mask_too_large")
        
        # Check if mask coverage is reasonable
        if mask_coverage < 0.1:
            is_valid = False
            issues.append("low_mask_coverage")
        
        metrics = {
            "foreground_ratio": float(foreground_ratio),
            "mask_coverage": mask_coverage,
            "is_valid": is_valid,
            "issues": issues,
        }
        
        return is_valid, metrics

    def _calculate_crop_bounds(
        self, 
        mask: torch.Tensor, 
        threshold: float
    ) -> dict:
        """
        Calculate crop bounds with proper dimension handling.
        
        Args:
            mask: Mask tensor (H, W) - already resized to input_image_size
            threshold: Threshold for bbox calculation
            
        Returns:
            Dictionary with crop arguments
        """
        # Get mask dimensions (already resized)
        mask_h, mask_w = mask.shape
        
        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > threshold)
        
        if len(bbox_indices) == 0:
            # No foreground detected, return full image
            return dict(
                top=0,
                left=0,
                height=mask_h,
                width=mask_w
            )
        
        # Extract coordinates - bbox_indices is (N, 2) where [:, 0] is height, [:, 1] is width
        h_coords = bbox_indices[:, 0]  # Height dimension
        w_coords = bbox_indices[:, 1]  # Width dimension
        
        h_min, h_max = torch.aminmax(h_coords)
        w_min, w_max = torch.aminmax(w_coords)
        
        # Calculate dimensions
        height = int(h_max - h_min) + 1
        width = int(w_max - w_min) + 1
        
        # Calculate center
        center_h = (h_max + h_min) / 2.0
        center_w = (w_max + w_min) / 2.0
        
        # Calculate size with padding
        size = max(height, width)
        padded_size = int(size * (1 + self.padding_percentage))
        
        # Calculate crop bounds
        top = int(center_h - padded_size // 2)
        left = int(center_w - padded_size // 2)
        bottom = int(center_h + padded_size // 2)
        right = int(center_w + padded_size // 2)
        
        # Clamp to image bounds if limit_padding is enabled
        if self.limit_padding:
            top = max(0, top)
            left = max(0, left)
            bottom = min(mask_h, bottom)
            right = min(mask_w, right)
        
        # Ensure valid dimensions
        height = max(1, bottom - top)
        width = max(1, right - left)
        
        return dict(
            top=top,
            left=left,
            height=height,
            width=width
        )

    def _remove_background_with_threshold(
        self, 
        image_tensor: torch.Tensor,
        threshold: float
    ) -> tuple[torch.Tensor, dict]:
        """
        Remove background with specific threshold and return result with metrics.
        
        Args:
            image_tensor: Input image tensor (C, H, W) - already resized to input_image_size
            threshold: Mask threshold to use
            
        Returns:
            Tuple of (output_tensor, metrics_dict)
        """
        # Normalize and prepare input
        input_tensor = self.normalize(image_tensor).unsqueeze(0)
        
        with torch.no_grad():
            # Get mask from model
            preds = self.model(input_tensor)[-1].sigmoid()
            mask = preds[0].squeeze()  # (H, W)
        
        # Validate mask quality
        is_valid, quality_metrics = self._validate_mask_quality(mask, threshold)
        quality_metrics["threshold_used"] = threshold
        
        # Refine mask (always refine for better quality, even if invalid)
        mask_np = mask.cpu().numpy()
        if is_valid:
            refined_mask_np = self._refine_mask(mask_np, threshold)
        else:
            # If invalid, try with lower threshold and refine
            lower_threshold = max(0.3, threshold - 0.2)
            binary_mask = (mask_np > lower_threshold).astype(np.float32)
            # Refine the binary mask too
            refined_mask_np = self._refine_mask(binary_mask, 0.5)
            quality_metrics["fallback_threshold"] = lower_threshold
        
        mask = torch.from_numpy(refined_mask_np).to(mask.device).float()
        
        # Calculate crop bounds (use mask dimensions, not original size)
        crop_args = self._calculate_crop_bounds(mask, threshold)
        
        # Apply mask to image
        mask_3d = mask.unsqueeze(0)  # (1, H, W) for broadcasting
        masked_image = image_tensor * mask_3d  # (C, H, W)
        
        # Create RGBA tensor
        tensor_rgba = torch.cat([
            masked_image,  # RGB channels
            mask  # Alpha channel
        ], dim=0)  # (4, H, W)
        
        # Crop and resize
        # Note: resized_crop expects (C, H, W) format
        output = resized_crop(
            tensor_rgba,
            **crop_args,
            size=self.output_size,
            antialias=True  # Use antialiasing for better quality
        )
        
        return output, quality_metrics

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove background from image with retry logic and quality validation.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Image with background removed (RGBA)
        """
        try:
            t1 = time.time()
            
            # Check if image already has alpha channel
            if image.mode == "RGBA":
                alpha = np.array(image)[:, :, 3]
                if not np.all(alpha == 255):
                    # Already has transparency, return as-is
                    logger.info("Image already has alpha channel, skipping background removal")
                    return image
            
            # Convert to RGB
            rgb_image = image.convert('RGB')
            
            # Transform to tensor (already resized to input_image_size)
            rgb_tensor = self.transforms(rgb_image).to(self.device)
            
            # Get initial mask to calculate adaptive threshold
            input_tensor = self.normalize(rgb_tensor).unsqueeze(0)
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid()
                initial_mask = preds[0].squeeze()
            
            # Calculate adaptive threshold
            adaptive_threshold = self._calculate_adaptive_threshold(initial_mask)
            
            # Try multiple thresholds for best result
            thresholds_to_try = [
                self.mask_threshold,  # Primary threshold (configured)
                adaptive_threshold,  # Adaptive threshold (calculated)
                max(0.5, self.mask_threshold - 0.1),  # Slightly lower
                min(0.9, self.mask_threshold + 0.1),  # Slightly higher
            ]
            # Remove duplicates while preserving order
            thresholds_to_try = list(dict.fromkeys(thresholds_to_try))
            
            best_result = None
            best_quality_score = -1.0
            best_metrics = None
            
            for threshold in thresholds_to_try:
                try:
                    output_tensor, metrics = self._remove_background_with_threshold(
                        rgb_tensor,
                        threshold
                    )
                    
                    # Calculate quality score
                    foreground_ratio = metrics.get("foreground_ratio", 0)
                    mask_coverage = metrics.get("mask_coverage", 0)
                    is_valid = metrics.get("is_valid", False)
                    
                    # Quality score: prefer valid masks with reasonable coverage
                    if is_valid:
                        # Good coverage (10-90% of image)
                        coverage_score = 1.0 - abs(foreground_ratio - 0.5) * 2
                        coverage_score = max(0, coverage_score)
                        
                        # Mask quality (higher coverage is better, but not too high)
                        quality_score = coverage_score * 0.6 + mask_coverage * 0.4
                    else:
                        # Invalid mask gets low score
                        quality_score = foreground_ratio * 0.3
                    
                    logger.debug(
                        f"Background removal attempt (threshold={threshold:.2f}): "
                        f"quality_score={quality_score:.3f}, "
                        f"foreground_ratio={foreground_ratio:.3f}, "
                        f"valid={is_valid}, issues={metrics.get('issues', [])}"
                    )
                    
                    # Track best result
                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        best_result = output_tensor
                        best_metrics = metrics
                        
                        # Early exit if we get a very good result
                        if is_valid and quality_score > 0.8:
                            logger.info(f"Found high-quality mask with threshold {threshold:.2f}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Error with threshold {threshold:.2f}: {e}")
                    continue
            
            # Convert best result to PIL Image
            if best_result is not None:
                # Ensure we have all 4 channels (RGBA)
                if best_result.shape[0] == 4:
                    # Convert tensor to numpy for proper RGBA handling
                    # Clamp values to [0, 1] range
                    result_np = best_result.cpu().clamp(0, 1).numpy()
                    
                    # Convert to uint8 (0-255)
                    result_np = (result_np * 255).astype(np.uint8)
                    
                    # Transpose from (C, H, W) to (H, W, C)
                    result_np = np.transpose(result_np, (1, 2, 0))
                    
                    # Create PIL Image from array
                    image_without_background = Image.fromarray(result_np, mode='RGBA')
                else:
                    # Fallback: create RGBA from RGB
                    rgb_result = to_pil_image(best_result[:3])
                    # Create solid alpha channel
                    alpha = Image.new('L', rgb_result.size, 255)
                    image_without_background = Image.merge('RGBA', (*rgb_result.split(), alpha))
            else:
                # Fallback: return original image with full alpha
                logger.warning("All background removal attempts failed, returning original image")
                image_without_background = image.convert('RGBA')
            
            removal_time = time.time() - t1
            logger.success(
                f"Background removal completed - Time: {removal_time:.2f}s, "
                f"OutputSize: {image_without_background.size}, "
                f"InputSize: {image.size}, "
                f"Quality: {best_quality_score:.3f if best_metrics else 'N/A'}, "
                f"ForegroundRatio: {best_metrics.get('foreground_ratio', 0):.3f if best_metrics else 'N/A'}"
            )
            
            return image_without_background
            
        except Exception as e:
            logger.error(f"Error removing background: {e}", exc_info=True)
            # Return original image as RGBA on error
            try:
                return image.convert('RGBA')
            except Exception:
                return image
