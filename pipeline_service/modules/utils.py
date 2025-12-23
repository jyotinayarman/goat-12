from PIL import Image
from PIL import ImageStat

import io
import base64
from datetime import datetime
from typing import Optional, Tuple
import os
import random
import numpy as np
import torch

from logger_config import logger
from schemas.trellis_schemas import TrellisResult

from config import settings

def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)

def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_image(prompt: str) -> Image.Image:
    """
    Decode the image from the base64 string.

    Args:
        prompt: The base64 string of the image.

    Returns:
        The image.
    """
    # Decode the image from the base64 string
    image_bytes = base64.b64decode(prompt)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def to_png_base64(image: Image.Image) -> str:
    """
    Convert the image to PNG format and encode to base64.

    Args:
        image: The image to convert.

    Returns:
        Base64 encoded PNG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Convert to base64 from bytes to string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")

def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")

def save_files(
    trellis_result: Optional[TrellisResult], 
    image_edited: Image.Image, 
    image_without_background: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        trellis_result: The Trellis result to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
    """
    # Save the Trellis result if available
    if trellis_result:
        if trellis_result.ply_file:
            save_file_bytes(trellis_result.ply_file, "ply", "mesh", suffix=".ply")

    # Save the images using PIL Image.save()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)


def _calculate_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate a simplified SSIM-like metric between two images.
    Uses mean, variance, and covariance to estimate structural similarity.
    
    Args:
        img1: First image array (normalized 0-1)
        img2: Second image array (normalized 0-1)
        
    Returns:
        SSIM-like score between 0 and 1 (higher is better)
    """
    try:
        # Ensure same size
        if img1.shape != img2.shape:
            # Resize img2 to match img1
            img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
            target_size = (img1.shape[1], img1.shape[0])  # PIL uses (width, height)
            img2_pil = img2_pil.resize(target_size, Image.Resampling.LANCZOS)
            img2 = np.array(img2_pil).astype(np.float32) / 255.0
        
        # Constants for SSIM calculation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Calculate means
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # Calculate variances and covariance
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim = numerator / (denominator + 1e-10)
        return float(np.clip(ssim, 0.0, 1.0))
    except Exception:
        return 0.5  # Return neutral score on error


def _calculate_color_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate color histogram similarity between two images.
    
    Args:
        img1: First image array (0-255)
        img2: Second image array (0-255)
        
    Returns:
        Color similarity score between 0 and 1 (higher is better)
    """
    try:
        # Calculate histograms for each channel
        hist1 = [np.histogram(img1[:, :, i], bins=32, range=(0, 256))[0] for i in range(3)]
        hist2 = [np.histogram(img2[:, :, i], bins=32, range=(0, 256))[0] for i in range(3)]
        
        # Normalize histograms
        hist1 = [h / (np.sum(h) + 1e-10) for h in hist1]
        hist2 = [h / (np.sum(h) + 1e-10) for h in hist2]
        
        # Calculate correlation coefficient for each channel
        similarities = []
        for h1, h2 in zip(hist1, hist2):
            # Use correlation coefficient
            corr = np.corrcoef(h1, h2)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            similarities.append((corr + 1.0) / 2.0)  # Normalize to 0-1
        
        return float(np.mean(similarities))
    except Exception:
        return 0.5  # Return neutral score on error


def validate_image_quality(image: Image.Image, reference_image: Optional[Image.Image] = None) -> Tuple[bool, dict]:
    """
    Validate image quality by checking for noise, artifacts, and other issues.
    Now includes reference image comparison for better quality assessment.
    
    Args:
        image: The image to validate
        reference_image: Optional reference image to compare against
        
    Returns:
        Tuple of (is_valid, quality_metrics) where:
        - is_valid: True if image passes quality checks
        - quality_metrics: Dictionary with quality metrics
    """
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image.convert("RGB"))
        
        # Calculate basic statistics
        stats = ImageStat.Stat(image)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        std_dev = sum(stats.stddev) / len(stats.stddev)
        
        # Calculate variance (high variance can indicate noise)
        variance = np.var(img_array)
        
        # Calculate Laplacian variance (detects blur/noise) - improved calculation
        # Convert to grayscale for Laplacian
        gray = np.mean(img_array, axis=2).astype(np.float32)
        
        # Improved Laplacian variance calculation using proper kernel
        if gray.size > 1 and gray.shape[0] > 2 and gray.shape[1] > 2:
            try:
                # Calculate second derivatives (Laplacian approximation)
                # Only use n=2 if image is large enough
                if gray.shape[0] > 2 and gray.shape[1] > 2:
                    h_diff = np.diff(gray, axis=0, n=2)
                    w_diff = np.diff(gray, axis=1, n=2)
                    if h_diff.size > 0 and w_diff.size > 0:
                        laplacian_var = float(np.var(h_diff) + np.var(w_diff))
                    else:
                        # Fallback to simple difference variance
                        h_diff = np.diff(gray, axis=0)
                        w_diff = np.diff(gray, axis=1)
                        laplacian_var = float(np.var(h_diff) + np.var(w_diff))
                else:
                    # Fallback for small images
                    h_diff = np.diff(gray, axis=0)
                    w_diff = np.diff(gray, axis=1)
                    laplacian_var = float(np.var(h_diff) + np.var(w_diff))
            except Exception:
                # Fallback to simple difference variance on any error
                h_diff = np.diff(gray, axis=0)
                w_diff = np.diff(gray, axis=1)
                laplacian_var = float(np.var(h_diff) + np.var(w_diff))
        else:
            laplacian_var = 0.0
        
        # Check for extreme values (potential artifacts)
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        extreme_pixels = np.sum((img_array < 5) | (img_array > 250))
        extreme_ratio = extreme_pixels / img_array.size
        
        # Check for uniform regions (potential corruption)
        # Calculate local variance in small patches
        patch_size = 8
        h, w = gray.shape
        local_vars = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                local_vars.append(np.var(patch))
        
        avg_local_var = np.mean(local_vars) if local_vars else 0
        
        # Calculate edge sharpness (using gradient magnitude)
        if gray.size > 1:
            grad_y = np.gradient(gray, axis=0)
            grad_x = np.gradient(gray, axis=1)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_sharpness = float(np.mean(gradient_magnitude))
        else:
            edge_sharpness = 0.0
        
        # Reference image comparison metrics
        ssim_score = None
        color_similarity = None
        if reference_image is not None:
            try:
                ref_array = np.array(reference_image.convert("RGB"))
                # Normalize to 0-1 for SSIM
                img_norm = img_array.astype(np.float32) / 255.0
                ref_norm = ref_array.astype(np.float32) / 255.0
                # Convert to grayscale for SSIM
                img_gray = np.mean(img_norm, axis=2)
                ref_gray = np.mean(ref_norm, axis=2)
                ssim_score = _calculate_ssim_simple(img_gray, ref_gray)
                color_similarity = _calculate_color_similarity(img_array, ref_array)
            except Exception as e:
                logger.debug(f"Error calculating reference comparison: {e}")
        
        # Adaptive thresholds based on image characteristics
        # For high-resolution images, variance threshold should be higher
        image_size = img_array.size
        adaptive_variance_threshold = 10000 * (1 + image_size / (512 * 512))  # Scale with image size
        adaptive_local_var_threshold = 10 * (1 + image_size / (1024 * 1024))  # Scale with image size
        
        # Quality metrics
        quality_metrics = {
            "mean_brightness": mean_brightness,
            "std_dev": std_dev,
            "variance": float(variance),
            "laplacian_variance": float(laplacian_var),
            "extreme_pixel_ratio": extreme_ratio,
            "avg_local_variance": float(avg_local_var),
            "edge_sharpness": edge_sharpness,
            "ssim_score": ssim_score,
            "color_similarity": color_similarity,
        }
        
        # Validation criteria with adaptive thresholds
        is_valid = True
        issues = []
        
        # Check for excessive noise (adaptive threshold)
        if variance > adaptive_variance_threshold:
            is_valid = False
            issues.append("excessive_variance")
        
        # Check for too many extreme pixels (potential artifacts)
        if extreme_ratio > 0.1:  # More than 10% extreme pixels
            is_valid = False
            issues.append("excessive_extreme_pixels")
        
        # Check for too uniform (adaptive threshold)
        if avg_local_var < adaptive_local_var_threshold:
            is_valid = False
            issues.append("too_uniform")
        
        # Check for reasonable brightness range
        if mean_brightness < 10 or mean_brightness > 245:
            is_valid = False
            issues.append("extreme_brightness")
        
        # Check for blur (low Laplacian variance relative to image size)
        laplacian_threshold = 50 * (image_size / (512 * 512))
        if laplacian_var < laplacian_threshold and image_size > 256 * 256:
            is_valid = False
            issues.append("too_blurry")
        
        # Check edge sharpness
        if edge_sharpness < 5.0 and image_size > 256 * 256:
            is_valid = False
            issues.append("low_edge_sharpness")
        
        quality_metrics["is_valid"] = is_valid
        quality_metrics["issues"] = issues
        quality_metrics["adaptive_variance_threshold"] = adaptive_variance_threshold
        quality_metrics["adaptive_local_var_threshold"] = adaptive_local_var_threshold
        
        return is_valid, quality_metrics
        
    except Exception as e:
        logger.warning(f"Error validating image quality: {e}")
        # On error, assume valid but log the issue
        return True, {"error": str(e), "is_valid": True}


def preprocess_input_image(image: Image.Image) -> Image.Image:
    """
    Preprocess input image to improve editing quality.
    
    Args:
        image: Input image to preprocess
        
    Returns:
        Preprocessed image
    """
    try:
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get image statistics
        stats = ImageStat.Stat(image)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        
        # Normalize brightness if too dark or too bright
        if mean_brightness < 30:
            # Too dark - apply slight brightening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            logger.debug("Applied brightness enhancement to dark image")
        elif mean_brightness > 225:
            # Too bright - apply slight darkening
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.9)
            logger.debug("Applied brightness reduction to bright image")
        
        return image
        
    except Exception as e:
        logger.warning(f"Error preprocessing image: {e}, returning original")
        return image.convert("RGB") if image.mode != "RGB" else image

