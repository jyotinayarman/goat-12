from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.gs_generator.reconviagen_manager import ReconViaGenManager
from modules.utils import (
    secure_randint, 
    set_random_seed, 
    decode_image, 
    to_png_base64, 
    save_files,
    validate_image_quality,
    preprocess_input_image
)


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)
        self.reconviagen = ReconViaGenManager(settings)
        
        # Image enhancement settings
        self.enable_image_enhancement = True
        self.enhancement_sharpening_factor = 1.2
        self.enhancement_contrast_factor = 1.1

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        await self.reconviagen.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()
        await self.reconviagen.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def _edit_image_with_retry(
        self, 
        image: Image.Image, 
        base_seed: int,
        max_retries: int = 3,
        quality_threshold: float = 0.7
    ) -> Image.Image:
        """
        Edit image with retry logic to ensure quality.
        Enhanced with improved quality scoring and reference image comparison.
        
        Args:
            image: Input image to edit
            base_seed: Base seed for generation
            max_retries: Maximum number of retry attempts
            quality_threshold: Minimum quality score threshold (0-1)
            
        Returns:
            Edited image that passes quality checks
        """
        best_image = None
        best_quality_score = 0.0
        best_quality_metrics = None
        last_exception = None
        
        # Improved seed variation strategy: use different prime numbers for better variation
        # Each attempt uses a different increment to maximize diversity
        seed_increments = [17, 31, 47, 61, 79]  # Prime numbers for better variation
        
        for attempt in range(max_retries):
            try:
                # Use improved seed variation strategy
                # Each attempt uses a different prime increment for maximum variation
                seed_increment = seed_increments[attempt % len(seed_increments)]
                retry_seed = base_seed + seed_increment * (attempt + 1)
                set_random_seed(retry_seed)
                
                logger.info(f"Editing image (attempt {attempt + 1}/{max_retries}, seed: {retry_seed})")
                
                # Edit the image
                image_edited = self.qwen_edit.edit_image(prompt_image=image, seed=retry_seed)
                
                # Validate quality with reference image comparison
                is_valid, quality_metrics = validate_image_quality(image_edited, reference_image=image)
                
                # Enhanced quality score calculation (higher is better)
                # Uses adaptive thresholds and includes reference comparison
                
                # 1. Variance score (adaptive threshold)
                adaptive_variance_threshold = quality_metrics.get("adaptive_variance_threshold", 10000)
                variance = quality_metrics.get("variance", 0)
                variance_score = 1.0 - min(variance / max(adaptive_variance_threshold, 1), 1.0)
                
                # 2. Extreme pixel score
                extreme_ratio = quality_metrics.get("extreme_pixel_ratio", 0)
                extreme_pixel_score = 1.0 - min(extreme_ratio * 10, 1.0)
                
                # 3. Local variance score (adaptive threshold)
                adaptive_local_var_threshold = quality_metrics.get("adaptive_local_var_threshold", 10)
                avg_local_var = quality_metrics.get("avg_local_variance", 0)
                local_var_score = min(avg_local_var / max(adaptive_local_var_threshold, 1), 1.0)
                
                # 4. Laplacian variance score (sharpness indicator)
                laplacian_var = quality_metrics.get("laplacian_variance", 0)
                # Normalize based on typical good values (50-500 range)
                laplacian_score = min(laplacian_var / 500.0, 1.0) if laplacian_var > 0 else 0.0
                
                # 5. Edge sharpness score
                edge_sharpness = quality_metrics.get("edge_sharpness", 0)
                edge_score = min(edge_sharpness / 20.0, 1.0)  # Normalize to 0-1
                
                # 6. Reference image comparison scores (if available)
                ssim_score = quality_metrics.get("ssim_score")
                color_similarity = quality_metrics.get("color_similarity")
                
                # Calculate composite quality score with adaptive weights
                # Base scores (always used) - weights sum to 1.0
                base_score_components = [
                    (variance_score, 0.25),      # Reduced from 0.4
                    (extreme_pixel_score, 0.20), # Reduced from 0.3
                    (local_var_score, 0.20),     # Reduced from 0.3
                    (laplacian_score, 0.20),     # NEW: sharpness indicator (increased from 0.15)
                    (edge_score, 0.15),          # NEW: edge quality (increased from 0.10)
                ]
                # Verify weights sum to 1.0
                base_weight_sum = sum(w for _, w in base_score_components)
                
                # Reference comparison scores (if available, boost their weight)
                if ssim_score is not None and color_similarity is not None:
                    # When reference is available, give it significant weight (30%)
                    reference_weight = 0.3
                    # Normalize base weights to make room for reference (70% total)
                    normalized_base_weights = [(score, w * (1 - reference_weight) / base_weight_sum) 
                                             for score, w in base_score_components]
                    
                    # Combine reference scores (SSIM weighted more than color)
                    reference_score = (ssim_score * 0.6 + color_similarity * 0.4)
                    
                    quality_score = (
                        sum(score * weight for score, weight in normalized_base_weights) +
                        reference_score * reference_weight
                    )
                else:
                    # No reference image, normalize base weights to sum to 1.0
                    normalized_base_weights = [(score, w / base_weight_sum) 
                                             for score, w in base_score_components]
                    quality_score = sum(score * weight for score, weight in normalized_base_weights)
                
                # Penalty for validation issues
                issues = quality_metrics.get("issues", [])
                issue_penalty = len(issues) * 0.1
                quality_score = max(0.0, quality_score - issue_penalty)
                
                logger.info(
                    f"Image quality check (attempt {attempt + 1}): "
                    f"valid={is_valid}, score={quality_score:.3f}, "
                    f"variance={variance:.1f}, laplacian={laplacian_var:.1f}, "
                    f"edge_sharpness={edge_sharpness:.2f}, "
                    f"ssim={ssim_score:.3f if ssim_score is not None else 'N/A'}, "
                    f"color_sim={color_similarity:.3f if color_similarity is not None else 'N/A'}, "
                    f"issues={issues}"
                )
                
                # Track best result
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_image = image_edited
                    best_quality_metrics = quality_metrics.copy()
                
                # Early exit strategy: return if quality is good enough
                # But also consider continuing if we're early in retries and score is close to threshold
                if is_valid and quality_score >= quality_threshold:
                    # If score is significantly above threshold, return immediately
                    if quality_score >= quality_threshold + 0.15:
                        logger.success(
                            f"Image editing successful on attempt {attempt + 1} "
                            f"with quality score {quality_score:.3f} (well above threshold)"
                        )
                        return image_edited
                    # If score is just above threshold and we have more retries, continue to find better
                    elif attempt < max_retries - 1 and quality_score < quality_threshold + 0.1:
                        logger.info(
                            f"Quality acceptable (score={quality_score:.3f}), "
                            f"but continuing to search for better result..."
                        )
                        continue
                    else:
                        logger.success(
                            f"Image editing successful on attempt {attempt + 1} "
                            f"with quality score {quality_score:.3f}"
                        )
                        return image_edited
                
                # If this is the last attempt, return best result
                if attempt == max_retries - 1:
                    if best_image is not None:
                        logger.warning(
                            f"Using best result after {max_retries} attempts "
                            f"(quality score: {best_quality_score:.3f}, "
                            f"issues: {best_quality_metrics.get('issues', []) if best_quality_metrics else []})"
                        )
                        return best_image
                    else:
                        # Fallback: return the last attempt even if quality is poor
                        logger.warning(
                            f"Returning last attempt result despite quality issues "
                            f"(score: {quality_score:.3f}, issues: {issues})"
                        )
                        return image_edited
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Error during image editing (attempt {attempt + 1}/{max_retries}): {e}")
                
                # If this is the last attempt, raise the exception
                if attempt == max_retries - 1:
                    if best_image is not None:
                        logger.warning("Returning best result from previous attempts despite error")
                        return best_image
                    raise e
                
                # Clean GPU memory before retry
                self._clean_gpu_memory()
        
        # Should not reach here, but return best image if available
        if best_image is not None:
            return best_image
        
        # Final fallback: raise last exception or return original
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Failed to edit image after all retry attempts")

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(64,64),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes,seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            
        Returns:
            PLY file as bytes
        """
        # Validate input image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Reopen after verify
        except Exception as e:
            logger.error(f"Invalid image format: {e}")
            raise ValueError(f"Invalid image format: {e}")
        
        # Check minimum image size
        min_size = 256
        if image.width < min_size or image.height < min_size:
            logger.warning(f"Image size ({image.width}x{image.height}) is below recommended minimum ({min_size}x{min_size})")
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerateRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )
        
        # Generate
        response = await self.generate_gs(request)
        
        # Return binary PLY - ensure it's bytes
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")
        
        # Handle both bytes and base64 string cases
        if isinstance(response.ply_file_base64, bytes):
            return response.ply_file_base64
        elif isinstance(response.ply_file_base64, str):
            # If it's a base64 string, decode it
            return base64.b64decode(response.ply_file_base64)
        else:
            raise ValueError(f"Unexpected PLY file type: {type(response.ply_file_base64)}")

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)
        
        # Validate input image quality
        if image.width < 64 or image.height < 64:
            raise ValueError(f"Image too small: {image.width}x{image.height}. Minimum size is 64x64")
        if image.width > 4096 or image.height > 4096:
            logger.warning(f"Image very large: {image.width}x{image.height}. This may cause memory issues.")

        # 0. Enhance image quality before editing (if enabled)
        if self.enable_image_enhancement:
            image = self._enhance_image(image)
        
        # Preprocess input image for better editing quality (if enabled)
        if hasattr(self.settings, 'enable_image_preprocessing') and self.settings.enable_image_preprocessing:
            image = preprocess_input_image(image)

        # 1. Edit the image using Qwen Edit
        image_edited = self.qwen_edit.edit_image(prompt_image=image, seed=request.seed)
        
        # Validate edited image
        if not image_edited or image_edited.size[0] == 0 or image_edited.size[1] == 0:
            raise ValueError("Image editing failed: invalid output image")

        # 2. Remove background
        image_without_background = self.rmbg.remove_background(image_edited)
        
        # Validate background-removed image
        if not image_without_background or image_without_background.size[0] == 0 or image_without_background.size[1] == 0:
            logger.warning("Background removal produced invalid image, using edited image instead")
            image_without_background = image_edited

        trellis_result: Optional[TrellisResult] = None
        
        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
       
        # 3. Generate the 3D model
        # Ensure image is in RGB format and has valid dimensions
        if image_without_background.mode != "RGB":
            image_without_background = image_without_background.convert("RGB")
        
        # Validate image before 3D generation
        min_3d_size = 256
        if image_without_background.width < min_3d_size or image_without_background.height < min_3d_size:
            logger.warning(f"Image size ({image_without_background.width}x{image_without_background.height}) is below recommended minimum for 3D generation ({min_3d_size}x{min_3d_size})")
        
        # Choose between Trellis and ReconViaGen based on settings
        use_reconviagen = getattr(self.settings, 'use_reconviagen', False)
        
        if use_reconviagen and self.reconviagen.is_ready():
            logger.info("Using ReconViaGen for 3D generation")
            trellis_result = self.reconviagen.generate(
                TrellisRequest(
                    image=image_without_background,
                    seed=request.seed,
                    params=trellis_params
                )
            )
        else:
            logger.info("Using standard Trellis for 3D generation")
            trellis_result = self.trellis.generate(
                TrellisRequest(
                    image=image_without_background,
                    seed=request.seed,
                    params=trellis_params
                )
            )
        
        # Validate 3D generation result
        if not trellis_result or not trellis_result.ply_file:
            raise ValueError("3D model generation failed: no PLY file produced")

        # Save generated files
        if self.settings.save_generated_files:
            save_files(trellis_result, image_edited, image_without_background)
        
        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64 if self.settings.send_generated_files else None,
            image_without_background_file_base64=image_without_background_base64 if self.settings.send_generated_files else None,
        )
        return response

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply image enhancement operations to improve quality before Qwen Edit.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        try:
            enhanced = image.copy()
            
            # Convert to RGB if needed
            if enhanced.mode != 'RGB':
                enhanced = enhanced.convert('RGB')
            
            # 1. Exposure correction (auto-adjust brightness)
            enhanced = self._correct_exposure(enhanced)
            
            # 2. Contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.enhancement_contrast_factor)
            
            # 3. Sharpening (should be last)
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(
                radius=1,
                percent=int((self.enhancement_sharpening_factor - 1.0) * 100),
                threshold=3
            ))
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error enhancing image, using original: {e}")
            return image

    def _correct_exposure(self, image: Image.Image) -> Image.Image:
        """
        Auto-correct exposure (brightness) if image is too dark or too bright.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Exposure-corrected PIL Image
        """
        try:
            # Convert to numpy for processing
            img_array = np.array(image, dtype=np.float32)
            
            # Calculate mean brightness
            mean_brightness = np.mean(img_array)
            
            # Target brightness (middle gray)
            target_brightness = 128.0
            
            # Adjust if too dark or too bright
            if mean_brightness < 100:  # Too dark
                brightness_factor = target_brightness / mean_brightness
                img_array = np.clip(img_array * brightness_factor, 0, 255)
            elif mean_brightness > 180:  # Too bright
                brightness_factor = target_brightness / mean_brightness
                img_array = np.clip(img_array * brightness_factor, 0, 255)
            
            return Image.fromarray(img_array.astype(np.uint8))
        except Exception as e:
            logger.warning(f"Error correcting exposure: {e}")
            return image

