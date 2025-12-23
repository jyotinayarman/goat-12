"""
ReconViaGen Manager for improved-miner12
Handles ReconViaGen pipeline for single image 3D generation
"""

from __future__ import annotations

import time
from typing import Optional
from PIL import Image

from config import Settings
from logger_config import logger
from schemas import TrellisResult, TrellisRequest
from libs.reconviagen.pipeline import ReconViaGenSingleImagePipeline


class ReconViaGenManager:
    """Manager for ReconViaGen single image 3D generation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[ReconViaGenSingleImagePipeline] = None
        self.gpu = getattr(settings, 'reconviagen_gpu', 0)
        
    async def startup(self) -> None:
        """Initialize ReconViaGen pipeline"""
        logger.info("Loading ReconViaGen pipeline...")
        
        try:
            model_path = getattr(self.settings, 'reconviagen_model_id', "Stable-X/trellis-image-large")
            self.pipeline = ReconViaGenSingleImagePipeline(model_path)
            
            # Load pipeline on startup for faster inference
            if not self.pipeline.load_pipeline():
                raise RuntimeError("Failed to load ReconViaGen pipeline")
                
            logger.success("ReconViaGen pipeline ready.")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReconViaGen pipeline: {e}")
            self.pipeline = None
    
    async def shutdown(self) -> None:
        """Shutdown ReconViaGen pipeline"""
        if self.pipeline:
            self.pipeline.unload_pipeline()
            self.pipeline = None
        logger.info("ReconViaGen pipeline closed.")
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        return self.pipeline is not None and self.pipeline.is_loaded
    
    def generate(self, request: TrellisRequest) -> TrellisResult:
        """
        Generate 3D model using ReconViaGen
        
        Args:
            request: TrellisRequest with image and parameters
            
        Returns:
            TrellisResult with generated PLY data
        """
        if not self.pipeline:
            raise RuntimeError("ReconViaGen pipeline not loaded.")
        
        logger.info(f"Generating ReconViaGen {request.seed=} and image size {request.image.size}")
        
        # Extract parameters from request
        params = request.params
        ss_guidance_strength = getattr(params, 'sparse_structure_cfg_strength', 7.5)
        ss_sampling_steps = getattr(params, 'sparse_structure_steps', 30)
        slat_guidance_strength = getattr(params, 'slat_cfg_strength', 3.0)
        slat_sampling_steps = getattr(params, 'slat_steps', 12)
        
        start = time.time()
        try:
            ply_data = self.pipeline.generate_3d_from_single_image(
                image=request.image,
                seed=request.seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_sampling_steps=ss_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                slat_sampling_steps=slat_sampling_steps,
                preprocess_image=False  # We handle preprocessing in the main pipeline
            )
            
            if ply_data is None:
                raise RuntimeError("ReconViaGen generation failed")
            
            generation_time = time.time() - start
            
            result = TrellisResult(
                ply_file=ply_data  # bytes
            )
            
            logger.success(f"ReconViaGen finished generation in {generation_time:.2f}s.")
            return result
            
        except Exception as e:
            logger.error(f"ReconViaGen generation failed: {e}")
            raise

