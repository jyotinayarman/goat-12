"""
ReconViaGen Pipeline Integration for Single Image Input
Simplified version of ReconViaGen that works with single images
"""

from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os
import sys

# Add the ReconViaGen path
reconviagen_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "generation", "ReconViaGen")
if reconviagen_path not in sys.path:
    sys.path.append(reconviagen_path)

try:
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
    from trellis.representations import Gaussian
    RECONVIAGEN_AVAILABLE = True
except ImportError as e:
    print(f"ReconViaGen not available: {e}")
    RECONVIAGEN_AVAILABLE = False
    TrellisImageTo3DPipeline = None
    Gaussian = None


class ReconViaGenSingleImagePipeline:
    """
    ReconViaGen pipeline wrapper for single image input
    """
    
    def __init__(self, model_path: str = "Stable-X/trellis-image-large"):
        self.model_path = model_path
        self.pipeline = None
        self.is_loaded = False
        
    def load_pipeline(self) -> bool:
        """Load the ReconViaGen pipeline"""
        if not RECONVIAGEN_AVAILABLE:
            print("ReconViaGen dependencies not available")
            return False
            
        try:
            print(f"Loading ReconViaGen pipeline from {self.model_path}...")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_path)
            self.pipeline.cuda()
            self.is_loaded = True
            print("ReconViaGen pipeline loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load ReconViaGen pipeline: {e}")
            return False
    
    def unload_pipeline(self):
        """Unload the pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            torch.cuda.empty_cache()
            print("ReconViaGen pipeline unloaded")
    
    def generate_3d_from_single_image(
        self,
        image: Image.Image,
        seed: int = 42,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 30,
        slat_guidance_strength: float = 3.0,
        slat_sampling_steps: int = 12,
        preprocess_image: bool = True
    ) -> Optional[bytes]:
        """
        Generate 3D model from single image using ReconViaGen
        
        Args:
            image: Input PIL Image
            seed: Random seed
            ss_guidance_strength: Sparse structure guidance strength
            ss_sampling_steps: Sparse structure sampling steps
            slat_guidance_strength: SLat guidance strength
            slat_sampling_steps: SLat sampling steps
            preprocess_image: Whether to preprocess the image
            
        Returns:
            PLY file as bytes or None if failed
        """
        if not self.is_loaded:
            if not self.load_pipeline():
                raise RuntimeError("Failed to load ReconViaGen pipeline")
        
        try:
            print(f"Generating 3D model from single image using ReconViaGen...")
            
            # Generate 3D model using ReconViaGen single image pipeline
            outputs = self.pipeline.run(
                image=image,
                seed=seed,
                formats=["gaussian"],
                preprocess_image=preprocess_image,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )
            
            # Extract Gaussian
            gs = outputs['gaussian'][0]
            
            # Generate PLY data
            ply_buffer = io.BytesIO()
            gs.save_ply(ply_buffer)
            ply_buffer.seek(0)
            ply_data = ply_buffer.getvalue()
            
            print(f"Generated 3D model: {len(ply_data)} bytes PLY")
            return ply_data
            
        except Exception as e:
            print(f"Error generating 3D model with ReconViaGen: {e}")
            return None
    
    @property
    def device(self):
        """Get the device of the pipeline"""
        if self.pipeline is not None:
            return next(self.pipeline.parameters()).device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

