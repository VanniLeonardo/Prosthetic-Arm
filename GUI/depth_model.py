import os
import torch
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image
from typing import Optional, Union, Dict, Any, Tuple, cast
from functools import lru_cache
import logging

# For whoever is reading this, "_func" means the function "func" is internal to the class and not to be used outside of it

# TODO: handle other models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DepthEstimator')


class DepthEstimator:
    # Class-level model cache (implemented since it may be useful for further developments)
    _model_cache = {}

    def __init__(self, model_size: str = 'small', device: Optional[str] = None, name: str = 'DepthAnythingv2'):
        """
        Args:
            model_size: Size of the model to use ('small', 'base', 'large')
            device: Computing device to use ('cuda', 'mps', 'cpu'). If None, selects automatically.
            name: Model architecture name.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 
                                'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                                'cpu')
        
        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            self.pipeline_device = 'cpu' if name == 'DepthAnythingv2' else self.device
        else:
            self.pipeline_device = self.device
            
        logger.info(f'Using device: {self.device} with pipeline device: {self.pipeline_device}')
        
        size_map = {
            'DepthAnythingv2': {
                'small': 'depth-anything/Depth-Anything-V2-Small-hf',
                'base': 'depth-anything/Depth-Anything-V2-Base-hf',
                'large': 'depth-anything/Depth-Anything-V2-Large-hf'
            }
        }
        
        model_size_lower = model_size.lower()
        if name in size_map:
            model_name = size_map[name].get(model_size_lower, size_map[name]['small'])
            if model_size_lower not in size_map[name]:
                logger.warning(f'Unknown model size \'{model_size}\', defaulting to \'small\'')
        else:
            model_name = name
            
        # Load model (with caching)
        self.pipeline = self._get_model(model_name, self.pipeline_device)
        
    @classmethod
    @lru_cache(maxsize=2)
    def _get_model(cls, model_name: str, device: str):
        """
        Cache models to avoid reloading the same model multiple times.
        
        Args:
            model_name: The name/path of the model to load
            device: The device to load the model onto
            
        Returns:
            The loaded pipeline or None if loading failed
        """
        try:
            logger.info(f'Loading model: {model_name} on device {device}')
            return pipeline('depth-estimation', model=model_name, device=device)
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            return None

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            image: Input BGR image as numpy array
            
        Returns:
            Normalized depth map (0-255) or None if estimation failed
        """
        if self.pipeline is None:
            logger.error('Model pipeline not initialized')
            return None
            
        if image is None or image.size == 0:
            logger.error('Invalid input image')
            return None

        with torch.no_grad():
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                depth_map = self.pipeline(pil_image)
                depth = depth_map['depth']
                
                if isinstance(depth, Image.Image):
                    depth = np.array(depth)
                elif isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                
                normalized_depth = np.zeros(depth.shape, dtype=np.uint8)
                cv2.normalize(depth, normalized_depth, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                return normalized_depth
                
            except Exception as e:
                logger.error(f'Error estimating depth: {e}')
                return None
    
    def colorize_depth(self, depth: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> Optional[np.ndarray]:
        """
        Apply a colormap to a depth image.
        
        Args:
            depth: Depth image as a numpy array
            cmap: OpenCV colormap to apply
            
        Returns:
            Colorized depth map or None if input was invalid
        """
        if depth is None or depth.size == 0:
            logger.warning('Cannot colorize: empty or None depth map')
            return None
        return cv2.applyColorMap(depth, cmap)
    
    def depth_at_point(self, depth: np.ndarray, x: int, y: int) -> Optional[float]:
        """
        Get depth value at a specific point.
        
        Args:
            depth: Depth image as a numpy array
            x: X-coordinate of the point
            y: Y-coordinate of the point
            
        Returns:
            Depth value at the specified point or None if coordinates are invalid
        """
        if depth is None or depth.size == 0:
            logger.warning('Invalid depth map provided')
            return None
            
        if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
            logger.warning(f'Coordinates ({x}, {y}) out of bounds for depth map of shape {depth.shape}')
            return None
            
        return float(depth[y, x])
    
    def depth_in_region(self, depth: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Calculate median depth in a rectangular region.
        
        Args:
            depth: Depth image as a numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Median depth value in the region or None if region is invalid
        """
        if depth is None or depth.size == 0:
            logger.warning('Invalid depth map provided')
            return None
        
        x1, y1, x2, y2 = [int(val) for val in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2:
            logger.warning(f'Invalid bounding box: {bbox} '
                           f'(after clipping: {(x1, y1, x2, y2)})')
            return None
            
        region = depth[y1:y2, x1:x2]
        return float(np.median(region)) if region.size > 0 else None
    
    def depth_in_roi(self, depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """
        Calculate median depth in a masked region of interest.
        
        Args:
            depth: Depth image as a numpy array
            mask: Binary mask of the same size as depth image
            
        Returns:
            Median depth value in the masked region or None if inputs are invalid
        """
        if depth is None or mask is None:
            logger.warning('Invalid inputs: depth or mask is None')
            return None
            
        if depth.size == 0 or mask.size == 0:
            logger.warning('Empty depth map or mask')
            return None
            
        if depth.shape[:2] != mask.shape[:2]:
            logger.warning(f'Shape mismatch: depth {depth.shape[:2]} '
                           f'vs mask {mask.shape[:2]}')
            return None
        
        try:
            roi_mask = mask.astype(bool)
            roi_values = depth[roi_mask]
            if roi_values.size == 0:
                logger.warning('No pixels in ROI')
                return None
            return float(np.median(roi_values))
        except Exception as e:
            logger.error(f'Error getting depth in ROI: {e}')
            return None
