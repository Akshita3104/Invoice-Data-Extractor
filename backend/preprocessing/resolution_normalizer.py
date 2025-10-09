"""
Resolution Normalizer
Normalizes image resolution to target DPI using intelligent upscaling/downscaling
Uses super-resolution techniques for upscaling when needed
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ResolutionNormalizer:
    """
    Normalizes document resolution to target DPI
    """
    
    def __init__(self, target_dpi: int = 300):
        """
        Initialize resolution normalizer
        
        Args:
            target_dpi: Target DPI for output images
        """
        self.target_dpi = target_dpi
        self.a4_width_inches = 8.27
        self.a4_height_inches = 11.69
        self.min_dpi = 150
        self.max_dpi = 600
    
    def normalize(
        self,
        image: np.ndarray,
        target_dpi: Optional[int] = None,
        method: str = 'auto'
    ) -> np.ndarray:
        """
        Normalize image resolution
        
        Args:
            image: Input image
            target_dpi: Target DPI (uses instance default if None)
            method: Resizing method ('auto', 'bicubic', 'lanczos', 'super_resolution')
            
        Returns:
            Normalized image
        """
        if target_dpi is None:
            target_dpi = self.target_dpi
        
        # Estimate current DPI
        current_dpi = self.estimate_dpi(image)
        
        print(f"Current DPI: ~{current_dpi}, Target DPI: {target_dpi}")
        
        # Check if normalization is needed
        if abs(current_dpi - target_dpi) < 20:
            print("DPI is already close to target, skipping normalization")
            return image
        
        # Calculate scale factor
        scale_factor = target_dpi / current_dpi
        
        # Choose method based on scale factor
        if method == 'auto':
            if scale_factor > 1.5:
                # Significant upscaling - use advanced method
                method = 'lanczos'
            elif scale_factor > 1.0:
                # Moderate upscaling
                method = 'cubic'
            else:
                # Downscaling
                method = 'area'
        
        # Perform resizing
        if method == 'super_resolution':
            normalized = self._super_resolution_upscale(image, scale_factor)
        else:
            normalized = self._standard_resize(image, scale_factor, method)
        
        return normalized
    
    def estimate_dpi(self, image: np.ndarray) -> int:
        """
        Estimate DPI of image (assumes A4 page)
        
        Args:
            image: Input image
            
        Returns:
            Estimated DPI
        """
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Estimate based on A4 dimensions
        estimated_dpi = int(np.sqrt(total_pixels / (self.a4_width_inches * self.a4_height_inches)))
        
        # Clamp to reasonable range
        estimated_dpi = max(self.min_dpi, min(estimated_dpi, self.max_dpi))
        
        return estimated_dpi
    
    def _standard_resize(
        self,
        image: np.ndarray,
        scale_factor: float,
        method: str
    ) -> np.ndarray:
        """
        Standard resize using OpenCV interpolation
        """
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Choose interpolation method
        if method == 'cubic' or method == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        elif method == 'lanczos':
            interpolation = cv2.INTER_LANCZOS4
        elif method == 'area':
            interpolation = cv2.INTER_AREA
        elif method == 'linear':
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_CUBIC
        
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation
        )
        
        return resized
    
    def _super_resolution_upscale(
        self,
        image: np.ndarray,
        scale_factor: float
    ) -> np.ndarray:
        """
        Advanced upscaling using edge-directed interpolation
        Simpler alternative to deep learning super-resolution
        """
        # For now, use Lanczos as it's high quality
        # TODO: Integrate deep learning super-resolution models
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Use Lanczos for high-quality upscaling
        upscaled = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Apply sharpening to enhance details
        upscaled = self._sharpen_upscaled(upscaled)
        
        return upscaled
    
    def _sharpen_upscaled(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen upscaled image to enhance details
        """
        # Create sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) / 1.0
        
        # Apply sharpening
        if len(image.shape) == 3:
            sharpened = cv2.filter2D(image, -1, kernel)
        else:
            sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend with original (50-50)
        result = cv2.addWeighted(image, 0.5, sharpened, 0.5, 0)
        
        return result
    
    def resize_to_fixed_dimensions(
        self,
        image: np.ndarray,
        width: int,
        height: int,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize to specific dimensions
        
        Args:
            image: Input image
            width: Target width
            height: Target height
            maintain_aspect: If True, pad to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            return self._resize_with_padding(image, (width, height))
        else:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    
    def _resize_with_padding(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize with padding to maintain aspect ratio
        """
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate scale factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Create white canvas
        if len(image.shape) == 3:
            canvas = np.ones((target_height, target_width, image.shape[2]), dtype=np.uint8) * 255
        else:
            canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
        
        # Center image on canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def upscale_if_needed(
        self,
        image: np.ndarray,
        min_dpi: int = 200
    ) -> np.ndarray:
        """
        Upscale image only if below minimum DPI
        
        Args:
            image: Input image
            min_dpi: Minimum acceptable DPI
            
        Returns:
            Upscaled image if needed, otherwise original
        """
        current_dpi = self.estimate_dpi(image)
        
        if current_dpi < min_dpi:
            scale_factor = min_dpi / current_dpi
            return self._standard_resize(image, scale_factor, 'lanczos')
        
        return image
    
    def downscale_if_needed(
        self,
        image: np.ndarray,
        max_dpi: int = 400
    ) -> np.ndarray:
        """
        Downscale image only if above maximum DPI
        
        Args:
            image: Input image
            max_dpi: Maximum acceptable DPI
            
        Returns:
            Downscaled image if needed, otherwise original
        """
        current_dpi = self.estimate_dpi(image)
        
        if current_dpi > max_dpi:
            scale_factor = max_dpi / current_dpi
            return self._standard_resize(image, scale_factor, 'area')
        
        return image