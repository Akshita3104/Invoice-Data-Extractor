"""
Noise Reducer
Implements various noise reduction techniques:
- Gaussian filtering
- Bilateral filtering (preserves edges)
- Non-local means denoising
- Morphological filtering
"""

import cv2
import numpy as np
from typing import Literal


class NoiseReducer:
    """
    Reduces noise in document images while preserving text clarity
    """
    
    def __init__(self):
        self.strength_params = {
            'light': {'h': 5, 'kernel': 3, 'd': 5},
            'medium': {'h': 10, 'kernel': 5, 'd': 7},
            'strong': {'h': 15, 'kernel': 7, 'd': 9}
        }
    
    def reduce_noise(
        self,
        image: np.ndarray,
        method: Literal['gaussian', 'bilateral', 'nlm', 'morphological'] = 'bilateral',
        strength: Literal['light', 'medium', 'strong'] = 'medium'
    ) -> np.ndarray:
        """
        Reduce noise using specified method
        
        Args:
            image: Input image
            method: Denoising method
                - 'gaussian': Fast, blurs everything
                - 'bilateral': Preserves edges (recommended for documents)
                - 'nlm': Non-local means, slow but high quality
                - 'morphological': Good for salt-and-pepper noise
            strength: Denoising strength
            
        Returns:
            Denoised image
        """
        params = self.strength_params[strength]
        
        if method == 'gaussian':
            return self._gaussian_denoise(image, params['kernel'])
        elif method == 'bilateral':
            return self._bilateral_denoise(image, params['d'])
        elif method == 'nlm':
            return self._nlm_denoise(image, params['h'])
        elif method == 'morphological':
            return self._morphological_denoise(image, params['kernel'])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _gaussian_denoise(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Gaussian blur denoising (fast but blurs edges)
        """
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _bilateral_denoise(self, image: np.ndarray, d: int) -> np.ndarray:
        """
        Bilateral filtering (preserves edges while reducing noise)
        Best for document images
        """
        if len(image.shape) == 3:
            # Apply to each channel
            channels = cv2.split(image)
            denoised_channels = []
            
            for channel in channels:
                denoised = cv2.bilateralFilter(
                    channel, 
                    d=d,              # Diameter of pixel neighborhood
                    sigmaColor=75,    # Filter sigma in color space
                    sigmaSpace=75     # Filter sigma in coordinate space
                )
                denoised_channels.append(denoised)
            
            return cv2.merge(denoised_channels)
        else:
            return cv2.bilateralFilter(image, d=d, sigmaColor=75, sigmaSpace=75)
    
    def _nlm_denoise(self, image: np.ndarray, h: int) -> np.ndarray:
        """
        Non-local means denoising (slow but high quality)
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=h,           # Filter strength
                hColor=h,      # Filter strength for color
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=h,
                templateWindowSize=7,
                searchWindowSize=21
            )
    
    def _morphological_denoise(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Morphological filtering (good for salt-and-pepper noise)
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening: erosion followed by dilation (removes white noise)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing: dilation followed by erosion (removes black noise)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def adaptive_denoise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Adaptive denoising based on estimated noise level
        
        Args:
            image: Input image
            noise_level: Estimated noise level (0.0 to 1.0)
            
        Returns:
            Denoised image
        """
        if noise_level < 0.3:
            # Low noise - light denoising
            return self.reduce_noise(image, method='bilateral', strength='light')
        elif noise_level < 0.6:
            # Medium noise
            return self.reduce_noise(image, method='bilateral', strength='medium')
        else:
            # High noise - strong denoising
            return self.reduce_noise(image, method='nlm', strength='strong')
    
    def median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter (good for salt-and-pepper noise)
        
        Args:
            image: Input image
            kernel_size: Kernel size (must be odd)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.medianBlur(image, kernel_size)
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in image
        
        Returns:
            Noise level from 0.0 (clean) to 1.0 (very noisy)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate difference (noise)
        noise = cv2.absdiff(gray, blurred)
        noise_level = np.std(noise) / 255.0
        
        return min(noise_level * 2, 1.0)  # Normalize to 0-1