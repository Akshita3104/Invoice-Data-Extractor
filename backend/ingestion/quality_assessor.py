"""
Quality Assessor
Assesses image quality using multiple metrics:
- Blur detection (Laplacian variance)
- Resolution check
- Noise level estimation
- Contrast measurement
- Brightness assessment
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple


class QualityAssessor:
    """
    Assesses document image quality across multiple dimensions
    """
    
    def __init__(self):
        self.min_dpi = 150
        self.optimal_dpi = 300
        self.blur_threshold = 100.0
        self.noise_threshold = 20.0
        
    def assess(self, image: np.ndarray) -> Dict:
        """
        Comprehensive quality assessment
        
        Args:
            image: Input image as numpy array (RGB or grayscale)
            
        Returns:
            Dictionary containing quality metrics and overall score
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate individual metrics
        blur_score = self._assess_blur(gray)
        resolution_score = self._assess_resolution(image)
        noise_score = self._assess_noise(gray)
        contrast_score = self._assess_contrast(gray)
        brightness_score = self._assess_brightness(gray)
        
        # Calculate overall quality score (weighted average)
        overall_score = (
            blur_score * 0.30 +
            resolution_score * 0.25 +
            noise_score * 0.20 +
            contrast_score * 0.15 +
            brightness_score * 0.10
        )
        
        # Determine quality level
        quality_level = self._get_quality_level(overall_score)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            blur_score, resolution_score, noise_score, 
            contrast_score, brightness_score
        )
        
        return {
            'overall_score': round(overall_score, 3),
            'quality_level': quality_level,
            'metrics': {
                'blur_score': round(blur_score, 3),
                'resolution_score': round(resolution_score, 3),
                'noise_score': round(noise_score, 3),
                'contrast_score': round(contrast_score, 3),
                'brightness_score': round(brightness_score, 3)
            },
            'dimensions': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2] if len(image.shape) == 3 else 1
            },
            'recommendations': recommendations
        }
    
    def _assess_blur(self, gray_image: np.ndarray) -> float:
        """
        Detect blur using Laplacian variance method
        Higher variance = sharper image
        
        Returns:
            Score from 0.0 (blurry) to 1.0 (sharp)
        """
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # Normalize score (values typically range 0-500+)
        # Good images usually have variance > 100
        score = min(laplacian_var / 200.0, 1.0)
        
        return score
    
    def _assess_resolution(self, image: np.ndarray) -> float:
        """
        Assess image resolution
        
        Returns:
            Score from 0.0 (low res) to 1.0 (high res)
        """
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Estimate DPI (assuming A4 page: 8.27 x 11.69 inches)
        # This is approximate - real DPI would need metadata
        estimated_dpi = int(np.sqrt(total_pixels / (8.27 * 11.69)))
        
        # Score based on DPI
        if estimated_dpi >= self.optimal_dpi:
            score = 1.0
        elif estimated_dpi >= self.min_dpi:
            score = (estimated_dpi - self.min_dpi) / (self.optimal_dpi - self.min_dpi)
        else:
            score = estimated_dpi / self.min_dpi * 0.5
        
        return min(score, 1.0)
    
    def _assess_noise(self, gray_image: np.ndarray) -> float:
        """
        Estimate noise level using standard deviation of Laplacian
        
        Returns:
            Score from 0.0 (noisy) to 1.0 (clean)
        """
        # Apply Gaussian blur to remove texture
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Calculate difference (noise)
        noise = cv2.absdiff(gray_image, blurred)
        noise_level = np.std(noise)
        
        # Normalize (typical noise levels: 0-50)
        score = 1.0 - min(noise_level / 50.0, 1.0)
        
        return score
    
    def _assess_contrast(self, gray_image: np.ndarray) -> float:
        """
        Measure contrast using standard deviation
        Higher std = better contrast
        
        Returns:
            Score from 0.0 (low contrast) to 1.0 (high contrast)
        """
        std_dev = np.std(gray_image)
        
        # Normalize (ideal std is around 50-80 for 8-bit images)
        # Maximum possible std is ~73 for uniform distribution
        score = min(std_dev / 70.0, 1.0)
        
        return score
    
    def _assess_brightness(self, gray_image: np.ndarray) -> float:
        """
        Assess brightness level
        Optimal brightness is around 127 (mid-gray)
        
        Returns:
            Score from 0.0 (too dark/bright) to 1.0 (optimal)
        """
        mean_brightness = np.mean(gray_image)
        
        # Optimal range: 100-160
        if 100 <= mean_brightness <= 160:
            score = 1.0
        elif mean_brightness < 100:
            score = mean_brightness / 100.0
        else:  # > 160
            score = 1.0 - ((mean_brightness - 160) / 95.0)
        
        return max(min(score, 1.0), 0.0)
    
    def _get_quality_level(self, score: float) -> str:
        """
        Convert numeric score to quality level
        """
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        elif score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_recommendations(
        self, 
        blur: float, 
        resolution: float, 
        noise: float,
        contrast: float, 
        brightness: float
    ) -> list:
        """
        Generate preprocessing recommendations based on scores
        """
        recommendations = []
        
        if blur < 0.5:
            recommendations.append('apply_sharpening')
        
        if resolution < 0.6:
            recommendations.append('upscale_image')
        
        if noise < 0.6:
            recommendations.append('apply_denoising')
        
        if contrast < 0.5:
            recommendations.append('enhance_contrast')
        
        if brightness < 0.5:
            recommendations.append('adjust_brightness')
        
        return recommendations
    
    def is_acceptable_quality(self, quality_metrics: Dict, threshold: float = 0.4) -> bool:
        """
        Determine if document quality is acceptable for OCR
        
        Args:
            quality_metrics: Output from assess()
            threshold: Minimum acceptable quality score
            
        Returns:
            True if quality is acceptable
        """
        return quality_metrics['overall_score'] >= threshold