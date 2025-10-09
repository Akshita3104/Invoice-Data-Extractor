"""
Quality Enhancer
Orchestrates adaptive preprocessing based on quality metrics
Selectively applies enhancement techniques based on detected issues
"""

import numpy as np
from typing import Dict, List
import cv2

from .noise_reducer import NoiseReducer
from .skew_corrector import SkewCorrector
from .resolution_normalizer import ResolutionNormalizer
from .contrast_adjuster import ContrastAdjuster


class QualityEnhancer:
    """
    Adaptive quality enhancement based on quality assessment
    """
    
    def __init__(self, aggressive: bool = False):
        """
        Initialize quality enhancer
        
        Args:
            aggressive: If True, apply more aggressive enhancements
        """
        self.aggressive = aggressive
        self.noise_reducer = NoiseReducer()
        self.skew_corrector = SkewCorrector()
        self.resolution_normalizer = ResolutionNormalizer()
        self.contrast_adjuster = ContrastAdjuster()
        
    def enhance(
        self, 
        image: np.ndarray, 
        quality_metrics: Dict
    ) -> np.ndarray:
        """
        Apply adaptive enhancements based on quality metrics
        
        Args:
            image: Input image as numpy array
            quality_metrics: Quality metrics from QualityAssessor
            
        Returns:
            Enhanced image
        """
        enhanced_image = image.copy()
        applied_enhancements = []
        
        # Extract scores
        metrics = quality_metrics.get('metrics', {})
        recommendations = quality_metrics.get('recommendations', [])
        
        # Get individual scores
        blur_score = metrics.get('blur_score', 1.0)
        noise_score = metrics.get('noise_score', 1.0)
        contrast_score = metrics.get('contrast_score', 1.0)
        brightness_score = metrics.get('brightness_score', 1.0)
        resolution_score = metrics.get('resolution_score', 1.0)
        
        # 1. Correct skew first (affects all other processing)
        if 'apply_skew_correction' in recommendations or self.aggressive:
            enhanced_image = self.skew_corrector.correct(enhanced_image)
            applied_enhancements.append('skew_correction')
        
        # 2. Denoise if needed
        if noise_score < 0.6 or 'apply_denoising' in recommendations:
            # Choose denoising strength based on score
            if noise_score < 0.3:
                strength = 'strong'
            elif noise_score < 0.5:
                strength = 'medium'
            else:
                strength = 'light'
            
            enhanced_image = self.noise_reducer.reduce_noise(
                enhanced_image, 
                method='bilateral',
                strength=strength
            )
            applied_enhancements.append(f'denoising_{strength}')
        
        # 3. Normalize resolution if needed
        if resolution_score < 0.6 or 'upscale_image' in recommendations:
            enhanced_image = self.resolution_normalizer.normalize(
                enhanced_image,
                target_dpi=300
            )
            applied_enhancements.append('resolution_normalization')
        
        # 4. Adjust contrast if needed
        if contrast_score < 0.5 or 'enhance_contrast' in recommendations:
            # Choose method based on image characteristics
            if contrast_score < 0.3:
                enhanced_image = self.contrast_adjuster.enhance_contrast(
                    enhanced_image,
                    method='clahe',
                    clip_limit=3.0
                )
                applied_enhancements.append('contrast_clahe')
            else:
                enhanced_image = self.contrast_adjuster.enhance_contrast(
                    enhanced_image,
                    method='adaptive_histogram'
                )
                applied_enhancements.append('contrast_adaptive')
        
        # 5. Adjust brightness if needed
        if brightness_score < 0.5 or 'adjust_brightness' in recommendations:
            enhanced_image = self.contrast_adjuster.adjust_brightness(
                enhanced_image,
                auto=True
            )
            applied_enhancements.append('brightness_adjustment')
        
        # 6. Sharpen if blurry
        if blur_score < 0.5 or 'apply_sharpening' in recommendations:
            if blur_score < 0.3:
                amount = 2.0  # Strong sharpening
            else:
                amount = 1.5  # Moderate sharpening
            
            enhanced_image = self._sharpen_image(enhanced_image, amount)
            applied_enhancements.append(f'sharpening_{amount}')
        
        # 7. Final cleanup - remove small artifacts
        if self.aggressive:
            enhanced_image = self._remove_artifacts(enhanced_image)
            applied_enhancements.append('artifact_removal')
        
        print(f"Applied enhancements: {', '.join(applied_enhancements)}")
        
        return enhanced_image
    
    def enhance_custom(
        self,
        image: np.ndarray,
        operations: List[str]
    ) -> np.ndarray:
        """
        Apply specific enhancement operations
        
        Args:
            image: Input image
            operations: List of operations to apply
                       Options: 'denoise', 'skew', 'contrast', 'brightness', 
                                'sharpen', 'resolution'
        
        Returns:
            Enhanced image
        """
        enhanced_image = image.copy()
        
        for operation in operations:
            if operation == 'denoise':
                enhanced_image = self.noise_reducer.reduce_noise(
                    enhanced_image, 
                    method='bilateral'
                )
            
            elif operation == 'skew':
                enhanced_image = self.skew_corrector.correct(enhanced_image)
            
            elif operation == 'contrast':
                enhanced_image = self.contrast_adjuster.enhance_contrast(
                    enhanced_image, 
                    method='clahe'
                )
            
            elif operation == 'brightness':
                enhanced_image = self.contrast_adjuster.adjust_brightness(
                    enhanced_image,
                    auto=True
                )
            
            elif operation == 'sharpen':
                enhanced_image = self._sharpen_image(enhanced_image, amount=1.5)
            
            elif operation == 'resolution':
                enhanced_image = self.resolution_normalizer.normalize(
                    enhanced_image,
                    target_dpi=300
                )
            
            else:
                print(f"Warning: Unknown operation '{operation}'")
        
        return enhanced_image
    
    def _sharpen_image(self, image: np.ndarray, amount: float = 1.5) -> np.ndarray:
        """
        Sharpen image using unsharp mask
        
        Args:
            image: Input image
            amount: Sharpening amount (1.0 = no change, >1.0 = sharper)
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            # Apply to each channel
            channels = cv2.split(image)
            sharpened_channels = []
            
            for channel in channels:
                blurred = cv2.GaussianBlur(channel, (0, 0), 3)
                sharpened = cv2.addWeighted(
                    channel, amount, 
                    blurred, 1 - amount, 0
                )
                sharpened_channels.append(sharpened)
            
            return cv2.merge(sharpened_channels)
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            return cv2.addWeighted(image, amount, blurred, 1 - amount, 0)
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small artifacts and noise spots
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply morphological opening to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Convert back to original color space if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        
        return cleaned
    
    def batch_enhance(
        self,
        images: List[np.ndarray],
        quality_metrics_list: List[Dict]
    ) -> List[np.ndarray]:
        """
        Enhance multiple images
        
        Args:
            images: List of images
            quality_metrics_list: List of quality metrics for each image
            
        Returns:
            List of enhanced images
        """
        enhanced_images = []
        
        for image, metrics in zip(images, quality_metrics_list):
            enhanced = self.enhance(image, metrics)
            enhanced_images.append(enhanced)
        
        return enhanced_images