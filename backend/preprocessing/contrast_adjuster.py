"""
Contrast Adjuster
Enhances contrast and adjusts brightness using various techniques:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram equalization
- Adaptive histogram equalization
- Gamma correction
- Automatic brightness adjustment
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ContrastAdjuster:
    """
    Adjusts contrast and brightness in document images
    """
    
    def __init__(self):
        self.optimal_brightness = 127  # Mid-gray for 8-bit images
        self.optimal_contrast_std = 60  # Target standard deviation
    
    def enhance_contrast(
        self,
        image: np.ndarray,
        method: str = 'clahe',
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image: Input image
            method: Enhancement method
                - 'clahe': CLAHE (best for documents)
                - 'histogram_eq': Global histogram equalization
                - 'adaptive_histogram': Adaptive histogram equalization
                - 'gamma': Gamma correction
            clip_limit: CLAHE clip limit (higher = more contrast)
            tile_size: CLAHE tile grid size
            
        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            return self._apply_clahe(image, clip_limit, tile_size)
        elif method == 'histogram_eq':
            return self._apply_histogram_equalization(image)
        elif method == 'adaptive_histogram':
            return self._apply_adaptive_histogram(image)
        elif method == 'gamma':
            return self._apply_gamma_correction(image, gamma=1.2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float,
        tile_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Best method for document images - prevents over-amplification
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # Convert back to RGB
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            return result
        else:
            # Grayscale image
            return clahe.apply(image)
    
    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply global histogram equalization
        Simple but can over-enhance
        """
        if len(image.shape) == 3:
            # Convert to YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return cv2.equalizeHist(image)
    
    def _apply_adaptive_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive histogram equalization with higher clip limit than CLAHE
        """
        return self._apply_clahe(image, clip_limit=4.0, tile_size=(8, 8))
    
    def _apply_gamma_correction(
        self,
        image: np.ndarray,
        gamma: float = 1.0
    ) -> np.ndarray:
        """
        Apply gamma correction
        gamma < 1: brighten image
        gamma > 1: darken image
        
        Args:
            image: Input image
            gamma: Gamma value
            
        Returns:
            Gamma-corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Apply gamma correction using LUT
        return cv2.LUT(image, table)
    
    def adjust_brightness(
        self,
        image: np.ndarray,
        target_brightness: Optional[int] = None,
        auto: bool = True
    ) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Input image
            target_brightness: Target mean brightness (0-255)
            auto: If True, automatically determine target brightness
            
        Returns:
            Brightness-adjusted image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        current_brightness = np.mean(gray)
        
        if auto:
            # Determine target based on current brightness
            if current_brightness < 80:
                target_brightness = 127  # Brighten dark images
            elif current_brightness > 180:
                target_brightness = 127  # Darken bright images
            else:
                target_brightness = current_brightness  # Already good
        
        if target_brightness is None:
            target_brightness = self.optimal_brightness
        
        # Calculate adjustment
        adjustment = int(target_brightness - current_brightness)
        
        if abs(adjustment) < 5:
            print("Brightness is already optimal")
            return image
        
        print(f"Adjusting brightness by {adjustment}")
        
        # Apply brightness adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=1, beta=adjustment)
        
        return adjusted
    
    def auto_contrast_brightness(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Automatically adjust both contrast and brightness
        
        Args:
            image: Input image
            
        Returns:
            Adjusted image
        """
        # First adjust brightness
        adjusted = self.adjust_brightness(image, auto=True)
        
        # Then enhance contrast
        adjusted = self.enhance_contrast(adjusted, method='clahe', clip_limit=2.0)
        
        return adjusted
    
    def normalize_intensity(
        self,
        image: np.ndarray,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> np.ndarray:
        """
        Normalize image intensity using percentile stretching
        Robust to outliers
        
        Args:
            image: Input image
            lower_percentile: Lower percentile for normalization
            upper_percentile: Upper percentile for normalization
            
        Returns:
            Normalized image
        """
        if len(image.shape) == 3:
            # Process each channel
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = self._normalize_channel(
                    image[:, :, i],
                    lower_percentile,
                    upper_percentile
                )
            return result
        else:
            return self._normalize_channel(image, lower_percentile, upper_percentile)
    
    def _normalize_channel(
        self,
        channel: np.ndarray,
        lower_percentile: float,
        upper_percentile: float
    ) -> np.ndarray:
        """
        Normalize a single channel
        """
        # Calculate percentiles
        p_low = np.percentile(channel, lower_percentile)
        p_high = np.percentile(channel, upper_percentile)
        
        # Avoid division by zero
        if p_high - p_low < 1:
            return channel
        
        # Stretch intensity
        normalized = np.clip((channel - p_low) / (p_high - p_low) * 255, 0, 255)
        
        return normalized.astype(np.uint8)
    
    def adaptive_gamma_correction(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Automatically determine and apply optimal gamma correction
        
        Args:
            image: Input image
            
        Returns:
            Gamma-corrected image
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Determine gamma based on brightness
        if mean_brightness < 80:
            # Dark image - brighten (gamma < 1)
            gamma = 0.7
        elif mean_brightness > 180:
            # Bright image - darken (gamma > 1)
            gamma = 1.5
        else:
            # Normal brightness
            gamma = 1.0
        
        if gamma == 1.0:
            return image
        
        print(f"Applying adaptive gamma correction: {gamma:.2f}")
        return self._apply_gamma_correction(image, gamma)
    
    def enhance_text_contrast(
        self,
        image: np.ndarray,
        aggressive: bool = False
    ) -> np.ndarray:
        """
        Specifically enhance text contrast for better OCR
        
        Args:
            image: Input image
            aggressive: If True, apply more aggressive enhancement
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding to find text regions
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        if aggressive:
            # Apply morphological operations to enhance text
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Apply CLAHE for final enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced