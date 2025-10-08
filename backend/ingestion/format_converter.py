"""
Format Converter
Standardizes documents to a uniform format for processing
Handles DPI normalization, color space conversion, and format standardization
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple


class FormatConverter:
    """
    Converts and standardizes document images to a uniform format
    """
    
    def __init__(self, target_dpi: int = 300, target_format: str = 'RGB'):
        """
        Initialize format converter
        
        Args:
            target_dpi: Target DPI for standardization (default: 300)
            target_format: Target color format - 'RGB', 'GRAY', or 'BGR'
        """
        self.target_dpi = target_dpi
        self.target_format = target_format
        self.a4_width_inches = 8.27
        self.a4_height_inches = 11.69
        
    def standardize(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Standardize image to uniform format
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Standardized numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure correct color space
        image = self._convert_color_space(image)
        
        # Normalize DPI (resize if needed)
        image = self._normalize_dpi(image)
        
        # Normalize bit depth to 8-bit
        image = self._normalize_bit_depth(image)
        
        return image
    
    def _convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to target color space
        """
        # Detect current format
        if len(image.shape) == 2:
            # Grayscale image
            if self.target_format == 'RGB':
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif self.target_format == 'BGR':
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                return image
        
        elif len(image.shape) == 3:
            channels = image.shape[2]
            
            if channels == 3:
                # Assume RGB from PIL
                if self.target_format == 'BGR':
                    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif self.target_format == 'GRAY':
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    return image
            
            elif channels == 4:
                # RGBA - remove alpha channel
                if self.target_format == 'RGB':
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif self.target_format == 'BGR':
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif self.target_format == 'GRAY':
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        
        return image
    
    def _normalize_dpi(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image DPI by resizing
        This is an approximation - real DPI would require metadata
        """
        height, width = image.shape[:2]
        
        # Calculate current approximate DPI (assuming A4 page)
        current_dpi_width = width / self.a4_width_inches
        current_dpi_height = height / self.a4_height_inches
        current_dpi = (current_dpi_width + current_dpi_height) / 2
        
        # Only resize if significantly different from target
        if abs(current_dpi - self.target_dpi) > 50:
            scale_factor = self.target_dpi / current_dpi
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Choose interpolation method based on scaling
            if scale_factor > 1:
                interpolation = cv2.INTER_CUBIC  # Upscaling
            else:
                interpolation = cv2.INTER_AREA   # Downscaling
            
            image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        return image
    
    def _normalize_bit_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 8-bit depth
        """
        if image.dtype == np.uint16:
            # Convert 16-bit to 8-bit
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            # Convert float to 8-bit
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            # Convert any other type to 8-bit
            image = image.astype(np.uint8)
        
        return image
    
    def resize_to_fixed_size(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image to fixed size
        
        Args:
            image: Input image
            target_size: (width, height) target size
            maintain_aspect: If True, maintain aspect ratio with padding
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            return self._resize_with_aspect_ratio(image, target_size)
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    def _resize_with_aspect_ratio(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize while maintaining aspect ratio, pad with white
        """
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate scale factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Create white canvas
        if len(image.shape) == 3:
            canvas = np.ones((target_height, target_width, image.shape[2]), dtype=np.uint8) * 255
        else:
            canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
        
        # Calculate padding
        pad_top = (target_height - new_height) // 2
        pad_left = (target_width - new_width) // 2
        
        # Place resized image on canvas
        canvas[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = resized
        
        return canvas
    
    def convert_to_binary(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Convert image to binary (black and white)
        
        Args:
            image: Input image
            method: Thresholding method - 'otsu', 'adaptive', or 'fixed'
            
        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'fixed':
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
        
        return binary
    
    def save_standardized_image(
        self, 
        image: np.ndarray, 
        output_path: str,
        quality: int = 95
    ) -> None:
        """
        Save standardized image to disk
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (0-100)
        """
        # Convert BGR to RGB if needed for saving with PIL
        if len(image.shape) == 3 and self.target_format == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        pil_image.save(output_path, quality=quality, dpi=(self.target_dpi, self.target_dpi))
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get statistics about the image
        """
        return {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': int(np.min(image)),
            'max_value': int(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image))
        }