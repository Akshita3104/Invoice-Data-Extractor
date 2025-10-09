"""
Skew Corrector
Detects and corrects document skew/rotation using multiple methods:
- Hough Line Transform
- Projection Profile
- Contour-based detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


class SkewCorrector:
    """
    Detects and corrects skew in document images
    """
    
    def __init__(self, min_angle: float = -45.0, max_angle: float = 45.0):
        """
        Initialize skew corrector
        
        Args:
            min_angle: Minimum rotation angle to consider
            max_angle: Maximum rotation angle to consider
        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_threshold = 0.5  # Don't correct if angle < threshold
    
    def correct(
        self,
        image: np.ndarray,
        method: str = 'hough',
        auto_detect_method: bool = True
    ) -> np.ndarray:
        """
        Detect and correct skew
        
        Args:
            image: Input image
            method: Detection method ('hough', 'projection', 'contour')
            auto_detect_method: If True, automatically choose best method
            
        Returns:
            Corrected image
        """
        # Detect skew angle
        if auto_detect_method:
            angle = self._detect_skew_auto(image)
        else:
            angle = self.detect_skew_angle(image, method=method)
        
        # Only correct if angle is significant
        if abs(angle) < self.angle_threshold:
            print(f"Skew angle {angle:.2f}° is below threshold, skipping correction")
            return image
        
        print(f"Detected skew angle: {angle:.2f}°")
        
        # Rotate image
        corrected = self.rotate_image(image, angle)
        
        return corrected
    
    def detect_skew_angle(
        self,
        image: np.ndarray,
        method: str = 'hough'
    ) -> float:
        """
        Detect skew angle using specified method
        
        Args:
            image: Input image
            method: Detection method
            
        Returns:
            Skew angle in degrees (positive = clockwise)
        """
        if method == 'hough':
            return self._detect_skew_hough(image)
        elif method == 'projection':
            return self._detect_skew_projection(image)
        elif method == 'contour':
            return self._detect_skew_contour(image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_skew_auto(self, image: np.ndarray) -> float:
        """
        Automatically detect skew using multiple methods and choose best result
        """
        angles = []
        
        try:
            angle_hough = self._detect_skew_hough(image)
            angles.append(angle_hough)
        except:
            pass
        
        try:
            angle_projection = self._detect_skew_projection(image)
            angles.append(angle_projection)
        except:
            pass
        
        if not angles:
            return 0.0
        
        # Return median angle (robust to outliers)
        return float(np.median(angles))
    
    def _detect_skew_hough(self, image: np.ndarray) -> float:
        """
        Detect skew using Hough Line Transform
        Best for documents with clear lines
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(gray.shape) // 4,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Calculate angles of all lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize angle to -45 to 45 range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            angles.append(angle)
        
        # Return median angle (robust to outliers)
        return float(np.median(angles))
    
    def _detect_skew_projection(self, image: np.ndarray) -> float:
        """
        Detect skew using projection profile method
        Best for text-heavy documents
        """
        # Convert to grayscale and binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try different angles and find the one with maximum variance in projection
        best_angle = 0.0
        max_variance = 0.0
        
        for angle in np.arange(self.min_angle, self.max_angle, 0.5):
            # Rotate image
            rotated = ndimage.rotate(binary, angle, reshape=False, order=0)
            
            # Calculate horizontal projection (sum of pixels in each row)
            projection = np.sum(rotated, axis=1)
            
            # Calculate variance (higher variance = better alignment)
            variance = np.var(projection)
            
            if variance > max_variance:
                max_variance = variance
                best_angle = angle
        
        return best_angle
    
    def _detect_skew_contour(self, image: np.ndarray) -> float:
        """
        Detect skew using contour analysis
        Best for documents with clear boundaries
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find largest contour (likely the document)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalize angle
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        return angle
    
    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Rotate image by specified angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = clockwise)
            background_color: Color for background (default: white)
            
        Returns:
            Rotated image
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to prevent cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )
        
        return rotated
    
    def auto_crop_after_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically crop white borders after rotation
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Find non-white pixels
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get bounding box of all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        
        # Add small margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        
        return cropped