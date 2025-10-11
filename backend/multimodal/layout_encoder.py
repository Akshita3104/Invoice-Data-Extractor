"""
Layout Encoder
Encodes spatial layout and positional features
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class LayoutEncoder:
    """
    Encodes layout and spatial features from document elements
    """
    
    def __init__(self):
        """Initialize layout encoder"""
        self.feature_dim = 12  # Number of layout features
    
    def encode(
        self,
        ocr_result: Dict,
        zones: List = None
    ) -> Dict:
        """
        Encode layout features from OCR result
        
        Args:
            ocr_result: OCR result with bounding boxes
            zones: Optional layout zones
            
        Returns:
            Layout features
        """
        features = {
            'element_features': [],
            'global_features': {}
        }
        
        # Extract features for each element
        elements = ocr_result.get('words', []) or ocr_result.get('lines', [])
        
        if not elements:
            return {'element_features': [], 'global_features': {}}
        
        # Determine document dimensions
        image_shape = self._estimate_document_size(elements)
        
        # Extract features for each element
        for element in elements:
            bbox = element.get('bbox', {})
            element_features = self.encode_bbox(bbox, image_shape)
            features['element_features'].append(element_features)
        
        # Extract global layout features
        features['global_features'] = self._extract_global_features(elements, zones)
        
        return features
    
    def encode_bbox(
        self,
        bbox: Dict[str, int],
        image_shape: Optional[Tuple] = None
    ) -> Dict:
        """
        Encode features for a single bounding box
        
        Args:
            bbox: Bounding box with x, y, width, height
            image_shape: Optional (height, width) for normalization
            
        Returns:
            Layout features
        """
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        # Normalize by image dimensions if available
        if image_shape:
            img_height, img_width = image_shape[:2]
            x_norm = x / img_width if img_width > 0 else 0
            y_norm = y / img_height if img_height > 0 else 0
            width_norm = width / img_width if img_width > 0 else 0
            height_norm = height / img_height if img_height > 0 else 0
        else:
            # Use raw values (less ideal)
            x_norm = x / 1000.0
            y_norm = y / 1000.0
            width_norm = width / 1000.0
            height_norm = height / 1000.0
        
        # Calculate center
        center_x = x_norm + width_norm / 2
        center_y = y_norm + height_norm / 2
        
        # Calculate area and aspect ratio
        area = width_norm * height_norm
        aspect_ratio = width / max(height, 1)
        
        # Position features (quadrant, relative position)
        quadrant = self._get_quadrant(center_x, center_y)
        
        # Distance from corners
        dist_top_left = np.sqrt(center_x**2 + center_y**2)
        dist_top_right = np.sqrt((1 - center_x)**2 + center_y**2)
        dist_bottom_left = np.sqrt(center_x**2 + (1 - center_y)**2)
        dist_bottom_right = np.sqrt((1 - center_x)**2 + (1 - center_y)**2)
        
        # Create feature vector
        feature_vector = np.array([
            x_norm,
            y_norm,
            width_norm,
            height_norm,
            center_x,
            center_y,
            area,
            aspect_ratio,
            dist_top_left,
            dist_top_right,
            dist_bottom_left,
            dist_bottom_right
        ])
        
        return {
            'feature_vector': feature_vector,
            'quadrant': quadrant,
            'normalized_bbox': {
                'x': x_norm,
                'y': y_norm,
                'width': width_norm,
                'height': height_norm
            }
        }
    
    def _estimate_document_size(self, elements: List[Dict]) -> Tuple[int, int]:
        """
        Estimate document size from bounding boxes
        """
        if not elements:
            return (1000, 1000)
        
        max_x = 0
        max_y = 0
        
        for element in elements:
            bbox = element.get('bbox', {})
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)
        
        return (max_y, max_x)
    
    def _get_quadrant(self, center_x: float, center_y: float) -> int:
        """
        Get quadrant (0-3) for a point
        
        0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        """
        if center_x < 0.5 and center_y < 0.5:
            return 0
        elif center_x >= 0.5 and center_y < 0.5:
            return 1
        elif center_x < 0.5 and center_y >= 0.5:
            return 2
        else:
            return 3
    
    def _extract_global_features(
        self,
        elements: List[Dict],
        zones: List = None
    ) -> Dict:
        """
        Extract global layout features
        """
        features = {}
        
        # Number of elements
        features['num_elements'] = len(elements)
        
        # Bounding box statistics
        all_bboxes = [e.get('bbox', {}) for e in elements]
        
        if all_bboxes:
            x_coords = [b.get('x', 0) for b in all_bboxes]
            y_coords = [b.get('y', 0) for b in all_bboxes]
            widths = [b.get('width', 0) for b in all_bboxes]
            heights = [b.get('height', 0) for b in all_bboxes]
            
            features['x_mean'] = np.mean(x_coords)
            features['x_std'] = np.std(x_coords)
            features['y_mean'] = np.mean(y_coords)
            features['y_std'] = np.std(y_coords)
            features['width_mean'] = np.mean(widths)
            features['width_std'] = np.std(widths)
            features['height_mean'] = np.mean(heights)
            features['height_std'] = np.std(heights)
        
        # Zone information
        if zones:
            features['num_zones'] = len(zones)
            zone_types = {}
            for zone in zones:
                zone_type = zone.zone_type if hasattr(zone, 'zone_type') else 'unknown'
                zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
            features['zone_types'] = zone_types
        
        # Alignment features
        features['alignment_score'] = self._calculate_alignment_score(elements)
        
        # Density features
        features['spatial_density'] = self._calculate_spatial_density(elements)
        
        return features
    
    def _calculate_alignment_score(self, elements: List[Dict]) -> float:
        """
        Calculate how well elements are aligned
        
        Returns:
            Alignment score (0-1, higher = better alignment)
        """
        if len(elements) < 2:
            return 1.0
        
        x_coords = []
        y_coords = []
        
        for element in elements:
            bbox = element.get('bbox', {})
            x_coords.append(bbox.get('x', 0))
            y_coords.append(bbox.get('y', 0))
        
        # Calculate coefficient of variation (lower = better alignment)
        x_cv = np.std(x_coords) / (np.mean(x_coords) + 1e-10)
        y_cv = np.std(y_coords) / (np.mean(y_coords) + 1e-10)
        
        # Convert to score (0-1)
        alignment_score = 1.0 / (1.0 + (x_cv + y_cv) / 2)
        
        return float(alignment_score)
    
    def _calculate_spatial_density(self, elements: List[Dict]) -> float:
        """
        Calculate spatial density of elements
        
        Returns:
            Density score (0-1)
        """
        if not elements:
            return 0.0
        
        # Calculate document bounds
        image_shape = self._estimate_document_size(elements)
        total_area = image_shape[0] * image_shape[1]
        
        # Calculate total element area
        element_area = 0
        for element in elements:
            bbox = element.get('bbox', {})
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            element_area += width * height
        
        # Density = element_area / total_area
        density = element_area / max(total_area, 1)
        
        return float(min(density, 1.0))
    
    def encode_relative_position(
        self,
        bbox1: Dict,
        bbox2: Dict
    ) -> Dict:
        """
        Encode relative position between two bounding boxes
        
        Returns:
            Relative position features
        """
        # Get centers
        x1 = bbox1.get('x', 0) + bbox1.get('width', 0) / 2
        y1 = bbox1.get('y', 0) + bbox1.get('height', 0) / 2
        
        x2 = bbox2.get('x', 0) + bbox2.get('width', 0) / 2
        y2 = bbox2.get('y', 0) + bbox2.get('height', 0) / 2
        
        # Calculate relative position
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Determine direction
        direction = self._get_direction(dx, dy)
        
        return {
            'dx': dx,
            'dy': dy,
            'distance': distance,
            'angle': angle,
            'direction': direction
        }
    
    def _get_direction(self, dx: float, dy: float) -> str:
        """
        Get cardinal direction from delta x and delta y
        """
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'below' if dy > 0 else 'above'
    
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return self.feature_dim
    
    def visualize_layout(
        self,
        image: np.ndarray,
        elements: List[Dict],
        zones: List = None
    ) -> np.ndarray:
        """
        Visualize layout features on image
        """
        import cv2
        
        vis_image = image.copy()
        
        # Draw elements with color based on quadrant
        for element in elements:
            bbox = element.get('bbox', {})
            x, y = bbox.get('x', 0), bbox.get('y', 0)
            w, h = bbox.get('width', 0), bbox.get('height', 0)
            
            # Encode to get quadrant
            layout_features = self.encode_bbox(bbox, image.shape)
            quadrant = layout_features['quadrant']
            
            # Color by quadrant
            colors = [
                (255, 0, 0),    # Top-left: Red
                (0, 255, 0),    # Top-right: Green
                (0, 0, 255),    # Bottom-left: Blue
                (255, 255, 0)   # Bottom-right: Yellow
            ]
            color = colors[quadrant]
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        return vis_image