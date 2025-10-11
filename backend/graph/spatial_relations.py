"""
Spatial Relations Extractor
Extracts spatial relationships between document elements:
- above, below, left, right
- aligned_horizontally, aligned_vertically
- near, far
- overlaps
"""

import numpy as np
from typing import Dict, Tuple


class SpatialRelationExtractor:
    """
    Extracts spatial relationships between bounding boxes
    """
    
    def __init__(self):
        self.near_threshold = 50  # pixels
        self.alignment_threshold = 10  # pixels for alignment detection
    
    def extract_relations(
        self,
        bbox1: Dict[str, int],
        bbox2: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Extract all spatial relations between two bounding boxes
        
        Returns:
            Dictionary mapping relation types to confidence scores (0-1)
        """
        relations = {}
        
        # Basic directional relations
        relations['above'] = self._is_above(bbox1, bbox2)
        relations['below'] = self._is_below(bbox1, bbox2)
        relations['left_of'] = self._is_left_of(bbox1, bbox2)
        relations['right_of'] = self._is_right_of(bbox1, bbox2)
        
        # Alignment relations
        relations['aligned_horizontally'] = self._is_aligned_horizontally(bbox1, bbox2)
        relations['aligned_vertically'] = self._is_aligned_vertically(bbox1, bbox2)
        
        # Proximity relations
        relations['near'] = self._is_near(bbox1, bbox2)
        relations['far'] = 1.0 - relations['near']
        
        # Overlap relation
        relations['overlaps'] = self._overlaps(bbox1, bbox2)
        
        # Diagonal relations
        relations['top_left'] = self._is_top_left(bbox1, bbox2)
        relations['top_right'] = self._is_top_right(bbox1, bbox2)
        relations['bottom_left'] = self._is_bottom_left(bbox1, bbox2)
        relations['bottom_right'] = self._is_bottom_right(bbox1, bbox2)
        
        return relations
    
    def _is_above(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is above bbox2"""
        y1_bottom = bbox1.get('y', 0) + bbox1.get('height', 0)
        y2_top = bbox2.get('y', 0)
        
        if y1_bottom <= y2_top:
            # Calculate confidence based on distance
            distance = y2_top - y1_bottom
            confidence = 1.0 / (1.0 + distance / 100.0)  # Decay with distance
            return confidence
        
        return 0.0
    
    def _is_below(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is below bbox2"""
        return self._is_above(bbox2, bbox1)
    
    def _is_left_of(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is left of bbox2"""
        x1_right = bbox1.get('x', 0) + bbox1.get('width', 0)
        x2_left = bbox2.get('x', 0)
        
        if x1_right <= x2_left:
            distance = x2_left - x1_right
            confidence = 1.0 / (1.0 + distance / 100.0)
            return confidence
        
        return 0.0
    
    def _is_right_of(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is right of bbox2"""
        return self._is_left_of(bbox2, bbox1)
    
    def _is_aligned_horizontally(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bboxes are horizontally aligned (same Y)"""
        y1_center = bbox1.get('y', 0) + bbox1.get('height', 0) / 2
        y2_center = bbox2.get('y', 0) + bbox2.get('height', 0) / 2
        
        diff = abs(y1_center - y2_center)
        
        if diff <= self.alignment_threshold:
            return 1.0
        elif diff <= self.alignment_threshold * 3:
            return 1.0 - (diff - self.alignment_threshold) / (self.alignment_threshold * 2)
        
        return 0.0
    
    def _is_aligned_vertically(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bboxes are vertically aligned (same X)"""
        x1_center = bbox1.get('x', 0) + bbox1.get('width', 0) / 2
        x2_center = bbox2.get('x', 0) + bbox2.get('width', 0) / 2
        
        diff = abs(x1_center - x2_center)
        
        if diff <= self.alignment_threshold:
            return 1.0
        elif diff <= self.alignment_threshold * 3:
            return 1.0 - (diff - self.alignment_threshold) / (self.alignment_threshold * 2)
        
        return 0.0
    
    def _is_near(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bboxes are near each other"""
        distance = self._calculate_distance(bbox1, bbox2)
        
        if distance <= self.near_threshold:
            return 1.0 - (distance / self.near_threshold)
        
        return 0.0
    
    def _overlaps(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bboxes overlap"""
        x1_left = bbox1.get('x', 0)
        x1_right = x1_left + bbox1.get('width', 0)
        y1_top = bbox1.get('y', 0)
        y1_bottom = y1_top + bbox1.get('height', 0)
        
        x2_left = bbox2.get('x', 0)
        x2_right = x2_left + bbox2.get('width', 0)
        y2_top = bbox2.get('y', 0)
        y2_bottom = y2_top + bbox2.get('height', 0)
        
        # Calculate intersection
        x_overlap = max(0, min(x1_right, x2_right) - max(x1_left, x2_left))
        y_overlap = max(0, min(y1_bottom, y2_bottom) - max(y1_top, y2_top))
        
        intersection = x_overlap * y_overlap
        
        if intersection == 0:
            return 0.0
        
        # Calculate union
        area1 = bbox1.get('width', 0) * bbox1.get('height', 0)
        area2 = bbox2.get('width', 0) * bbox2.get('height', 0)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_top_left(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is top-left of bbox2"""
        above = self._is_above(bbox1, bbox2)
        left = self._is_left_of(bbox1, bbox2)
        
        return (above + left) / 2 if above > 0 and left > 0 else 0.0
    
    def _is_top_right(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is top-right of bbox2"""
        above = self._is_above(bbox1, bbox2)
        right = self._is_right_of(bbox1, bbox2)
        
        return (above + right) / 2 if above > 0 and right > 0 else 0.0
    
    def _is_bottom_left(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is bottom-left of bbox2"""
        below = self._is_below(bbox1, bbox2)
        left = self._is_left_of(bbox1, bbox2)
        
        return (below + left) / 2 if below > 0 and left > 0 else 0.0
    
    def _is_bottom_right(self, bbox1: Dict, bbox2: Dict) -> float:
        """Check if bbox1 is bottom-right of bbox2"""
        below = self._is_below(bbox1, bbox2)
        right = self._is_right_of(bbox1, bbox2)
        
        return (below + right) / 2 if below > 0 and right > 0 else 0.0
    
    def _calculate_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate minimum distance between two bounding boxes"""
        # Get centers
        x1_center = bbox1.get('x', 0) + bbox1.get('width', 0) / 2
        y1_center = bbox1.get('y', 0) + bbox1.get('height', 0) / 2
        
        x2_center = bbox2.get('x', 0) + bbox2.get('width', 0) / 2
        y2_center = bbox2.get('y', 0) + bbox2.get('height', 0) / 2
        
        # Euclidean distance between centers
        distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
        
        return distance
    
    def get_relative_position(self, bbox1: Dict, bbox2: Dict) -> Tuple[str, float]:
        """
        Get the strongest relative position relation
        
        Returns:
            Tuple of (relation_name, confidence)
        """
        relations = self.extract_relations(bbox1, bbox2)
        
        # Filter out weak relations
        strong_relations = {k: v for k, v in relations.items() if v > 0.3}
        
        if not strong_relations:
            return ('none', 0.0)
        
        # Return strongest relation
        best_relation = max(strong_relations.items(), key=lambda x: x[1])
        
        return best_relation