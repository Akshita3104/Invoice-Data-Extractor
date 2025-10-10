"""
Zone Segmenter
Segments document into functional zones:
- Header (company info, invoice number, date)
- Body (line items, main content)
- Footer (totals, signatures, terms)
- Sidebar (additional information)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Zone:
    """Represents a document zone"""
    zone_type: str  # 'header', 'body', 'footer', 'sidebar', 'table'
    bbox: Dict[str, int]  # {x, y, width, height}
    confidence: float
    text_density: float
    content: str = ""


class ZoneSegmenter:
    """
    Segments document into functional zones using layout analysis
    """
    
    def __init__(self, method: str = 'hybrid'):
        """
        Initialize zone segmenter
        
        Args:
            method: Segmentation method
                - 'rule_based': Simple rule-based segmentation
                - 'projection': Projection profile-based
                - 'hybrid': Combination of methods (recommended)
        """
        self.method = method
        
        # Zone detection parameters (relative to page height)
        self.header_threshold = 0.2  # Top 20% is typically header
        self.footer_threshold = 0.85  # Bottom 15% is typically footer
        self.min_zone_height = 20  # Minimum height for a zone
    
    def segment(
        self,
        image: np.ndarray,
        ocr_result: Dict = None
    ) -> List[Zone]:
        """
        Segment document into zones
        
        Args:
            image: Input document image
            ocr_result: Optional OCR result for text-aware segmentation
            
        Returns:
            List of detected zones
        """
        if self.method == 'rule_based':
            zones = self._segment_rule_based(image)
        elif self.method == 'projection':
            zones = self._segment_projection(image)
        elif self.method == 'hybrid':
            zones = self._segment_hybrid(image, ocr_result)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Add text content to zones if OCR result available
        if ocr_result:
            zones = self._add_text_to_zones(zones, ocr_result)
        
        return zones
    
    def _segment_rule_based(self, image: np.ndarray) -> List[Zone]:
        """
        Simple rule-based segmentation using position heuristics
        """
        height, width = image.shape[:2]
        zones = []
        
        # Header zone (top 20%)
        header_height = int(height * self.header_threshold)
        zones.append(Zone(
            zone_type='header',
            bbox={'x': 0, 'y': 0, 'width': width, 'height': header_height},
            confidence=0.8,
            text_density=self._calculate_text_density(
                image[0:header_height, :]
            )
        ))
        
        # Footer zone (bottom 15%)
        footer_y = int(height * self.footer_threshold)
        footer_height = height - footer_y
        zones.append(Zone(
            zone_type='footer',
            bbox={'x': 0, 'y': footer_y, 'width': width, 'height': footer_height},
            confidence=0.8,
            text_density=self._calculate_text_density(
                image[footer_y:, :]
            )
        ))
        
        # Body zone (middle)
        body_height = footer_y - header_height
        zones.append(Zone(
            zone_type='body',
            bbox={'x': 0, 'y': header_height, 'width': width, 'height': body_height},
            confidence=0.9,
            text_density=self._calculate_text_density(
                image[header_height:footer_y, :]
            )
        ))
        
        return zones
    
    def _segment_projection(self, image: np.ndarray) -> List[Zone]:
        """
        Segmentation using horizontal projection profile
        Detects gaps between zones
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale and binarize
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate horizontal projection (sum of black pixels per row)
        projection = np.sum(binary, axis=1)
        
        # Smooth projection
        kernel_size = max(5, height // 100)
        projection = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find gaps (regions with low projection values)
        threshold = np.mean(projection) * 0.3
        gaps = projection < threshold
        
        # Find continuous gap regions
        gap_regions = self._find_continuous_regions(gaps, min_length=10)
        
        # Create zones between gaps
        zones = []
        prev_y = 0
        
        for gap_start, gap_end in gap_regions:
            if gap_start - prev_y > self.min_zone_height:
                zone_type = self._classify_zone_by_position(prev_y, gap_start, height)
                
                zones.append(Zone(
                    zone_type=zone_type,
                    bbox={'x': 0, 'y': prev_y, 'width': width, 'height': gap_start - prev_y},
                    confidence=0.85,
                    text_density=self._calculate_text_density(
                        image[prev_y:gap_start, :]
                    )
                ))
            
            prev_y = gap_end
        
        # Add final zone
        if height - prev_y > self.min_zone_height:
            zone_type = self._classify_zone_by_position(prev_y, height, height)
            zones.append(Zone(
                zone_type=zone_type,
                bbox={'x': 0, 'y': prev_y, 'width': width, 'height': height - prev_y},
                confidence=0.85,
                text_density=self._calculate_text_density(
                    image[prev_y:, :]
                )
            ))
        
        # If no zones detected, fall back to rule-based
        if len(zones) == 0:
            return self._segment_rule_based(image)
        
        return zones
    
    def _segment_hybrid(
        self,
        image: np.ndarray,
        ocr_result: Dict = None
    ) -> List[Zone]:
        """
        Hybrid segmentation combining multiple methods
        """
        height, width = image.shape[:2]
        
        # Start with projection-based segmentation
        zones = self._segment_projection(image)
        
        # Refine zones using text blocks from OCR
        if ocr_result and ocr_result.get('blocks'):
            zones = self._refine_with_text_blocks(zones, ocr_result['blocks'], image.shape)
        
        # Detect special zones (tables, signatures, etc.)
        special_zones = self._detect_special_zones(image)
        zones.extend(special_zones)
        
        # Merge overlapping zones
        zones = self._merge_overlapping_zones(zones)
        
        # Re-classify zones based on content and position
        zones = self._reclassify_zones(zones, height)
        
        return sorted(zones, key=lambda z: z.bbox['y'])
    
    def _find_continuous_regions(
        self,
        binary_array: np.ndarray,
        min_length: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Find continuous True regions in binary array
        """
        regions = []
        in_region = False
        start = 0
        
        for i, value in enumerate(binary_array):
            if value and not in_region:
                start = i
                in_region = True
            elif not value and in_region:
                if i - start >= min_length:
                    regions.append((start, i))
                in_region = False
        
        # Handle last region
        if in_region and len(binary_array) - start >= min_length:
            regions.append((start, len(binary_array)))
        
        return regions
    
    def _classify_zone_by_position(
        self,
        y_start: int,
        y_end: int,
        page_height: int
    ) -> str:
        """
        Classify zone type based on vertical position
        """
        y_center = (y_start + y_end) / 2
        relative_position = y_center / page_height
        
        if relative_position < 0.25:
            return 'header'
        elif relative_position > 0.85:
            return 'footer'
        else:
            return 'body'
    
    def _calculate_text_density(self, image_region: np.ndarray) -> float:
        """
        Calculate text density in an image region
        """
        if image_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_region
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate density (ratio of black pixels)
        black_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        
        return black_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _refine_with_text_blocks(
        self,
        zones: List[Zone],
        text_blocks: List[Dict],
        image_shape: Tuple
    ) -> List[Zone]:
        """
        Refine zone boundaries using OCR text blocks
        """
        if not text_blocks:
            return zones
        
        # Group text blocks by zone
        for zone in zones:
            zone_blocks = []
            for block in text_blocks:
                bbox = block.get('bbox', {})
                block_center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
                
                # Check if block center is in zone
                zone_y_start = zone.bbox['y']
                zone_y_end = zone.bbox['y'] + zone.bbox['height']
                
                if zone_y_start <= block_center_y <= zone_y_end:
                    zone_blocks.append(block)
            
            # Adjust zone boundaries to fit text blocks
            if zone_blocks:
                min_y = min(b['bbox']['y'] for b in zone_blocks)
                max_y = max(b['bbox']['y'] + b['bbox']['height'] for b in zone_blocks)
                
                # Update zone boundaries with some padding
                padding = 10
                zone.bbox['y'] = max(0, min_y - padding)
                zone.bbox['height'] = min(image_shape[0], max_y + padding) - zone.bbox['y']
        
        return zones
    
    def _detect_special_zones(self, image: np.ndarray) -> List[Zone]:
        """
        Detect special zones like tables, signatures, logos
        """
        special_zones = []
        
        # Detect table-like regions (high density of horizontal/vertical lines)
        table_regions = self._detect_table_regions(image)
        for region in table_regions:
            special_zones.append(Zone(
                zone_type='table',
                bbox=region,
                confidence=0.7,
                text_density=0.0
            ))
        
        return special_zones
    
    def _detect_table_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect table-like regions using line detection
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert to bounding boxes
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w > 100 and h > 50:
                table_regions.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
        
        return table_regions
    
    def _merge_overlapping_zones(self, zones: List[Zone]) -> List[Zone]:
        """
        Merge zones that overlap significantly
        """
        if len(zones) <= 1:
            return zones
        
        merged = []
        zones_sorted = sorted(zones, key=lambda z: z.bbox['y'])
        
        current = zones_sorted[0]
        
        for next_zone in zones_sorted[1:]:
            # Check overlap
            overlap = self._calculate_overlap(current.bbox, next_zone.bbox)
            
            if overlap > 0.3:  # 30% overlap threshold
                # Merge zones
                current = self._merge_two_zones(current, next_zone)
            else:
                merged.append(current)
                current = next_zone
        
        merged.append(current)
        
        return merged
    
    def _calculate_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        Calculate overlap ratio between two bounding boxes
        """
        # Calculate intersection
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
        y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_two_zones(self, zone1: Zone, zone2: Zone) -> Zone:
        """
        Merge two zones into one
        """
        # Calculate merged bounding box
        x1 = min(zone1.bbox['x'], zone2.bbox['x'])
        y1 = min(zone1.bbox['y'], zone2.bbox['y'])
        x2 = max(zone1.bbox['x'] + zone1.bbox['width'], zone2.bbox['x'] + zone2.bbox['width'])
        y2 = max(zone1.bbox['y'] + zone1.bbox['height'], zone2.bbox['y'] + zone2.bbox['height'])
        
        # Determine zone type (prefer more specific types)
        type_priority = {'table': 3, 'header': 2, 'footer': 2, 'body': 1}
        zone_type = zone1.zone_type if type_priority.get(zone1.zone_type, 0) >= type_priority.get(zone2.zone_type, 0) else zone2.zone_type
        
        return Zone(
            zone_type=zone_type,
            bbox={'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1},
            confidence=(zone1.confidence + zone2.confidence) / 2,
            text_density=(zone1.text_density + zone2.text_density) / 2,
            content=zone1.content + '\n' + zone2.content
        )
    
    def _reclassify_zones(self, zones: List[Zone], page_height: int) -> List[Zone]:
        """
        Re-classify zones based on content and position
        """
        for zone in zones:
            # Skip if already classified as special zone
            if zone.zone_type in ['table', 'signature', 'logo']:
                continue
            
            # Reclassify based on position
            y_center = zone.bbox['y'] + zone.bbox['height'] / 2
            relative_position = y_center / page_height
            
            if relative_position < 0.2:
                zone.zone_type = 'header'
            elif relative_position > 0.85:
                zone.zone_type = 'footer'
            else:
                zone.zone_type = 'body'
        
        return zones
    
    def _add_text_to_zones(
        self,
        zones: List[Zone],
        ocr_result: Dict
    ) -> List[Zone]:
        """
        Add extracted text to zones based on spatial overlap
        """
        text_blocks = ocr_result.get('blocks', [])
        
        for zone in zones:
            zone_texts = []
            
            for block in text_blocks:
                bbox = block.get('bbox', {})
                
                # Check if block overlaps with zone
                if self._bbox_in_zone(bbox, zone.bbox):
                    zone_texts.append(block.get('text', ''))
            
            zone.content = '\n'.join(zone_texts)
        
        return zones
    
    def _bbox_in_zone(self, bbox: Dict, zone_bbox: Dict) -> bool:
        """
        Check if bounding box is within zone
        """
        bbox_center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
        
        zone_y_start = zone_bbox['y']
        zone_y_end = zone_bbox['y'] + zone_bbox['height']
        
        return zone_y_start <= bbox_center_y <= zone_y_end
    
    def visualize_zones(self, image: np.ndarray, zones: List[Zone]) -> np.ndarray:
        """
        Visualize detected zones on image
        
        Returns:
            Image with zones drawn
        """
        vis_image = image.copy()
        
        # Color mapping for zone types
        colors = {
            'header': (255, 0, 0),      # Red
            'body': (0, 255, 0),        # Green
            'footer': (0, 0, 255),      # Blue
            'table': (255, 255, 0),     # Yellow
            'sidebar': (255, 0, 255)    # Magenta
        }
        
        for zone in zones:
            color = colors.get(zone.zone_type, (128, 128, 128))
            bbox = zone.bbox
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color,
                2
            )
            
            # Add label
            label = f"{zone.zone_type} ({zone.confidence:.2f})"
            cv2.putText(
                vis_image,
                label,
                (bbox['x'] + 5, bbox['y'] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return vis_image