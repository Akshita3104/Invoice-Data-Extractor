"""
Reading Order Detector
Determines the logical reading order of text blocks in a document
Handles:
- Single column documents
- Multi-column documents
- Mixed layouts
- Tables and special elements
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class TextBlock:
    """Represents a text block with reading order"""
    bbox: Dict[str, int]
    text: str
    order: int
    confidence: float
    block_type: str = 'text'  # 'text', 'heading', 'table', 'caption'


class ReadingOrderDetector:
    """
    Determines reading order of text blocks in a document
    """
    
    def __init__(self):
        self.column_threshold = 0.3  # Relative X overlap for column detection
        self.heading_height_ratio = 1.5  # Height ratio for heading detection
    
    def detect_order(
        self,
        ocr_result: Dict,
        zones: List = None,
        image_shape: Tuple = None
    ) -> List[TextBlock]:
        """
        Detect reading order of text blocks
        
        Args:
            ocr_result: OCR result with blocks/lines
            zones: Optional zones from ZoneSegmenter
            image_shape: Optional image shape (height, width)
            
        Returns:
            List of TextBlocks sorted by reading order
        """
        # Extract text blocks from OCR result
        blocks = self._extract_blocks(ocr_result)
        
        if not blocks:
            return []
        
        # Detect document layout (single/multi-column)
        layout_type = self._detect_layout(blocks, image_shape)
        
        # Sort blocks based on layout type
        if layout_type == 'single_column':
            ordered_blocks = self._order_single_column(blocks)
        elif layout_type == 'two_column':
            ordered_blocks = self._order_multi_column(blocks, n_columns=2)
        elif layout_type == 'multi_column':
            ordered_blocks = self._order_multi_column(blocks, n_columns=3)
        else:
            ordered_blocks = self._order_complex(blocks)
        
        # Assign order numbers
        for i, block in enumerate(ordered_blocks):
            block.order = i + 1
        
        # Classify block types
        ordered_blocks = self._classify_block_types(ordered_blocks)
        
        return ordered_blocks
    
    def _extract_blocks(self, ocr_result: Dict) -> List[TextBlock]:
        """
        Extract text blocks from OCR result
        """
        blocks = []
        
        # Try to get blocks from OCR result
        ocr_blocks = ocr_result.get('blocks', [])
        
        if ocr_blocks:
            for block_data in ocr_blocks:
                blocks.append(TextBlock(
                    bbox=block_data.get('bbox', {}),
                    text=block_data.get('text', ''),
                    order=0,
                    confidence=block_data.get('confidence', 0.8)
                ))
        else:
            # Fall back to lines
            lines = ocr_result.get('lines', [])
            for line_data in lines:
                blocks.append(TextBlock(
                    bbox=line_data.get('bbox', {}),
                    text=line_data.get('text', ''),
                    order=0,
                    confidence=line_data.get('confidence', 0.8)
                ))
        
        return blocks
    
    def _detect_layout(
        self,
        blocks: List[TextBlock],
        image_shape: Tuple = None
    ) -> str:
        """
        Detect document layout type
        
        Returns:
            Layout type: 'single_column', 'two_column', 'multi_column', 'complex'
        """
        if not blocks:
            return 'single_column'
        
        # Get X coordinates of blocks
        x_coords = [block.bbox.get('x', 0) for block in blocks]
        
        # Cluster X coordinates to find columns
        columns = self._cluster_coordinates(x_coords, threshold=50)
        
        n_columns = len(columns)
        
        if n_columns <= 1:
            return 'single_column'
        elif n_columns == 2:
            return 'two_column'
        elif n_columns >= 3:
            return 'multi_column'
        else:
            return 'complex'
    
    def _cluster_coordinates(
        self,
        coords: List[int],
        threshold: int = 50
    ) -> List[int]:
        """
        Cluster coordinates to find distinct positions (e.g., columns)
        """
        if not coords:
            return []
        
        sorted_coords = sorted(coords)
        clusters = []
        current_cluster = [sorted_coords[0]]
        
        for coord in sorted_coords[1:]:
            if coord - current_cluster[-1] <= threshold:
                current_cluster.append(coord)
            else:
                # Save cluster center
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [coord]
        
        # Save last cluster
        clusters.append(int(np.mean(current_cluster)))
        
        return clusters
    
    def _order_single_column(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Order blocks in single column layout (simple top-to-bottom)
        """
        return sorted(blocks, key=lambda b: b.bbox.get('y', 0))
    
    def _order_multi_column(
        self,
        blocks: List[TextBlock],
        n_columns: int = 2
    ) -> List[TextBlock]:
        """
        Order blocks in multi-column layout
        """
        if not blocks:
            return []
        
        # Get page width
        max_x = max(b.bbox.get('x', 0) + b.bbox.get('width', 0) for b in blocks)
        
        # Divide into columns based on X position
        column_width = max_x / n_columns
        
        # Assign blocks to columns
        columns = [[] for _ in range(n_columns)]
        
        for block in blocks:
            block_center_x = block.bbox.get('x', 0) + block.bbox.get('width', 0) / 2
            column_idx = min(int(block_center_x / column_width), n_columns - 1)
            columns[column_idx].append(block)
        
        # Sort each column by Y coordinate
        for col in columns:
            col.sort(key=lambda b: b.bbox.get('y', 0))
        
        # Combine columns in reading order (left to right)
        ordered = []
        for col in columns:
            ordered.extend(col)
        
        return ordered
    
    def _order_complex(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Order blocks in complex layout using XY-cut algorithm
        """
        if not blocks:
            return []
        
        # Get bounding box of all blocks
        min_x = min(b.bbox.get('x', 0) for b in blocks)
        min_y = min(b.bbox.get('y', 0) for b in blocks)
        max_x = max(b.bbox.get('x', 0) + b.bbox.get('width', 0) for b in blocks)
        max_y = max(b.bbox.get('y', 0) + b.bbox.get('height', 0) for b in blocks)
        
        # Apply XY-cut recursively
        ordered = self._xy_cut(blocks, min_x, min_y, max_x, max_y)
        
        return ordered
    
    def _xy_cut(
        self,
        blocks: List[TextBlock],
        x1: int, y1: int,
        x2: int, y2: int,
        direction: str = 'horizontal'
    ) -> List[TextBlock]:
        """
        Recursive XY-cut algorithm for reading order
        """
        if len(blocks) <= 1:
            return blocks
        
        if direction == 'horizontal':
            # Find horizontal cut (Y axis)
            cut_y = self._find_horizontal_cut(blocks, y1, y2)
            
            if cut_y is not None:
                # Split blocks into top and bottom
                top_blocks = [b for b in blocks if b.bbox.get('y', 0) + b.bbox.get('height', 0) <= cut_y]
                bottom_blocks = [b for b in blocks if b.bbox.get('y', 0) >= cut_y]
                
                # Recursively process with vertical direction
                top_ordered = self._xy_cut(top_blocks, x1, y1, x2, cut_y, 'vertical')
                bottom_ordered = self._xy_cut(bottom_blocks, x1, cut_y, x2, y2, 'vertical')
                
                return top_ordered + bottom_ordered
            else:
                # No cut found, try vertical
                return self._xy_cut(blocks, x1, y1, x2, y2, 'vertical')
        
        else:  # vertical
            # Find vertical cut (X axis)
            cut_x = self._find_vertical_cut(blocks, x1, x2)
            
            if cut_x is not None:
                # Split blocks into left and right
                left_blocks = [b for b in blocks if b.bbox.get('x', 0) + b.bbox.get('width', 0) <= cut_x]
                right_blocks = [b for b in blocks if b.bbox.get('x', 0) >= cut_x]
                
                # Recursively process with horizontal direction
                left_ordered = self._xy_cut(left_blocks, x1, y1, cut_x, y2, 'horizontal')
                right_ordered = self._xy_cut(right_blocks, cut_x, y1, x2, y2, 'horizontal')
                
                return left_ordered + right_ordered
            else:
                # No cut found, sort by position
                return sorted(blocks, key=lambda b: (b.bbox.get('y', 0), b.bbox.get('x', 0)))
    
    def _find_horizontal_cut(
        self,
        blocks: List[TextBlock],
        y1: int, y2: int
    ) -> int:
        """
        Find horizontal cutting line with no blocks
        """
        # Create projection
        projection = np.zeros(y2 - y1)
        
        for block in blocks:
            block_y1 = block.bbox.get('y', 0) - y1
            block_y2 = block_y1 + block.bbox.get('height', 0)
            
            block_y1 = max(0, block_y1)
            block_y2 = min(len(projection), block_y2)
            
            projection[block_y1:block_y2] = 1
        
        # Find largest gap
        gaps = self._find_gaps(projection)
        
        if gaps:
            # Return middle of largest gap
            largest_gap = max(gaps, key=lambda g: g[1] - g[0])
            return y1 + (largest_gap[0] + largest_gap[1]) // 2
        
        return None
    
    def _find_vertical_cut(
        self,
        blocks: List[TextBlock],
        x1: int, x2: int
    ) -> int:
        """
        Find vertical cutting line with no blocks
        """
        # Create projection
        projection = np.zeros(x2 - x1)
        
        for block in blocks:
            block_x1 = block.bbox.get('x', 0) - x1
            block_x2 = block_x1 + block.bbox.get('width', 0)
            
            block_x1 = max(0, block_x1)
            block_x2 = min(len(projection), block_x2)
            
            projection[block_x1:block_x2] = 1
        
        # Find largest gap
        gaps = self._find_gaps(projection)
        
        if gaps:
            # Return middle of largest gap
            largest_gap = max(gaps, key=lambda g: g[1] - g[0])
            return x1 + (largest_gap[0] + largest_gap[1]) // 2
        
        return None
    
    def _find_gaps(self, projection: np.ndarray, min_gap: int = 10) -> List[Tuple[int, int]]:
        """
        Find gaps (zeros) in projection
        """
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(projection):
            if val == 0 and not in_gap:
                gap_start = i
                in_gap = True
            elif val > 0 and in_gap:
                if i - gap_start >= min_gap:
                    gaps.append((gap_start, i))
                in_gap = False
        
        # Handle last gap
        if in_gap and len(projection) - gap_start >= min_gap:
            gaps.append((gap_start, len(projection)))
        
        return gaps
    
    def _classify_block_types(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Classify blocks into types (heading, text, table, etc.)
        """
        if not blocks:
            return blocks
        
        # Calculate average height
        heights = [b.bbox.get('height', 0) for b in blocks if b.bbox.get('height', 0) > 0]
        avg_height = np.mean(heights) if heights else 20
        
        for block in blocks:
            block_height = block.bbox.get('height', 0)
            text = block.text.strip()
            
            # Detect heading (larger text, short, often capitalized)
            if (block_height > avg_height * self.heading_height_ratio and
                len(text) < 100 and
                text.isupper()):
                block.block_type = 'heading'
            
            # Detect caption (smaller text, starts with Fig/Table)
            elif (block_height < avg_height * 0.8 and
                  (text.startswith('Fig') or text.startswith('Table') or text.startswith('Caption'))):
                block.block_type = 'caption'
            
            else:
                block.block_type = 'text'
        
        return blocks
    
    def visualize_order(
        self,
        image: np.ndarray,
        blocks: List[TextBlock]
    ) -> np.ndarray:
        """
        Visualize reading order on image
        """
        vis_image = image.copy()
        
        for block in blocks:
            bbox = block.bbox
            
            # Color based on block type
            if block.block_type == 'heading':
                color = (255, 0, 0)  # Red
            elif block.block_type == 'caption':
                color = (255, 165, 0)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color,
                2
            )
            
            # Add order number
            cv2.putText(
                vis_image,
                str(block.order),
                (bbox['x'] + 5, bbox['y'] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
        
        return vis_image