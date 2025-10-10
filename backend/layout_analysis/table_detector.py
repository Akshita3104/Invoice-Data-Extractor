"""
Table Detector
Detects and extracts tables from documents using:
- Line-based detection (for bordered tables)
- Contour-based detection
- Text alignment detection (for borderless tables)
- Table structure analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TableCell:
    """Represents a table cell"""
    row: int
    col: int
    bbox: Dict[str, int]
    text: str = ""
    confidence: float = 0.0


@dataclass
class Table:
    """Represents a detected table"""
    bbox: Dict[str, int]
    rows: int
    cols: int
    cells: List[TableCell]
    confidence: float
    has_borders: bool


class TableDetector:
    """
    Detects and extracts tables from document images
    """
    
    def __init__(self, method: str = 'hybrid'):
        """
        Initialize table detector
        
        Args:
            method: Detection method
                - 'line_based': For bordered tables
                - 'contour': For tables with visible structure
                - 'text_alignment': For borderless tables
                - 'hybrid': Combination of methods (recommended)
        """
        self.method = method
        self.min_table_width = 200
        self.min_table_height = 100
        self.min_cells = 4  # Minimum cells to consider it a table
    
    def detect(
        self,
        image: np.ndarray,
        zones: List = None,
        ocr_result: Dict = None
    ) -> List[Table]:
        """
        Detect tables in document
        
        Args:
            image: Input document image
            zones: Optional list of zones from ZoneSegmenter
            ocr_result: Optional OCR result for text-aware detection
            
        Returns:
            List of detected tables
        """
        if self.method == 'line_based':
            tables = self._detect_line_based(image)
        elif self.method == 'contour':
            tables = self._detect_contour_based(image)
        elif self.method == 'text_alignment':
            if ocr_result is None:
                raise ValueError("OCR result required for text_alignment method")
            tables = self._detect_text_alignment(image, ocr_result)
        elif self.method == 'hybrid':
            tables = self._detect_hybrid(image, ocr_result)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Add text to cells if OCR result available
        if ocr_result:
            tables = self._add_text_to_cells(tables, ocr_result)
        
        return tables
    
    def _detect_line_based(self, image: np.ndarray) -> List[Table]:
        """
        Detect tables using line detection (for bordered tables)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines to find table grid
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours of tables
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size
            if w < self.min_table_width or h < self.min_table_height:
                continue
            
            # Extract table region
            table_region = image[y:y+h, x:x+w]
            
            # Analyze table structure
            rows, cols = self._analyze_table_structure(
                table_region,
                horizontal_lines[y:y+h, x:x+w],
                vertical_lines[y:y+h, x:x+w]
            )
            
            if rows * cols >= self.min_cells:
                # Generate cell information
                cells = self._generate_cells(x, y, w, h, rows, cols)
                
                tables.append(Table(
                    bbox={'x': x, 'y': y, 'width': w, 'height': h},
                    rows=rows,
                    cols=cols,
                    cells=cells,
                    confidence=0.9,
                    has_borders=True
                ))
        
        return tables
    
    def _detect_contour_based(self, image: np.ndarray) -> List[Table]:
        """
        Detect tables using contour analysis
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangular contours that might be tables
        tables = []
        for i, contour in enumerate(contours):
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w < self.min_table_width or h < self.min_table_height:
                    continue
                
                # Check if it has child contours (cells)
                if hierarchy is not None:
                    children = np.where(hierarchy[0][:, 3] == i)[0]
                    
                    if len(children) >= self.min_cells:
                        # Estimate rows and cols from children
                        rows, cols = self._estimate_grid_from_children(
                            children, contours, x, y, w, h
                        )
                        
                        cells = self._generate_cells(x, y, w, h, rows, cols)
                        
                        tables.append(Table(
                            bbox={'x': x, 'y': y, 'width': w, 'height': h},
                            rows=rows,
                            cols=cols,
                            cells=cells,
                            confidence=0.8,
                            has_borders=True
                        ))
        
        return tables
    
    def _detect_text_alignment(
        self,
        image: np.ndarray,
        ocr_result: Dict
    ) -> List[Table]:
        """
        Detect tables using text alignment (for borderless tables)
        """
        words = ocr_result.get('words', [])
        if not words:
            return []
        
        # Group words by Y coordinate (rows)
        rows = self._group_words_into_rows(words, threshold=10)
        
        # Detect aligned columns
        columns = self._detect_aligned_columns(rows, threshold=15)
        
        if len(rows) < 2 or len(columns) < 2:
            return []
        
        # Create table from aligned text
        table_bbox = self._calculate_table_bbox(rows, columns)
        
        cells = []
        for row_idx, row_words in enumerate(rows):
            for col_idx, col_x in enumerate(columns):
                # Find word in this cell
                cell_word = self._find_word_at_position(row_words, col_x, threshold=15)
                
                if cell_word:
                    bbox = cell_word.get('bbox', {})
                    cells.append(TableCell(
                        row=row_idx,
                        col=col_idx,
                        bbox=bbox,
                        text=cell_word.get('text', ''),
                        confidence=cell_word.get('confidence', 0.0)
                    ))
        
        if len(cells) >= self.min_cells:
            return [Table(
                bbox=table_bbox,
                rows=len(rows),
                cols=len(columns),
                cells=cells,
                confidence=0.7,
                has_borders=False
            )]
        
        return []
    
    def _detect_hybrid(
        self,
        image: np.ndarray,
        ocr_result: Dict = None
    ) -> List[Table]:
        """
        Hybrid detection combining multiple methods
        """
        tables = []
        
        # Try line-based detection first (most reliable for bordered tables)
        line_tables = self._detect_line_based(image)
        tables.extend(line_tables)
        
        # Try contour-based detection
        contour_tables = self._detect_contour_based(image)
        
        # Add non-overlapping contour tables
        for ctable in contour_tables:
            if not self._overlaps_with_existing(ctable, tables):
                tables.append(ctable)
        
        # Try text alignment for borderless tables
        if ocr_result:
            text_tables = self._detect_text_alignment(image, ocr_result)
            
            for ttable in text_tables:
                if not self._overlaps_with_existing(ttable, tables):
                    tables.append(ttable)
        
        return tables
    
    def _analyze_table_structure(
        self,
        table_image: np.ndarray,
        horizontal_lines: np.ndarray,
        vertical_lines: np.ndarray
    ) -> Tuple[int, int]:
        """
        Analyze table structure to count rows and columns
        """
        # Count horizontal lines (rows)
        h_projection = np.sum(horizontal_lines, axis=1)
        h_peaks = self._find_peaks(h_projection, min_distance=10)
        rows = max(len(h_peaks) - 1, 1)
        
        # Count vertical lines (columns)
        v_projection = np.sum(vertical_lines, axis=0)
        v_peaks = self._find_peaks(v_projection, min_distance=10)
        cols = max(len(v_peaks) - 1, 1)
        
        return rows, cols
    
    def _find_peaks(
        self,
        projection: np.ndarray,
        min_distance: int = 10
    ) -> List[int]:
        """
        Find peaks in projection profile
        """
        peaks = []
        threshold = np.max(projection) * 0.3
        
        i = 0
        while i < len(projection):
            if projection[i] > threshold:
                # Found a peak, find its maximum
                peak_start = i
                peak_max = i
                peak_max_val = projection[i]
                
                while i < len(projection) and projection[i] > threshold:
                    if projection[i] > peak_max_val:
                        peak_max = i
                        peak_max_val = projection[i]
                    i += 1
                
                peaks.append(peak_max)
                i += min_distance  # Skip minimum distance
            else:
                i += 1
        
        return peaks
    
    def _generate_cells(
        self,
        table_x: int,
        table_y: int,
        table_w: int,
        table_h: int,
        rows: int,
        cols: int
    ) -> List[TableCell]:
        """
        Generate cell information for a table
        """
        cells = []
        cell_height = table_h / rows
        cell_width = table_w / cols
        
        for row in range(rows):
            for col in range(cols):
                cell_x = table_x + int(col * cell_width)
                cell_y = table_y + int(row * cell_height)
                
                cells.append(TableCell(
                    row=row,
                    col=col,
                    bbox={
                        'x': cell_x,
                        'y': cell_y,
                        'width': int(cell_width),
                        'height': int(cell_height)
                    }
                ))
        
        return cells
    
    def _estimate_grid_from_children(
        self,
        children_indices: np.ndarray,
        contours: List,
        parent_x: int,
        parent_y: int,
        parent_w: int,
        parent_h: int
    ) -> Tuple[int, int]:
        """
        Estimate grid dimensions from child contours
        """
        # Get child bounding boxes
        child_boxes = []
        for idx in children_indices:
            x, y, w, h = cv2.boundingRect(contours[idx])
            child_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
        
        # Group by Y coordinate to find rows
        y_coords = sorted([box['y'] for box in child_boxes])
        rows = 1
        prev_y = y_coords[0]
        for y in y_coords[1:]:
            if y - prev_y > 20:  # Threshold for new row
                rows += 1
                prev_y = y
        
        # Group by X coordinate to find columns
        x_coords = sorted([box['x'] for box in child_boxes])
        cols = 1
        prev_x = x_coords[0]
        for x in x_coords[1:]:
            if x - prev_x > 20:  # Threshold for new column
                cols += 1
                prev_x = x
        
        return max(rows, 1), max(cols, 1)
    
    def _group_words_into_rows(
        self,
        words: List[Dict],
        threshold: int = 10
    ) -> List[List[Dict]]:
        """
        Group words into rows based on Y coordinate
        """
        if not words:
            return []
        
        # Sort by Y coordinate
        sorted_words = sorted(words, key=lambda w: w.get('bbox', {}).get('y', 0))
        
        rows = []
        current_row = [sorted_words[0]]
        current_y = sorted_words[0].get('bbox', {}).get('y', 0)
        
        for word in sorted_words[1:]:
            word_y = word.get('bbox', {}).get('y', 0)
            
            if abs(word_y - current_y) <= threshold:
                current_row.append(word)
            else:
                rows.append(current_row)
                current_row = [word]
                current_y = word_y
        
        rows.append(current_row)
        
        return rows
    
    def _detect_aligned_columns(
        self,
        rows: List[List[Dict]],
        threshold: int = 15
    ) -> List[int]:
        """
        Detect aligned columns from rows of words
        """
        # Collect all X coordinates
        x_coords = []
        for row in rows:
            for word in row:
                x = word.get('bbox', {}).get('x', 0)
                x_coords.append(x)
        
        # Cluster X coordinates
        x_coords = sorted(x_coords)
        columns = []
        
        if x_coords:
            current_col = x_coords[0]
            col_group = [x_coords[0]]
            
            for x in x_coords[1:]:
                if x - current_col <= threshold:
                    col_group.append(x)
                else:
                    # Save average of group as column position
                    columns.append(int(np.mean(col_group)))
                    col_group = [x]
                    current_col = x
            
            columns.append(int(np.mean(col_group)))
        
        return columns
    
    def _calculate_table_bbox(
        self,
        rows: List[List[Dict]],
        columns: List[int]
    ) -> Dict[str, int]:
        """
        Calculate bounding box for table from rows and columns
        """
        all_words = [word for row in rows for word in row]
        
        if not all_words:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        # Find min/max coordinates
        min_x = min(w.get('bbox', {}).get('x', 0) for w in all_words)
        min_y = min(w.get('bbox', {}).get('y', 0) for w in all_words)
        
        max_x = max(w.get('bbox', {}).get('x', 0) + w.get('bbox', {}).get('width', 0) 
                   for w in all_words)
        max_y = max(w.get('bbox', {}).get('y', 0) + w.get('bbox', {}).get('height', 0) 
                   for w in all_words)
        
        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def _find_word_at_position(
        self,
        words: List[Dict],
        target_x: int,
        threshold: int = 15
    ) -> Optional[Dict]:
        """
        Find word closest to target X position
        """
        closest_word = None
        min_distance = float('inf')
        
        for word in words:
            word_x = word.get('bbox', {}).get('x', 0)
            distance = abs(word_x - target_x)
            
            if distance < min_distance and distance <= threshold:
                min_distance = distance
                closest_word = word
        
        return closest_word
    
    def _overlaps_with_existing(
        self,
        new_table: Table,
        existing_tables: List[Table]
    ) -> bool:
        """
        Check if new table overlaps significantly with existing tables
        """
        for existing in existing_tables:
            overlap = self._calculate_bbox_overlap(new_table.bbox, existing.bbox)
            if overlap > 0.5:  # 50% overlap threshold
                return True
        
        return False
    
    def _calculate_bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
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
    
    def _add_text_to_cells(
        self,
        tables: List[Table],
        ocr_result: Dict
    ) -> List[Table]:
        """
        Add text content to table cells using OCR result
        """
        words = ocr_result.get('words', [])
        
        for table in tables:
            for cell in table.cells:
                # Find words that overlap with cell
                cell_words = []
                
                for word in words:
                    word_bbox = word.get('bbox', {})
                    
                    if self._point_in_bbox(
                        word_bbox.get('x', 0) + word_bbox.get('width', 0) // 2,
                        word_bbox.get('y', 0) + word_bbox.get('height', 0) // 2,
                        cell.bbox
                    ):
                        cell_words.append(word.get('text', ''))
                
                cell.text = ' '.join(cell_words)
        
        return tables
    
    def _point_in_bbox(self, x: int, y: int, bbox: Dict) -> bool:
        """
        Check if point is inside bounding box
        """
        return (bbox['x'] <= x <= bbox['x'] + bbox['width'] and
                bbox['y'] <= y <= bbox['y'] + bbox['height'])
    
    def extract_table_data(self, table: Table) -> List[List[str]]:
        """
        Extract table data as 2D array
        
        Returns:
            List of rows, each row is a list of cell texts
        """
        # Initialize 2D array
        data = [['' for _ in range(table.cols)] for _ in range(table.rows)]
        
        # Fill with cell data
        for cell in table.cells:
            if 0 <= cell.row < table.rows and 0 <= cell.col < table.cols:
                data[cell.row][cell.col] = cell.text
        
        return data
    
    def visualize_tables(
        self,
        image: np.ndarray,
        tables: List[Table]
    ) -> np.ndarray:
        """
        Visualize detected tables on image
        """
        vis_image = image.copy()
        
        for i, table in enumerate(tables):
            bbox = table.bbox
            
            # Draw table boundary
            color = (0, 255, 0) if table.has_borders else (255, 165, 0)
            cv2.rectangle(
                vis_image,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color,
                3
            )
            
            # Add label
            label = f"Table {i+1} ({table.rows}x{table.cols})"
            cv2.putText(
                vis_image,
                label,
                (bbox['x'] + 5, bbox['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            # Draw cells
            for cell in table.cells:
                cell_bbox = cell.bbox
                cv2.rectangle(
                    vis_image,
                    (cell_bbox['x'], cell_bbox['y']),
                    (cell_bbox['x'] + cell_bbox['width'], 
                     cell_bbox['y'] + cell_bbox['height']),
                    (200, 200, 200),
                    1
                )
        
        return vis_image