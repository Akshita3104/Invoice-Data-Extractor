"""
CascadeTabNet Model
Advanced table detection using Cascade Mask R-CNN
Based on: https://github.com/DevashishPrasad/CascadeTabNet

Detects:
- Bordered tables
- Borderless tables  
- Table cells
- Table structure (rows and columns)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CascadeTabNetModel:
    """
    CascadeTabNet model for advanced table detection and structure recognition
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize CascadeTabNet model
        
        Args:
            model_path: Path to pre-trained weights
            confidence_threshold: Minimum confidence for detections
            use_gpu: Use GPU if available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and torchvision required for CascadeTabNet. "
                "Install with: pip install torch torchvision"
            )
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Load weights if provided
        if model_path:
            self._load_weights(model_path)
        
        self.model.eval()
        
        # Class labels
        self.classes = {
            0: 'background',
            1: 'bordered_table',
            2: 'borderless_table',
            3: 'table_cell'
        }
        
        print(f"CascadeTabNet model initialized on {self.device}")
    
    def _build_model(self, num_classes: int = 4):
        """
        Build Mask R-CNN model for table detection
        """
        # Load pre-trained Mask R-CNN model
        model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
        return model
    
    def _load_weights(self, model_path: str):
        """
        Load pre-trained weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}: {e}")
    
    def detect_tables(
        self,
        image: np.ndarray,
        detect_cells: bool = True
    ) -> Dict:
        """
        Detect tables and optionally table cells
        
        Args:
            image: Input image (RGB)
            detect_cells: Whether to detect individual cells
            
        Returns:
            Dictionary containing:
                - tables: List of detected tables
                - cells: List of detected cells (if detect_cells=True)
        """
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)[0]
        
        # Post-process predictions
        result = self._postprocess(predictions, image.shape)
        
        # Separate tables and cells
        tables = []
        cells = []
        
        for detection in result['detections']:
            class_name = detection['class_name']
            
            if class_name in ['bordered_table', 'borderless_table']:
                tables.append(detection)
            elif class_name == 'table_cell' and detect_cells:
                cells.append(detection)
        
        # Group cells into tables
        if cells:
            tables = self._assign_cells_to_tables(tables, cells)
        
        return {
            'tables': tables,
            'cells': cells
        }
    
    def _preprocess(self, image: np.ndarray) -> List[torch.Tensor]:
        """
        Preprocess image for model input
        """
        # Convert to float and normalize
        img_float = image.astype(np.float32) / 255.0
        
        # Convert to tensor (CHW format)
        tensor = torch.from_numpy(img_float).permute(2, 0, 1)
        
        return [tensor.to(self.device)]
    
    def _postprocess(
        self,
        predictions: Dict,
        image_shape: Tuple
    ) -> Dict:
        """
        Post-process model predictions
        """
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()
        
        detections = []
        
        for i in range(len(boxes)):
            if scores[i] < self.confidence_threshold:
                continue
            
            box = boxes[i]
            label = labels[i]
            score = scores[i]
            mask = masks[i, 0]
            
            # Convert box to integer coordinates
            x1, y1, x2, y2 = map(int, box)
            
            detection = {
                'bbox': {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                },
                'class_id': int(label),
                'class_name': self.classes.get(int(label), 'unknown'),
                'confidence': float(score),
                'mask': (mask > 0.5).astype(np.uint8)
            }
            
            detections.append(detection)
        
        return {'detections': detections}
    
    def _assign_cells_to_tables(
        self,
        tables: List[Dict],
        cells: List[Dict]
    ) -> List[Dict]:
        """
        Assign detected cells to their parent tables
        """
        for table in tables:
            table['cells'] = []
            table_bbox = table['bbox']
            
            for cell in cells:
                cell_bbox = cell['bbox']
                
                # Check if cell is inside table
                if self._bbox_contains(table_bbox, cell_bbox):
                    # Calculate cell position in table grid
                    cell_position = self._calculate_cell_position(
                        cell_bbox,
                        table_bbox,
                        table.get('cells', [])
                    )
                    
                    cell['row'] = cell_position['row']
                    cell['col'] = cell_position['col']
                    
                    table['cells'].append(cell)
            
            # Estimate table structure
            if table['cells']:
                table['rows'], table['cols'] = self._estimate_table_structure(
                    table['cells']
                )
        
        return tables
    
    def _bbox_contains(self, parent_bbox: Dict, child_bbox: Dict) -> bool:
        """
        Check if child bounding box is inside parent bounding box
        """
        px1, py1 = parent_bbox['x'], parent_bbox['y']
        px2 = px1 + parent_bbox['width']
        py2 = py1 + parent_bbox['height']
        
        cx1, cy1 = child_bbox['x'], child_bbox['y']
        cx2 = cx1 + child_bbox['width']
        cy2 = cy1 + child_bbox['height']
        
        # Check if child center is in parent
        center_x = (cx1 + cx2) / 2
        center_y = (cy1 + cy2) / 2
        
        return px1 <= center_x <= px2 and py1 <= center_y <= py2
    
    def _calculate_cell_position(
        self,
        cell_bbox: Dict,
        table_bbox: Dict,
        existing_cells: List[Dict]
    ) -> Dict:
        """
        Calculate cell position (row, col) in table grid
        """
        # Get cell center relative to table
        cell_center_x = cell_bbox['x'] + cell_bbox['width'] / 2
        cell_center_y = cell_bbox['y'] + cell_bbox['height'] / 2
        
        # Get all Y coordinates of existing cells
        y_coords = sorted(set(
            c['bbox']['y'] for c in existing_cells
        ))
        
        # Find row
        row = 0
        for i, y in enumerate(y_coords):
            if cell_center_y >= y:
                row = i
        
        # Get all X coordinates of cells in same row
        x_coords = sorted(set(
            c['bbox']['x'] for c in existing_cells
            if abs(c['bbox']['y'] - cell_bbox['y']) < 20
        ))
        
        # Find column
        col = 0
        for i, x in enumerate(x_coords):
            if cell_center_x >= x:
                col = i
        
        return {'row': row, 'col': col}
    
    def _estimate_table_structure(self, cells: List[Dict]) -> Tuple[int, int]:
        """
        Estimate number of rows and columns from detected cells
        """
        if not cells:
            return 0, 0
        
        # Get unique row and column indices
        rows = set(cell.get('row', 0) for cell in cells)
        cols = set(cell.get('col', 0) for cell in cells)
        
        return len(rows), len(cols)
    
    def extract_table_structure(self, table: Dict) -> List[List[str]]:
        """
        Extract table as 2D array structure
        
        Args:
            table: Table detection with cells
            
        Returns:
            2D list representing table structure
        """
        if 'cells' not in table or not table['cells']:
            return []
        
        rows = table.get('rows', 0)
        cols = table.get('cols', 0)
        
        if rows == 0 or cols == 0:
            return []
        
        # Initialize 2D array
        structure = [['' for _ in range(cols)] for _ in range(rows)]
        
        # Fill with cell content
        for cell in table['cells']:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            
            if 0 <= row < rows and 0 <= col < cols:
                structure[row][col] = cell.get('text', '')
        
        return structure
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: Dict,
        show_cells: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on image
        """
        vis_image = image.copy()
        
        # Draw tables
        for table in detections['tables']:
            bbox = table['bbox']
            class_name = table['class_name']
            confidence = table['confidence']
            
            # Color based on table type
            color = (0, 255, 0) if class_name == 'bordered_table' else (255, 165, 0)
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color,
                3
            )
            
            # Draw label
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(
                vis_image,
                label,
                (bbox['x'], bbox['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw cells
            if show_cells and 'cells' in table:
                for cell in table['cells']:
                    cell_bbox = cell['bbox']
                    cv2.rectangle(
                        vis_image,
                        (cell_bbox['x'], cell_bbox['y']),
                        (cell_bbox['x'] + cell_bbox['width'],
                         cell_bbox['y'] + cell_bbox['height']),
                        (200, 200, 200),
                        1
                    )
        
        return vis_image