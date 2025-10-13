"""
Field Detector
Uses YOLO-like object detection for field detection in documents
Detects key fields like invoice number, date, amounts, etc.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FieldDetector:
    """
    Detects key fields in document images
    Uses bounding box detection approach similar to YOLO
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize field detector
        
        Args:
            model_path: Path to pre-trained model weights
            confidence_threshold: Minimum confidence for detections
            use_gpu: Use GPU if available
        """
        self.confidence_threshold = confidence_threshold
        
        # Field types we want to detect
        self.field_types = [
            'invoice_number',
            'date',
            'company_name',
            'amount',
            'total',
            'tax',
            'address',
            'phone',
            'email',
            'item_description',
            'quantity',
            'price'
        ]
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            self.use_deep_detection = True
            
            # Initialize model
            self.model = SimpleFieldDetector(num_classes=len(self.field_types))
            self.model.to(self.device)
            self.model.eval()
            
            if model_path:
                self._load_weights(model_path)
            
            print(f"Field detector initialized on {self.device}")
        else:
            self.use_deep_detection = False
            print("Warning: PyTorch not available. Using rule-based field detection.")
    
    def detect_fields(
        self,
        image: np.ndarray,
        ocr_result: Dict
    ) -> List[Dict]:
        """
        Detect fields in document
        
        Args:
            image: Document image
            ocr_result: OCR result with text and bounding boxes
            
        Returns:
            List of detected fields with bounding boxes and types
        """
        if self.use_deep_detection:
            return self._deep_detection(image, ocr_result)
        else:
            return self._rule_based_detection(ocr_result)
    
    def _deep_detection(
        self,
        image: np.ndarray,
        ocr_result: Dict
    ) -> List[Dict]:
        """
        Deep learning-based field detection
        """
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Run detection
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Post-process predictions
        fields = self._postprocess_predictions(predictions, image.shape, ocr_result)
        
        return fields
    
    def _rule_based_detection(self, ocr_result: Dict) -> List[Dict]:
        """
        Rule-based field detection using text patterns
        """
        import re
        
        fields = []
        
        # Get all text elements
        elements = ocr_result.get('words', []) or ocr_result.get('lines', [])
        
        for element in elements:
            text = element.get('text', '').strip()
            bbox = element.get('bbox', {})
            
            if not text:
                continue
            
            text_lower = text.lower()
            
            # Detect field type based on patterns
            field_type = None
            confidence = 0.0
            
            # Invoice number
            if re.search(r'inv(oice)?\s*(no|number|#)', text_lower):
                field_type = 'invoice_number'
                confidence = 0.8
            
            # Date
            elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                field_type = 'date'
                confidence = 0.9
            
            # Company name (has Ltd, Pvt, etc.)
            elif re.search(r'(pvt|ltd|llp|inc|corp|company)', text_lower):
                field_type = 'company_name'
                confidence = 0.7
            
            # Amount/Price
            elif re.search(r'[₹$€£]\s*[\d,]+\.?\d*|^\d+[,\d]*\.\d{2}$', text):
                # Check context to determine if it's amount, total, or price
                if 'total' in text_lower:
                    field_type = 'total'
                    confidence = 0.9
                elif 'tax' in text_lower or 'gst' in text_lower:
                    field_type = 'tax'
                    confidence = 0.8
                else:
                    field_type = 'amount'
                    confidence = 0.6
            
            # Quantity
            elif re.search(r'(qty|quantity|count)', text_lower) or (text.isdigit() and len(text) <= 3):
                field_type = 'quantity'
                confidence = 0.6
            
            # Email
            elif re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
                field_type = 'email'
                confidence = 0.95
            
            # Phone
            elif re.search(r'\+?\d{10,15}', text.replace(' ', '').replace('-', '')):
                field_type = 'phone'
                confidence = 0.85
            
            # Add field if detected
            if field_type:
                fields.append({
                    'field_type': field_type,
                    'text': text,
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        return fields
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        """
        # Resize to model input size
        resized = cv2.resize(image, (640, 640))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _postprocess_predictions(
        self,
        predictions: torch.Tensor,
        image_shape: Tuple,
        ocr_result: Dict
    ) -> List[Dict]:
        """
        Post-process model predictions
        """
        # This is a placeholder - actual implementation would depend on model architecture
        fields = []
        
        # Combine with OCR results
        elements = ocr_result.get('words', []) or ocr_result.get('lines', [])
        
        for element in elements:
            # Simple assignment based on text pattern
            text = element.get('text', '')
            bbox = element.get('bbox', {})
            
            # Use rule-based classification for now
            field_type = self._classify_field(text)
            
            if field_type:
                fields.append({
                    'field_type': field_type,
                    'text': text,
                    'bbox': bbox,
                    'confidence': 0.7
                })
        
        return fields
    
    def _classify_field(self, text: str) -> Optional[str]:
        """
        Classify text into field type
        """
        import re
        
        text_lower = text.lower().strip()
        
        # Invoice number
        if 'invoice' in text_lower or 'inv' in text_lower:
            return 'invoice_number'
        
        # Date
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
            return 'date'
        
        # Company
        if re.search(r'(pvt|ltd|llp|inc|corp)', text_lower):
            return 'company_name'
        
        # Total
        if 'total' in text_lower and re.search(r'\d+', text):
            return 'total'
        
        # Amount
        if re.search(r'[₹$€£]\s*[\d,]+', text):
            return 'amount'
        
        return None
    
    def _load_weights(self, model_path: str):
        """
        Load pre-trained weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    def visualize_detections(
        self,
        image: np.ndarray,
        fields: List[Dict]
    ) -> np.ndarray:
        """
        Visualize detected fields on image
        """
        vis_image = image.copy()
        
        # Color map for field types
        colors = {
            'invoice_number': (255, 0, 0),
            'date': (0, 255, 0),
            'company_name': (0, 0, 255),
            'amount': (255, 255, 0),
            'total': (255, 0, 255),
            'tax': (0, 255, 255),
            'quantity': (128, 128, 0),
            'price': (128, 0, 128)
        }
        
        for field in fields:
            bbox = field['bbox']
            field_type = field['field_type']
            confidence = field.get('confidence', 0.0)
            
            x, y = bbox.get('x', 0), bbox.get('y', 0)
            w, h = bbox.get('width', 0), bbox.get('height', 0)
            
            color = colors.get(field_type, (200, 200, 200))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{field_type} ({confidence:.2f})"
            cv2.putText(
                vis_image, label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        
        return vis_image


class SimpleFieldDetector(nn.Module):
    """
    Simple CNN-based field detector
    (Simplified version - production would use proper object detection like YOLO)
    """
    
    def __init__(self, num_classes: int):
        super(SimpleFieldDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Simple CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, 3, H, W)
            
        Returns:
            Class predictions (batch, num_classes)
        """
        features = self.features(x)
        output = self.classifier(features)
        return output