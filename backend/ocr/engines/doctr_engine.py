"""
DocTR OCR Engine
Deep Learning based OCR using DocTR (Document Text Recognition)
"""

import numpy as np
from typing import Dict, List
import cv2

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False


class DocTREngine:
    """
    DocTR OCR engine wrapper
    Uses deep learning for robust OCR
    """
    
    def __init__(self, det_arch: str = 'db_resnet50', reco_arch: str = 'crnn_vgg16_bn'):
        """
        Initialize DocTR engine
        
        Args:
            det_arch: Detection architecture
                Options: 'db_resnet50', 'db_mobilenet_v3_large', 'linknet_resnet18'
            reco_arch: Recognition architecture
                Options: 'crnn_vgg16_bn', 'crnn_mobilenet_v3_small', 'sar_resnet31'
        """
        if not DOCTR_AVAILABLE:
            raise ImportError(
                "DocTR is not installed. Install it with: pip install python-doctr[torch]"
            )
        
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        
        # Initialize predictor (will download models on first use)
        print(f"Loading DocTR model (det: {det_arch}, reco: {reco_arch})...")
        self.model = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True
        )
        print("DocTR model loaded successfully")
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract text from image using DocTR
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Dictionary containing:
                - text: Extracted text
                - words: List of word-level data
                - lines: List of line-level data
                - blocks: List of block-level data
        """
        # Convert numpy array to format expected by DocTR
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Process image with DocTR
        doc = DocumentFile.from_images([image])
        result = self.model(doc)
        
        # Parse results
        text = self._extract_full_text(result)
        words = self._parse_words(result, image.shape)
        lines = self._parse_lines(result, image.shape)
        blocks = self._parse_blocks(result, image.shape)
        
        return {
            'text': text,
            'words': words,
            'lines': lines,
            'blocks': blocks,
            'raw_result': result
        }
    
    def _extract_full_text(self, result) -> str:
        """
        Extract all text from DocTR result
        """
        text_parts = []
        
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ' '.join([word.value for word in line.words])
                    text_parts.append(line_text)
        
        return '\n'.join(text_parts)
    
    def _parse_words(self, result, image_shape: tuple) -> List[Dict]:
        """
        Parse word-level data from DocTR result
        """
        words = []
        height, width = image_shape[:2]
        
        for page in result.pages:
            for block_idx, block in enumerate(page.blocks):
                for line_idx, line in enumerate(block.lines):
                    for word_idx, word in enumerate(line.words):
                        # Get bounding box (normalized 0-1)
                        x1, y1, x2, y2 = word.geometry
                        
                        # Convert to pixel coordinates
                        bbox = {
                            'x': int(x1 * width),
                            'y': int(y1 * height),
                            'width': int((x2 - x1) * width),
                            'height': int((y2 - y1) * height)
                        }
                        
                        words.append({
                            'text': word.value,
                            'confidence': float(word.confidence),
                            'bbox': bbox,
                            'block_num': block_idx,
                            'line_num': line_idx,
                            'word_num': word_idx
                        })
        
        return words
    
    def _parse_lines(self, result, image_shape: tuple) -> List[Dict]:
        """
        Parse line-level data from DocTR result
        """
        lines = []
        height, width = image_shape[:2]
        
        for page in result.pages:
            for block_idx, block in enumerate(page.blocks):
                for line_idx, line in enumerate(block.lines):
                    # Get line bounding box
                    x1, y1, x2, y2 = line.geometry
                    
                    bbox = {
                        'x': int(x1 * width),
                        'y': int(y1 * height),
                        'width': int((x2 - x1) * width),
                        'height': int((y2 - y1) * height)
                    }
                    
                    # Extract text and confidence
                    text = ' '.join([word.value for word in line.words])
                    confidences = [word.confidence for word in line.words]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    lines.append({
                        'text': text,
                        'confidence': float(avg_confidence),
                        'bbox': bbox,
                        'block_num': block_idx,
                        'line_num': line_idx,
                        'words': [word.value for word in line.words]
                    })
        
        return lines
    
    def _parse_blocks(self, result, image_shape: tuple) -> List[Dict]:
        """
        Parse block-level data from DocTR result
        """
        blocks = []
        height, width = image_shape[:2]
        
        for page in result.pages:
            for block_idx, block in enumerate(page.blocks):
                # Get block bounding box
                x1, y1, x2, y2 = block.geometry
                
                bbox = {
                    'x': int(x1 * width),
                    'y': int(y1 * height),
                    'width': int((x2 - x1) * width),
                    'height': int((y2 - y1) * height)
                }
                
                # Extract all text in block
                text_lines = []
                all_confidences = []
                
                for line in block.lines:
                    line_text = ' '.join([word.value for word in line.words])
                    text_lines.append(line_text)
                    all_confidences.extend([word.confidence for word in line.words])
                
                text = '\n'.join(text_lines)
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
                
                blocks.append({
                    'text': text,
                    'confidence': float(avg_confidence),
                    'bbox': bbox,
                    'block_num': block_idx,
                    'lines': text_lines
                })
        
        return blocks
    
    def export_result(self, result) -> str:
        """
        Export DocTR result as formatted string
        """
        return result.export()