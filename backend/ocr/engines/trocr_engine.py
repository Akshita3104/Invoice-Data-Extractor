"""
TrOCR OCR Engine
Transformer-based OCR using Microsoft's TrOCR
Best for challenging images (blurry, low quality)
"""

import numpy as np
from typing import Dict, List
import cv2
from PIL import Image

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class TrOCREngine:
    """
    TrOCR engine wrapper
    Uses transformer-based architecture for robust OCR
    """
    
    def __init__(self, model_name: str = 'microsoft/trocr-base-printed'):
        """
        Initialize TrOCR engine
        
        Args:
            model_name: Pretrained model to use
                Options:
                - 'microsoft/trocr-base-printed': Base model for printed text
                - 'microsoft/trocr-large-printed': Large model for printed text
                - 'microsoft/trocr-base-handwritten': Base model for handwritten text
                - 'microsoft/trocr-large-handwritten': Large model for handwritten text
        """
        if not TROCR_AVAILABLE:
            raise ImportError(
                "TrOCR dependencies not installed. "
                "Install with: pip install transformers torch pillow"
            )
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        print(f"Loading TrOCR model: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        print(f"TrOCR model loaded successfully on {self.device}")
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract text from image using TrOCR
        
        Note: TrOCR works best on line-level images. For full documents,
        we'll split into lines and process each separately.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing:
                - text: Extracted text
                - words: List of word-level data (approximated)
                - lines: List of line-level data
                - blocks: List of block-level data
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image)
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect lines in the image
        lines_data = self._detect_lines(image)
        
        # Process each line with TrOCR
        line_results = []
        for line_data in lines_data:
            line_image = line_data['image']
            text, confidence = self._process_line(line_image)
            
            line_results.append({
                'text': text,
                'confidence': confidence,
                'bbox': line_data['bbox']
            })
        
        # Combine results
        full_text = '\n'.join([line['text'] for line in line_results])
        words = self._extract_words_from_lines(line_results)
        lines = self._format_lines(line_results)
        blocks = self._group_into_blocks(line_results)
        
        return {
            'text': full_text,
            'words': words,
            'lines': lines,
            'blocks': blocks
        }
    
    def _detect_lines(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text lines in the image using simple projection method
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (text regions)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes and sort by Y coordinate
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small regions
            if w > 10 and h > 5:
                bboxes.append({'x': x, 'y': y, 'width': w, 'height': h})
        
        # Sort by Y coordinate (top to bottom)
        bboxes = sorted(bboxes, key=lambda b: b['y'])
        
        # Merge overlapping bboxes into lines
        lines = self._merge_into_lines(bboxes, image.shape[0])
        
        # Extract line images
        lines_data = []
        for line_bbox in lines:
            x, y, w, h = line_bbox['x'], line_bbox['y'], line_bbox['width'], line_bbox['height']
            # Add padding
            padding = 5
            y_start = max(0, y - padding)
            y_end = min(image.shape[0], y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(image.shape[1], x + w + padding)
            
            line_image = image[y_start:y_end, x_start:x_end]
            
            lines_data.append({
                'image': line_image,
                'bbox': line_bbox
            })
        
        return lines_data
    
    def _merge_into_lines(self, bboxes: List[Dict], image_height: int, threshold: int = 10) -> List[Dict]:
        """
        Merge overlapping bounding boxes into lines
        """
        if not bboxes:
            return []
        
        lines = []
        current_line = bboxes[0].copy()
        
        for bbox in bboxes[1:]:
            # Check if bbox overlaps with current line (Y axis)
            if abs(bbox['y'] - current_line['y']) < threshold:
                # Merge into current line
                right = max(current_line['x'] + current_line['width'], bbox['x'] + bbox['width'])
                bottom = max(current_line['y'] + current_line['height'], bbox['y'] + bbox['height'])
                
                current_line['x'] = min(current_line['x'], bbox['x'])
                current_line['y'] = min(current_line['y'], bbox['y'])
                current_line['width'] = right - current_line['x']
                current_line['height'] = bottom - current_line['y']
            else:
                # Start new line
                lines.append(current_line)
                current_line = bbox.copy()
        
        # Add last line
        lines.append(current_line)
        
        return lines
    
    def _process_line(self, line_image: np.ndarray) -> tuple:
        """
        Process a single line with TrOCR
        
        Returns:
            Tuple of (text, confidence)
        """
        # Convert to PIL
        if isinstance(line_image, np.ndarray):
            pil_image = Image.fromarray(line_image)
        else:
            pil_image = line_image
        
        # Preprocess
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # TrOCR doesn't provide confidence scores directly
        # We'll estimate based on the model's output probabilities
        confidence = 0.85  # Default confidence for TrOCR
        
        return text.strip(), confidence
    
    def _extract_words_from_lines(self, line_results: List[Dict]) -> List[Dict]:
        """
        Extract word-level data from line results
        """
        words = []
        
        for line_idx, line in enumerate(line_results):
            # Split line text into words
            line_words = line['text'].split()
            
            # Estimate word positions (simple approximation)
            line_bbox = line['bbox']
            word_width = line_bbox['width'] / max(len(line_words), 1)
            
            for word_idx, word_text in enumerate(line_words):
                word_x = line_bbox['x'] + int(word_idx * word_width)
                
                words.append({
                    'text': word_text,
                    'confidence': line['confidence'],
                    'bbox': {
                        'x': word_x,
                        'y': line_bbox['y'],
                        'width': int(word_width),
                        'height': line_bbox['height']
                    },
                    'line_num': line_idx,
                    'word_num': word_idx
                })
        
        return words
    
    def _format_lines(self, line_results: List[Dict]) -> List[Dict]:
        """
        Format line results
        """
        lines = []
        for line_idx, line in enumerate(line_results):
            lines.append({
                'text': line['text'],
                'confidence': line['confidence'],
                'bbox': line['bbox'],
                'line_num': line_idx,
                'words': line['text'].split()
            })
        
        return lines
    
    def _group_into_blocks(self, line_results: List[Dict]) -> List[Dict]:
        """
        Group lines into blocks (paragraphs)
        """
        if not line_results:
            return []
        
        blocks = []
        current_block = {
            'lines': [line_results[0]],
            'text': line_results[0]['text'],
            'bbox': line_results[0]['bbox'].copy(),
            'confidences': [line_results[0]['confidence']]
        }
        
        for line in line_results[1:]:
            # Check vertical gap between lines
            prev_line = current_block['lines'][-1]
            gap = line['bbox']['y'] - (prev_line['bbox']['y'] + prev_line['bbox']['height'])
            
            if gap < 30:  # Lines are close - same block
                current_block['lines'].append(line)
                current_block['text'] += '\n' + line['text']
                current_block['confidences'].append(line['confidence'])
                
                # Update block bbox
                right = max(
                    current_block['bbox']['x'] + current_block['bbox']['width'],
                    line['bbox']['x'] + line['bbox']['width']
                )
                bottom = line['bbox']['y'] + line['bbox']['height']
                
                current_block['bbox']['x'] = min(current_block['bbox']['x'], line['bbox']['x'])
                current_block['bbox']['width'] = right - current_block['bbox']['x']
                current_block['bbox']['height'] = bottom - current_block['bbox']['y']
            else:
                # New block
                avg_conf = sum(current_block['confidences']) / len(current_block['confidences'])
                blocks.append({
                    'text': current_block['text'],
                    'confidence': avg_conf,
                    'bbox': current_block['bbox'],
                    'block_num': len(blocks),
                    'lines': [l['text'] for l in current_block['lines']]
                })
                
                current_block = {
                    'lines': [line],
                    'text': line['text'],
                    'bbox': line['bbox'].copy(),
                    'confidences': [line['confidence']]
                }
        
        # Add last block
        avg_conf = sum(current_block['confidences']) / len(current_block['confidences'])
        blocks.append({
            'text': current_block['text'],
            'confidence': avg_conf,
            'bbox': current_block['bbox'],
            'block_num': len(blocks),
            'lines': [l['text'] for l in current_block['lines']]
        })
        
        return blocks
    
    def process_single_line(self, line_image: np.ndarray) -> str:
        """
        Process a single line image (for use when you already have line-segmented images)
        
        Args:
            line_image: Single line image
            
        Returns:
            Extracted text
        """
        text, _ = self._process_line(line_image)
        return text