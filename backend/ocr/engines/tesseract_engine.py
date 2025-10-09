"""
Tesseract OCR Engine
Wrapper for Tesseract OCR with enhanced configuration
"""

import pytesseract
import numpy as np
import cv2
from typing import Dict, List, Tuple
from PIL import Image


class TesseractEngine:
    """
    Tesseract OCR engine wrapper
    """
    
    def __init__(self, lang: str = 'eng', config: str = None):
        """
        Initialize Tesseract engine
        
        Args:
            lang: Language code (default: 'eng')
            config: Custom Tesseract config string
        """
        self.lang = lang
        
        # Default config optimized for documents
        if config is None:
            self.config = (
                '--oem 3 '  # OCR Engine Mode: LSTM only
                '--psm 6 '  # Page Segmentation Mode: Assume uniform block of text
                '-c preserve_interword_spaces=1'
            )
        else:
            self.config = config
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract text from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing:
                - text: Extracted text
                - words: List of word-level data
                - lines: List of line-level data
                - blocks: List of block-level data
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB to BGR for PIL
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Extract text
        text = pytesseract.image_to_string(
            pil_image,
            lang=self.lang,
            config=self.config
        )
        
        # Extract detailed data (word and line level)
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )
        
        # Parse detailed data
        words = self._parse_word_data(data)
        lines = self._parse_line_data(data)
        blocks = self._parse_block_data(data)
        
        return {
            'text': text.strip(),
            'words': words,
            'lines': lines,
            'blocks': blocks,
            'raw_data': data
        }
    
    def _parse_word_data(self, data: Dict) -> List[Dict]:
        """
        Parse word-level data from Tesseract output
        """
        words = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            # Skip empty text
            if not data['text'][i].strip():
                continue
            
            word = {
                'text': data['text'][i],
                'confidence': float(data['conf'][i]),
                'bbox': {
                    'x': int(data['left'][i]),
                    'y': int(data['top'][i]),
                    'width': int(data['width'][i]),
                    'height': int(data['height'][i])
                },
                'block_num': int(data['block_num'][i]),
                'line_num': int(data['line_num'][i]),
                'word_num': int(data['word_num'][i])
            }
            words.append(word)
        
        return words
    
    def _parse_line_data(self, data: Dict) -> List[Dict]:
        """
        Parse line-level data from Tesseract output
        """
        lines = {}
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if not data['text'][i].strip():
                continue
            
            line_id = f"{data['block_num'][i]}-{data['line_num'][i]}"
            
            if line_id not in lines:
                lines[line_id] = {
                    'text': '',
                    'words': [],
                    'bbox': {
                        'x': int(data['left'][i]),
                        'y': int(data['top'][i]),
                        'width': 0,
                        'height': int(data['height'][i])
                    },
                    'block_num': int(data['block_num'][i]),
                    'line_num': int(data['line_num'][i]),
                    'confidences': []
                }
            
            # Update line data
            lines[line_id]['text'] += data['text'][i] + ' '
            lines[line_id]['words'].append(data['text'][i])
            lines[line_id]['confidences'].append(float(data['conf'][i]))
            
            # Update bounding box
            right = int(data['left'][i] + data['width'][i])
            if right > lines[line_id]['bbox']['x'] + lines[line_id]['bbox']['width']:
                lines[line_id]['bbox']['width'] = right - lines[line_id]['bbox']['x']
        
        # Calculate average confidence for each line
        for line in lines.values():
            line['text'] = line['text'].strip()
            if line['confidences']:
                line['confidence'] = sum(line['confidences']) / len(line['confidences'])
            else:
                line['confidence'] = 0.0
            del line['confidences']
        
        return list(lines.values())
    
    def _parse_block_data(self, data: Dict) -> List[Dict]:
        """
        Parse block-level data from Tesseract output
        """
        blocks = {}
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if not data['text'][i].strip():
                continue
            
            block_id = data['block_num'][i]
            
            if block_id not in blocks:
                blocks[block_id] = {
                    'text': '',
                    'lines': [],
                    'bbox': {
                        'x': int(data['left'][i]),
                        'y': int(data['top'][i]),
                        'width': 0,
                        'height': 0
                    },
                    'block_num': int(block_id),
                    'confidences': []
                }
            
            # Update block data
            blocks[block_id]['text'] += data['text'][i] + ' '
            blocks[block_id]['confidences'].append(float(data['conf'][i]))
            
            # Update bounding box
            right = int(data['left'][i] + data['width'][i])
            bottom = int(data['top'][i] + data['height'][i])
            
            if right > blocks[block_id]['bbox']['x'] + blocks[block_id]['bbox']['width']:
                blocks[block_id]['bbox']['width'] = right - blocks[block_id]['bbox']['x']
            
            if bottom > blocks[block_id]['bbox']['y'] + blocks[block_id]['bbox']['height']:
                blocks[block_id]['bbox']['height'] = bottom - blocks[block_id]['bbox']['y']
        
        # Calculate average confidence for each block
        for block in blocks.values():
            block['text'] = block['text'].strip()
            if block['confidences']:
                block['confidence'] = sum(block['confidences']) / len(block['confidences'])
            else:
                block['confidence'] = 0.0
            del block['confidences']
        
        return list(blocks.values())
    
    def extract_with_psm(self, image: np.ndarray, psm: int) -> Dict:
        """
        Extract text with specific Page Segmentation Mode
        
        Args:
            image: Input image
            psm: Page Segmentation Mode
                0 = Orientation and script detection only
                1 = Automatic page segmentation with OSD
                3 = Fully automatic page segmentation (no OSD)
                4 = Assume a single column of text
                5 = Assume a single uniform block of vertically aligned text
                6 = Assume a single uniform block of text (default)
                7 = Treat the image as a single text line
                8 = Treat the image as a single word
                9 = Treat the image as a single word in a circle
                10 = Treat the image as a single character
                11 = Sparse text. Find as much text as possible
                12 = Sparse text with OSD
                13 = Raw line. Treat image as single text line
        """
        custom_config = f'--oem 3 --psm {psm}'
        
        original_config = self.config
        self.config = custom_config
        
        result = self.extract(image)
        
        self.config = original_config
        
        return result
    
    def detect_language(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Detect language(s) in the image
        
        Returns:
            List of (language_code, confidence) tuples
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect using OSD (Orientation and Script Detection)
        try:
            osd_data = pytesseract.image_to_osd(pil_image)
            # Parse OSD output (basic parsing)
            lines = osd_data.split('\n')
            script_line = [l for l in lines if 'Script:' in l][0]
            script = script_line.split(':')[1].strip()
            
            return [(script, 1.0)]
        except:
            return [('eng', 1.0)]  # Default to English
    
    def get_hocr(self, image: np.ndarray) -> str:
        """
        Get hOCR (HTML-based OCR) output
        
        Returns:
            hOCR formatted string
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        hocr = pytesseract.image_to_pdf_or_hocr(
            pil_image,
            lang=self.lang,
            config=self.config,
            extension='hocr'
        )
        
        return hocr.decode('utf-8')
    
    def get_searchable_pdf(self, image: np.ndarray) -> bytes:
        """
        Generate searchable PDF from image
        
        Returns:
            PDF bytes
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        pdf = pytesseract.image_to_pdf_or_hocr(
            pil_image,
            lang=self.lang,
            config=self.config,
            extension='pdf'
        )
        
        return pdf