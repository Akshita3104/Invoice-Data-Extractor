"""
Format Handler
Handles multiple document formats (PDF, JPEG, PNG, TIFF) and orchestrates
quality assessment and format conversion
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
from PIL import Image
import cv2
from pdf2image import convert_from_path

from .quality_assessor import QualityAssessor
from .format_converter import FormatConverter


class FormatHandler:
    """
    Main handler for multi-format document ingestion
    Supports: PDF, JPEG, JPG, PNG, TIFF, TIF
    """
    
    SUPPORTED_FORMATS = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    def __init__(self, target_dpi: int = 300):
        """
        Initialize format handler
        
        Args:
            target_dpi: Target DPI for standardization (default: 300)
        """
        self.target_dpi = target_dpi
        self.quality_assessor = QualityAssessor()
        self.format_converter = FormatConverter(target_dpi=target_dpi)
        
    def process(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Process document of any supported format
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple of (standardized_image, quality_metrics)
            - standardized_image: numpy array of processed image
            - quality_metrics: dictionary containing quality scores
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_ext}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        # Process based on format
        if file_ext == '.pdf':
            images = self._handle_pdf(str(file_path))
        else:
            images = self._handle_image(str(file_path))
        
        # Process each page/image
        processed_images = []
        quality_reports = []
        
        for img in images:
            # Convert to standardized format
            standardized_img = self.format_converter.standardize(img)
            
            # Assess quality
            quality_metrics = self.quality_assessor.assess(standardized_img)
            
            processed_images.append(standardized_img)
            quality_reports.append(quality_metrics)
        
        # For multi-page documents, return first page for now
        # TODO: Handle multi-page documents properly
        return processed_images[0], quality_reports[0]
    
    def process_multiple_pages(self, file_path: str) -> Tuple[list, list]:
        """
        Process all pages of a multi-page document
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple of (list of standardized_images, list of quality_metrics)
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            images = self._handle_pdf(str(file_path))
        else:
            images = self._handle_image(str(file_path))
        
        processed_images = []
        quality_reports = []
        
        for img in images:
            standardized_img = self.format_converter.standardize(img)
            quality_metrics = self.quality_assessor.assess(standardized_img)
            
            processed_images.append(standardized_img)
            quality_reports.append(quality_metrics)
        
        return processed_images, quality_reports
    
    def _handle_pdf(self, pdf_path: str) -> list:
        """
        Convert PDF to images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Images
        """
        try:
            # Convert PDF to images at target DPI
            images = convert_from_path(
                pdf_path, 
                dpi=self.target_dpi,
                fmt='jpeg'
            )
            return images
        except Exception as e:
            raise RuntimeError(f"Error converting PDF to images: {e}")
    
    def _handle_image(self, image_path: str) -> list:
        """
        Load image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            List containing single PIL Image
        """
        try:
            img = Image.open(image_path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return [img]
        except Exception as e:
            raise RuntimeError(f"Error loading image: {e}")
    
    def get_document_info(self, file_path: str) -> Dict:
        """
        Get metadata about the document
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with document metadata
        """
        file_path = Path(file_path)
        
        info = {
            'filename': file_path.name,
            'format': file_path.suffix.lower(),
            'size_bytes': file_path.stat().st_size,
            'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2)
        }
        
        # Get page count for PDFs
        if file_path.suffix.lower() == '.pdf':
            try:
                import pdfplumber
                with pdfplumber.open(str(file_path)) as pdf:
                    info['page_count'] = len(pdf.pages)
            except:
                info['page_count'] = 'unknown'
        else:
            info['page_count'] = 1
        
        return info
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """
        Check if file format is supported
        
        Args:
            file_path: Path to file
            
        Returns:
            True if format is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in FormatHandler.SUPPORTED_FORMATS