"""
OCR Router
Intelligently selects the best OCR engine based on:
- Image quality metrics
- Document type
- Text density
- Language detection
Uses multiple engines and ensemble methods for best results
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2

from .engines.tesseract_engine import TesseractEngine
from .engines.doctr_engine import DocTREngine
from .engines.trocr_engine import TrOCREngine
from .ensemble import OCREnsemble
from .confidence_scorer import ConfidenceScorer


class OCRRouter:
    """
    Routes OCR requests to the most appropriate engine
    """
    
    def __init__(self, enable_ensemble: bool = True):
        """
        Initialize OCR router
        
        Args:
            enable_ensemble: If True, use ensemble of multiple engines for better accuracy
        """
        self.enable_ensemble = enable_ensemble
        
        # Initialize engines
        print("Initializing OCR engines...")
        self.tesseract = TesseractEngine()
        
        # DocTR and TrOCR are optional (require additional dependencies)
        try:
            self.doctr = DocTREngine()
            self.has_doctr = True
        except Exception as e:
            print(f"DocTR not available: {e}")
            self.has_doctr = False
        
        try:
            self.trocr = TrOCREngine()
            self.has_trocr = True
        except Exception as e:
            print(f"TrOCR not available: {e}")
            self.has_trocr = False
        
        # Initialize ensemble and scorer
        self.ensemble = OCREnsemble() if enable_ensemble else None
        self.scorer = ConfidenceScorer()
        
        print(f"Available engines: Tesseract{', DocTR' if self.has_doctr else ''}{', TrOCR' if self.has_trocr else ''}")
    
    def extract_text(
        self,
        image: np.ndarray,
        quality_metrics: Optional[Dict] = None,
        use_ensemble: Optional[bool] = None
    ) -> Dict:
        """
        Extract text using optimal OCR engine(s)
        
        Args:
            image: Input image as numpy array
            quality_metrics: Optional quality metrics from QualityAssessor
            use_ensemble: Override ensemble setting
            
        Returns:
            Dictionary containing:
                - text: Extracted text
                - confidence: Overall confidence score
                - engine: Engine(s) used
                - word_boxes: List of word bounding boxes with text and confidence
                - line_boxes: List of line bounding boxes
        """
        # Determine if we should use ensemble
        if use_ensemble is None:
            use_ensemble = self.enable_ensemble
        
        # Select best engine based on image characteristics
        selected_engine = self._select_engine(image, quality_metrics)
        
        print(f"Selected OCR engine: {selected_engine}")
        
        # If ensemble is enabled and we have multiple engines
        if use_ensemble and self._can_use_ensemble():
            return self._extract_with_ensemble(image)
        else:
            return self._extract_with_single_engine(image, selected_engine)
    
    def _select_engine(
        self,
        image: np.ndarray,
        quality_metrics: Optional[Dict]
    ) -> str:
        """
        Select best OCR engine based on image characteristics
        
        Returns:
            Engine name: 'tesseract', 'doctr', or 'trocr'
        """
        # Default to Tesseract
        if not self.has_doctr and not self.has_trocr:
            return 'tesseract'
        
        # If no quality metrics, use Tesseract (most reliable)
        if quality_metrics is None:
            return 'tesseract'
        
        overall_quality = quality_metrics.get('overall_score', 0.5)
        blur_score = quality_metrics.get('metrics', {}).get('blur_score', 0.5)
        noise_score = quality_metrics.get('metrics', {}).get('noise_score', 0.5)
        
        # Decision logic
        if overall_quality >= 0.7:
            # High quality - Tesseract works best
            return 'tesseract'
        
        elif blur_score < 0.4 and self.has_trocr:
            # Blurry image - TrOCR handles blur better
            return 'trocr'
        
        elif noise_score < 0.4 and self.has_doctr:
            # Noisy image - DocTR is more robust
            return 'doctr'
        
        elif overall_quality < 0.4 and self.has_doctr:
            # Poor quality - DocTR for challenging images
            return 'doctr'
        
        else:
            # Default to Tesseract
            return 'tesseract'
    
    def _can_use_ensemble(self) -> bool:
        """
        Check if ensemble is possible (need at least 2 engines)
        """
        available_engines = 1  # Tesseract is always available
        if self.has_doctr:
            available_engines += 1
        if self.has_trocr:
            available_engines += 1
        
        return available_engines >= 2
    
    def _extract_with_single_engine(
        self,
        image: np.ndarray,
        engine_name: str
    ) -> Dict:
        """
        Extract text using a single OCR engine
        """
        if engine_name == 'tesseract':
            result = self.tesseract.extract(image)
        elif engine_name == 'doctr' and self.has_doctr:
            result = self.doctr.extract(image)
        elif engine_name == 'trocr' and self.has_trocr:
            result = self.trocr.extract(image)
        else:
            # Fallback to Tesseract
            result = self.tesseract.extract(image)
        
        # Add confidence scores
        result['confidence'] = self.scorer.calculate_confidence(result)
        result['engine'] = engine_name
        
        return result
    
    def _extract_with_ensemble(self, image: np.ndarray) -> Dict:
        """
        Extract text using ensemble of multiple engines
        """
        results = []
        
        # Run all available engines
        print("Running ensemble OCR...")
        
        # Tesseract (always available)
        tesseract_result = self.tesseract.extract(image)
        tesseract_result['engine'] = 'tesseract'
        results.append(tesseract_result)
        
        # DocTR (if available)
        if self.has_doctr:
            doctr_result = self.doctr.extract(image)
            doctr_result['engine'] = 'doctr'
            results.append(doctr_result)
        
        # TrOCR (if available)
        if self.has_trocr:
            trocr_result = self.trocr.extract(image)
            trocr_result['engine'] = 'trocr'
            results.append(trocr_result)
        
        # Combine results using ensemble
        combined_result = self.ensemble.combine(results)
        
        # Calculate ensemble confidence
        combined_result['confidence'] = self.scorer.calculate_ensemble_confidence(results)
        combined_result['engine'] = 'ensemble'
        combined_result['individual_results'] = results
        
        return combined_result
    
    def extract_with_fallback(
        self,
        image: np.ndarray,
        min_confidence: float = 0.6
    ) -> Dict:
        """
        Extract text with automatic fallback to other engines if confidence is low
        
        Args:
            image: Input image
            min_confidence: Minimum acceptable confidence
            
        Returns:
            OCR result with highest confidence
        """
        # Try primary engine first
        primary_result = self.extract_text(image, use_ensemble=False)
        
        if primary_result['confidence'] >= min_confidence:
            return primary_result
        
        print(f"Low confidence ({primary_result['confidence']:.2f}), trying fallback engines...")
        
        # Try other engines
        all_results = [primary_result]
        
        if self.has_doctr and primary_result['engine'] != 'doctr':
            doctr_result = self._extract_with_single_engine(image, 'doctr')
            all_results.append(doctr_result)
        
        if self.has_trocr and primary_result['engine'] != 'trocr':
            trocr_result = self._extract_with_single_engine(image, 'trocr')
            all_results.append(trocr_result)
        
        # Return result with highest confidence
        best_result = max(all_results, key=lambda x: x['confidence'])
        print(f"Best result from {best_result['engine']} with confidence {best_result['confidence']:.2f}")
        
        return best_result
    
    def compare_engines(self, image: np.ndarray) -> Dict:
        """
        Run all engines and compare results (for debugging/analysis)
        
        Returns:
            Dictionary with results from all engines
        """
        comparison = {}
        
        # Tesseract
        comparison['tesseract'] = self.tesseract.extract(image)
        comparison['tesseract']['confidence'] = self.scorer.calculate_confidence(
            comparison['tesseract']
        )
        
        # DocTR
        if self.has_doctr:
            comparison['doctr'] = self.doctr.extract(image)
            comparison['doctr']['confidence'] = self.scorer.calculate_confidence(
                comparison['doctr']
            )
        
        # TrOCR
        if self.has_trocr:
            comparison['trocr'] = self.trocr.extract(image)
            comparison['trocr']['confidence'] = self.scorer.calculate_confidence(
                comparison['trocr']
            )
        
        return comparison
    
    def batch_extract(
        self,
        images: List[np.ndarray],
        quality_metrics_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Extract text from multiple images
        
        Args:
            images: List of images
            quality_metrics_list: Optional list of quality metrics
            
        Returns:
            List of OCR results
        """
        results = []
        
        for i, image in enumerate(images):
            quality_metrics = None
            if quality_metrics_list and i < len(quality_metrics_list):
                quality_metrics = quality_metrics_list[i]
            
            result = self.extract_text(image, quality_metrics)
            results.append(result)
        
        return results