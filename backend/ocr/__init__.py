"""
OCR Module
Multi-engine OCR with intelligent routing:
- Tesseract OCR
- DocTR (Deep OCR)
- TrOCR (Transformer-based OCR)
- Ensemble methods
- Confidence scoring
"""

from .ocr_router import OCRRouter
from .ensemble import OCREnsemble
from .confidence_scorer import ConfidenceScorer

__all__ = [
    'OCRRouter',
    'OCREnsemble', 
    'ConfidenceScorer'
]