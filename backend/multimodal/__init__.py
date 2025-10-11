"""
Multimodal Module
Integrates visual, textual, layout, and graph features:
- Feature extraction from multiple modalities
- Attention-based fusion
- Field detection (YOLO-based)
- Entity classification (BERT-NER)
"""

from .feature_extractor import MultimodalFeatureExtractor
from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder
from .layout_encoder import LayoutEncoder
from .fusion_layer import FusionLayer
from .field_detector import FieldDetector
from .entity_classifier import EntityClassifier

__all__ = [
    'MultimodalFeatureExtractor',
    'VisualEncoder',
    'TextEncoder',
    'LayoutEncoder',
    'FusionLayer',
    'FieldDetector',
    'EntityClassifier'
]