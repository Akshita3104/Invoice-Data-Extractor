"""
Models package for Invoice Extractor
Contains pre-trained model configurations and weight management
"""

from pathlib import Path

# Model directories
MODELS_DIR = Path(__file__).parent
WEIGHTS_DIR = MODELS_DIR / 'weights'
CONFIGS_DIR = MODELS_DIR / 'configs'

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Model registry
MODEL_REGISTRY = {
    'tablenet': {
        'weight_file': 'tablenet.pth',
        'config_file': 'tablenet_config.yaml',
        'url': 'https://github.com/example/tablenet/releases/download/v1.0/tablenet.pth',
        'description': 'TableNet model for table detection and structure recognition'
    },
    'yolov5': {
        'weight_file': 'yolov5.pt',
        'config_file': 'yolo_config.yaml',
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt',
        'description': 'YOLOv5 model for field detection in invoices'
    },
    'bert_ner': {
        'weight_file': 'bert-ner.bin',
        'config_file': 'bert_config.json',
        'url': 'https://huggingface.co/dslim/bert-base-NER/resolve/main/pytorch_model.bin',
        'description': 'BERT model fine-tuned for Named Entity Recognition'
    },
    'ocr_router': {
        'weight_file': 'ocr_router.pkl',
        'config_file': 'ocr_router_config.yaml',
        'url': None,  # Custom trained model
        'description': 'ML-based OCR engine selector'
    },
    'doctr': {
        'weight_file': 'doctr_db_resnet50.pt',
        'config_file': 'doctr_config.yaml',
        'url': 'https://github.com/mindee/doctr/releases/download/v0.5.0/db_resnet50.pt',
        'description': 'DocTR detection and recognition models'
    },
    'layoutlm': {
        'weight_file': 'layoutlm_base.bin',
        'config_file': 'layoutlm_config.json',
        'url': 'https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/pytorch_model.bin',
        'description': 'LayoutLM for document understanding'
    }
}

__all__ = [
    'MODELS_DIR',
    'WEIGHTS_DIR',
    'CONFIGS_DIR',
    'MODEL_REGISTRY'
]