# Models Directory

This directory contains pre-trained model weights and configurations for the Invoice Extractor.

## Directory Structure

```
models/
├── __init__.py              # Model registry and initialization
├── model_downloader.py      # Download model weights
├── model_loader.py          # Load and manage models
├── model_manager.py         # High-level model management
├── weights/                 # Model weight files
│   ├── tablenet.pth
│   ├── yolov5.pt
│   ├── bert-ner.bin
│   ├── ocr_router.pkl
│   ├── doctr_db_resnet50.pt
│   └── layoutlm_base.bin
└── configs/                 # Model configuration files
    ├── tablenet_config.yaml
    ├── yolo_config.yaml
    ├── bert_config.json
    ├── ocr_router_config.yaml
    ├── doctr_config.yaml
    └── layoutlm_config.json
```

## Available Models

### 1. TableNet
- **Purpose**: Table detection and structure recognition
- **Architecture**: VGG-19 based encoder-decoder
- **Weight File**: `tablenet.pth`
- **Config**: `tablenet_config.yaml`
- **Input Size**: 1024x768
- **Use Case**: Detecting and extracting table regions from invoices

### 2. YOLOv5
- **Purpose**: Invoice field detection
- **Architecture**: YOLOv5s (small)
- **Weight File**: `yolov5.pt`
- **Config**: `yolo_config.yaml`
- **Input Size**: 640x640
- **Classes**: 12 invoice fields (company name, date, total, etc.)
- **Use Case**: Locating specific fields in invoice documents

### 3. BERT-NER
- **Purpose**: Named Entity Recognition
- **Architecture**: BERT base model fine-tuned for NER
- **Weight File**: `bert-ner.bin`
- **Config**: `bert_config.json`
- **Use Case**: Extracting entities like organization names, locations, dates

### 4. OCR Router
- **Purpose**: Intelligent OCR engine selection
- **Architecture**: Random Forest classifier
- **Weight File**: `ocr_router.pkl`
- **Config**: `ocr_router_config.yaml`
- **Use Case**: Selecting the best OCR engine based on document characteristics

### 5. DocTR
- **Purpose**: Document Text Recognition
- **Architecture**: DB-ResNet50 for detection + CRNN for recognition
- **Weight File**: `doctr_db_resnet50.pt`
- **Config**: `doctr_config.yaml`
- **Use Case**: End-to-end text detection and recognition

### 6. LayoutLM
- **Purpose**: Document understanding with layout
- **Architecture**: BERT + 2D position embeddings
- **Weight File**: `layoutlm_base.bin`
- **Config**: `layoutlm_config.json`
- **Use Case**: Understanding document structure and extracting information

## Usage

### Download Models

```bash
# Download a specific model
python model_downloader.py download --model tablenet

# Download all models
python model_downloader.py download-all

# List available models
python model_downloader.py list

# Get model info
python model_downloader.py info --model yolov5
```

### Load Models in Code

```python
from backend.models import get_model_registry

# Get model registry
registry = get_model_registry()

# Load a specific model
tablenet = registry.get_model('tablenet')

# Preload multiple models
registry.preload_models(['tablenet', 'yolov5', 'bert_ner'])

# List loaded models
loaded = registry.list_loaded_models()
print(f"Loaded models: {loaded}")
```

### Using Model Loader

```python
from backend.models.model_loader import ModelLoader

loader = ModelLoader()

# Load model
model = loader.load_model('yolov5')

# Load configuration
config = loader.load_config('yolov5')

# Get metadata
metadata = loader.get_model_metadata('tablenet')
print(f"Model size: {metadata['size_mb']:.2f} MB")
```

## Model Download Sources

Models are downloaded from the following sources:

- **TableNet**: Custom trained model (provide your own URL)
- **YOLOv5**: Official Ultralytics repository
- **BERT-NER**: HuggingFace Hub (dslim/bert-base-NER)
- **DocTR**: Mindee DocTR repository
- **LayoutLM**: HuggingFace Hub (microsoft/layoutlm-base-uncased)
- **OCR Router**: Custom trained (train your own)

## Training Your Own Models

### OCR Router

```python
from backend.ocr.ocr_router import train_ocr_router

# Prepare training data
training_data = [
    {
        'features': {...},  # Image quality features
        'best_engine': 'tesseract'
    },
    # ... more examples
]

# Train router
router = train_ocr_router(training_data)
router.save('backend/models/weights/ocr_router.pkl')
```

### Fine-tune YOLOv5

```bash
# Prepare dataset in YOLO format
# Train
python -m yolov5.train \
    --img 640 \
    --batch 16 \
    --epochs 300 \
    --data invoice_fields.yaml \
    --weights yolov5s.pt \
    --project models/weights \
    --name yolov5_invoice
```

## Model Performance

| Model | Size | Inference Time | Accuracy |
|-------|------|----------------|----------|
| TableNet | ~100 MB | ~200ms | 95%+ |
| YOLOv5s | ~14 MB | ~20ms | 92%+ |
| BERT-NER | ~420 MB | ~100ms | 94%+ |
| DocTR | ~200 MB | ~300ms | 93%+ |
| LayoutLM | ~440 MB | ~150ms | 96%+ |
| OCR Router | <1 MB | <5ms | 88%+ |

*Note: Times are approximate on GPU (RTX 3080)*

## GPU Requirements

- **Minimum**: 4GB VRAM for inference
- **Recommended**: 8GB VRAM for batch processing
- **Optimal**: 12GB+ VRAM for multiple models

Models automatically fall back to CPU if GPU is not available.

## Troubleshooting

### Model Not Found
```
Error: Model weights not found: tablenet.pth
```
**Solution**: Run `python model_downloader.py download --model tablenet`

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size in config files
- Use smaller models (e.g., YOLOv5n instead of YOLOv5s)
- Unload unused models: `registry.unload_all()`

### Slow Inference
**Solution**:
- Enable mixed precision in configs
- Use GPU if available
- Enable model compilation (PyTorch 2.0+)

## License

Model weights are subject to their respective licenses:
- YOLOv5: GPL-3.0
- BERT models: Apache 2.0
- DocTR: Apache 2.0
- Custom models: MIT

## Contributing

To add a new model:

1. Add entry to `MODEL_REGISTRY` in `__init__.py`
2. Create configuration file in `configs/`
3. Implement loading logic in `model_loader.py`
4. Update this README

## Support

For issues with models, please check:
- [GitHub Issues](https://github.com/Cherry28831/Invoice-Data-Extractor/issues)
- [Documentation](https://github.com/Cherry28831/Invoice-Data-Extractor/wiki)