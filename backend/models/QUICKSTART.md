# Models Quick Start Guide

Get up and running with models in 5 minutes!

## Installation

### 1. Install Dependencies

```bash
# Install model requirements
pip install -r backend/models/requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Setup Models

```bash
# Run automated setup
cd backend/models
python setup_models.py
```

This will:
- Create necessary directories
- Check dependencies
- Download essential models
- Verify installation

### 3. Quick Test

```python
from backend.models.model_manager import get_model_status

# Check what's installed
status = get_model_status()
print(f"Downloaded: {status['num_downloaded']} models")
print(f"Device: {status['device']}")
```

## Common Operations

### Download a Model

```bash
# Download specific model
python model_downloader.py download --model yolov5

# Download all models
python model_downloader.py download-all

# Check what's available
python model_downloader.py list
```

### Load a Model in Code

```python
from backend.models import get_model_registry

registry = get_model_registry()

# Load YOLOv5 for field detection
yolov5 = registry.get_model('yolov5')

# Use the model
results = yolov5(image)
```

### Setup a Pipeline

```python
from backend.models.model_manager import setup_inference_pipeline

# Basic pipeline (lightweight)
manager = setup_inference_pipeline('basic', auto_download=True)

# Advanced pipeline (includes table detection, field detection)
manager = setup_inference_pipeline('advanced', auto_download=True)

# Full pipeline (all models)
manager = setup_inference_pipeline('full', auto_download=True)
```

### Get Model Info

```bash
# Get detailed info about a model
python model_downloader.py info --model tablenet
```

### Benchmark Models

```bash
# Benchmark all loaded models
python model_manager.py benchmark

# Benchmark specific models
python model_manager.py benchmark --models yolov5 tablenet
```

## Pipeline Types

### Basic Pipeline
- **Models**: OCR Router
- **Size**: < 1 MB
- **Use Case**: Simple OCR with intelligent engine selection
- **Recommended For**: Quick testing, minimal setup

### Advanced Pipeline
- **Models**: TableNet, YOLOv5, OCR Router
- **Size**: ~114 MB
- **Use Case**: Table detection + field detection + OCR
- **Recommended For**: Production use, invoice processing

### Full Pipeline
- **Models**: All models (TableNet, YOLOv5, BERT-NER, LayoutLM, DocTR, OCR Router)
- **Size**: ~1.2 GB
- **Use Case**: Complete document understanding with layout analysis
- **Recommended For**: Maximum accuracy, research

## Code Examples

### Example 1: Field Detection

```python
from backend.models import get_model_registry
import cv2

registry = get_model_registry()
yolov5 = registry.get_model('yolov5')

# Load image
image = cv2.imread('invoice.jpg')

# Detect fields
results = yolov5(image)

# Get bounding boxes
boxes = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    label = yolov5.names[int(cls)]
    print(f"Found {label} with confidence {conf:.2f}")
```

### Example 2: Table Detection

```python
from backend.models import get_model_registry
import cv2

registry = get_model_registry()
tablenet = registry.get_model('tablenet')

# Load image
image = cv2.imread('invoice.jpg')

# Detect tables
table_mask = tablenet(image)

# Process table regions
# ... extract table data
```

### Example 3: Named Entity Recognition

```python
from backend.models import get_model_registry
from transformers import AutoTokenizer

registry = get_model_registry()
bert_ner = registry.get_model('bert_ner')
tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')

text = "Invoice from ABC Company dated 15/10/2025"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Extract entities
outputs = bert_ner(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = [bert_ner.config.id2label[p.item()] for p in predictions[0]]

# Print entities
for token, label in zip(tokens, labels):
    if label != 'O':
        print(f"{token}: {label}")
```

### Example 4: Complete Pipeline

```python
from backend.models.model_manager import ModelManager

# Initialize manager
manager = ModelManager(auto_download=True)

# Setup pipeline
models = manager.setup_pipeline('advanced')

# Get status
status = manager.get_status()
print(f"Loaded {status['num_loaded']} models")

# Use models
yolov5 = models['yolov5']
tablenet = models['tablenet']

# ... process invoice

# Cleanup when done
manager.cleanup()
```

## Troubleshooting

### Model Not Found
```
Error: Model weights not found
```
**Solution**: Download the model
```bash
python model_downloader.py download --model <model_name>
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
1. Reduce batch size in config files
2. Use CPU: `export CUDA_VISIBLE_DEVICES=""`
3. Unload unused models: `manager.unload_models(['model_name'])`

### Slow Download
**Solutions**:
1. Use a download manager
2. Download from alternative sources
3. Check your internet connection

### Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution**: Install missing packages
```bash
pip install -r backend/models/requirements.txt
```

## Performance Tips

1. **Use GPU**: Models run 10-100x faster on GPU
2. **Batch Processing**: Process multiple images at once
3. **Model Caching**: Models are cached after first load
4. **Mixed Precision**: Enable in config files for 2x speedup
5. **Preload Models**: Load all models once at startup

## Next Steps

- Read the [full documentation](README.md)
- Check [configuration files](configs/)
- Explore [model architectures](../layout_analysis/models/)
- Join the [community discussions](https://github.com/Cherry28831/Invoice-Data-Extractor/discussions)

## Support

- **Issues**: [GitHub Issues](https://github.com/Cherry28831/Invoice-Data-Extractor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Cherry28831/Invoice-Data-Extractor/discussions)
- **Email**: support@example.com