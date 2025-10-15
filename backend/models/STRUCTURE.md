# Models Folder Complete Structure

Complete directory structure and file descriptions for the models module.

## 📁 Directory Tree

```
backend/models/
│
├── __init__.py                  # Package initialization, MODEL_REGISTRY
├── model_downloader.py          # Download model weights from various sources
├── model_loader.py              # Load and cache models in memory
├── model_manager.py             # High-level model management interface
├── config_validator.py          # Validate configuration files
├── setup_models.py              # Initial setup script
├── requirements.txt             # Python dependencies for models
├── README.md                    # Complete documentation
├── QUICKSTART.md                # Quick start guide
├── STRUCTURE.md                 # This file
│
├── weights/                     # Model weight files (NOT in git)
│   ├── README.md                # Weight folder documentation
│   ├── .gitkeep                 # Keep directory in git
│   ├── .gitignore               # Ignore weight files
│   ├── checksums.md5            # MD5 checksums for verification
│   ├── download_weights.sh      # Bash script to download weights
│   ├── model_info.json          # Detailed model information
│   ├── tablenet.pth             # TableNet weights (~100 MB)
│   ├── yolov5.pt                # YOLOv5 weights (~14 MB)
│   ├── bert-ner.bin             # BERT-NER weights (~420 MB)
│   ├── ocr_router.pkl           # OCR Router weights (<1 MB)
│   ├── doctr_db_resnet50.pt     # DocTR detection (~100 MB)
│   ├── doctr_crnn_vgg16.pt      # DocTR recognition (~100 MB)
│   └── layoutlm_base.bin        # LayoutLM weights (~440 MB)
│
└── configs/                     # Model configuration files
    ├── tablenet_config.yaml     # TableNet configuration
    ├── yolo_config.yaml         # YOLOv5 configuration
    ├── bert_config.json         # BERT-NER configuration
    ├── ocr_router_config.yaml   # OCR Router configuration
    ├── doctr_config.yaml        # DocTR configuration
    └── layoutlm_config.json     # LayoutLM configuration
```

## 📄 File Descriptions

### Core Python Files

#### `__init__.py`
- **Purpose**: Package initialization
- **Exports**: MODEL_REGISTRY, paths (WEIGHTS_DIR, CONFIGS_DIR)
- **Size**: ~2 KB
- **Usage**: Central registry of all available models

#### `model_downloader.py`
- **Purpose**: Download model weights from various sources
- **Classes**: `ModelDownloader`
- **Features**:
  - Progress bar for downloads
  - Retry logic
  - HuggingFace Hub support
  - Google Drive support
  - Checksum verification
- **CLI**: Can be run directly
- **Size**: ~10 KB

#### `model_loader.py`
- **Purpose**: Load models from weights with caching
- **Classes**: `ModelLoader`, `ModelRegistry`
- **Features**:
  - LRU cache for loaded models
  - Automatic device selection (GPU/CPU)
  - Lazy loading
  - Memory management
- **Size**: ~8 KB

#### `model_manager.py`
- **Purpose**: High-level model management
- **Classes**: `ModelManager`
- **Features**:
  - Pipeline setup (basic/advanced/full)
  - Benchmark models
  - Memory monitoring
  - Model optimization
  - Batch operations
- **CLI**: Can be run directly
- **Size**: ~12 KB

#### `config_validator.py`
- **Purpose**: Validate configuration files
- **Classes**: `ConfigValidator`
- **Features**:
  - Schema validation
  - Type checking
  - Required field checking
  - Warning system
- **CLI**: Can be run directly
- **Size**: ~10 KB

#### `setup_models.py`
- **Purpose**: Interactive setup wizard
- **Features**:
  - Dependency checking
  - Directory creation
  - Model downloading
  - Installation verification
  - Usage instructions
- **Run once**: After installation
- **Size**: ~8 KB

### Configuration Files

#### `tablenet_config.yaml`
- **Model**: TableNet
- **Format**: YAML
- **Sections**: model, input, output, training, augmentation, postprocessing, performance
- **Size**: ~1 KB

#### `yolo_config.yaml`
- **Model**: YOLOv5
- **Format**: YAML
- **Sections**: model, input, classes, detection, training, augmentation, performance
- **Classes**: 12 invoice fields
- **Size**: ~2 KB

#### `bert_config.json`
- **Model**: BERT-NER
- **Format**: JSON
- **Sections**: model, labels, tokenizer, inference, training
- **Labels**: 9 entity types
- **Size**: ~1 KB

#### `ocr_router_config.yaml`
- **Model**: OCR Router
- **Format**: YAML
- **Sections**: model, features, engines, routing, quality_thresholds, ensemble
- **Features**: 15 image quality metrics
- **Size**: ~2 KB

#### `doctr_config.yaml`
- **Model**: DocTR
- **Format**: YAML
- **Sections**: detection, recognition, postprocessing, performance, output
- **Size**: ~1 KB

#### `layoutlm_config.json`
- **Model**: LayoutLM
- **Format**: JSON
- **Sections**: model, labels, preprocessing, tokenizer, inference, training, optimization
- **Labels**: 25 invoice field labels
- **Size**: ~2 KB

### Documentation Files

#### `README.md` (main)
- **Purpose**: Complete documentation
- **Sections**:
  - Directory structure
  - Available models
  - Usage examples
  - Download instructions
  - Training guides
  - Performance metrics
  - Troubleshooting
- **Size**: ~15 KB

#### `QUICKSTART.md`
- **Purpose**: Quick start guide (5 minutes)
- **Sections**:
  - Installation
  - Common operations
  - Pipeline types
  - Code examples
  - Troubleshooting
- **Size**: ~10 KB

#### `weights/README.md`
- **Purpose**: Weight folder documentation
- **Sections**:
  - Download methods
  - File integrity
  - Storage requirements
  - Manual download links
  - Troubleshooting
- **Size**: ~8 KB

### Weight Files

All weight files are in `weights/` directory:

| File | Size | Format | Required | Pipeline |
|------|------|--------|----------|----------|
| `tablenet.pth` | ~100 MB | PyTorch | No | Advanced, Full |
| `yolov5.pt` | ~14 MB | PyTorch | No | Advanced, Full |
| `bert-ner.bin` | ~420 MB | PyTorch | No | Full |
| `ocr_router.pkl` | <1 MB | Pickle | Yes | All |
| `doctr_db_resnet50.pt` | ~100 MB | PyTorch | No | Full |
| `doctr_crnn_vgg16.pt` | ~100 MB | PyTorch | No | Full |
| `layoutlm_base.bin` | ~440 MB | PyT