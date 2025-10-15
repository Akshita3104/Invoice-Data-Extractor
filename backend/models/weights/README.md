# Model Weights Directory

This directory stores pre-trained model weight files.

## Directory Structure

```
weights/
├── README.md                    # This file
├── .gitkeep                     # Keep empty directory in git
├── tablenet.pth                 # TableNet weights (~100 MB)
├── yolov5.pt                    # YOLOv5 weights (~14 MB)
├── bert-ner.bin                 # BERT-NER weights (~420 MB)
├── ocr_router.pkl               # OCR Router weights (<1 MB)
├── doctr_db_resnet50.pt         # DocTR detection weights (~100 MB)
├── doctr_crnn_vgg16.pt          # DocTR recognition weights (~100 MB)
└── layoutlm_base.bin            # LayoutLM weights (~440 MB)
```

## Important Notes

⚠️ **Model weights are NOT included in the repository due to their large size.**

### Total Size
- **Minimum (Basic)**: < 1 MB (ocr_router only)
- **Recommended (Advanced)**: ~214 MB (tablenet + yolov5 + ocr_router)
- **Complete (Full)**: ~1.2 GB (all models)

## Downloading Weights

### Method 1: Automated Download (Recommended)

```bash
# Download all models
cd backend/models
python model_downloader.py download-all

# Download specific model
python model_downloader.py download --model yolov5

# Verify downloads
python model_downloader.py list
```

### Method 2: Manual Setup Script

```bash
cd backend/models
python setup_models.py
```

This interactive script will:
- Check your system
- Download essential models
- Ask about optional models
- Verify installation

### Method 3: Manual Download

If automated download fails, download manually from these sources:

#### 1. **tablenet.pth** (~100 MB)
```bash
# Option A: From GitHub Release (if available)
wget https://github.com/YOUR_REPO/releases/download/v1.0/tablenet.pth

# Option B: From Google Drive
# Download from: https://drive.google.com/file/d/YOUR_FILE_ID/view
gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O tablenet.pth

# Option C: Train your own
# See: backend/layout_analysis/models/tablenet_model.py
```

#### 2. **yolov5.pt** (~14 MB)
```bash
# Official YOLOv5 weights
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O yolov5.pt

# Or use PyTorch Hub (downloads automatically)
# The model loader will handle this
```

#### 3. **bert-ner.bin** (~420 MB)
```bash
# Download from HuggingFace Hub
# This happens automatically via transformers library
# Or manually:
wget https://huggingface.co/dslim/bert-base-NER/resolve/main/pytorch_model.bin -O bert-ner.bin
```

#### 4. **ocr_router.pkl** (<1 MB)
```bash
# This is a custom trained model
# You need to train it yourself or use the provided one

# If provided in release:
wget https://github.com/YOUR_REPO/releases/download/v1.0/ocr_router.pkl
```

#### 5. **doctr_db_resnet50.pt** (~100 MB)
```bash
# Download from DocTR
wget https://doctr-static.mindee.com/models?id=v0.5.1/db_resnet50-79bd7d70.pt -O doctr_db_resnet50.pt

# Or let the library download it automatically
```

#### 6. **layoutlm_base.bin** (~440 MB)
```bash
# Download from HuggingFace Hub
wget https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/pytorch_model.bin -O layoutlm_base.bin
```

## File Integrity Verification

After downloading, verify file integrity:

```bash
# Check file sizes
ls -lh *.pt *.pth *.bin *.pkl

# Expected sizes:
# tablenet.pth        : ~100 MB
# yolov5.pt          : ~14 MB
# bert-ner.bin       : ~420 MB
# ocr_router.pkl     : <1 MB
# doctr_*.pt         : ~100 MB each
# layoutlm_base.bin  : ~440 MB
```

### MD5 Checksums

```bash
# Verify checksums (if provided)
md5sum -c checksums.md5
```

## Storage Requirements

### By Pipeline Type

**Basic Pipeline** (Minimal)
- ocr_router.pkl: <1 MB
- **Total: <1 MB**

**Advanced Pipeline** (Recommended)
- tablenet.pth: ~100 MB
- yolov5.pt: ~14 MB
- ocr_router.pkl: <1 MB
- **Total: ~214 MB**

**Full Pipeline** (Complete)
- All models
- **Total: ~1.2 GB**

### Disk Space Recommendations

- **Minimum**: 500 MB free (for downloads + temp files)
- **Recommended**: 2 GB free
- **Optimal**: 5 GB free (for training/fine-tuning)

## Git LFS (Large File Storage)

If you're contributing model weights to the repository:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.pkl"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model weights"

# Add model files
git add tablenet.pth
git commit -m "Add TableNet weights"
git push
```

## Troubleshooting

### Issue: Download Failed
```
Error: Failed to download model
```

**Solutions:**
1. Check internet connection
2. Use VPN if region-blocked
3. Download manually and place in weights folder
4. Check disk space

### Issue: Corrupted File
```
Error: Cannot load model weights
```

**Solutions:**
1. Re-download the file
2. Verify file size matches expected
3. Check MD5 checksum
4. Ensure download completed fully

### Issue: Out of Disk Space
```
Error: No space left on device
```

**Solutions:**
1. Free up disk space
2. Download only essential models
3. Use external storage
4. Clean up temporary files

### Issue: Permission Denied
```
Error: Permission denied
```

**Solutions:**
```bash
# Fix permissions
chmod 644 *.pth *.pt *.bin *.pkl
chmod 755 .
```

## Model Updates

Models may be updated periodically. To update:

```bash
# Re-download with force flag
python model_downloader.py download --model yolov5 --force

# Or delete old weights and re-download
rm yolov5.pt
python model_downloader.py download --model yolov5
```

## Custom Models

To add your own trained models:

1. **Train your model**
2. **Save weights** in PyTorch format (`.pth` or `.pt`)
3. **Copy to weights folder**
4. **Update** `MODEL_REGISTRY` in `__init__.py`
5. **Create config file** in `configs/`
6. **Test loading**:
```python
from backend.models import get_model_registry
registry = get_model_registry()
model = registry.get_model('your_model_name')
```

## Backup

Important: **Backup your trained models!**

```bash
# Backup to external drive
cp *.pth *.pt *.bin /path/to/backup/

# Backup to cloud
rclone copy . remote:invoice-models/weights/

# Create archive
tar -czf model_weights_backup.tar.gz *.pth *.pt *.bin *.pkl
```

## Security

⚠️ **Security Considerations:**

- Model weights can contain sensitive information
- Only download from trusted sources
- Verify checksums before using
- Scan for malware if downloading from third parties
- Don't commit large files to git without LFS

## License Information

Model weights are subject to their original licenses:

- **YOLOv5**: GPL-3.0
- **BERT/LayoutLM**: Apache 2.0
- **DocTR**: Apache 2.0
- **Custom models**: Specify your license

## Support

For issues with model weights:

1. Check this README
2. See [QUICKSTART.md](../QUICKSTART.md)
3. Open an [issue](https://github.com/Cherry28831/Invoice-Data-Extractor/issues)
4. Join [discussions](https://github.com/Cherry28831/Invoice-Data-Extractor/discussions)

---

**Note**: This directory should contain only model weight files. Configuration files go in `configs/` directory.