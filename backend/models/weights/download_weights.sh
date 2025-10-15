#!/bin/bash
# Quick script to download all model weights
# Usage: bash download_weights.sh

set -e  # Exit on error

echo "=========================================="
echo "Invoice Extractor - Weight Downloader"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOADER="wget"
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl"
    DOWNLOAD_CMD="curl -L -o"
else
    echo -e "${RED}Error: Neither wget nor curl found. Please install one.${NC}"
    exit 1
fi

echo "Using $DOWNLOADER for downloads..."
echo ""

# Function to download file
download_file() {
    local url=$1
    local filename=$2
    local description=$3
    
    if [ -f "$filename" ]; then
        echo -e "${YELLOW}⚠ $filename already exists. Skipping...${NC}"
        return 0
    fi
    
    echo -e "${GREEN}Downloading $description...${NC}"
    if $DOWNLOAD_CMD "$filename" "$url"; then
        echo -e "${GREEN}✓ Downloaded $filename${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to download $filename${NC}"
        return 1
    fi
}

# Download YOLOv5 (Official, always available)
echo "=== YOLOv5 Weights ==="
download_file \
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt" \
    "yolov5.pt" \
    "YOLOv5s weights (~14 MB)"
echo ""

# DocTR weights (if URL is available)
echo "=== DocTR Weights ==="
echo -e "${YELLOW}Note: DocTR weights are typically downloaded automatically by the library${NC}"
echo -e "${YELLOW}If you need to pre-download, visit: https://github.com/mindee/doctr/releases${NC}"
echo ""

# HuggingFace models (requires transformers library)
echo "=== HuggingFace Models ==="
echo -e "${YELLOW}BERT-NER and LayoutLM weights are downloaded automatically via transformers${NC}"
echo -e "${YELLOW}No manual download needed${NC}"
echo ""

# Custom models (TableNet, OCR Router)
echo "=== Custom Models ==="
echo -e "${YELLOW}TableNet and OCR Router require custom training or specific release URLs${NC}"
echo ""
echo "To download these models, use one of these methods:"
echo "1. Use the Python downloader: python ../model_downloader.py download-all"
echo "2. Download from GitHub releases (if available)"
echo "3. Train your own models"
echo ""

# Summary
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo ""

check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo -e "${GREEN}✓ $1 ($size)${NC}"
    else
        echo -e "${RED}✗ $1 (not found)${NC}"
    fi
}

check_file "tablenet.pth"
check_file "yolov5.pt"
check_file "bert-ner.bin"
check_file "ocr_router.pkl"
check_file "doctr_db_resnet50.pt"
check_file "layoutlm_base.bin"

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. For missing models, use: python ../model_downloader.py download --model <name>"
echo "2. Or run the setup script: python ../setup_models.py"
echo "3. Verify models: python ../model_downloader.py list"
echo ""
echo "For more information, see: README.md"
echo ""