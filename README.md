# 🚀 Advanced Invoice Extraction System

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Cherry28831/Invoice-Data-Extractor)
![MIT License](https://img.shields.io/github/license/Cherry28831/Invoice-Data-Extractor)
![Platform](https://img.shields.io/badge/platform-Windows-blue)

---

A production-ready, multi-modal invoice extraction system with state-of-the-art document understanding capabilities.

## 📋 Features

### Core Capabilities
- ✅ **Multi-format Support**: PDF, JPEG, PNG, TIFF
- ✅ **Quality Assessment**: Automatic quality detection and adaptive preprocessing
- ✅ **Multi-Engine OCR**: Tesseract, DocTR, TrOCR with intelligent routing
- ✅ **Layout Analysis**: Zone segmentation, table detection, reading order
- ✅ **Document Graph**: Graph Neural Networks for structural reasoning
- ✅ **Multimodal Fusion**: Visual + Text + Layout + Graph features
- ✅ **Hybrid Extraction**: LLM (Gemini) + Rule-based for best results
- ✅ **Multi-Layer Validation**: Arithmetic, format, consistency, plausibility
- ✅ **Multiple Export Formats**: Excel, CSV, JSON, PDF reports

### Advanced Features
- 🔥 Adaptive preprocessing based on image quality
- 🔥 Ensemble OCR with confidence scoring
- 🔥 Attention-based multimodal fusion
- 🔥 Graph Neural Network reasoning
- 🔥 Automatic field detection and entity classification
- 🔥 Cross-validation between extraction methods
- 🔥 Comprehensive confidence scoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Input                           │
│              (PDF, JPEG, PNG, TIFF)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  INGESTION: Format handling, Quality assessment             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PREPROCESSING: Adaptive enhancement (denoise, skew, etc)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  OCR: Multi-engine routing (Tesseract/DocTR/TrOCR)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYOUT ANALYSIS: Zones, Tables, Reading order              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  GRAPH: Document graph + GNN reasoning                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MULTIMODAL: Feature fusion (Visual+Text+Layout+Graph)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  EXTRACTION: Hybrid LLM + Rule-based                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  VALIDATION: Arithmetic, Format, Consistency, Plausibility  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  EXPORT: Excel, CSV, JSON, PDF                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd invoice-extractor

# Install basic dependencies
pip install -r requirements.txt

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr poppler-utils

# MacOS:
brew install tesseract poppler
```

### 2. Basic Usage

```python
from backend.backend import InvoiceExtractionPipeline

# Initialize pipeline
pipeline = InvoiceExtractionPipeline(
    api_key="your-gemini-api-key",
    enable_advanced_features=False,  # Set True for full features
    use_gpu=False
)

# Process single document
result = pipeline.process_document(
    document_path="path/to/invoice.pdf",
    output_folder="./output",
    filename="invoice_data.xlsx"
)

if result['success']:
    print(f"Extracted {len(result['extracted_data'])} items")
    print(f"Confidence: {result['confidence']['overall_confidence']:.1%}")
```

### 3. Process Multiple Documents

```python
# Process batch
result = pipeline.process_multiple_documents(
    document_paths=["invoice1.pdf", "invoice2.pdf", "invoice3.pdf"],
    output_folder="./output",
    filename="combined_invoices.xlsx"
)

print(f"Processed {result['successful']}/{result['total_documents']} documents")
print(f"Total items: {result['total_items']}")
```

## 📁 Project Structure

```
invoice-extractor/
├── backend/
│   ├── backend.py              # Main pipeline orchestrator
│   ├── ingestion/              # Multi-format document handling
│   ├── preprocessing/          # Adaptive quality enhancement
│   ├── ocr/                    # Multi-engine OCR
│   ├── layout_analysis/        # Document structure analysis
│   ├── graph/                  # Document graph & GNN
│   ├── multimodal/             # Feature fusion
│   ├── extraction/             # Data extraction
│   ├── validation/             # Multi-layer validation
│   └── export/                 # Multiple output formats
├── frontend/                   # Electron UI
├── main.js                     # Electron main process
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚙️ Configuration

### Enable Advanced Features

```python
pipeline = InvoiceExtractionPipeline(
    api_key="your-api-key",
    enable_advanced_features=True,  # Enable GNN, multimodal fusion
    use_gpu=True  # Use GPU acceleration
)
```

### Custom Validation Thresholds

```python
from validation import ArithmeticValidator

validator = ArithmeticValidator(tolerance=0.05)  # 5% tolerance
```

### Export Options

```python
from export import ExcelExporter, CSVExporter, JSONExporter

# Excel with validation
excel_exporter.export(
    data=extracted_data,
    output_folder="./output",
    include_validation=True,
    validation_issues=issues,
    confidence_scores=confidence
)

# JSON with grouping
json_exporter.export_structured(
    data=extracted_data,
    output_path="./output/invoices.json",
    group_by_invoice=True
)
```

## 📊 Output

The system generates:

1. **Excel File** (Multi-sheet):
   - Invoice Data
   - Validation Issues
   - Confidence Scores
   - Summary Statistics

2. **JSON File**:
   - Structured data with metadata
   - Validation issues
   - Confidence metrics

3. **PDF Report** (Optional):
   - Summary
   - Data tables
   - Validation results

## 🎯 Extracted Fields

- Goods Description
- HSN/SAC Code
- Quantity
- Weight (with unit conversion)
- Rate (per unit)
- Amount
- Company Name
- Invoice Number
- FSSAI Number
- Date of Invoice (DD/MM/YYYY)

## 🔧 Advanced Usage

### Using Individual Modules

```python
# OCR only
from ocr import OCRRouter

router = OCRRouter(enable_ensemble=True)
result = router.extract_text(image)

# Validation only
from validation import ArithmeticValidator, FormatValidator

arith = ArithmeticValidator()
data, issues = arith.validate(extracted_data)

# Export only
from export import ExcelExporter

exporter = ExcelExporter()
exporter.export(data, "./output", "invoices.xlsx")
```

## 📈 Performance

- **Basic Mode** (No advanced features): ~5-10 seconds/page
- **Advanced Mode** (Full features): ~15-30 seconds/page
- **Accuracy**: 95%+ on good quality documents
- **Supported Languages**: English (extendable)

## 🐛 Troubleshooting

### Common Issues

**"Tesseract not found"**
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr
```

**"Module not found"**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**Low extraction accuracy**
- Check image quality
- Enable preprocessing
- Try ensemble OCR
- Enable advanced features

## 🧪 Want to Build from Source?

git clone https://github.com/Cherry28831/Invoice-Data-Extractor.git
cd Invoice-Data-Extractor
pip install -r requirements.txt
npm run dist  # if using Electron or similar packaging tools

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional OCR engines
- More validation rules
- Support for more languages
- UI improvements

## 🚀 Getting Started

### 🔽 Download the Executable

[⬇ Click here to download the `.exe` file (v1.0.1)](https://github.com/Cherry28831/Invoice-Data-Extractor/releases/tag/v1.0.1)

---

### 🔑 Required: Google Gemini API Key

This app requires access to Google's Generative AI API (Gemini). You can get a **free API key** by following the guide below:

📄 [Read API Key Setup Guide](https://github.com/Cherry28831/Invoice-Data-Extractor/blob/main/API%20Documentation.docx)

---

## 🛠 How to Use

1. Generate a free API key from Google Cloud Console.
2. Download and run the `.exe` file.
3. When prompted, paste your API key.
4. Upload a PDF or JPG invoice.
5. The app will extract the data and save it to Excel!

---

## 🙏 Acknowledgments

- Google Gemini for LLM
- Tesseract OCR
- PyTorch ecosystem
- ReportLab for PDF generation

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ for accurate invoice extraction**