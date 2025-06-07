# 🧾 Invoice Data Extractor

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Cherry28831/Invoice-Data-Extractor)
![MIT License](https://img.shields.io/github/license/Cherry28831/Invoice-Data-Extractor)
![Platform](https://img.shields.io/badge/platform-Windows-blue)

A lightweight yet powerful desktop app that extracts data from invoices in **PDF** or **JPG** format with over **95% accuracy**. This is possible using a mix of **OCR**, **Generative AI (Gemini)**, and smart processing — all bundled into a single `.exe` file!

---

## 📦 Features

- 💡 Intelligent extraction using **Google Generative AI**
- 📄 Supports **PDF** and **JPG** invoice formats
- 🧠 Uses `pytesseract`, `pdfplumber`, and `pdf2image`
- 📊 Exports structured data to **Excel**
- 🖼️ Intuitive **Tkinter GUI**
- ⚙️ Backend logic is bundled — **no additional installation needed!**

---

## 🧠 Tech Stack & Libraries Used

- [`pdfplumber`](https://github.com/jsvine/pdfplumber)
- [`pdf2image`](https://github.com/Belval/pdf2image)
- [`pytesseract`](https://github.com/madmaze/pytesseract)
- [`google-generativeai`](https://github.com/google/generative-ai-python)
- `tkinter` (GUI)
- `pandas`, `openpyxl` (Excel generation)

---

🧠 How It Works
- PDFs are read using pdfplumber. If text extraction fails, it switches to OCR using pytesseract.
- Extracted text is sent to the Gemini API with a prompt for field extraction.
- Parsed response is saved to combined_data.json and converted to an Excel file.
- All weights are converted to kilograms.

> ✅ Since we bundled `backend.py` into the final executable, there is **no need to install any libraries** or set up Python locally. Everything runs from the `.exe` — plug and play!

---

## 🚀 Getting Started

### 🔽 Download the Executable

[⬇ Click here to download the `.exe` file (v1.0.0)](https://github.com/Cherry28831/Invoice-Data-Extractor/releases/tag/v1.0.0)

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

📧 Output
- `combined_data.json`: Raw structured data
- `combined_invoice_data.xlsx`: Final Excel export
- JSON is deleted after Excel is generated.

---

## 🧪 Want to Build from Source?

git clone https://github.com/Akshita3104/Invoice-Data-Extractor.git
cd Invoice-Data-Extractor
pip install -r requirements.txt
npm run dist  # if using Electron or similar packaging tools


Built with 💙 by Akshita3104, Cherry28831, and itisar-345
Open for contributions, forks, and feedback!
