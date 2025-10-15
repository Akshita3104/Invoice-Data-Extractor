"""
Common helper utilities for the Invoice Extractor
Includes file operations, data validation, and conversion functions
"""

import os
import re
import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal, InvalidOperation
from .logger import get_logger

logger = get_logger(__name__)


# ===================== File Operations =====================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(filepath: Union[str, Path]) -> str:
    """Get file extension in lowercase"""
    return Path(filepath).suffix.lower()


def is_valid_file(filepath: Union[str, Path], extensions: List[str]) -> bool:
    """Check if file exists and has valid extension"""
    path = Path(filepath)
    return path.exists() and path.is_file() and get_file_extension(path) in extensions


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Get file size in megabytes"""
    return Path(filepath).stat().st_size / (1024 * 1024)


def list_files(directory: Union[str, Path], extensions: Optional[List[str]] = None, recursive: bool = False) -> List[Path]:
    """List files in directory with optional filtering"""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    pattern = '**/*' if recursive else '*'
    files = []
    
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            if extensions is None or get_file_extension(file_path) in extensions:
                files.append(file_path)
    
    return sorted(files)


def safe_read_json(filepath: Union[str, Path]) -> Optional[Dict]:
    """Safely read JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {filepath}: {e}")
        return None


def safe_write_json(data: Dict, filepath: Union[str, Path], indent: int = 4):
    """Safely write JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"JSON written to {filepath}")
    except Exception as e:
        logger.error(f"Error writing JSON file {filepath}: {e}")


# ===================== Data Validation =====================

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number (Indian format)"""
    # Remove spaces and special characters
    phone_clean = re.sub(r'[^\d+]', '', phone)
    # Check if valid Indian phone number
    pattern = r'^(\+91|91)?[6-9]\d{9}$'
    return bool(re.match(pattern, phone_clean))


def validate_gst(gst: str) -> bool:
    """Validate GST number format"""
    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
    return bool(re.match(pattern, gst.upper().replace(' ', '')))


def validate_pan(pan: str) -> bool:
    """Validate PAN number format"""
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    return bool(re.match(pattern, pan.upper().replace(' ', '')))


def validate_fssai(fssai: str) -> bool:
    """Validate FSSAI number format"""
    fssai_clean = re.sub(r'[^\d]', '', fssai)
    return len(fssai_clean) == 14 and fssai_clean.isdigit()


def validate_date(date_str: str, format: str = '%d/%m/%Y') -> bool:
    """Validate date string"""
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False


# ===================== Data Conversion =====================

def parse_date(date_str: str, output_format: str = '%d/%m/%Y') -> Optional[str]:
    """Parse date from various formats to standard format"""
    if not date_str:
        return None
    
    # Common date formats
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
        '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d',
        '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',
        '%d %B %Y', '%d %b %Y',
        '%B %d, %Y', '%b %d, %Y'
    ]
    
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str.strip(), fmt)
            return date_obj.strftime(output_format)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def parse_amount(amount_str: str) -> Optional[Decimal]:
    """Parse amount from string to Decimal"""
    if not amount_str:
        return None
    
    try:
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[₹$€£,\s]', '', str(amount_str))
        # Handle parentheses for negative numbers
        if '(' in cleaned and ')' in cleaned:
            cleaned = '-' + cleaned.replace('(', '').replace(')', '')
        return Decimal(cleaned)
    except (InvalidOperation, ValueError) as e:
        logger.warning(f"Could not parse amount: {amount_str}")
        return None


def parse_quantity(qty_str: str) -> Optional[float]:
    """Parse quantity from string"""
    if not qty_str:
        return None
    
    try:
        # Extract first number from string
        match = re.search(r'[\d,.]+', str(qty_str))
        if match:
            cleaned = match.group().replace(',', '')
            return float(cleaned)
    except ValueError:
        pass
    
    logger.warning(f"Could not parse quantity: {qty_str}")
    return None


def convert_weight_to_kg(weight_str: str) -> Optional[float]:
    """Convert weight from various units to kilograms"""
    if not weight_str:
        return None
    
    # Remove commas and extra spaces
    weight_str = weight_str.replace(',', '').strip()
    
    # Extract number and unit
    match = re.match(r'([\d.]+)\s*([a-zA-Z]*)', weight_str)
    if not match:
        return None
    
    try:
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        # Conversion factors
        conversions = {
            'kg': 1.0,
            'kgs': 1.0,
            'kilogram': 1.0,
            'kilograms': 1.0,
            'g': 0.001,
            'gm': 0.001,
            'gms': 0.001,
            'gram': 0.001,
            'grams': 0.001,
            'quintal': 100.0,
            'qtl': 100.0,
            'ton': 1000.0,
            'tons': 1000.0,
            'mt': 1000.0,
            'tonne': 1000.0,
            'tonnes': 1000.0,
            'lb': 0.453592,
            'lbs': 0.453592,
            'pound': 0.453592,
            'pounds': 0.453592,
            'oz': 0.0283495,
            'ounce': 0.0283495,
            'ounces': 0.0283495
        }
        
        factor = conversions.get(unit, 1.0)
        return value * factor
        
    except ValueError as e:
        logger.warning(f"Could not convert weight: {weight_str}")
        return None


def normalize_text(text: str) -> str:
    """Normalize text by removing extra spaces and special characters"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep common punctuation
    text = re.sub(r'[^\w\s.,:-]', '', text)
    return text.strip()


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    if not text:
        return []
    
    # Find all numbers (including decimals)
    matches = re.findall(r'[\d,]+\.?\d*', text)
    numbers = []
    
    for match in matches:
        try:
            cleaned = match.replace(',', '')
            numbers.append(float(cleaned))
        except ValueError:
            continue
    
    return numbers


# ===================== Text Processing =====================

def clean_ocr_text(text: str) -> str:
    """Clean OCR text by removing noise"""
    if not text:
        return ""
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()


def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from text"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(pattern, text)
    return match.group() if match else None


def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number from text"""
    # Look for Indian phone numbers
    pattern = r'(\+91[\s-]?)?[6-9]\d{9}'
    match = re.search(pattern, text)
    return match.group() if match else None


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ===================== Data Structures =====================

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """Unflatten dictionary with dot-separated keys"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries (later dicts override earlier ones)"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def remove_none_values(d: Dict) -> Dict:
    """Remove keys with None values from dictionary"""
    return {k: v for k, v in d.items() if v is not None}


# ===================== Performance =====================

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure"""
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def timeout(seconds: int):
    """Decorator to add timeout to function"""
    import functools
    import signal
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator