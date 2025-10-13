"""
Rule-Based Extractor
Extracts invoice data using pattern matching and rules
Useful as fallback when LLM is unavailable or for simple invoices
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class RuleBasedExtractor:
    """
    Extracts invoice data using rule-based pattern matching
    """
    
    def __init__(self):
        """Initialize rule-based extractor"""
        # Define extraction patterns
        self.patterns = {
            'invoice_number': [
                r'invoice\s*(?:no|number|#)?\s*:?\s*([A-Z0-9/-]+)',
                r'inv\s*(?:no|#)?\s*:?\s*([A-Z0-9/-]+)',
                r'bill\s*no\s*:?\s*([A-Z0-9/-]+)'
            ],
            'date': [
                r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'dated\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
            ],
            'company': [
                r'([A-Z][a-zA-Z\s&]+(?:Pvt\.?|Ltd\.?|LLC|Inc\.?|Corp\.?))',
            ],
            'fssai': [
                r'fssai\s*(?:no|number|lic|license)?\s*:?\s*(\d{14})',
                r'license\s*no\s*:?\s*(\d{14})'
            ],
            'gstin': [
                r'gstin\s*:?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})',
                r'gst\s*no\s*:?\s*(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})'
            ],
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'phone': [
                r'(?:phone|mobile|tel|contact)\s*:?\s*([\+]?\d{10,15})',
                r'(\+?\d{10,15})'
            ],
            'amount': [
                r'[₹Rs\.]\s*([\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)\s*(?:INR|Rs\.?)'
            ]
        }
    
    def extract(
        self,
        text: str,
        ocr_result: Dict = None
    ) -> List[Dict]:
        """
        Extract invoice data using rules
        
        Args:
            text: Text to extract from
            ocr_result: Optional OCR result with structure
            
        Returns:
            List of extracted items
        """
        # Extract invoice-level fields
        invoice_fields = self._extract_invoice_fields(text)
        
        # Extract line items
        line_items = self._extract_line_items(text, ocr_result)
        
        # Combine invoice fields with line items
        if line_items:
            for item in line_items:
                item.update(invoice_fields)
        else:
            # No line items found, return invoice fields only
            line_items = [invoice_fields]
        
        return line_items
    
    def _extract_invoice_fields(self, text: str) -> Dict:
        """
        Extract invoice-level fields (company, invoice number, date, etc.)
        """
        fields = {}
        
        # Invoice number
        inv_num = self._extract_field(text, 'invoice_number')
        fields['Invoice Number'] = inv_num if inv_num else 'N/A'
        
        # Date
        date = self._extract_field(text, 'date')
        if date:
            # Standardize date format
            date = self._standardize_date(date)
        fields['Date of Invoice'] = date if date else 'N/A'
        
        # Company name
        company = self._extract_field(text, 'company')
        fields['Company Name'] = company if company else 'N/A'
        
        # FSSAI
        fssai = self._extract_field(text, 'fssai')
        fields['FSSAI Number'] = fssai if fssai else 'N/A'
        
        # GSTIN
        gstin = self._extract_field(text, 'gstin')
        if gstin:
            fields['GSTIN'] = gstin
        
        # Email
        email = self._extract_field(text, 'email')
        if email:
            fields['Email'] = email
        
        # Phone
        phone = self._extract_field(text, 'phone')
        if phone:
            fields['Phone'] = phone
        
        return fields
    
    def _extract_line_items(
        self,
        text: str,
        ocr_result: Dict = None
    ) -> List[Dict]:
        """
        Extract line items from invoice
        """
        line_items = []
        
        # Try to extract from structured OCR data first
        if ocr_result and ocr_result.get('lines'):
            line_items = self._extract_from_ocr_lines(ocr_result['lines'])
        
        # If no items from OCR, try pattern-based extraction
        if not line_items:
            line_items = self._extract_from_patterns(text)
        
        return line_items
    
    def _extract_from_ocr_lines(self, lines: List[Dict]) -> List[Dict]:
        """
        Extract line items from OCR lines
        """
        items = []
        
        # Look for table-like structure
        # Typically: Description | Quantity | Rate | Amount
        
        for line in lines:
            line_text = line.get('text', '').strip()
            
            if not line_text:
                continue
            
            # Skip header lines
            if any(keyword in line_text.lower() for keyword in ['description', 'quantity', 'rate', 'amount', 's.no', 'total']):
                continue
            
            # Try to parse line as item
            item = self._parse_line_as_item(line_text)
            
            if item:
                items.append(item)
        
        return items
    
    def _parse_line_as_item(self, line_text: str) -> Optional[Dict]:
        """
        Parse a single line as an invoice item
        """
        # Split line into parts
        parts = re.split(r'\s{2,}|\t', line_text)
        
        if len(parts) < 3:
            return None
        
        item = {}
        
        # Try to identify fields
        # Typically first part is description, last few are numbers
        
        # Description (usually first non-numeric part)
        for i, part in enumerate(parts):
            if not part.replace('.', '').replace(',', '').isdigit():
                item['Goods Description'] = part
                parts = parts[i+1:]
                break
        
        # Extract numbers (quantity, rate, amount)
        numbers = []
        for part in parts:
            number = self._extract_number(part)
            if number is not None:
                numbers.append(number)
        
        if len(numbers) >= 2:
            # Assume last is amount, second last is rate
            item['Amount'] = str(numbers[-1])
            item['Rate'] = str(numbers[-2])
            
            if len(numbers) >= 3:
                item['Quantity'] = str(numbers[-3])
        
        # Only return if we have minimum required fields
        if 'Goods Description' in item and 'Amount' in item:
            # Fill in missing fields
            if 'Quantity' not in item:
                item['Quantity'] = 'N/A'
            if 'Rate' not in item:
                item['Rate'] = 'N/A'
            
            item['HSN/SAC Code'] = 'N/A'
            item['Weight'] = 'N/A'
            
            return item
        
        return None
    
    def _extract_from_patterns(self, text: str) -> List[Dict]:
        """
        Extract line items using patterns (less reliable)
        """
        items = []
        
        # Look for product descriptions followed by numbers
        pattern = r'([A-Za-z][A-Za-z\s]+?)\s+([\d,]+)\s+.*?([\d,]+\.?\d*)'
        
        matches = re.finditer(pattern, text)
        
        for match in matches:
            description = match.group(1).strip()
            quantity = match.group(2).strip()
            amount = match.group(3).strip()
            
            # Filter out obvious non-items
            if any(keyword in description.lower() for keyword in ['total', 'subtotal', 'tax', 'invoice']):
                continue
            
            item = {
                'Goods Description': description,
                'Quantity': quantity,
                'Rate': 'N/A',
                'Amount': amount,
                'HSN/SAC Code': 'N/A',
                'Weight': 'N/A'
            }
            
            items.append(item)
        
        return items
    
    def _extract_field(self, text: str, field_type: str) -> Optional[str]:
        """
        Extract a specific field using patterns
        """
        patterns = self.patterns.get(field_type, [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract number from text
        """
        # Remove currency symbols and commas
        cleaned = re.sub(r'[₹$€£¥,\s]', '', text)
        cleaned = re.sub(r'Rs\.?', '', cleaned, flags=re.IGNORECASE)
        
        # Extract number
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
    
    def _standardize_date(self, date_str: str) -> str:
        """
        Standardize date to DD/MM/YYYY format
        """
        formats = [
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%d.%m.%Y',
            '%Y-%m-%d',
            '%d/%m/%y',
            '%d-%m-%y'
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        return date_str  # Return as-is if can't parse
    
    def extract_table_data(
        self,
        table_cells: List[Dict]
    ) -> List[Dict]:
        """
        Extract data from detected table cells
        
        Args:
            table_cells: List of table cells with text and position
            
        Returns:
            List of extracted items
        """
        # Group cells by row
        rows = {}
        for cell in table_cells:
            row = cell.get('row', 0)
            if row not in rows:
                rows[row] = []
            rows[row].append(cell)
        
        # Sort cells in each row by column
        for row_idx in rows:
            rows[row_idx] = sorted(rows[row_idx], key=lambda c: c.get('col', 0))
        
        # Extract header (typically first row)
        if 0 in rows:
            headers = [cell.get('text', '').lower() for cell in rows[0]]
        else:
            headers = []
        
        # Extract data rows
        items = []
        for row_idx in sorted(rows.keys())[1:]:  # Skip header row
            row_cells = rows[row_idx]
            
            item = {}
            for i, cell in enumerate(row_cells):
                text = cell.get('text', '').strip()
                
                # Map to field based on header
                if i < len(headers):
                    header = headers[i]
                    field_name = self._map_header_to_field(header)
                    item[field_name] = text
            
            if item:
                items.append(item)
        
        return items
    
    def _map_header_to_field(self, header: str) -> str:
        """
        Map table header to standard field name
        """
        mapping = {
            'description': 'Goods Description',
            'item': 'Goods Description',
            'product': 'Goods Description',
            'goods': 'Goods Description',
            'quantity': 'Quantity',
            'qty': 'Quantity',
            'rate': 'Rate',
            'price': 'Rate',
            'amount': 'Amount',
            'total': 'Amount',
            'hsn': 'HSN/SAC Code',
            'sac': 'HSN/SAC Code',
            'weight': 'Weight',
            'wt': 'Weight'
        }
        
        for key, value in mapping.items():
            if key in header:
                return value
        
        return header.capitalize()