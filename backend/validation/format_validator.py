"""
Format Validator
Validates format of various fields:
- Date formats (DD/MM/YYYY, etc.)
- Number formats
- Code formats (HSN, FSSAI, GSTIN)
- Email, phone formats
"""

import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class FormatValidator:
    """
    Validates format of extracted fields
    """
    
    def __init__(self):
        """Initialize format validator"""
        self.issues = []
        
        # Define validation patterns
        self.patterns = {
            'date': [
                r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY
                r'\d{1,2}-\d{1,2}-\d{4}',  # DD-MM-YYYY
                r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            ],
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[(]?\d{2,4}[)]?[-\s\.]?\d{3,4}[-\s\.]?\d{4,6}$',
            'hsn': r'^\d{4,8}$',  # 4-8 digit HSN code
            'sac': r'^\d{6}$',  # 6 digit SAC code
            'gstin': r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$',  # 15 char GSTIN
            'fssai': r'^\d{14}$',  # 14 digit FSSAI number
            'pincode': r'^\d{6}$',  # 6 digit pincode (India)
            'invoice_number': r'^[A-Z0-9/-]{3,20}$',  # Alphanumeric invoice number
        }
    
    def validate(self, extracted_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate formats in extracted data
        
        Args:
            extracted_data: List of extracted invoice items
            
        Returns:
            Tuple of (corrected_data, issues)
        """
        self.issues = []
        corrected_data = []
        
        for item in extracted_data:
            corrected_item = self._validate_item(item.copy())
            corrected_data.append(corrected_item)
        
        return corrected_data, self.issues
    
    def _validate_item(self, item: Dict) -> Dict:
        """
        Validate formats in a single item
        """
        # Validate date
        if 'Date of Invoice' in item:
            item = self._validate_date(item, 'Date of Invoice')
        
        # Validate HSN/SAC code
        if 'HSN/SAC Code' in item:
            item = self._validate_code(item, 'HSN/SAC Code', ['hsn', 'sac'])
        
        # Validate FSSAI number
        if 'FSSAI Number' in item:
            item = self._validate_code(item, 'FSSAI Number', ['fssai'])
        
        # Validate invoice number
        if 'Invoice Number' in item:
            item = self._validate_invoice_number(item)
        
        # Validate company name (should not be empty or just numbers)
        if 'Company Name' in item:
            item = self._validate_company_name(item)
        
        return item
    
    def _validate_date(self, item: Dict, field_name: str) -> Dict:
        """
        Validate and standardize date format
        """
        date_str = str(item.get(field_name, '')).strip()
        
        if not date_str or date_str == 'N/A':
            self.issues.append({
                'type': 'missing_date',
                'field': field_name,
                'message': f'{field_name} is missing',
                'severity': 'high'
            })
            return item
        
        # Try to parse date
        parsed_date = self._parse_date(date_str)
        
        if not parsed_date:
            self.issues.append({
                'type': 'invalid_date_format',
                'field': field_name,
                'value': date_str,
                'message': f'Invalid date format: {date_str}',
                'severity': 'high'
            })
        else:
            # Standardize to DD/MM/YYYY format
            standardized = parsed_date.strftime('%d/%m/%Y')
            if standardized != date_str:
                item[field_name] = standardized
                item['_date_corrected'] = True
                self.issues.append({
                    'type': 'date_format_corrected',
                    'field': field_name,
                    'original': date_str,
                    'corrected': standardized,
                    'severity': 'low'
                })
            
            # Validate date is reasonable (not in future, not too old)
            if not self._is_reasonable_date(parsed_date):
                self.issues.append({
                    'type': 'unreasonable_date',
                    'field': field_name,
                    'value': date_str,
                    'message': f'Date seems unreasonable: {standardized}',
                    'severity': 'medium'
                })
        
        return item
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Try to parse date from various formats
        """
        # Common date formats
        formats = [
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%d.%m.%Y',
            '%Y-%m-%d',
            '%d/%m/%y',
            '%d-%m-%y',
            '%d %B %Y',
            '%d %b %Y',
            '%B %d, %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _is_reasonable_date(self, date: datetime) -> bool:
        """
        Check if date is reasonable (not in future, not too old)
        """
        now = datetime.now()
        
        # Not in future
        if date > now:
            return False
        
        # Not more than 10 years old
        if (now - date).days > 3650:
            return False
        
        return True
    
    def _validate_code(
        self,
        item: Dict,
        field_name: str,
        code_types: List[str]
    ) -> Dict:
        """
        Validate HSN/SAC/FSSAI codes
        """
        code = str(item.get(field_name, '')).strip()
        
        if not code or code == 'N/A':
            return item
        
        # Remove spaces and special characters
        cleaned_code = re.sub(r'[^A-Z0-9]', '', code.upper())
        
        # Check against patterns
        valid = False
        for code_type in code_types:
            pattern = self.patterns.get(code_type)
            if pattern and re.match(pattern, cleaned_code):
                valid = True
                break
        
        if not valid:
            self.issues.append({
                'type': 'invalid_code_format',
                'field': field_name,
                'value': code,
                'message': f'Invalid {field_name} format: {code}',
                'severity': 'medium'
            })
        elif cleaned_code != code:
            # Code was cleaned
            item[field_name] = cleaned_code
            item[f'_{field_name}_corrected'] = True
        
        return item
    
    def _validate_invoice_number(self, item: Dict) -> Dict:
        """
        Validate invoice number format
        """
        invoice_num = str(item.get('Invoice Number', '')).strip()
        
        if not invoice_num or invoice_num == 'N/A':
            self.issues.append({
                'type': 'missing_invoice_number',
                'field': 'Invoice Number',
                'message': 'Invoice number is missing',
                'severity': 'critical'
            })
            return item
        
        # Clean invoice number
        cleaned = invoice_num.replace(' ', '').upper()
        
        # Check format
        if not re.match(self.patterns['invoice_number'], cleaned):
            self.issues.append({
                'type': 'invalid_invoice_number',
                'field': 'Invoice Number',
                'value': invoice_num,
                'message': f'Invalid invoice number format: {invoice_num}',
                'severity': 'high'
            })
        elif cleaned != invoice_num:
            item['Invoice Number'] = cleaned
            item['_invoice_number_corrected'] = True
        
        return item
    
    def _validate_company_name(self, item: Dict) -> Dict:
        """
        Validate company name
        """
        company_name = str(item.get('Company Name', '')).strip()
        
        if not company_name or company_name == 'N/A':
            self.issues.append({
                'type': 'missing_company_name',
                'field': 'Company Name',
                'message': 'Company name is missing',
                'severity': 'high'
            })
            return item
        
        # Check if it's just numbers
        if company_name.replace(' ', '').isdigit():
            self.issues.append({
                'type': 'invalid_company_name',
                'field': 'Company Name',
                'value': company_name,
                'message': 'Company name appears to be just numbers',
                'severity': 'high'
            })
        
        # Check minimum length
        if len(company_name) < 3:
            self.issues.append({
                'type': 'suspicious_company_name',
                'field': 'Company Name',
                'value': company_name,
                'message': 'Company name is too short',
                'severity': 'medium'
            })
        
        return item
    
    def validate_email(self, email: str) -> bool:
        """
        Validate email format
        """
        if not email:
            return False
        
        pattern = self.patterns['email']
        return bool(re.match(pattern, email.strip()))
    
    def validate_phone(self, phone: str) -> bool:
        """
        Validate phone number format
        """
        if not phone:
            return False
        
        pattern = self.patterns['phone']
        return bool(re.match(pattern, phone.strip()))
    
    def validate_gstin(self, gstin: str) -> bool:
        """
        Validate GSTIN format and checksum
        """
        if not gstin:
            return False
        
        gstin = gstin.strip().upper()
        
        # Check format
        if not re.match(self.patterns['gstin'], gstin):
            return False
        
        # Additional validation: Check state code
        state_code = int(gstin[:2])
        if state_code < 1 or state_code > 37:
            return False
        
        return True
    
    def get_summary(self) -> Dict:
        """
        Get validation summary
        """
        return {
            'total_issues': len(self.issues),
            'critical': len([i for i in self.issues if i['severity'] == 'critical']),
            'high_severity': len([i for i in self.issues if i['severity'] == 'high']),
            'medium_severity': len([i for i in self.issues if i['severity'] == 'medium']),
            'low_severity': len([i for i in self.issues if i['severity'] == 'low']),
            'corrected': len([i for i in self.issues if 'corrected' in i or 'original' in i]),
            'issues_by_type': self._group_issues_by_type()
        }
    
    def _group_issues_by_type(self) -> Dict:
        """
        Group issues by type
        """
        grouped = {}
        for issue in self.issues:
            issue_type = issue['type']
            grouped[issue_type] = grouped.get(issue_type, 0) + 1
        return grouped