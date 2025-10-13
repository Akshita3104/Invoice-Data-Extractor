"""
Consistency Validator
Validates cross-field consistency:
- Company name consistency across invoices
- Date range validation
- Invoice number uniqueness
- Related fields consistency
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime, timedelta


class ConsistencyValidator:
    """
    Validates consistency across fields and multiple invoices
    """
    
    def __init__(self):
        """Initialize consistency validator"""
        self.issues = []
        self.invoice_history = []  # Track previously seen invoices
    
    def validate(self, extracted_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate consistency in extracted data
        
        Args:
            extracted_data: List of extracted invoice items
            
        Returns:
            Tuple of (data, issues)
        """
        self.issues = []
        
        # Check for duplicate entries
        self._check_duplicates(extracted_data)
        
        # Check company name consistency
        self._check_company_consistency(extracted_data)
        
        # Check date consistency
        self._check_date_consistency(extracted_data)
        
        # Check quantity/weight consistency
        self._check_quantity_weight_consistency(extracted_data)
        
        # Check invoice number format consistency
        self._check_invoice_number_consistency(extracted_data)
        
        return extracted_data, self.issues
    
    def _check_duplicates(self, data: List[Dict]):
        """
        Check for duplicate items
        """
        descriptions = []
        for item in data:
            desc = item.get('Goods Description', '').lower().strip()
            if desc:
                descriptions.append(desc)
        
        # Count occurrences
        counts = Counter(descriptions)
        duplicates = {desc: count for desc, count in counts.items() if count > 1}
        
        if duplicates:
            self.issues.append({
                'type': 'duplicate_items',
                'duplicates': duplicates,
                'message': f'Found duplicate items: {list(duplicates.keys())}',
                'severity': 'medium'
            })
    
    def _check_company_consistency(self, data: List[Dict]):
        """
        Check if company name is consistent
        """
        company_names = []
        for item in data:
            name = item.get('Company Name', '').strip()
            if name and name != 'N/A':
                company_names.append(name)
        
        if not company_names:
            return
        
        # Check if all names are similar
        unique_names = set(company_names)
        
        if len(unique_names) > 1:
            # Multiple different company names
            self.issues.append({
                'type': 'inconsistent_company_name',
                'names': list(unique_names),
                'message': f'Multiple company names found: {list(unique_names)}',
                'severity': 'high'
            })
    
    def _check_date_consistency(self, data: List[Dict]):
        """
        Check date consistency and reasonableness
        """
        dates = []
        for item in data:
            date_str = item.get('Date of Invoice', '')
            if date_str and date_str != 'N/A':
                try:
                    # Try to parse date
                    date = self._parse_date(date_str)
                    if date:
                        dates.append(date)
                except:
                    pass
        
        if not dates:
            return
        
        # Check if all dates are the same (should be for same invoice)
        unique_dates = set(dates)
        
        if len(unique_dates) > 1:
            self.issues.append({
                'type': 'inconsistent_dates',
                'dates': [d.strftime('%d/%m/%Y') for d in unique_dates],
                'message': 'Multiple invoice dates found in same document',
                'severity': 'high'
            })
        
        # Check if date is in reasonable range
        latest_date = max(dates)
        now = datetime.now()
        
        if latest_date > now:
            self.issues.append({
                'type': 'future_date',
                'date': latest_date.strftime('%d/%m/%Y'),
                'message': 'Invoice date is in the future',
                'severity': 'critical'
            })
        
        if (now - latest_date).days > 1825:  # 5 years
            self.issues.append({
                'type': 'very_old_invoice',
                'date': latest_date.strftime('%d/%m/%Y'),
                'message': 'Invoice date is more than 5 years old',
                'severity': 'medium'
            })
    
    def _check_quantity_weight_consistency(self, data: List[Dict]):
        """
        Check if quantity and weight fields are consistent
        """
        for item in data:
            quantity_str = str(item.get('Quantity', '')).strip()
            weight_str = str(item.get('Weight', '')).strip()
            
            if not quantity_str or not weight_str:
                continue
            
            if quantity_str == 'N/A' or weight_str == 'N/A':
                continue
            
            # Parse quantity
            try:
                quantity = float(quantity_str.replace(',', ''))
            except:
                continue
            
            # Check if weight makes sense for quantity
            # (This is domain-specific logic)
            if quantity > 1000:
                self.issues.append({
                    'type': 'unusually_high_quantity',
                    'item': item.get('Goods Description', 'Unknown'),
                    'quantity': quantity,
                    'message': f'Unusually high quantity: {quantity}',
                    'severity': 'medium'
                })
            
            # Check if weight is 'N/A' but quantity exists
            if weight_str == 'N/A' and quantity > 0:
                self.issues.append({
                    'type': 'missing_weight',
                    'item': item.get('Goods Description', 'Unknown'),
                    'message': 'Weight is missing but quantity is present',
                    'severity': 'low'
                })
    
    def _check_invoice_number_consistency(self, data: List[Dict]):
        """
        Check invoice number format consistency
        """
        invoice_numbers = []
        for item in data:
            inv_num = item.get('Invoice Number', '').strip()
            if inv_num and inv_num != 'N/A':
                invoice_numbers.append(inv_num)
        
        if not invoice_numbers:
            return
        
        # Check if all invoice numbers are the same
        unique_numbers = set(invoice_numbers)
        
        if len(unique_numbers) > 1:
            self.issues.append({
                'type': 'multiple_invoice_numbers',
                'numbers': list(unique_numbers),
                'message': 'Multiple invoice numbers found in same document',
                'severity': 'critical'
            })
        
        # Check against history for duplicates
        if self.invoice_history:
            for inv_num in unique_numbers:
                if inv_num in self.invoice_history:
                    self.issues.append({
                        'type': 'duplicate_invoice_number',
                        'invoice_number': inv_num,
                        'message': f'Invoice number already seen: {inv_num}',
                        'severity': 'critical'
                    })
        
        # Add to history
        self.invoice_history.extend(unique_numbers)
    
    def validate_field_relationships(self, item: Dict) -> List[Dict]:
        """
        Validate relationships between related fields
        
        Returns:
            List of validation issues
        """
        field_issues = []
        
        # Rate and Amount should have same currency
        rate = str(item.get('Rate', ''))
        amount = str(item.get('Amount', ''))
        
        rate_currency = self._extract_currency(rate)
        amount_currency = self._extract_currency(amount)
        
        if rate_currency and amount_currency and rate_currency != amount_currency:
            field_issues.append({
                'type': 'currency_mismatch',
                'item': item.get('Goods Description', 'Unknown'),
                'message': f'Currency mismatch: Rate has {rate_currency}, Amount has {amount_currency}',
                'severity': 'high'
            })
        
        # HSN and SAC are mutually exclusive
        hsn = item.get('HSN/SAC Code', '')
        if hsn and hsn != 'N/A':
            # Determine if it's HSN or SAC
            if len(hsn) == 6:
                # Likely SAC (services)
                if 'goods' in item.get('Goods Description', '').lower():
                    field_issues.append({
                        'type': 'hsn_sac_mismatch',
                        'message': 'SAC code used for goods (should be HSN)',
                        'severity': 'medium'
                    })
            elif len(hsn) in [4, 8]:
                # Likely HSN (goods)
                if 'service' in item.get('Goods Description', '').lower():
                    field_issues.append({
                        'type': 'hsn_sac_mismatch',
                        'message': 'HSN code used for service (should be SAC)',
                        'severity': 'medium'
                    })
        
        return field_issues
    
    def check_buyer_seller_consistency(
        self,
        buyer_info: Dict,
        seller_info: Dict
    ) -> List[Dict]:
        """
        Check consistency between buyer and seller information
        """
        issues = []
        
        # Buyer and seller should not have same GSTIN
        buyer_gstin = buyer_info.get('gstin', '')
        seller_gstin = seller_info.get('gstin', '')
        
        if buyer_gstin and seller_gstin and buyer_gstin == seller_gstin:
            issues.append({
                'type': 'same_buyer_seller_gstin',
                'message': 'Buyer and seller have same GSTIN',
                'severity': 'critical'
            })
        
        # Check state codes in GSTIN
        if buyer_gstin and seller_gstin:
            buyer_state = buyer_gstin[:2]
            seller_state = seller_gstin[:2]
            
            # This determines if IGST or CGST/SGST should be used
            if buyer_state == seller_state:
                # Same state - should have CGST and SGST
                issues.append({
                    'type': 'intra_state_transaction',
                    'message': 'Intra-state transaction (should have CGST + SGST)',
                    'severity': 'info'
                })
            else:
                # Different states - should have IGST
                issues.append({
                    'type': 'inter_state_transaction',
                    'message': 'Inter-state transaction (should have IGST)',
                    'severity': 'info'
                })
        
        return issues
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date from string
        """
        formats = [
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d',
            '%d/%m/%y',
            '%d-%m-%y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_currency(self, value_str: str) -> Optional[str]:
        """
        Extract currency symbol from value string
        """
        currencies = {
            '₹': 'INR',
            'Rs': 'INR',
            ': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY'
        }
        
        for symbol, code in currencies.items():
            if symbol in value_str:
                return code
        
        return None
    
    def validate_sequence(self, invoice_numbers: List[str]) -> List[Dict]:
        """
        Validate invoice number sequence
        (Useful when processing multiple invoices from same vendor)
        """
        issues = []
        
        # Extract numeric parts
        numeric_parts = []
        for inv_num in invoice_numbers:
            # Extract last number in invoice number
            import re
            numbers = re.findall(r'\d+', inv_num)
            if numbers:
                numeric_parts.append((inv_num, int(numbers[-1])))
        
        if len(numeric_parts) < 2:
            return issues
        
        # Check for gaps in sequence
        sorted_nums = sorted(numeric_parts, key=lambda x: x[1])
        
        for i in range(len(sorted_nums) - 1):
            current = sorted_nums[i][1]
            next_num = sorted_nums[i + 1][1]
            
            if next_num - current > 1:
                issues.append({
                    'type': 'invoice_sequence_gap',
                    'gap': (sorted_nums[i][0], sorted_nums[i + 1][0]),
                    'message': f'Gap in invoice sequence between {sorted_nums[i][0]} and {sorted_nums[i + 1][0]}',
                    'severity': 'low'
                })
        
        return issues
    
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
            'info': len([i for i in self.issues if i['severity'] == 'info']),
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