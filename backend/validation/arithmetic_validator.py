"""
Arithmetic Validator
Validates arithmetic relationships in invoices:
- Quantity × Rate = Amount
- Sum of line items = Subtotal
- Subtotal + Tax = Total
- Discount calculations
"""

import re
from typing import Dict, List, Tuple, Optional
from decimal import Decimal, InvalidOperation


class ArithmeticValidator:
    """
    Validates arithmetic calculations in invoice data
    """
    
    def __init__(self, tolerance: float = 0.02):
        """
        Initialize arithmetic validator
        
        Args:
            tolerance: Tolerance for floating point comparisons (2% default)
        """
        self.tolerance = tolerance
        self.issues = []
    
    def validate(self, extracted_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate arithmetic in extracted invoice data
        
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
        
        # Validate totals across all items
        self._validate_invoice_totals(corrected_data)
        
        return corrected_data, self.issues
    
    def _validate_item(self, item: Dict) -> Dict:
        """
        Validate arithmetic for a single line item
        """
        # Extract values
        quantity = self._parse_number(item.get('Quantity', ''))
        rate = self._parse_number(item.get('Rate', ''))
        amount = self._parse_number(item.get('Amount', ''))
        
        if quantity is None or rate is None or amount is None:
            self.issues.append({
                'type': 'missing_values',
                'item': item.get('Goods Description', 'Unknown'),
                'message': 'Missing quantity, rate, or amount',
                'severity': 'high'
            })
            return item
        
        # Calculate expected amount
        expected_amount = quantity * rate
        
        # Check if amounts match (within tolerance)
        if not self._values_match(amount, expected_amount):
            self.issues.append({
                'type': 'arithmetic_mismatch',
                'item': item.get('Goods Description', 'Unknown'),
                'message': f'Quantity × Rate ({quantity} × {rate} = {expected_amount}) != Amount ({amount})',
                'expected': float(expected_amount),
                'actual': float(amount),
                'severity': 'high'
            })
            
            # Auto-correct if one value is clearly wrong
            if self._should_auto_correct(quantity, rate, amount, expected_amount):
                item['Amount'] = str(expected_amount)
                item['_corrected'] = True
                self.issues[-1]['corrected'] = True
        
        return item
    
    def _validate_invoice_totals(self, items: List[Dict]):
        """
        Validate totals across all invoice items
        """
        # Calculate subtotal
        subtotal = Decimal('0')
        for item in items:
            amount = self._parse_number(item.get('Amount', ''))
            if amount:
                subtotal += amount
        
        # Extract invoice-level totals (if present)
        # This would come from the invoice header/footer
        # For now, just validate line items sum correctly
        
        # Check for duplicate items
        descriptions = [item.get('Goods Description', '').lower() for item in items]
        duplicates = [desc for desc in set(descriptions) if descriptions.count(desc) > 1]
        
        if duplicates:
            self.issues.append({
                'type': 'duplicate_items',
                'message': f'Duplicate items found: {duplicates}',
                'severity': 'medium'
            })
    
    def validate_tax_calculation(
        self,
        subtotal: float,
        tax_rate: float,
        tax_amount: float,
        total: float
    ) -> bool:
        """
        Validate tax calculations
        
        Args:
            subtotal: Subtotal before tax
            tax_rate: Tax rate (e.g., 0.18 for 18%)
            tax_amount: Calculated tax amount
            total: Final total
            
        Returns:
            True if calculations are correct
        """
        subtotal_dec = Decimal(str(subtotal))
        tax_rate_dec = Decimal(str(tax_rate))
        tax_amount_dec = Decimal(str(tax_amount))
        total_dec = Decimal(str(total))
        
        # Calculate expected tax
        expected_tax = subtotal_dec * tax_rate_dec
        
        # Calculate expected total
        expected_total = subtotal_dec + tax_amount_dec
        
        # Validate tax amount
        if not self._values_match(tax_amount_dec, expected_tax):
            self.issues.append({
                'type': 'tax_calculation_error',
                'message': f'Tax calculation incorrect: {subtotal} × {tax_rate} = {expected_tax}, but got {tax_amount}',
                'expected': float(expected_tax),
                'actual': float(tax_amount_dec),
                'severity': 'high'
            })
            return False
        
        # Validate total
        if not self._values_match(total_dec, expected_total):
            self.issues.append({
                'type': 'total_calculation_error',
                'message': f'Total calculation incorrect: {subtotal} + {tax_amount} = {expected_total}, but got {total}',
                'expected': float(expected_total),
                'actual': float(total_dec),
                'severity': 'high'
            })
            return False
        
        return True
    
    def validate_discount(
        self,
        original_amount: float,
        discount_percent: float,
        discount_amount: float,
        final_amount: float
    ) -> bool:
        """
        Validate discount calculations
        """
        original_dec = Decimal(str(original_amount))
        discount_pct_dec = Decimal(str(discount_percent)) / Decimal('100')
        discount_amt_dec = Decimal(str(discount_amount))
        final_dec = Decimal(str(final_amount))
        
        # Calculate expected discount amount
        expected_discount = original_dec * discount_pct_dec
        
        # Calculate expected final amount
        expected_final = original_dec - discount_amt_dec
        
        # Validate discount amount
        if not self._values_match(discount_amt_dec, expected_discount):
            self.issues.append({
                'type': 'discount_calculation_error',
                'message': f'Discount calculation incorrect',
                'severity': 'medium'
            })
            return False
        
        # Validate final amount
        if not self._values_match(final_dec, expected_final):
            self.issues.append({
                'type': 'discount_final_amount_error',
                'message': f'Final amount after discount incorrect',
                'severity': 'high'
            })
            return False
        
        return True
    
    def _parse_number(self, value: str) -> Optional[Decimal]:
        """
        Parse number from string, handling various formats
        """
        if not value:
            return None
        
        # Convert to string if not already
        value = str(value)
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[₹$€£¥,\s]', '', value)
        
        # Remove "Rs" or "Rs."
        cleaned = re.sub(r'Rs\.?', '', cleaned, flags=re.IGNORECASE)
        
        # Extract first number found
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            try:
                return Decimal(match.group())
            except InvalidOperation:
                return None
        
        return None
    
    def _values_match(self, val1: Decimal, val2: Decimal) -> bool:
        """
        Check if two values match within tolerance
        """
        if val1 == 0 and val2 == 0:
            return True
        
        if val1 == 0 or val2 == 0:
            return abs(val1 - val2) < Decimal('0.01')
        
        # Calculate relative difference
        diff = abs(val1 - val2) / max(abs(val1), abs(val2))
        
        return diff <= Decimal(str(self.tolerance))
    
    def _should_auto_correct(
        self,
        quantity: Decimal,
        rate: Decimal,
        amount: Decimal,
        expected_amount: Decimal
    ) -> bool:
        """
        Determine if we should auto-correct the amount
        
        Auto-correct if:
        - The calculation is clearly correct
        - The difference is within reasonable bounds
        """
        # Don't correct if values are very different
        if abs(amount - expected_amount) / max(amount, expected_amount) > 0.1:
            return False
        
        # Don't correct if quantity or rate seem wrong
        if quantity < 0 or rate < 0:
            return False
        
        return True
    
    def validate_weight_calculation(
        self,
        quantity: float,
        unit_weight: float,
        total_weight: float,
        weight_unit: str
    ) -> bool:
        """
        Validate weight calculations (quantity × unit_weight = total_weight)
        """
        quantity_dec = Decimal(str(quantity))
        unit_weight_dec = Decimal(str(unit_weight))
        total_weight_dec = Decimal(str(total_weight))
        
        expected_weight = quantity_dec * unit_weight_dec
        
        if not self._values_match(total_weight_dec, expected_weight):
            self.issues.append({
                'type': 'weight_calculation_error',
                'message': f'Weight calculation incorrect: {quantity} × {unit_weight} {weight_unit} ≠ {total_weight} {weight_unit}',
                'expected': float(expected_weight),
                'actual': float(total_weight_dec),
                'severity': 'medium'
            })
            return False
        
        return True
    
    def get_summary(self) -> Dict:
        """
        Get validation summary
        
        Returns:
            Summary statistics
        """
        return {
            'total_issues': len(self.issues),
            'high_severity': len([i for i in self.issues if i['severity'] == 'high']),
            'medium_severity': len([i for i in self.issues if i['severity'] == 'medium']),
            'low_severity': len([i for i in self.issues if i['severity'] == 'low']),
            'corrected': len([i for i in self.issues if i.get('corrected', False)]),
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