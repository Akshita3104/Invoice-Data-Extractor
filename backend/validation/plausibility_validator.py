"""
Plausibility Validator
Uses LLM and heuristics to check if extracted values make business sense:
- Reasonable prices for products
- Logical quantities
- Appropriate tax rates
- Business rule validation
"""

from typing import Dict, List, Tuple, Optional
import re


class PlausibilityValidator:
    """
    Validates plausibility of extracted data using business logic and LLM
    """
    
    def __init__(self, use_llm: bool = False, api_key: str = None):
        """
        Initialize plausibility validator
        
        Args:
            use_llm: Use LLM for advanced plausibility checks
            api_key: API key for LLM service
        """
        self.use_llm = use_llm
        self.api_key = api_key
        self.issues = []
        
        # Define reasonable ranges (can be customized per domain)
        self.ranges = {
            'quantity': (0, 10000),
            'rate_per_unit': (0.01, 1000000),
            'amount': (1, 10000000),
            'tax_rate': (0, 0.30),  # 0-30%
            'weight_kg': (0.001, 100000)
        }
    
    def validate(self, extracted_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate plausibility of extracted data
        
        Args:
            extracted_data: List of extracted invoice items
            
        Returns:
            Tuple of (data, issues)
        """
        self.issues = []
        
        for item in extracted_data:
            self._validate_item_plausibility(item)
        
        # Validate overall invoice plausibility
        self._validate_invoice_plausibility(extracted_data)
        
        return extracted_data, self.issues
    
    def _validate_item_plausibility(self, item: Dict):
        """
        Validate plausibility of a single line item
        """
        description = item.get('Goods Description', 'Unknown')
        
        # Check quantity
        quantity = self._parse_number(item.get('Quantity', ''))
        if quantity is not None:
            if not self._is_in_range(quantity, 'quantity'):
                self.issues.append({
                    'type': 'implausible_quantity',
                    'item': description,
                    'value': quantity,
                    'message': f'Quantity {quantity} seems implausible',
                    'severity': 'medium'
                })
        
        # Check rate
        rate = self._parse_number(item.get('Rate', ''))
        if rate is not None:
            if not self._is_in_range(rate, 'rate_per_unit'):
                self.issues.append({
                    'type': 'implausible_rate',
                    'item': description,
                    'value': rate,
                    'message': f'Rate {rate} seems implausible',
                    'severity': 'medium'
                })
            
            # Check for suspicious patterns (e.g., rate too low)
            if rate < 0.01:
                self.issues.append({
                    'type': 'suspiciously_low_rate',
                    'item': description,
                    'value': rate,
                    'message': f'Rate {rate} is suspiciously low',
                    'severity': 'high'
                })
        
        # Check amount
        amount = self._parse_number(item.get('Amount', ''))
        if amount is not None:
            if not self._is_in_range(amount, 'amount'):
                self.issues.append({
                    'type': 'implausible_amount',
                    'item': description,
                    'value': amount,
                    'message': f'Amount {amount} seems implausible',
                    'severity': 'medium'
                })
        
        # Check weight
        weight_str = str(item.get('Weight', '')).upper()
        if weight_str and weight_str != 'N/A':
            weight = self._parse_weight_to_kg(weight_str)
            if weight and not self._is_in_range(weight, 'weight_kg'):
                self.issues.append({
                    'type': 'implausible_weight',
                    'item': description,
                    'value': weight_str,
                    'message': f'Weight {weight_str} seems implausible',
                    'severity': 'low'
                })
        
        # Check description plausibility
        if description and description != 'Unknown':
            if len(description) < 3:
                self.issues.append({
                    'type': 'suspicious_description',
                    'item': description,
                    'message': 'Product description is too short',
                    'severity': 'medium'
                })
            
            if description.replace(' ', '').isdigit():
                self.issues.append({
                    'type': 'invalid_description',
                    'item': description,
                    'message': 'Product description is just numbers',
                    'severity': 'high'
                })
    
    def _validate_invoice_plausibility(self, data: List[Dict]):
        """
        Validate plausibility of overall invoice
        """
        # Calculate total amount
        total_amount = 0
        for item in data:
            amount = self._parse_number(item.get('Amount', ''))
            if amount:
                total_amount += amount
        
        # Check if total is reasonable
        if total_amount > 10000000:  # 1 crore
            self.issues.append({
                'type': 'very_high_total',
                'value': total_amount,
                'message': f'Total amount {total_amount} is very high',
                'severity': 'medium'
            })
        
        if total_amount < 1:
            self.issues.append({
                'type': 'very_low_total',
                'value': total_amount,
                'message': f'Total amount {total_amount} is very low',
                'severity': 'high'
            })
        
        # Check number of line items
        if len(data) > 100:
            self.issues.append({
                'type': 'too_many_items',
                'count': len(data),
                'message': f'Invoice has {len(data)} line items (unusually high)',
                'severity': 'low'
            })
        
        # Check for suspiciously round numbers
        amounts = [self._parse_number(item.get('Amount', '')) for item in data]
        amounts = [a for a in amounts if a is not None]
        
        if amounts:
            round_numbers = [a for a in amounts if a == int(a) and a % 100 == 0]
            if len(round_numbers) / len(amounts) > 0.8:
                self.issues.append({
                    'type': 'too_many_round_numbers',
                    'message': 'Most amounts are round numbers (suspicious)',
                    'severity': 'low'
                })
    
    def validate_business_rules(self, item: Dict) -> List[Dict]:
        """
        Validate domain-specific business rules
        """
        issues = []
        
        # Rule: Food items should have FSSAI number
        description = item.get('Goods Description', '').lower()
        fssai = item.get('FSSAI Number', '')
        
        food_keywords = ['food', 'rice', 'wheat', 'flour', 'oil', 'spice', 'grain']
        if any(keyword in description for keyword in food_keywords):
            if not fssai or fssai == 'N/A':
                issues.append({
                    'type': 'missing_fssai',
                    'item': item.get('Goods Description'),
                    'message': 'Food item missing FSSAI number',
                    'severity': 'high'
                })
        
        # Rule: Certain products should have specific HSN codes
        hsn = item.get('HSN/SAC Code', '')
        
        # Example: Edible oil should have HSN starting with 15
        if 'oil' in description and 'edible' in description:
            if hsn and not hsn.startswith('15'):
                issues.append({
                    'type': 'incorrect_hsn',
                    'item': item.get('Goods Description'),
                    'hsn': hsn,
                    'message': 'Edible oil should have HSN starting with 15',
                    'severity': 'medium'
                })
        
        return issues
    
    def validate_tax_plausibility(
        self,
        subtotal: float,
        cgst: float,
        sgst: float,
        igst: float,
        total: float
    ) -> List[Dict]:
        """
        Validate tax calculations make business sense
        """
        issues = []
        
        # CGST and SGST should be equal (typically)
        if cgst > 0 and sgst > 0:
            if abs(cgst - sgst) / max(cgst, sgst) > 0.01:
                issues.append({
                    'type': 'cgst_sgst_mismatch',
                    'cgst': cgst,
                    'sgst': sgst,
                    'message': 'CGST and SGST should typically be equal',
                    'severity': 'high'
                })
        
        # Can't have both IGST and CGST/SGST
        if igst > 0 and (cgst > 0 or sgst > 0):
            issues.append({
                'type': 'igst_with_cgst_sgst',
                'message': 'Cannot have both IGST and CGST/SGST',
                'severity': 'critical'
            })
        
        # Total tax rate should be reasonable (typically 5%, 12%, 18%, 28%)
        total_tax = cgst + sgst + igst
        tax_rate = total_tax / subtotal if subtotal > 0 else 0
        
        common_rates = [0.05, 0.12, 0.18, 0.28]
        if not any(abs(tax_rate - rate) < 0.01 for rate in common_rates):
            issues.append({
                'type': 'unusual_tax_rate',
                'tax_rate': tax_rate * 100,
                'message': f'Tax rate {tax_rate*100:.1f}% is unusual (common: 5%, 12%, 18%, 28%)',
                'severity': 'medium'
            })
        
        return issues
    
    def check_price_reasonableness(
        self,
        product: str,
        price: float,
        unit: str = 'kg'
    ) -> bool:
        """
        Check if price is reasonable for a product
        (Would use historical data or LLM in production)
        """
        # Simple heuristics for common products
        reasonable_ranges = {
            'rice': (20, 200),  # per kg
            'wheat': (15, 100),
            'oil': (50, 500),
            'sugar': (30, 100),
            'flour': (20, 80)
        }
        
        product_lower = product.lower()
        
        for keyword, (min_price, max_price) in reasonable_ranges.items():
            if keyword in product_lower:
                if unit.lower() == 'kg':
                    return min_price <= price <= max_price
        
        # If no specific range, use general bounds
        return 0.01 <= price <= 10000
    
    def _parse_number(self, value: str) -> Optional[float]:
        """
        Parse number from string
        """
        if not value or value == 'N/A':
            return None
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[₹$€£¥,\s]', '', str(value))
        cleaned = re.sub(r'Rs\.?', '', cleaned, flags=re.IGNORECASE)
        
        # Extract number
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
    
    def _parse_weight_to_kg(self, weight_str: str) -> Optional[float]:
        """
        Parse weight and convert to kg
        """
        weight_str = weight_str.upper().replace(' ', '')
        
        # Extract number
        match = re.search(r'(\d+\.?\d*)', weight_str)
        if not match:
            return None
        
        value = float(match.group())
        
        # Convert to kg based on unit
        if 'KG' in weight_str:
            return value
        elif 'G' in weight_str and 'KG' not in weight_str:
            return value / 1000
        elif 'QTL' in weight_str or 'QUINTAL' in weight_str:
            return value * 100
        elif 'TON' in weight_str or 'MT' in weight_str:
            return value * 1000
        
        return value  # Assume kg if no unit
    
    def _is_in_range(self, value: float, range_type: str) -> bool:
        """
        Check if value is in reasonable range
        """
        if range_type not in self.ranges:
            return True
        
        min_val, max_val = self.ranges[range_type]
        return min_val <= value <= max_val
    
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