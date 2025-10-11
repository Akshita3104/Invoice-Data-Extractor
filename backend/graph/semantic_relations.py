"""
Semantic Relations Extractor
Extracts semantic relationships between text elements:
- Key-value pairs
- Label-value pairs
- Entity relationships (company-address, product-price, etc.)
"""

import re
from typing import Dict, Optional, List


class SemanticRelationExtractor:
    """
    Extracts semantic relationships between text elements
    """
    
    def __init__(self):
        # Common key patterns for invoices
        self.key_patterns = {
            'invoice_number': r'invoice\s*(no|number|#)',
            'date': r'(date|dated|on)',
            'company': r'(company|firm|business|vendor|supplier)',
            'address': r'(address|location)',
            'phone': r'(phone|tel|mobile|contact)',
            'email': r'(email|e-mail)',
            'amount': r'(amount|total|sum|price|cost|value)',
            'quantity': r'(quantity|qty|count|no\.|number)',
            'description': r'(description|item|product|goods|particulars)',
            'tax': r'(tax|gst|vat|igst|cgst|sgst)',
            'discount': r'(discount|off)',
            'subtotal': r'(subtotal|sub-total|sub total)',
            'fssai': r'(fssai|license|lic)',
            'gstin': r'(gstin|gst\s*no|tax\s*id)',
        }
        
        # Value patterns
        self.value_patterns = {
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'amount': r'₹?\s*\d+[,\d]*\.?\d*',
            'phone': r'\+?\d{10,15}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'number': r'\d+',
            'alphanumeric': r'[A-Z0-9]+',
        }
        
        self.max_distance = 200  # Max distance for key-value pairing
    
    def detect_key_value_pair(
        self,
        text1: str,
        text2: str,
        bbox1: Dict,
        bbox2: Dict
    ) -> Optional[Dict]:
        """
        Detect if two text elements form a key-value pair
        
        Returns:
            Dictionary with relation info or None
        """
        # Normalize texts
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # Check if text1 is a key
        key_type = self._is_key(text1_lower)
        
        if not key_type:
            return None
        
        # Check if text2 is a value
        value_type = self._is_value(text2_lower, key_type)
        
        if not value_type:
            return None
        
        # Check spatial relationship (keys and values should be near)
        distance = self._calculate_distance(bbox1, bbox2)
        
        if distance > self.max_distance:
            return None
        
        # Check if they are aligned (horizontally or vertically)
        alignment = self._check_alignment(bbox1, bbox2)
        
        # Calculate confidence
        confidence = self._calculate_kv_confidence(
            key_type,
            value_type,
            distance,
            alignment
        )
        
        return {
            'type': 'key_value',
            'key': text1,
            'value': text2,
            'key_type': key_type,
            'value_type': value_type,
            'confidence': confidence,
            'alignment': alignment
        }
    
    def _is_key(self, text: str) -> Optional[str]:
        """
        Check if text is a potential key
        
        Returns:
            Key type or None
        """
        for key_type, pattern in self.key_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return key_type
        
        # Check for colon at end (common key indicator)
        if text.endswith(':'):
            return 'generic_key'
        
        return None
    
    def _is_value(self, text: str, key_type: str) -> Optional[str]:
        """
        Check if text is a valid value for the given key type
        """
        # Map key types to expected value patterns
        expected_patterns = {
            'date': ['date'],
            'amount': ['amount'],
            'phone': ['phone'],
            'email': ['email'],
            'invoice_number': ['alphanumeric', 'number'],
            'quantity': ['number', 'amount'],
            'fssai': ['number'],
            'gstin': ['alphanumeric'],
        }
        
        patterns_to_check = expected_patterns.get(key_type, ['alphanumeric', 'number'])
        
        for pattern_type in patterns_to_check:
            pattern = self.value_patterns.get(pattern_type)
            if pattern and re.search(pattern, text):
                return pattern_type
        
        # If no specific pattern matches but text is not empty, it might still be a value
        if text and len(text) > 0:
            return 'text'
        
        return None
    
    def _calculate_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate distance between two bounding boxes"""
        x1 = bbox1.get('x', 0) + bbox1.get('width', 0) / 2
        y1 = bbox1.get('y', 0) + bbox1.get('height', 0) / 2
        
        x2 = bbox2.get('x', 0) + bbox2.get('width', 0) / 2
        y2 = bbox2.get('y', 0) + bbox2.get('height', 0) / 2
        
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    
    def _check_alignment(self, bbox1: Dict, bbox2: Dict) -> str:
        """
        Check if bboxes are aligned
        
        Returns:
            'horizontal', 'vertical', or 'none'
        """
        y1_center = bbox1.get('y', 0) + bbox1.get('height', 0) / 2
        y2_center = bbox2.get('y', 0) + bbox2.get('height', 0) / 2
        
        x1_center = bbox1.get('x', 0) + bbox1.get('width', 0) / 2
        x2_center = bbox2.get('x', 0) + bbox2.get('width', 0) / 2
        
        y_diff = abs(y1_center - y2_center)
        x_diff = abs(x1_center - x2_center)
        
        if y_diff < 10:
            return 'horizontal'
        elif x_diff < 10:
            return 'vertical'
        
        return 'none'
    
    def _calculate_kv_confidence(
        self,
        key_type: str,
        value_type: str,
        distance: float,
        alignment: str
    ) -> float:
        """Calculate confidence score for key-value pair"""
        confidence = 0.5  # Base confidence
        
        # Boost for specific key-value matches
        if key_type in ['invoice_number', 'date', 'amount', 'phone', 'email']:
            confidence += 0.2
        
        # Boost for proper alignment
        if alignment in ['horizontal', 'vertical']:
            confidence += 0.2
        
        # Penalty for distance
        if distance < 50:
            confidence += 0.1
        elif distance > 150:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def extract_entity_relations(
        self,
        nodes: List[Dict]
    ) -> List[Dict]:
        """
        Extract entity relationships from multiple nodes
        (e.g., company -> address, product -> price)
        """
        relations = []
        
        # Find potential entities
        entities = {
            'company': [],
            'product': [],
            'amount': [],
            'date': [],
            'contact': []
        }
        
        for node in nodes:
            text = node.get('text', '').lower()
            
            # Classify node
            if re.search(r'(pvt|ltd|llp|inc|corp|company)', text, re.IGNORECASE):
                entities['company'].append(node)
            elif re.search(r'₹|\$|rs\.?\s*\d+', text):
                entities['amount'].append(node)
            elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                entities['date'].append(node)
            elif re.search(r'@|phone|tel|mobile', text, re.IGNORECASE):
                entities['contact'].append(node)
            else:
                entities['product'].append(node)
        
        # Create relations
        # Company -> Address (typically below company name)
        for company in entities['company']:
            nearby = self._find_nearby_nodes(company, nodes, direction='below')
            for node in nearby[:2]:  # Take first 2 nodes below
                relations.append({
                    'source': company.get('id'),
                    'target': node.get('id'),
                    'type': 'company_address',
                    'confidence': 0.7
                })
        
        # Product -> Amount (typically on same line or nearby)
        for product in entities['product']:
            nearby_amounts = [
                n for n in entities['amount']
                if self._calculate_distance(product.get('bbox'), n.get('bbox')) < 100
            ]
            for amount in nearby_amounts:
                relations.append({
                    'source': product.get('id'),
                    'target': amount.get('id'),
                    'type': 'product_price',
                    'confidence': 0.8
                })
        
        return relations
    
    def _find_nearby_nodes(
        self,
        reference_node: Dict,
        all_nodes: List[Dict],
        direction: str = 'any',
        max_distance: float = 100
    ) -> List[Dict]:
        """
        Find nodes near the reference node
        
        Args:
            direction: 'above', 'below', 'left', 'right', 'any'
        """
        ref_bbox = reference_node.get('bbox', {})
        ref_y = ref_bbox.get('y', 0)
        ref_x = ref_bbox.get('x', 0)
        
        nearby = []
        
        for node in all_nodes:
            if node.get('id') == reference_node.get('id'):
                continue
            
            node_bbox = node.get('bbox', {})
            node_y = node_bbox.get('y', 0)
            node_x = node_bbox.get('x', 0)
            
            distance = self._calculate_distance(ref_bbox, node_bbox)
            
            if distance > max_distance:
                continue
            
            # Check direction
            if direction == 'below' and node_y <= ref_y:
                continue
            elif direction == 'above' and node_y >= ref_y:
                continue
            elif direction == 'right' and node_x <= ref_x:
                continue
            elif direction == 'left' and node_x >= ref_x:
                continue
            
            nearby.append(node)
        
        # Sort by distance
        nearby.sort(key=lambda n: self._calculate_distance(ref_bbox, n.get('bbox')))
        
        return nearby
    
    def detect_table_relationships(self, cells: List[Dict]) -> List[Dict]:
        """
        Detect relationships between table cells
        (e.g., header-data, row relationships)
        """
        relations = []
        
        if not cells:
            return relations
        
        # Group cells by row
        rows = {}
        for cell in cells:
            row_idx = cell.get('row', 0)
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(cell)
        
        # First row is typically header
        if 0 in rows:
            headers = rows[0]
            
            # Create header-data relations
            for row_idx in range(1, len(rows)):
                if row_idx not in rows:
                    continue
                
                data_cells = rows[row_idx]
                
                for i, header in enumerate(headers):
                    if i < len(data_cells):
                        relations.append({
                            'source': header.get('id'),
                            'target': data_cells[i].get('id'),
                            'type': 'header_data',
                            'confidence': 0.9
                        })
        
        return relations
    
    def classify_field_type(self, text: str) -> str:
        """
        Classify the type of field based on text content
        
        Returns:
            Field type: 'company', 'date', 'amount', 'email', etc.
        """
        text_lower = text.lower().strip()
        
        # Check against patterns
        if re.search(self.value_patterns['email'], text):
            return 'email'
        elif re.search(self.value_patterns['phone'], text):
            return 'phone'
        elif re.search(self.value_patterns['date'], text):
            return 'date'
        elif re.search(self.value_patterns['amount'], text):
            return 'amount'
        elif re.search(r'(pvt|ltd|llp|inc|corp)', text_lower):
            return 'company'
        elif text.replace(',', '').replace('.', '').isdigit():
            return 'number'
        else:
            return 'text'