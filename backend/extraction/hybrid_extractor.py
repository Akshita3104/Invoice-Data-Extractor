"""
Hybrid Extractor
Combines LLM-based and rule-based extraction for best results
Uses rule-based as fallback and for validation
"""

from typing import Dict, List, Optional
import numpy as np

from .gemini_extractor import GeminiExtractor
from .rule_based_extractor import RuleBasedExtractor


class HybridExtractor:
    """
    Combines multiple extraction methods for robust results
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        prefer_llm: bool = True
    ):
        """
        Initialize hybrid extractor
        
        Args:
            api_key: API key for LLM (optional)
            prefer_llm: Prefer LLM over rules when both available
        """
        self.prefer_llm = prefer_llm
        
        # Initialize extractors
        if api_key:
            try:
                self.gemini = GeminiExtractor(api_key)
                self.has_llm = True
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self.has_llm = False
        else:
            self.has_llm = False
        
        self.rule_based = RuleBasedExtractor()
        
        print(f"Hybrid extractor initialized (LLM: {self.has_llm})")
    
    def extract(
        self,
        text: str,
        ocr_result: Dict = None,
        zones: List = None,
        tables: List = None,
        graph_features: Dict = None
    ) -> List[Dict]:
        """
        Extract using hybrid approach
        
        Args:
            text: OCR extracted text
            ocr_result: Full OCR result with structure
            zones: Layout zones
            tables: Detected tables
            graph_features: Document graph features
            
        Returns:
            Extracted invoice items
        """
        extracted_data = []
        
        # Try LLM extraction first if available and preferred
        if self.has_llm and self.prefer_llm:
            try:
                print("Attempting LLM extraction...")
                llm_data = self.gemini.extract_with_context(
                    text, zones, tables, graph_features
                )
                
                if llm_data:
                    extracted_data = llm_data
                    print(f"LLM extracted {len(llm_data)} items")
            except Exception as e:
                print(f"LLM extraction failed: {e}")
        
        # Try rule-based extraction
        print("Running rule-based extraction...")
        rule_data = self.rule_based.extract(text, ocr_result)
        
        # If we don't have LLM data, use rule-based
        if not extracted_data:
            extracted_data = rule_data
            print(f"Using rule-based extraction: {len(rule_data)} items")
        else:
            # We have both - merge or validate
            extracted_data = self._merge_extractions(extracted_data, rule_data)
        
        # Try table-based extraction if tables available
        if tables:
            table_data = self._extract_from_tables(tables)
            if table_data:
                extracted_data = self._merge_with_table_data(extracted_data, table_data)
        
        # Post-process and validate
        extracted_data = self._post_process(extracted_data)
        
        return extracted_data
    
    def _merge_extractions(
        self,
        llm_data: List[Dict],
        rule_data: List[Dict]
    ) -> List[Dict]:
        """
        Merge LLM and rule-based extractions
        
        Strategy:
        - Use LLM data as primary
        - Fill missing fields from rule-based data
        - Validate LLM results against rule-based
        """
        merged_data = []
        
        for i, llm_item in enumerate(llm_data):
            merged_item = llm_item.copy()
            
            # Try to find corresponding item in rule data
            if i < len(rule_data):
                rule_item = rule_data[i]
                
                # Fill missing fields from rule-based extraction
                for field, value in rule_item.items():
                    if field not in merged_item or merged_item[field] in ['N/A', '', None]:
                        merged_item[field] = value
                
                # Cross-validate numeric fields
                merged_item = self._cross_validate_item(merged_item, rule_item)
            
            merged_data.append(merged_item)
        
        # Add any extra items from rule-based that LLM missed
        if len(rule_data) > len(llm_data):
            for rule_item in rule_data[len(llm_data):]:
                merged_data.append(rule_item)
        
        return merged_data
    
    def _cross_validate_item(
        self,
        primary_item: Dict,
        secondary_item: Dict
    ) -> Dict:
        """
        Cross-validate numeric fields between two extractions
        """
        numeric_fields = ['Quantity', 'Amount']
        
        for field in numeric_fields:
            primary_val = self._extract_number(primary_item.get(field, ''))
            secondary_val = self._extract_number(secondary_item.get(field, ''))
            
            if primary_val and secondary_val:
                # Check if values are close
                if abs(primary_val - secondary_val) / max(primary_val, secondary_val) > 0.1:
                    # Significant difference - flag for review
                    primary_item[f'_{field}_conflict'] = {
                        'llm': primary_val,
                        'rule': secondary_val
                    }
        
        return primary_item
    
    def _extract_from_tables(self, tables: List) -> List[Dict]:
        """
        Extract data from detected tables
        """
        all_items = []
        
        for table in tables:
            # Get table cells
            cells = table.cells if hasattr(table, 'cells') else table.get('cells', [])
            
            if cells:
                items = self.rule_based.extract_table_data(cells)
                all_items.extend(items)
        
        return all_items
    
    def _merge_with_table_data(
        self,
        extracted_data: List[Dict],
        table_data: List[Dict]
    ) -> List[Dict]:
        """
        Merge extracted data with table data
        """
        # If table data has more items, use table data as base
        if len(table_data) > len(extracted_data):
            base_data = table_data
            supplement_data = extracted_data
        else:
            base_data = extracted_data
            supplement_data = table_data
        
        merged = []
        
        for i, item in enumerate(base_data):
            merged_item = item.copy()
            
            # Supplement with data from other source
            if i < len(supplement_data):
                supplement_item = supplement_data[i]
                
                for field, value in supplement_item.items():
                    if field not in merged_item or merged_item[field] in ['N/A', '', None]:
                        merged_item[field] = value
            
            merged.append(merged_item)
        
        return merged
    
    def _post_process(self, data: List[Dict]) -> List[Dict]:
        """
        Post-process extracted data
        """
        processed = []
        
        for item in data:
            # Ensure all required fields exist
            required_fields = [
                'Goods Description',
                'HSN/SAC Code',
                'Quantity',
                'Weight',
                'Rate',
                'Amount',
                'Company Name',
                'Invoice Number',
                'FSSAI Number',
                'Date of Invoice'
            ]
            
            for field in required_fields:
                if field not in item:
                    item[field] = 'N/A'
            
            # Clean values
            for field, value in item.items():
                if isinstance(value, str):
                    item[field] = value.strip()
                    
                    # Convert empty strings to N/A
                    if item[field] == '':
                        item[field] = 'N/A'
            
            # Remove internal fields (starting with _)
            cleaned_item = {
                k: v for k, v in item.items() 
                if not k.startswith('_')
            }
            
            processed.append(cleaned_item)
        
        return processed
    
    def _extract_number(self, value: str) -> Optional[float]:
        """
        Extract number from string
        """
        if not value or value == 'N/A':
            return None
        
        import re
        
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
    
    def extract_with_confidence(
        self,
        text: str,
        ocr_result: Dict = None,
        zones: List = None,
        tables: List = None
    ) -> tuple:
        """
        Extract data with confidence scores
        
        Returns:
            Tuple of (extracted_data, confidence_scores)
        """
        # Extract data
        extracted_data = self.extract(text, ocr_result, zones, tables)
        
        # Calculate confidence for each item
        confidence_scores = []
        
        for item in extracted_data:
            item_confidence = self._calculate_item_confidence(item)
            confidence_scores.append(item_confidence)
        
        return extracted_data, confidence_scores
    
    def _calculate_item_confidence(self, item: Dict) -> Dict:
        """
        Calculate confidence score for an item
        """
        # Count non-N/A fields
        total_fields = len(item)
        filled_fields = sum(1 for v in item.values() if v != 'N/A' and v)
        
        completeness = filled_fields / total_fields if total_fields > 0 else 0
        
        # Check for conflicts
        has_conflicts = any(k.endswith('_conflict') for k in item.keys())
        
        # Calculate overall confidence
        confidence = completeness
        
        if has_conflicts:
            confidence *= 0.7  # Reduce confidence if conflicts exist
        
        return {
            'overall': round(confidence, 3),
            'completeness': round(completeness, 3),
            'has_conflicts': has_conflicts
        }
    
    def compare_methods(
        self,
        text: str,
        ocr_result: Dict = None
    ) -> Dict:
        """
        Compare LLM vs rule-based extraction
        (For debugging/analysis)
        
        Returns:
            Dictionary with results from both methods
        """
        comparison = {}
        
        # LLM extraction
        if self.has_llm:
            try:
                llm_data = self.gemini.extract(text)
                comparison['llm'] = {
                    'data': llm_data,
                    'item_count': len(llm_data)
                }
            except Exception as e:
                comparison['llm'] = {
                    'error': str(e)
                }
        
        # Rule-based extraction
        try:
            rule_data = self.rule_based.extract(text, ocr_result)
            comparison['rule_based'] = {
                'data': rule_data,
                'item_count': len(rule_data)
            }
        except Exception as e:
            comparison['rule_based'] = {
                'error': str(e)
            }
        
        # Calculate agreement
        if 'llm' in comparison and 'rule_based' in comparison:
            if 'data' in comparison['llm'] and 'data' in comparison['rule_based']:
                agreement = self._calculate_agreement(
                    comparison['llm']['data'],
                    comparison['rule_based']['data']
                )
                comparison['agreement'] = agreement
        
        return comparison
    
    def _calculate_agreement(
        self,
        data1: List[Dict],
        data2: List[Dict]
    ) -> float:
        """
        Calculate agreement between two extractions
        """
        if not data1 or not data2:
            return 0.0
        
        # Compare key fields
        agreements = []
        
        for i in range(min(len(data1), len(data2))):
            item1 = data1[i]
            item2 = data2[i]
            
            # Compare amounts (most important)
            amount1 = self._extract_number(item1.get('Amount', ''))
            amount2 = self._extract_number(item2.get('Amount', ''))
            
            if amount1 and amount2:
                if abs(amount1 - amount2) / max(amount1, amount2) < 0.05:
                    agreements.append(1)
                else:
                    agreements.append(0)
        
        return np.mean(agreements) if agreements else 0.0
    
    def batch_extract(
        self,
        texts: List[str],
        ocr_results: List[Dict] = None
    ) -> List[List[Dict]]:
        """
        Extract from multiple documents
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"Processing document {i+1}/{len(texts)}...")
            
            ocr_result = ocr_results[i] if ocr_results and i < len(ocr_results) else None
            
            extracted = self.extract(text, ocr_result)
            results.append(extracted)
        
        return results