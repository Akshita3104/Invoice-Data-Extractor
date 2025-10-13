"""
Entity Classifier
Classifies entities using BERT-NER and LLM-based disambiguation
Identifies entities like company names, dates, amounts, etc.
"""

import re
from typing import Dict, List, Optional, Tuple

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EntityClassifier:
    """
    Classifies and disambiguates entities in document text
    Uses BERT-NER for initial classification and rules for disambiguation
    """
    
    def __init__(
        self,
        model_name: str = 'dbmdz/bert-large-cased-finetuned-conll03-english',
        use_gpu: bool = True
    ):
        """
        Initialize entity classifier
        
        Args:
            model_name: Pre-trained NER model
            use_gpu: Use GPU if available
        """
        self.entity_types = [
            'INVOICE_NUMBER',
            'DATE',
            'COMPANY',
            'PERSON',
            'AMOUNT',
            'QUANTITY',
            'ADDRESS',
            'PHONE',
            'EMAIL',
            'PRODUCT',
            'TAX_ID',
            'FSSAI'
        ]
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            self.use_ner = True
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"Entity classifier initialized with {model_name} on {self.device}")
            except Exception as e:
                print(f"Warning: Could not load NER model: {e}")
                self.use_ner = False
        else:
            self.use_ner = False
            print("Warning: Transformers not available. Using rule-based entity classification.")
    
    def classify_entities(
        self,
        text: str,
        context: Dict = None
    ) -> List[Dict]:
        """
        Classify entities in text
        
        Args:
            text: Input text
            context: Optional context information
            
        Returns:
            List of detected entities with types and positions
        """
        if self.use_ner:
            entities = self._ner_classification(text)
        else:
            entities = self._rule_based_classification(text)
        
        # Apply disambiguation
        entities = self._disambiguate_entities(entities, context)
        
        return entities
    
    def classify_element(
        self,
        element: Dict,
        neighbors: List[Dict] = None
    ) -> Dict:
        """
        Classify a single element (word, line, block)
        
        Args:
            element: Element with text and bbox
            neighbors: Optional neighboring elements for context
            
        Returns:
            Element with entity classification
        """
        text = element.get('text', '')
        
        # Get context from neighbors
        context_text = ""
        if neighbors:
            context_text = " ".join([n.get('text', '') for n in neighbors[:3]])
        
        # Classify
        entities = self.classify_entities(text, {'context': context_text})
        
        # Add classification to element
        if entities:
            element['entity_type'] = entities[0]['type']
            element['entity_confidence'] = entities[0]['confidence']
        else:
            element['entity_type'] = 'OTHER'
            element['entity_confidence'] = 0.0
        
        return element
    
    def _ner_classification(self, text: str) -> List[Dict]:
        """
        NER-based entity classification using BERT
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
        
        # Convert to entities
        entities = []
        current_entity = None
        
        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset[0] == 0 and offset[1] == 0:
                continue
            
            label_id = pred.item()
            label = self.model.config.id2label[label_id]
            
            # Skip O (outside) labels
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Parse B-/I- tags
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                start, end = offset
                current_entity = {
                    'type': self._map_ner_label(entity_type),
                    'text': text[start:end],
                    'start': int(start),
                    'end': int(end),
                    'confidence': 0.8
                }
            
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                start, end = offset
                current_entity['end'] = int(end)
                current_entity['text'] = text[current_entity['start']:end]
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _rule_based_classification(self, text: str) -> List[Dict]:
        """
        Rule-based entity classification
        """
        entities = []
        
        # Date
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        for match in re.finditer(date_pattern, text):
            entities.append({
                'type': 'DATE',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Amount
        amount_pattern = r'[₹$€£¥]\s*[\d,]+\.?\d*|Rs\.?\s*[\d,]+\.?\d*'
        for match in re.finditer(amount_pattern, text):
            entities.append({
                'type': 'AMOUNT',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # Email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'type': 'EMAIL',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Phone
        phone_pattern = r'\+?\d{10,15}|\d{3}[-.]?\d{3}[-.]?\d{4}'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                'type': 'PHONE',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.85
            })
        
        # Company (has Ltd, Pvt, etc.)
        company_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Pvt\.?|Ltd\.?|LLC|Inc\.?|Corp\.?)\b'
        for match in re.finditer(company_pattern, text):
            entities.append({
                'type': 'COMPANY',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })
        
        # Invoice number
        invoice_pattern = r'INV[-/]?\d+|Invoice\s*(?:No\.?|Number|#)?\s*:?\s*([A-Z0-9/-]+)'
        for match in re.finditer(invoice_pattern, text, re.IGNORECASE):
            entities.append({
                'type': 'INVOICE_NUMBER',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.85
            })
        
        # GSTIN
        gstin_pattern = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}'
        for match in re.finditer(gstin_pattern, text):
            entities.append({
                'type': 'TAX_ID',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # FSSAI
        fssai_pattern = r'\d{14}'
        for match in re.finditer(fssai_pattern, text):
            # Check if preceded by FSSAI keyword
            start = max(0, match.start() - 20)
            context = text[start:match.start()].lower()
            if 'fssai' in context or 'license' in context:
                entities.append({
                    'type': 'FSSAI',
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        return entities
    
    def _disambiguate_entities(
        self,
        entities: List[Dict],
        context: Dict = None
    ) -> List[Dict]:
        """
        Disambiguate entities based on context
        """
        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlaps(entities)
        
        # Apply domain-specific rules for disambiguation
        for entity in entities:
            # If amount near "total" keyword, classify as TOTAL_AMOUNT
            if entity['type'] == 'AMOUNT' and context:
                context_text = context.get('context', '')
                if 'total' in context_text.lower():
                    entity['type'] = 'TOTAL_AMOUNT'
                    entity['confidence'] *= 1.1
        
        return entities
    
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove overlapping entities, keeping higher confidence ones
        """
        # Sort by start position
        entities = sorted(entities, key=lambda e: e['start'])
        
        filtered = []
        for entity in entities:
            # Check if overlaps with any existing entity
            overlaps = False
            for existing in filtered:
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # Overlaps - keep higher confidence
                    if entity['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def _map_ner_label(self, ner_label: str) -> str:
        """
        Map standard NER labels to invoice-specific labels
        """
        mapping = {
            'PER': 'PERSON',
            'ORG': 'COMPANY',
            'LOC': 'ADDRESS',
            'MISC': 'OTHER'
        }
        
        return mapping.get(ner_label, ner_label)
    
    def batch_classify(
        self,
        texts: List[str]
    ) -> List[List[Dict]]:
        """
        Classify entities in multiple texts
        
        Returns:
            List of entity lists (one per text)
        """
        results = []
        
        for text in texts:
            entities = self.classify_entities(text)
            results.append(entities)
        
        return results