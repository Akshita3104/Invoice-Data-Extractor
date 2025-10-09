"""
Confidence Scorer
Calculates confidence scores for OCR results based on multiple factors:
- Word-level confidence
- Text characteristics (special characters, numbers, etc.)
- Consistency checks
- Linguistic features
"""

import numpy as np
import re
from typing import Dict, List
from collections import Counter


class ConfidenceScorer:
    """
    Scores OCR result confidence based on multiple heuristics
    """
    
    def __init__(self):
        # Common English words for dictionary check
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            # Add invoice-specific terms
            'invoice', 'date', 'total', 'amount', 'quantity', 'price', 'tax', 'subtotal',
            'company', 'name', 'address', 'phone', 'email', 'number', 'code', 'description'
        ])
    
    def calculate_confidence(self, ocr_result: Dict) -> float:
        """
        Calculate overall confidence score for OCR result
        
        Args:
            ocr_result: OCR result dictionary
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        scores = []
        weights = []
        
        # 1. Word-level confidence (if available)
        word_conf = self._word_confidence(ocr_result)
        if word_conf is not None:
            scores.append(word_conf)
            weights.append(0.4)
        
        # 2. Text quality score
        text_quality = self._text_quality_score(ocr_result.get('text', ''))
        scores.append(text_quality)
        weights.append(0.3)
        
        # 3. Dictionary match score
        dict_score = self._dictionary_match_score(ocr_result.get('text', ''))
        scores.append(dict_score)
        weights.append(0.2)
        
        # 4. Consistency score
        consistency = self._consistency_score(ocr_result)
        scores.append(consistency)
        weights.append(0.1)
        
        # Calculate weighted average
        if scores:
            weighted_score = np.average(scores, weights=weights)
            return float(weighted_score)
        else:
            return 0.5  # Default
    
    def _word_confidence(self, ocr_result: Dict) -> float:
        """
        Calculate average word-level confidence
        """
        words = ocr_result.get('words', [])
        if not words:
            return None
        
        confidences = [w.get('confidence', 0.5) for w in words]
        return np.mean(confidences)
    
    def _text_quality_score(self, text: str) -> float:
        """
        Score text quality based on characteristics
        Higher score = more natural, readable text
        """
        if not text or len(text) < 5:
            return 0.3
        
        score = 1.0
        
        # Penalty for too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text)
        if special_ratio > 0.3:
            score -= 0.3
        
        # Penalty for too many repeated characters
        repeated = re.findall(r'(.)\1{3,}', text)  # 4+ repeated chars
        if repeated:
            score -= 0.2 * min(len(repeated), 3)
        
        # Bonus for proper capitalization
        words = text.split()
        if words:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            cap_ratio = capitalized / len(words)
            if 0.1 < cap_ratio < 0.5:  # Reasonable capitalization
                score += 0.1
        
        # Penalty for excessive spaces
        if '  ' in text:  # Double spaces
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _dictionary_match_score(self, text: str) -> float:
        """
        Score based on how many words match dictionary
        """
        if not text:
            return 0.3
        
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        if not words:
            return 0.5
        
        # Count dictionary matches
        matches = sum(1 for word in words if word in self.common_words)
        match_ratio = matches / len(words)
        
        # Normalize to 0.5-1.0 range (some technical terms won't match)
        return 0.5 + (match_ratio * 0.5)
    
    def _consistency_score(self, ocr_result: Dict) -> float:
        """
        Score consistency across different levels (words, lines, blocks)
        """
        text = ocr_result.get('text', '')
        words = ocr_result.get('words', [])
        
        if not text or not words:
            return 0.5
        
        # Check if word texts match full text reasonably
        word_texts = ' '.join([w.get('text', '') for w in words])
        
        # Calculate similarity between reconstructed and original text
        # Simple character-level comparison
        text_normalized = ''.join(text.split()).lower()
        words_normalized = ''.join(word_texts.split()).lower()
        
        if not text_normalized:
            return 0.5
        
        # Calculate character overlap
        min_len = min(len(text_normalized), len(words_normalized))
        max_len = max(len(text_normalized), len(words_normalized))
        
        if max_len == 0:
            return 0.5
        
        matches = sum(1 for i in range(min_len) 
                     if text_normalized[i] == words_normalized[i])
        
        similarity = matches / max_len
        
        return similarity
    
    def calculate_ensemble_confidence(self, results: List[Dict]) -> float:
        """
        Calculate confidence for ensemble result based on agreement
        
        Args:
            results: List of OCR results from different engines
            
        Returns:
            Ensemble confidence score
        """
        if not results:
            return 0.0
        
        if len(results) == 1:
            return self.calculate_confidence(results[0])
        
        # Calculate individual confidences
        individual_confidences = [
            self.calculate_confidence(result) for result in results
        ]
        
        # Calculate agreement between results
        agreement = self._calculate_agreement(results)
        
        # Combine: higher agreement = higher confidence
        # Average confidence weighted by agreement
        avg_confidence = np.mean(individual_confidences)
        ensemble_confidence = avg_confidence * 0.6 + agreement * 0.4
        
        return float(ensemble_confidence)
    
    def _calculate_agreement(self, results: List[Dict]) -> float:
        """
        Calculate agreement between multiple OCR results
        """
        if len(results) < 2:
            return 1.0
        
        texts = [result.get('text', '') for result in results]
        
        # Calculate pairwise character overlap
        agreements = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1 = ''.join(texts[i].split()).lower()
                text2 = ''.join(texts[j].split()).lower()
                
                if not text1 or not text2:
                    continue
                
                # Simple character overlap
                min_len = min(len(text1), len(text2))
                max_len = max(len(text1), len(text2))
                
                if max_len > 0:
                    matches = sum(1 for k in range(min_len) 
                                 if text1[k] == text2[k])
                    agreement = matches / max_len
                    agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.5
    
    def score_word(self, word: str, confidence: float = None) -> float:
        """
        Score a single word
        
        Args:
            word: Word text
            confidence: Optional OCR engine confidence
            
        Returns:
            Word confidence score
        """
        if not word:
            return 0.0
        
        score = confidence if confidence is not None else 0.5
        
        # Bonus for dictionary match
        if word.lower() in self.common_words:
            score = min(1.0, score + 0.1)
        
        # Penalty for unusual characters
        if not word.replace('-', '').replace('.', '').isalnum():
            score *= 0.9
        
        # Penalty for single character (unless it's 'a' or 'I')
        if len(word) == 1 and word.lower() not in ['a', 'i']:
            score *= 0.8
        
        return score
    
    def score_field(self, field_value: str, field_type: str) -> float:
        """
        Score a specific field based on expected format
        
        Args:
            field_value: Extracted field value
            field_type: Type of field (e.g., 'date', 'amount', 'email')
            
        Returns:
            Field confidence score
        """
        if not field_value:
            return 0.0
        
        if field_type == 'date':
            return self._score_date(field_value)
        elif field_type == 'amount':
            return self._score_amount(field_value)
        elif field_type == 'email':
            return self._score_email(field_value)
        elif field_type == 'phone':
            return self._score_phone(field_value)
        elif field_type == 'number':
            return self._score_number(field_value)
        else:
            return 0.7  # Default confidence for unknown types
    
    def _score_date(self, date_str: str) -> float:
        """
        Score date field
        """
        # Check common date patterns
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}\s+\w+\s+\d{4}',  # DD Month YYYY
        ]
        
        for pattern in patterns:
            if re.match(pattern, date_str.strip()):
                return 0.9
        
        # Partial match
        if re.search(r'\d{4}', date_str):  # Has a year
            return 0.6
        
        return 0.3
    
    def _score_amount(self, amount_str: str) -> float:
        """
        Score monetary amount field
        """
        # Check for currency symbols and numbers
        pattern = r'[$₹€£¥]\s*[\d,]+\.?\d*|[\d,]+\.?\d*\s*[$₹€£¥]'
        
        if re.search(pattern, amount_str):
            return 0.9
        
        # Just numbers with decimal
        if re.match(r'[\d,]+\.\d{2}', amount_str):
            return 0.8
        
        # Just numbers
        if re.match(r'[\d,]+', amount_str):
            return 0.6
        
        return 0.3
    
    def _score_email(self, email_str: str) -> float:
        """
        Score email field
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        
        if re.match(pattern, email_str.strip()):
            return 0.95
        
        # Has @ and .
        if '@' in email_str and '.' in email_str:
            return 0.6
        
        return 0.2
    
    def _score_phone(self, phone_str: str) -> float:
        """
        Score phone number field
        """
        # Remove common separators
        digits = re.sub(r'[\s\-\(\)\+]', '', phone_str)
        
        # Check if mostly digits
        if digits.isdigit() and 7 <= len(digits) <= 15:
            return 0.9
        
        return 0.3
    
    def _score_number(self, number_str: str) -> float:
        """
        Score generic number field
        """
        # Remove commas and spaces
        cleaned = number_str.replace(',', '').replace(' ', '')
        
        try:
            float(cleaned)
            return 0.9
        except ValueError:
            return 0.3
    
    def get_low_confidence_words(
        self,
        ocr_result: Dict,
        threshold: float = 0.6
    ) -> List[Dict]:
        """
        Get list of words with confidence below threshold
        
        Args:
            ocr_result: OCR result
            threshold: Confidence threshold
            
        Returns:
            List of low-confidence words
        """
        words = ocr_result.get('words', [])
        
        low_conf_words = []
        for word in words:
            confidence = word.get('confidence', 0.5)
            if confidence < threshold:
                low_conf_words.append({
                    'text': word.get('text', ''),
                    'confidence': confidence,
                    'bbox': word.get('bbox')
                })
        
        return low_conf_words
    
    def suggest_reprocessing(self, ocr_result: Dict) -> Dict:
        """
        Analyze OCR result and suggest if reprocessing is needed
        
        Returns:
            Dictionary with suggestions
        """
        overall_confidence = self.calculate_confidence(ocr_result)
        low_conf_words = self.get_low_confidence_words(ocr_result, threshold=0.6)
        
        suggestions = {
            'overall_confidence': overall_confidence,
            'needs_reprocessing': overall_confidence < 0.6,
            'low_confidence_word_count': len(low_conf_words),
            'recommendations': []
        }
        
        if overall_confidence < 0.4:
            suggestions['recommendations'].append('Try preprocessing (denoise, enhance contrast)')
        
        if overall_confidence < 0.6:
            suggestions['recommendations'].append('Try different OCR engine')
        
        if len(low_conf_words) > 10:
            suggestions['recommendations'].append('Consider manual review of low-confidence words')
        
        return suggestions