"""
OCR Ensemble
Combines results from multiple OCR engines using voting and confidence weighting
"""

import numpy as np
from typing import Dict, List
from difflib import SequenceMatcher
from collections import Counter


class OCREnsemble:
    """
    Combines OCR results from multiple engines
    """
    
    def __init__(self, method: str = 'weighted_vote'):
        """
        Initialize ensemble
        
        Args:
            method: Combination method
                - 'weighted_vote': Weight by confidence scores
                - 'majority_vote': Simple majority voting
                - 'best_confidence': Take result with highest confidence
        """
        self.method = method
    
    def combine(self, results: List[Dict]) -> Dict:
        """
        Combine results from multiple OCR engines
        
        Args:
            results: List of OCR results from different engines
            
        Returns:
            Combined OCR result
        """
        if len(results) == 0:
            return {'text': '', 'words': [], 'lines': [], 'blocks': []}
        
        if len(results) == 1:
            return results[0]
        
        if self.method == 'weighted_vote':
            return self._weighted_vote(results)
        elif self.method == 'majority_vote':
            return self._majority_vote(results)
        elif self.method == 'best_confidence':
            return self._best_confidence(results)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _weighted_vote(self, results: List[Dict]) -> Dict:
        """
        Combine using weighted voting based on confidence scores
        """
        # Align words from different engines
        aligned_words = self._align_words(results)
        
        # Combine words using weighted voting
        combined_words = []
        for word_group in aligned_words:
            if word_group:
                combined_word = self._combine_word_group(word_group)
                combined_words.append(combined_word)
        
        # Reconstruct text
        combined_text = ' '.join([w['text'] for w in combined_words])
        
        # Combine lines and blocks similarly
        combined_lines = self._combine_lines(results)
        combined_blocks = self._combine_blocks(results)
        
        return {
            'text': combined_text,
            'words': combined_words,
            'lines': combined_lines,
            'blocks': combined_blocks
        }
    
    def _majority_vote(self, results: List[Dict]) -> Dict:
        """
        Combine using simple majority voting
        """
        # Get all texts
        texts = [result['text'] for result in results]
        
        # Split into words
        all_words = [text.split() for text in texts]
        
        # Find most common words at each position
        max_len = max(len(words) for words in all_words)
        combined_words = []
        
        for i in range(max_len):
            words_at_position = []
            for words in all_words:
                if i < len(words):
                    words_at_position.append(words[i])
            
            if words_at_position:
                # Get most common word
                most_common = Counter(words_at_position).most_common(1)[0][0]
                combined_words.append(most_common)
        
        combined_text = ' '.join(combined_words)
        
        return {
            'text': combined_text,
            'words': [{'text': w, 'confidence': 0.8} for w in combined_words],
            'lines': [],
            'blocks': []
        }
    
    def _best_confidence(self, results: List[Dict]) -> Dict:
        """
        Take the result with highest overall confidence
        """
        # Calculate average confidence for each result
        confidences = []
        for result in results:
            words = result.get('words', [])
            if words:
                avg_conf = np.mean([w.get('confidence', 0.5) for w in words])
            else:
                avg_conf = 0.5
            confidences.append(avg_conf)
        
        # Return result with highest confidence
        best_idx = np.argmax(confidences)
        return results[best_idx]
    
    def _align_words(self, results: List[Dict]) -> List[List[Dict]]:
        """
        Align words from different OCR engines
        Uses dynamic programming to find best alignment
        """
        # Get all word lists
        word_lists = []
        for result in results:
            words = result.get('words', [])
            if words:
                word_lists.append(words)
        
        if not word_lists:
            return []
        
        # Use first result as reference
        reference = word_lists[0]
        aligned = [[word] for word in reference]
        
        # Align other results to reference
        for word_list in word_lists[1:]:
            aligned = self._align_two_lists(aligned, word_list)
        
        return aligned
    
    def _align_two_lists(
        self,
        aligned: List[List[Dict]],
        word_list: List[Dict]
    ) -> List[List[Dict]]:
        """
        Align a new word list to existing aligned groups
        """
        result = []
        word_idx = 0
        
        for group in aligned:
            if word_idx >= len(word_list):
                result.append(group)
                continue
            
            # Get reference text from group
            ref_text = group[0]['text'].lower()
            
            # Find best matching word in word_list
            best_match = None
            best_similarity = 0.0
            
            # Check next few words
            for i in range(word_idx, min(word_idx + 3, len(word_list))):
                similarity = SequenceMatcher(None, ref_text, word_list[i]['text'].lower()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = i
            
            # Add to group if similarity is high enough
            if best_match is not None and best_similarity > 0.6:
                group.append(word_list[best_match])
                word_idx = best_match + 1
            
            result.append(group)
        
        # Add remaining words
        while word_idx < len(word_list):
            result.append([word_list[word_idx]])
            word_idx += 1
        
        return result
    
    def _combine_word_group(self, word_group: List[Dict]) -> Dict:
        """
        Combine a group of aligned words from different engines
        """
        # Weight by confidence
        weighted_texts = {}
        total_weight = 0.0
        
        for word in word_group:
            text = word['text']
            confidence = word.get('confidence', 0.5)
            
            if text in weighted_texts:
                weighted_texts[text] += confidence
            else:
                weighted_texts[text] = confidence
            
            total_weight += confidence
        
        # Choose text with highest weighted vote
        best_text = max(weighted_texts, key=weighted_texts.get)
        
        # Calculate average confidence
        avg_confidence = total_weight / len(word_group)
        
        # Average bounding box (if available)
        bboxes = [w.get('bbox') for w in word_group if w.get('bbox')]
        if bboxes:
            avg_bbox = {
                'x': int(np.mean([b['x'] for b in bboxes])),
                'y': int(np.mean([b['y'] for b in bboxes])),
                'width': int(np.mean([b['width'] for b in bboxes])),
                'height': int(np.mean([b['height'] for b in bboxes]))
            }
        else:
            avg_bbox = None
        
        return {
            'text': best_text,
            'confidence': avg_confidence,
            'bbox': avg_bbox,
            'sources': [w.get('engine', 'unknown') for w in word_group]
        }
    
    def _combine_lines(self, results: List[Dict]) -> List[Dict]:
        """
        Combine line-level data from multiple results
        """
        # Use lines from result with most lines
        line_lists = [r.get('lines', []) for r in results]
        if not line_lists or not any(line_lists):
            return []
        
        # Take lines from result with most lines
        longest_lines = max(line_lists, key=len)
        return longest_lines
    
    def _combine_blocks(self, results: List[Dict]) -> List[Dict]:
        """
        Combine block-level data from multiple results
        """
        # Use blocks from result with most blocks
        block_lists = [r.get('blocks', []) for r in results]
        if not block_lists or not any(block_lists):
            return []
        
        # Take blocks from result with most blocks
        longest_blocks = max(block_lists, key=len)
        return longest_blocks
    
    def calculate_agreement(self, results: List[Dict]) -> float:
        """
        Calculate agreement score between different OCR results
        
        Returns:
            Agreement score from 0.0 (no agreement) to 1.0 (perfect agreement)
        """
        if len(results) < 2:
            return 1.0
        
        texts = [result['text'] for result in results]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0