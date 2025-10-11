"""
Text Encoder
Extracts text features using BERT or similar transformers
"""

import numpy as np
import re
from typing import Dict, List, Optional

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TextEncoder:
    """
    Encodes text features using transformer models (BERT)
    Falls back to TF-IDF and handcrafted features if transformers not available
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        use_gpu: bool = True
    ):
        """
        Initialize text encoder
        
        Args:
            model_name: Transformer model to use
            use_gpu: Use GPU if available
        """
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: Transformers not available. Using handcrafted text features.")
            self.use_transformers = False
            return
        
        self.use_transformers = True
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = self.model.config.hidden_size
        
        print(f"Text encoder initialized with {model_name} on {self.device}")
    
    def encode(self, ocr_result: Dict) -> Dict:
        """
        Encode OCR result
        
        Args:
            ocr_result: OCR result with text
            
        Returns:
            Text features
        """
        text = ocr_result.get('text', '')
        
        if not text:
            return self._get_zero_features()
        
        return self.encode_text(text)
    
    def encode_text(self, text: str) -> Dict:
        """
        Encode a single text string
        
        Args:
            text: Input text
            
        Returns:
            Text features
        """
        if not text or len(text.strip()) == 0:
            return self._get_zero_features()
        
        if self.use_transformers:
            return self._encode_transformer_features(text)
        else:
            return self._encode_handcrafted_features(text)
    
    def _encode_transformer_features(self, text: str) -> Dict:
        """
        Extract features using BERT
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy().flatten()
        
        return {
            'transformer_features': embeddings_np,
            'feature_dim': len(embeddings_np)
        }
    
    def _encode_handcrafted_features(self, text: str) -> Dict:
        """
        Extract handcrafted text features (fallback)
        """
        features = {}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # Character type ratios
        features['digit_ratio'] = sum(c.isdigit() for c in text) / max(len(text), 1)
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / max(len(text), 1)
        features['upper_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
        features['space_ratio'] = sum(c.isspace() for c in text) / max(len(text), 1)
        features['punct_ratio'] = sum(c in '.,!?;:' for c in text) / max(len(text), 1)
        
        # Special patterns
        features['has_currency'] = float(bool(re.search(r'[$₹€£¥]', text)))
        features['has_date'] = float(bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)))
        features['has_email'] = float(bool(re.search(r'@', text)))
        features['has_phone'] = float(bool(re.search(r'\d{10}', text)))
        features['has_percentage'] = float(bool(re.search(r'%', text)))
        
        # Positional features
        features['starts_with_upper'] = float(text[0].isupper() if text else False)
        features['ends_with_punct'] = float(text[-1] in '.!?' if text else False)
        
        # Semantic indicators
        invoice_keywords = ['invoice', 'bill', 'receipt', 'date', 'total', 'amount', 'qty', 'price']
        features['keyword_count'] = sum(1 for kw in invoice_keywords if kw in text.lower())
        
        # Create feature vector
        feature_vector = np.array([
            features['length'] / 100.0,  # Normalize
            features['word_count'] / 20.0,
            features['avg_word_length'] / 10.0,
            features['digit_ratio'],
            features['alpha_ratio'],
            features['upper_ratio'],
            features['space_ratio'],
            features['punct_ratio'],
            features['has_currency'],
            features['has_date'],
            features['has_email'],
            features['has_phone'],
            features['has_percentage'],
            features['starts_with_upper'],
            features['ends_with_punct'],
            features['keyword_count'] / 5.0
        ])
        
        return {
            'handcrafted_features': feature_vector,
            'feature_dict': features,
            'feature_dim': len(feature_vector)
        }
    
    def encode_batch(self, texts: List[str]) -> List[Dict]:
        """
        Encode multiple texts in batch
        
        Args:
            texts: List of text strings
            
        Returns:
            List of feature dictionaries
        """
        if not self.use_transformers:
            return [self.encode_text(text) for text in texts]
        
        # Batch processing for transformers
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Convert to list of feature dicts
        results = []
        for i in range(len(texts)):
            feat = embeddings[i].cpu().numpy()
            results.append({
                'transformer_features': feat,
                'feature_dim': len(feat)
            })
        
        return results
    
    def _get_zero_features(self) -> Dict:
        """
        Return zero features for empty text
        """
        if self.use_transformers:
            return {
                'transformer_features': np.zeros(self.feature_dim),
                'feature_dim': self.feature_dim
            }
        else:
            return {
                'handcrafted_features': np.zeros(16),
                'feature_dim': 16
            }
    
    def get_feature_dim(self) -> int:
        """
        Get dimension of feature vector
        """
        if self.use_transformers:
            return self.feature_dim
        else:
            return 16  # Number of handcrafted features
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract top keywords from text
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'in', 'of'])
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        from collections import Counter
        word_freq = Counter(words)
        
        # Return top-k
        return [word for word, _ in word_freq.most_common(top_k)]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Returns:
            Similarity score (0-1)
        """
        if self.use_transformers:
            # Use cosine similarity of embeddings
            feat1 = self.encode_text(text1)['transformer_features']
            feat2 = self.encode_text(text2)['transformer_features']
            
            # Cosine similarity
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            return float(similarity)
        else:
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            
            return overlap / total if total > 0 else 0.0