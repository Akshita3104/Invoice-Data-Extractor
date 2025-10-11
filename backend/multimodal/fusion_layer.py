"""
Fusion Layer
Combines features from multiple modalities using attention-based fusion
"""

import numpy as np
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FusionLayer:
    """
    Fuses features from visual, text, layout, and graph modalities
    Uses attention mechanism to weight different modalities
    """
    
    def __init__(
        self,
        fusion_method: str = 'attention',
        use_gpu: bool = True
    ):
        """
        Initialize fusion layer
        
        Args:
            fusion_method: Fusion method ('concat', 'attention', 'weighted_sum')
            use_gpu: Use GPU if available
        """
        self.fusion_method = fusion_method
        
        if TORCH_AVAILABLE and fusion_method == 'attention':
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            self.use_attention = True
            
            # Initialize attention model (will be built dynamically)
            self.attention_model = None
            
            print(f"Fusion layer initialized with {fusion_method} on {self.device}")
        else:
            self.use_attention = False
            print(f"Fusion layer initialized with {fusion_method} (non-deep)")
    
    def fuse(self, features: Dict) -> Dict:
        """
        Fuse features from multiple modalities
        
        Args:
            features: Dictionary containing features from different modalities
                Keys: 'visual', 'text', 'layout', 'graph'
                
        Returns:
            Fused features
        """
        if self.fusion_method == 'concat':
            return self._concatenate_fusion(features)
        elif self.fusion_method == 'attention' and self.use_attention:
            return self._attention_fusion(features)
        elif self.fusion_method == 'weighted_sum':
            return self._weighted_sum_fusion(features)
        else:
            return self._concatenate_fusion(features)
    
    def _concatenate_fusion(self, features: Dict) -> Dict:
        """
        Simple concatenation of all features
        """
        feature_vectors = []
        modality_names = []
        
        # Extract feature vectors from each modality
        for modality, feat_dict in features.items():
            if isinstance(feat_dict, dict):
                # Try different feature key names
                for key in ['deep_features', 'transformer_features', 'handcrafted_features', 'feature_vector']:
                    if key in feat_dict:
                        feature_vectors.append(feat_dict[key])
                        modality_names.append(modality)
                        break
        
        if not feature_vectors:
            return {'fused_features': np.array([]), 'method': 'concat'}
        
        # Concatenate
        fused = np.concatenate([np.atleast_1d(f) for f in feature_vectors])
        
        return {
            'fused_features': fused,
            'method': 'concat',
            'modalities': modality_names,
            'feature_dim': len(fused)
        }
    
    def _attention_fusion(self, features: Dict) -> Dict:
        """
        Attention-based fusion
        Learns to weight different modalities based on their importance
        """
        # Extract feature vectors
        modality_features = {}
        
        for modality, feat_dict in features.items():
            if isinstance(feat_dict, dict):
                for key in ['deep_features', 'transformer_features', 'handcrafted_features', 'feature_vector']:
                    if key in feat_dict:
                        modality_features[modality] = feat_dict[key]
                        break
        
        if not modality_features:
            return self._concatenate_fusion(features)
        
        # Initialize attention model if needed
        if self.attention_model is None:
            feature_dims = {k: len(v) for k, v in modality_features.items()}
            self.attention_model = MultimodalAttention(feature_dims).to(self.device)
            self.attention_model.eval()
        
        # Convert to tensors
        tensors = {}
        for modality, feat in modality_features.items():
            tensors[modality] = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Apply attention fusion
        with torch.no_grad():
            fused, attention_weights = self.attention_model(tensors)
        
        # Convert back to numpy
        fused_np = fused.cpu().numpy().flatten()
        attention_weights_np = {k: v.cpu().numpy().flatten() for k, v in attention_weights.items()}
        
        return {
            'fused_features': fused_np,
            'method': 'attention',
            'modalities': list(modality_features.keys()),
            'attention_weights': attention_weights_np,
            'feature_dim': len(fused_np)
        }
    
    def _weighted_sum_fusion(self, features: Dict) -> Dict:
        """
        Weighted sum fusion with predefined weights
        """
        # Default weights
        weights = {
            'visual': 0.3,
            'text': 0.4,
            'layout': 0.2,
            'graph': 0.1
        }
        
        # Extract and normalize features
        modality_features = {}
        
        for modality, feat_dict in features.items():
            if isinstance(feat_dict, dict):
                for key in ['deep_features', 'transformer_features', 'handcrafted_features', 'feature_vector']:
                    if key in feat_dict:
                        feat = feat_dict[key]
                        # Normalize
                        feat_norm = feat / (np.linalg.norm(feat) + 1e-10)
                        modality_features[modality] = feat_norm
                        break
        
        if not modality_features:
            return self._concatenate_fusion(features)
        
        # Find common dimension (pad/truncate to match)
        max_dim = max(len(f) for f in modality_features.values())
        
        # Weighted sum
        fused = np.zeros(max_dim)
        total_weight = 0.0
        
        for modality, feat in modality_features.items():
            weight = weights.get(modality, 0.1)
            
            # Pad or truncate to max_dim
            if len(feat) < max_dim:
                feat_padded = np.pad(feat, (0, max_dim - len(feat)))
            else:
                feat_padded = feat[:max_dim]
            
            fused += weight * feat_padded
            total_weight += weight
        
        # Normalize by total weight
        fused = fused / max(total_weight, 1e-10)
        
        return {
            'fused_features': fused,
            'method': 'weighted_sum',
            'modalities': list(modality_features.keys()),
            'weights': weights,
            'feature_dim': len(fused)
        }
    
    def fuse_element_features(
        self,
        element_features_list: List[Dict]
    ) -> List[Dict]:
        """
        Fuse features for multiple elements
        
        Args:
            element_features_list: List of feature dictionaries for elements
            
        Returns:
            List of fused features
        """
        fused_list = []
        
        for element_features in element_features_list:
            fused = self.fuse(element_features)
            fused_list.append(fused)
        
        return fused_list


class MultimodalAttention(nn.Module):
    """
    Attention mechanism for multimodal fusion
    """
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 128):
        """
        Initialize attention model
        
        Args:
            feature_dims: Dictionary mapping modality names to feature dimensions
            hidden_dim: Hidden dimension for attention
        """
        super(MultimodalAttention, self).__init__()
        
        self.modalities = list(feature_dims.keys())
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Attention layers
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, features: Dict[str, torch.Tensor]):
        """
        Forward pass
        
        Args:
            features: Dictionary of tensors (one per modality)
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        # Project each modality to hidden space
        projected = {}
        for modality in self.modalities:
            if modality in features:
                feat = features[modality]
                projected[modality] = self.projections[modality](feat)
        
        if not projected:
            # Return zeros if no features
            batch_size = 1
            return torch.zeros(batch_size, self.hidden_dim), {}
        
        # Stack projected features
        stacked = torch.stack(list(projected.values()), dim=1)  # (batch, num_modalities, hidden_dim)
        
        # Compute attention
        # Query: mean of all modalities
        query = torch.mean(stacked, dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        query = self.attention_query(query)
        
        # Keys and values
        keys = self.attention_key(stacked)  # (batch, num_modalities, hidden_dim)
        values = self.attention_value(stacked)
        
        # Attention scores
        scores = torch.matmul(query, keys.transpose(1, 2))  # (batch, 1, num_modalities)
        scores = scores / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, values)  # (batch, 1, hidden_dim)
        attended = attended.squeeze(1)
        
        # Output projection
        output = self.output_proj(attended)
        
        # Convert attention weights to dict
        attention_dict = {}
        for i, modality in enumerate(projected.keys()):
            attention_dict[modality] = attention_weights[:, 0, i]
        
        return output, attention_dict