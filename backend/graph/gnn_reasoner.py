"""
GNN Reasoner
Graph Neural Network for reasoning over document graphs
Uses message passing to propagate information through the graph
"""

import re
import numpy as np
import networkx as nx
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GNNReasoner:
    """
    Applies Graph Neural Network reasoning to document graph
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_gpu: bool = True
    ):
        """
        Initialize GNN reasoner
        
        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            use_gpu: Use GPU if available
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Using rule-based reasoning.")
            self.use_gnn = False
            return
        
        self.use_gnn = True
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build GNN model
        self.model = DocumentGNN(
            node_feature_dim=20,  # Adjust based on features
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
        
        print(f"GNN Reasoner initialized on {self.device}")
    
    def reason(
        self,
        graph: nx.DiGraph,
        task: str = 'field_extraction'
    ) -> Dict:
        """
        Apply reasoning to extract information from graph
        
        Args:
            graph: Document graph
            task: Reasoning task ('field_extraction', 'relation_extraction')
            
        Returns:
            Reasoning results
        """
        if self.use_gnn:
            return self._gnn_reasoning(graph, task)
        else:
            return self._rule_based_reasoning(graph, task)
    
    def _gnn_reasoning(self, graph: nx.DiGraph, task: str) -> Dict:
        """GNN-based reasoning"""
        # Convert NetworkX graph to PyTorch Geometric format
        node_features, edge_index = self._graph_to_tensors(graph)
        
        # Run GNN
        with torch.no_grad():
            node_embeddings = self.model(node_features, edge_index)
        
        # Post-process based on task
        if task == 'field_extraction':
            results = self._extract_fields_from_embeddings(graph, node_embeddings)
        elif task == 'relation_extraction':
            results = self._extract_relations_from_embeddings(graph, node_embeddings)
        else:
            results = {'embeddings': node_embeddings.cpu().numpy()}
        
        return results
    
    def _rule_based_reasoning(self, graph: nx.DiGraph, task: str) -> Dict:
        """Fallback rule-based reasoning when PyTorch is not available"""
        if task == 'field_extraction':
            return self._rule_based_field_extraction(graph)
        elif task == 'relation_extraction':
            return self._rule_based_relation_extraction(graph)
        
        return {}
    
    def _graph_to_tensors(self, graph: nx.DiGraph):
        """Convert NetworkX graph to PyTorch tensors"""
        # Create node feature matrix
        node_list = list(graph.nodes())
        node_features = []
        
        for node_id in node_list:
            node_data = graph.nodes[node_id]
            features = self._extract_node_feature_vector(node_data)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        
        # Create edge index
        edge_list = []
        for source, target in graph.edges():
            source_idx = node_list.index(source)
            target_idx = node_list.index(target)
            edge_list.append([source_idx, target_idx])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        
        return node_features, edge_index
    
    def _extract_node_feature_vector(self, node_data: Dict) -> List[float]:
        """Extract feature vector from node data"""
        features = node_data.get('features', {})
        
        # Create feature vector (adjust size as needed)
        feature_vector = [
            features.get('x', 0) / 1000.0,  # Normalize spatial features
            features.get('y', 0) / 1000.0,
            features.get('width', 0) / 1000.0,
            features.get('height', 0) / 1000.0,
            features.get('area', 0) / 100000.0,
            features.get('aspect_ratio', 0),
            features.get('text_length', 0) / 100.0,
            float(features.get('is_numeric', False)),
            float(features.get('is_uppercase', False)),
            float(features.get('has_digits', False)),
            # Add more features as needed
        ]
        
        # Pad to fixed size
        while len(feature_vector) < 20:
            feature_vector.append(0.0)
        
        return feature_vector[:20]
    
    def _extract_fields_from_embeddings(
        self,
        graph: nx.DiGraph,
        embeddings: torch.Tensor
    ) -> Dict:
        """Extract fields using node embeddings"""
        node_list = list(graph.nodes())
        embeddings_np = embeddings.cpu().numpy()
        
        # Find nodes with similar embeddings (potential field candidates)
        fields = {
            'invoice_number': [],
            'date': [],
            'amount': [],
            'company': []
        }
        
        for i, node_id in enumerate(node_list):
            node_data = graph.nodes[node_id]
            text = node_data.get('text', '').lower()
            
            # Simple classification based on text patterns
            # (In practice, this would use the embeddings)
            if 'invoice' in text or 'inv' in text:
                fields['invoice_number'].append({
                    'node_id': node_id,
                    'text': node_data.get('text'),
                    'confidence': 0.8
                })
            elif any(d in text for d in ['date', 'dated']):
                fields['date'].append({
                    'node_id': node_id,
                    'text': node_data.get('text'),
                    'confidence': 0.8
                })
        
        return fields
    
    def _extract_relations_from_embeddings(
        self,
        graph: nx.DiGraph,
        embeddings: torch.Tensor
    ) -> Dict:
        """Extract relations using node embeddings"""
        node_list = list(graph.nodes())
        embeddings_np = embeddings.cpu().numpy()
        
        relations = []
        
        # Find pairs of nodes with complementary embeddings
        for i, source_id in enumerate(node_list):
            for j, target_id in enumerate(node_list):
                if i >= j:
                    continue
                
                # Calculate embedding similarity
                similarity = np.dot(embeddings_np[i], embeddings_np[j])
                
                if similarity > 0.8:  # Threshold
                    relations.append({
                        'source': source_id,
                        'target': target_id,
                        'type': 'related',
                        'confidence': float(similarity)
                    })
        
        return {'relations': relations}
    
    def _rule_based_field_extraction(self, graph: nx.DiGraph) -> Dict:
        """Extract fields using rules"""
        fields = {
            'invoice_number': [],
            'date': [],
            'company': [],
            'amount': [],
            'total': []
        }
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            text = node_data.get('text', '')
            text_lower = text.lower()
            
            # Invoice number
            if 'invoice' in text_lower and any(c.isdigit() for c in text):
                fields['invoice_number'].append({
                    'node_id': node_id,
                    'text': text,
                    'bbox': node_data.get('bbox'),
                    'confidence': 0.7
                })
            
            # Date
            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                fields['date'].append({
                    'node_id': node_id,
                    'text': text,
                    'bbox': node_data.get('bbox'),
                    'confidence': 0.8
                })
            
            # Company (usually has Ltd, Pvt, etc.)
            if re.search(r'(pvt|ltd|llp|inc|corp)', text_lower):
                fields['company'].append({
                    'node_id': node_id,
                    'text': text,
                    'bbox': node_data.get('bbox'),
                    'confidence': 0.75
                })
            
            # Amount
            if re.search(r'₹|rs\.?\s*\d+', text_lower) or (
                re.match(r'^\d+[,\d]*\.?\d*$', text) and len(text) > 2
            ):
                fields['amount'].append({
                    'node_id': node_id,
                    'text': text,
                    'bbox': node_data.get('bbox'),
                    'confidence': 0.6
                })
            
            # Total
            if 'total' in text_lower:
                # Find nearby amount
                neighbors = list(graph.neighbors(node_id))
                for neighbor_id in neighbors:
                    neighbor_text = graph.nodes[neighbor_id].get('text', '')
                    if re.search(r'\d+', neighbor_text):
                        fields['total'].append({
                            'node_id': neighbor_id,
                            'text': neighbor_text,
                            'bbox': graph.nodes[neighbor_id].get('bbox'),
                            'confidence': 0.9
                        })
        
        return fields
    
    def _rule_based_relation_extraction(self, graph: nx.DiGraph) -> Dict:
        """Extract relations using rules"""
        relations = []
        
        # Extract key-value relations
        for node_id in graph.nodes():
            # Get edges from this node
            for target_id in graph.neighbors(node_id):
                edge_data = graph[node_id][target_id]
                
                if edge_data.get('edge_type') == 'semantic':
                    relations.append({
                        'source': node_id,
                        'target': target_id,
                        'type': edge_data.get('relation'),
                        'confidence': edge_data.get('weight', 0.5)
                    })
        
        return {'relations': relations}


class DocumentGNN(nn.Module):
    """
    Graph Neural Network for document understanding
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super(DocumentGNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, node_features, edge_index):
        """
        Forward pass
        
        Args:
            node_features: (num_nodes, node_feature_dim)
            edge_index: (2, num_edges)
        """
        # Project input features
        x = self.input_proj(node_features)
        x = F.relu(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class GNNLayer(nn.Module):
    """Single GNN layer with message passing"""
    
    def __init__(self, hidden_dim: int):
        super(GNNLayer, self).__init__()
        
        self.message_fn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_fn = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, edge_index):
        """
        Message passing
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge indices (2, num_edges)
        """
        if edge_index.size(1) == 0:
            # No edges, return input
            return x
        
        # Gather source and target features
        source_features = x[edge_index[0]]
        target_features = x[edge_index[1]]
        
        # Compute messages
        edge_features = torch.cat([source_features, target_features], dim=1)
        messages = self.message_fn(edge_features)
        
        # Aggregate messages (sum)
        aggregated = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            target_idx = edge_index[1, i]
            aggregated[target_idx] += messages[i]
        
        # Update node features
        combined = torch.cat([x, aggregated], dim=1)
        updated = self.update_fn(combined)
        
        return updated
