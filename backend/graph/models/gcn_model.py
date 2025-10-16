"""
Graph Convolutional Network (GCN) Model
Implementation of GCN for document graph reasoning

Reference:
Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvLayer(nn.Module):
    """
    Single Graph Convolutional Layer
    
    Performs message passing:
    h_i^(l+1) = σ(Σ_{j∈N(i)} (1/√(d_i * d_j)) * W^(l) * h_j^(l))
    
    where:
    - h_i^(l) is the feature of node i at layer l
    - N(i) is the neighborhood of node i
    - d_i is the degree of node i
    - W^(l) is the weight matrix at layer l
    - σ is an activation function
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize GCN layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            dropout: Dropout rate
        """
        super(GraphConvLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes) or edge_index (2, num_edges)
        
        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Apply dropout to input
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        
        # Linear transformation: X * W
        support = torch.mm(x, self.weight)
        
        # Message passing: A * (X * W)
        if adj.dim() == 2 and adj.size(0) == adj.size(1):
            # Dense adjacency matrix
            output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        else:
            # Edge index format (2, num_edges)
            output = self._sparse_mm_from_edge_index(support, adj)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _sparse_mm_from_edge_index(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Sparse matrix multiplication using edge index"""
        num_nodes = x.size(0)
        
        # Aggregate messages
        output = torch.zeros_like(x)
        
        if edge_index.size(1) > 0:
            # Get source and target indices
            source_idx = edge_index[0]
            target_idx = edge_index[1]
            
            # Aggregate: sum of neighbor features
            output.index_add_(0, target_idx, x[source_idx])
        
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'


class GCNModel(nn.Module):
    """
    Multi-layer Graph Convolutional Network
    
    Architecture:
    - Input layer
    - Multiple GCN layers with ReLU activation
    - Optional batch normalization
    - Dropout for regularization
    - Output layer
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        output_dim: int = 16,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        """
        Initialize GCN model
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(GCNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Input layer
        self.layers.append(GraphConvLayer(input_dim, hidden_dims[0], dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1], dropout=dropout)
            )
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(
            GraphConvLayer(hidden_dims[-1], output_dim, dropout=0.0)
        )
        
        # Activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        return activations.get(activation, F.relu)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes) or edge_index (2, num_edges)
            return_all_layers: Return features from all layers
        
        Returns:
            Node embeddings (num_nodes, output_dim)
            Or list of embeddings from all layers if return_all_layers=True
        """
        layer_outputs = []
        
        # Process through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            
            # Batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            if return_all_layers:
                layer_outputs.append(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x, adj)
        
        if return_all_layers:
            layer_outputs.append(x)
            return layer_outputs
        
        return x
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Get node embeddings from a specific layer
        
        Args:
            x: Node features
            adj: Adjacency matrix
            layer: Layer index (-1 for final layer)
        
        Returns:
            Node embeddings from specified layer
        """
        all_outputs = self.forward(x, adj, return_all_layers=True)
        return all_outputs[layer]
    
    def normalize_adjacency(
        self,
        adj: torch.Tensor,
        add_self_loops: bool = True
    ) -> torch.Tensor:
        """
        Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        
        Args:
            adj: Adjacency matrix (num_nodes, num_nodes)
            add_self_loops: Whether to add self-loops
        
        Returns:
            Normalized adjacency matrix
        """
        if add_self_loops:
            # Add self-loops: A = A + I
            num_nodes = adj.size(0)
            adj = adj + torch.eye(num_nodes, device=adj.device)
        
        # Compute degree matrix D
        degree = adj.sum(dim=1)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # D^(-1/2) * A * D^(-1/2)
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
        
        return adj_normalized


class DocumentGCN(nn.Module):
    """
    GCN model specifically designed for document understanding
    
    Features:
    - Handles both spatial and semantic edges
    - Multi-task outputs (node classification, edge prediction)
    - Attention-based aggregation
    """
    
    def __init__(
        self,
        node_feature_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 10,
        dropout: float = 0.5
    ):
        """
        Initialize Document GCN
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            num_classes: Number of node classes (field types)
            dropout: Dropout rate
        """
        super(DocumentGCN, self).__init__()
        
        # GCN encoder
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.gcn = GCNModel(
            input_dim=node_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dim,
            dropout=dropout,
            use_batch_norm=True
        )
        
        # Task-specific heads
        # Node classification (field type prediction)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Edge prediction (relationship detection)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Field extraction head
        self.field_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        task: str = 'classify'
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features
            adj: Adjacency matrix
            task: Task type ('classify', 'edge_predict', 'extract')
        
        Returns:
            Task-specific output
        """
        # Get node embeddings
        embeddings = self.gcn(x, adj)
        
        if task == 'classify':
            # Node classification
            return self.node_classifier(embeddings)
        
        elif task == 'edge_predict':
            # Edge prediction (for all pairs)
            num_nodes = embeddings.size(0)
            edge_scores = []
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    edge_feat = torch.cat([embeddings[i], embeddings[j]])
                    score = self.edge_predictor(edge_feat)
                    edge_scores.append(score)
            
            return torch.stack(edge_scores) if edge_scores else torch.tensor([])
        
        elif task == 'extract':
            # Field extraction (importance score)
            return self.field_extractor(embeddings)
        
        else:
            return embeddings
    
    def predict_field_type(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        class_names: list = None
    ) -> dict:
        """
        Predict field types for all nodes
        
        Args:
            x: Node features
            adj: Adjacency matrix
            class_names: List of class names
        
        Returns:
            Dictionary mapping node indices to predicted classes
        """
        logits = self.forward(x, adj, task='classify')
        predictions = torch.argmax(logits, dim=1)
        confidences = F.softmax(logits, dim=1)
        
        results = {}
        for i in range(predictions.size(0)):
            pred_class = predictions[i].item()
            confidence = confidences[i, pred_class].item()
            
            results[i] = {
                'class_id': pred_class,
                'class_name': class_names[pred_class] if class_names else str(pred_class),
                'confidence': confidence
            }
        
        return results


def create_gcn_for_documents(
    node_feature_dim: int = 20,
    model_size: str = 'medium'
) -> DocumentGCN:
    """
    Factory function to create Document GCN models
    
    Args:
        node_feature_dim: Dimension of node features
        model_size: 'small', 'medium', or 'large'
    
    Returns:
        DocumentGCN model
    """
    configs = {
        'small': {
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.3
        },
        'medium': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.5
        },
        'large': {
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.6
        }
    }
    
    config = configs.get(model_size, configs['medium'])
    
    return DocumentGCN(
        node_feature_dim=node_feature_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=10,  # Adjust based on field types
        dropout=config['dropout']
    )


# Example usage
if __name__ == "__main__":
    # Create sample data
    num_nodes = 10
    num_edges = 15
    feature_dim = 20
    
    # Random node features
    x = torch.randn(num_nodes, feature_dim)
    
    # Random adjacency matrix
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = (adj + adj.t()) / 2  # Make symmetric
    
    # Create model
    model = create_gcn_for_documents(node_feature_dim=feature_dim)
    
    # Forward pass
    embeddings = model(x, adj)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Predict field types
    predictions = model.predict_field_type(x, adj)
    print(f"Predictions: {predictions}")