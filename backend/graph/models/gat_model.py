"""
Graph Attention Network (GAT) Model
Implementation of GAT with multi-head attention for document graphs

Reference:
Veličković et al. (2018) - Graph Attention Networks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer with multi-head attention
    
    Computes attention coefficients:
    α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
    
    where:
    - h_i is the feature of node i
    - W is a learnable weight matrix
    - a is a learnable attention vector
    - || denotes concatenation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize GAT layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate multi-head outputs (vs average)
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation: W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism: a
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            h: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Linear transformation: Wh
        Wh = torch.mm(h, self.W)  # (num_nodes, out_features)
        num_nodes = Wh.size(0)
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask attention for non-existing edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize attention coefficients using softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for attention mechanism by creating all combinations
        
        Args:
            Wh: Transformed features (num_nodes, out_features)
        
        Returns:
            Concatenated features for all pairs (num_nodes, num_nodes, 2*out_features)
        """
        num_nodes = Wh.size(0)
        
        # Repeat features for all pairs
        # Wh_repeated_in_chunks: (num_nodes, num_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=0)
        Wh_repeated_alternating = Wh.repeat(num_nodes, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating],
            dim=1
        )
        
        return all_combinations_matrix.view(num_nodes, num_nodes, 2 * self.out_features)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer
    
    Uses multiple attention heads and concatenates/averages their outputs
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat_heads: bool = True
    ):
        """
        Initialize multi-head GAT layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat_heads: Concatenate heads (True) or average (False)
        """
        super(MultiHeadGATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        # Create attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                out_features,
                dropout=dropout,
                alpha=alpha,
                concat=True
            ) for _ in range(num_heads)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through all attention heads
        
        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            Multi-head output (num_nodes, out_features * num_heads) if concat
            or (num_nodes, out_features) if average
        """
        # Apply all attention heads
        head_outputs = [att(x, adj) for att in self.attentions]
        
        if self.concat_heads:
            # Concatenate heads
            return torch.cat(head_outputs, dim=1)
        else:
            # Average heads
            return torch.mean(torch.stack(head_outputs), dim=0)


class GATModel(nn.Module):
    """
    Multi-layer Graph Attention Network
    
    Architecture:
    - Multiple GAT layers with multi-head attention
    - ELU activation between layers
    - Dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.6,
        alpha: float = 0.2
    ):
        """
        Initialize GAT model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (per head)
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
        """
        super(GATModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            MultiHeadGATLayer(
                input_dim,
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                alpha=alpha,
                concat_heads=True
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                MultiHeadGATLayer(
                    hidden_dim * num_heads,  # Input is concatenated heads
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    alpha=alpha,
                    concat_heads=True
                )
            )
        
        # Output layer (average heads instead of concat)
        self.layers.append(
            MultiHeadGATLayer(
                hidden_dim * num_heads if num_layers > 1 else input_dim,
                output_dim,
                num_heads=num_heads,
                dropout=dropout,
                alpha=alpha,
                concat_heads=False  # Average for output
            )
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, input_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            return_attention: Return attention weights
        
        Returns:
            Node embeddings (num_nodes, output_dim)
        """
        # Apply dropout to input
        x = self.dropout_layer(x)
        
        # Process through all layers except last
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = self.dropout_layer(x)
        
        # Output layer (no dropout after)
        x = self.layers[-1](x, adj)
        
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Get attention weights from a specific layer
        
        Args:
            x: Node features
            adj: Adjacency matrix
            layer_idx: Layer index to extract attention from
        
        Returns:
            Attention weights (num_nodes, num_nodes)
        """
        # Forward pass up to target layer
        for i, layer in enumerate(self.layers[:layer_idx + 1]):
            if i == layer_idx:
                # Extract attention weights from this layer
                # This requires modifying the forward pass
                pass
            x = layer(x, adj)
        
        # Note: Actual implementation would require modifying
        # GraphAttentionLayer to return attention weights
        return None


class DocumentGAT(nn.Module):
    """
    GAT model specifically for document understanding
    
    Features:
    - Edge type aware attention (spatial vs semantic edges)
    - Multi-task learning
    - Hierarchical attention
    """
    
    def __init__(
        self,
        node_feature_dim: int = 20,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.6
    ):
        """
        Initialize Document GAT
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension per head
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            num_classes: Number of node classes
            dropout: Dropout rate
        """
        super(DocumentGAT, self).__init__()
        
        # Main GAT encoder
        self.gat = GATModel(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Edge type embeddings (for different edge types)
        self.edge_type_embedding = nn.Embedding(3, hidden_dim)  # 3 edge types
        
        # Task-specific heads
        # Node classification
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Importance scoring (for field extraction)
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Hierarchical attention (for zone-level reasoning)
        self.hierarchical_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_types: Optional[torch.Tensor] = None,
        task: str = 'embed'
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, node_feature_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)
            edge_types: Edge type indices (num_edges,) - 0: spatial, 1: semantic, 2: hierarchical
            task: Task type ('embed', 'classify', 'predict_edges', 'score')
        
        Returns:
            Task-specific output
        """
        # Get node embeddings
        embeddings = self.gat(x, adj)
        
        if task == 'embed':
            return embeddings
        
        elif task == 'classify':
            # Node classification
            return self.node_classifier(embeddings)
        
        elif task == 'predict_edges':
            # Predict edges between all node pairs
            return self._predict_all_edges(embeddings)
        
        elif task == 'score':
            # Importance scoring for field extraction
            return self.importance_scorer(embeddings)
        
        elif task == 'hierarchical':
            # Hierarchical reasoning
            return self._hierarchical_reasoning(embeddings)
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _predict_all_edges(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict edges for all node pairs"""
        num_nodes = embeddings.size(0)
        edge_predictions = torch.zeros(num_nodes, num_nodes, device=embeddings.device)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_feat = torch.cat([embeddings[i], embeddings[j]])
                score = self.edge_predictor(edge_feat)
                edge_predictions[i, j] = score
                edge_predictions[j, i] = score  # Symmetric
        
        return edge_predictions
    
    def _hierarchical_reasoning(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical attention for zone-level reasoning"""
        # Reshape for attention: (seq_len, batch, embed_dim)
        embeddings_t = embeddings.unsqueeze(1)  # (num_nodes, 1, hidden_dim)
        
        # Apply multi-head attention
        attended, attention_weights = self.hierarchical_attention(
            embeddings_t, embeddings_t, embeddings_t
        )
        
        return attended.squeeze(1)
    
    def extract_key_value_pairs(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        node_texts: list,
        threshold: float = 0.7
    ) -> list:
        """
        Extract key-value pairs using attention weights
        
        Args:
            x: Node features
            adj: Adjacency matrix
            node_texts: List of text for each node
            threshold: Confidence threshold
        
        Returns:
            List of key-value pairs
        """
        # Get embeddings and edge predictions
        embeddings = self.forward(x, adj, task='embed')
        edge_scores = self._predict_all_edges(embeddings)
        
        # Extract pairs with high edge scores
        pairs = []
        num_nodes = edge_scores.size(0)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                score = edge_scores[i, j].item()
                
                if score > threshold:
                    pairs.append({
                        'key_idx': i,
                        'value_idx': j,
                        'key_text': node_texts[i] if i < len(node_texts) else '',
                        'value_text': node_texts[j] if j < len(node_texts) else '',
                        'confidence': score
                    })
        
        # Sort by confidence
        pairs.sort(key=lambda x: x['confidence'], reverse=True)
        
        return pairs
    
    def classify_nodes(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        class_names: list = None
    ) -> dict:
        """
        Classify all nodes
        
        Args:
            x: Node features
            adj: Adjacency matrix
            class_names: List of class names
        
        Returns:
            Dictionary with predictions
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
                'confidence': confidence,
                'all_scores': confidences[i].tolist()
            }
        
        return results


class EdgeTypeAwareGAT(nn.Module):
    """
    GAT with edge-type-aware attention
    Different attention mechanisms for different edge types
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.6
    ):
        """
        Initialize edge-type-aware GAT
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads per edge type
            num_edge_types: Number of edge types
            dropout: Dropout rate
        """
        super(EdgeTypeAwareGAT, self).__init__()
        
        self.num_edge_types = num_edge_types
        
        # Separate attention for each edge type
        self.edge_type_attentions = nn.ModuleList([
            MultiHeadGATLayer(
                input_dim,
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat_heads=True
            ) for _ in range(num_edge_types)
        ])
        
        # Fusion layer to combine outputs from different edge types
        self.fusion = nn.Linear(hidden_dim * num_heads * num_edge_types, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_list: list
    ) -> torch.Tensor:
        """
        Forward pass with multiple adjacency matrices
        
        Args:
            x: Node features
            adj_list: List of adjacency matrices, one per edge type
        
        Returns:
            Fused node embeddings
        """
        outputs = []
        
        for i, adj in enumerate(adj_list):
            if i < len(self.edge_type_attentions):
                output = self.edge_type_attentions[i](x, adj)
                outputs.append(output)
        
        # Concatenate and fuse
        if outputs:
            combined = torch.cat(outputs, dim=1)
            fused = self.fusion(combined)
            return F.elu(fused)
        
        return x


def create_gat_for_documents(
    node_feature_dim: int = 20,
    model_size: str = 'medium'
) -> DocumentGAT:
    """
    Factory function to create Document GAT models
    
    Args:
        node_feature_dim: Dimension of node features
        model_size: 'small', 'medium', or 'large'
    
    Returns:
        DocumentGAT model
    """
    configs = {
        'small': {
            'hidden_dim': 32,
            'num_heads': 2,
            'num_layers': 2,
            'dropout': 0.4
        },
        'medium': {
            'hidden_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.6
        },
        'large': {
            'hidden_dim': 128,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.6
        }
    }
    
    config = configs.get(model_size, configs['medium'])
    
    return DocumentGAT(
        node_feature_dim=node_feature_dim,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_classes=10,  # Invoice field types
        dropout=config['dropout']
    )


# Training utilities
class GATTrainer:
    """Trainer for GAT models"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.005,
        weight_decay: float = 5e-4
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_step(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x, adj, task='classify')
        
        # Compute loss (only on masked nodes)
        loss = F.cross_entropy(output[mask], labels[mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(x, adj, task='classify')
            loss = F.cross_entropy(output[mask], labels[mask])
            
            # Compute accuracy
            predictions = torch.argmax(output[mask], dim=1)
            accuracy = (predictions == labels[mask]).float().mean()
        
        return loss.item(), accuracy.item()


# Example usage
if __name__ == "__main__":
    # Create sample data
    num_nodes = 20
    feature_dim = 20
    num_classes = 5
    
    # Random node features
    x = torch.randn(num_nodes, feature_dim)
    
    # Random adjacency matrix
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = (adj + adj.t()) / 2  # Make symmetric
    
    # Random labels
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    # Create model
    print("Creating GAT model...")
    model = create_gat_for_documents(node_feature_dim=feature_dim, model_size='medium')
    
    # Forward pass
    print("\nForward pass...")
    embeddings = model(x, adj, task='embed')
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Node classification
    print("\nNode classification...")
    class_names = ['invoice_number', 'date', 'company', 'amount', 'other']
    predictions = model.classify_nodes(x, adj, class_names=class_names)
    print(f"Sample predictions: {list(predictions.items())[:3]}")
    
    # Extract key-value pairs
    print("\nExtracting key-value pairs...")
    node_texts = [f"Node_{i}" for i in range(num_nodes)]
    pairs = model.extract_key_value_pairs(x, adj, node_texts, threshold=0.5)
    print(f"Found {len(pairs)} potential key-value pairs")
    if pairs:
        print(f"Top pair: {pairs[0]}")
    
    # Training example
    print("\nTraining example...")
    trainer = GATTrainer(model, learning_rate=0.005)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.6 * num_nodes)] = True
    
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
    
    # Train for a few epochs
    for epoch in range(5):
        train_loss = trainer.train_step(x, adj, labels, train_mask)
        val_loss, val_acc = trainer.evaluate(x, adj, labels, val_mask)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")