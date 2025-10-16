"""
Graph Neural Network Models
Implementations of GNN architectures for document understanding
"""

try:
    from .gcn_model import GCNModel, GraphConvLayer
    from .gat_model import GATModel, GraphAttentionLayer
    
    __all__ = [
        'GCNModel',
        'GraphConvLayer',
        'GATModel',
        'GraphAttentionLayer'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import GNN models. PyTorch may not be installed: {e}")
    __all__ = []