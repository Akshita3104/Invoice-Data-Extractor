"""
Graph Module
Document graph construction and reasoning:
- Build document graphs encoding spatial, semantic, and hierarchical relationships
- Apply Graph Neural Networks for reasoning
- Extract relationships between document entities
"""

from .graph_builder import GraphBuilder
from .spatial_relations import SpatialRelationExtractor
from .semantic_relations import SemanticRelationExtractor
from .gnn_reasoner import GNNReasoner

__all__ = [
    'GraphBuilder',
    'SpatialRelationExtractor',
    'SemanticRelationExtractor',
    'GNNReasoner'
]