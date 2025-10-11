"""
Graph Builder
Constructs document graph from OCR results, layout zones, and tables
Nodes represent text elements, edges represent relationships
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from dataclasses import dataclass

from .spatial_relations import SpatialRelationExtractor
from .semantic_relations import SemanticRelationExtractor


@dataclass
class DocumentNode:
    """Represents a node in the document graph"""
    id: str
    node_type: str  # 'word', 'line', 'block', 'zone', 'table', 'cell'
    text: str
    bbox: Dict[str, int]
    features: Dict
    confidence: float = 0.0


@dataclass
class DocumentEdge:
    """Represents an edge in the document graph"""
    source: str
    target: str
    edge_type: str  # 'spatial', 'semantic', 'hierarchical'
    relation: str  # 'above', 'below', 'left', 'right', 'key-value', etc.
    weight: float = 1.0


class GraphBuilder:
    """
    Builds document graph from various document analysis results
    """
    
    def __init__(
        self,
        include_words: bool = True,
        include_lines: bool = True,
        include_blocks: bool = True
    ):
        """
        Initialize graph builder
        
        Args:
            include_words: Include word-level nodes
            include_lines: Include line-level nodes
            include_blocks: Include block-level nodes
        """
        self.include_words = include_words
        self.include_lines = include_lines
        self.include_blocks = include_blocks
        
        self.spatial_extractor = SpatialRelationExtractor()
        self.semantic_extractor = SemanticRelationExtractor()
        
        self.graph = None
        self.nodes = []
        self.edges = []
    
    def build(
        self,
        ocr_result: Dict,
        zones: List = None,
        tables: List = None
    ) -> nx.DiGraph:
        """
        Build document graph
        
        Args:
            ocr_result: OCR result containing words, lines, blocks
            zones: Optional list of zones from ZoneSegmenter
            tables: Optional list of tables from TableDetector
            
        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        self.nodes = []
        self.edges = []
        
        # Add nodes from OCR result
        if self.include_words and 'words' in ocr_result:
            self._add_word_nodes(ocr_result['words'])
        
        if self.include_lines and 'lines' in ocr_result:
            self._add_line_nodes(ocr_result['lines'])
        
        if self.include_blocks and 'blocks' in ocr_result:
            self._add_block_nodes(ocr_result['blocks'])
        
        # Add zone nodes
        if zones:
            self._add_zone_nodes(zones)
        
        # Add table nodes
        if tables:
            self._add_table_nodes(tables)
        
        # Add nodes to graph
        for node in self.nodes:
            self.graph.add_node(
                node.id,
                node_type=node.node_type,
                text=node.text,
                bbox=node.bbox,
                features=node.features,
                confidence=node.confidence
            )
        
        # Extract and add edges
        self._add_spatial_edges()
        self._add_semantic_edges()
        self._add_hierarchical_edges()
        
        # Add edges to graph
        for edge in self.edges:
            self.graph.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type,
                relation=edge.relation,
                weight=edge.weight
            )
        
        print(f"Built graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
        
        return self.graph
    
    def _add_word_nodes(self, words: List[Dict]):
        """Add word-level nodes"""
        for i, word in enumerate(words):
            node = DocumentNode(
                id=f"word_{i}",
                node_type='word',
                text=word.get('text', ''),
                bbox=word.get('bbox', {}),
                features=self._extract_node_features(word, 'word'),
                confidence=word.get('confidence', 0.0)
            )
            self.nodes.append(node)
    
    def _add_line_nodes(self, lines: List[Dict]):
        """Add line-level nodes"""
        for i, line in enumerate(lines):
            node = DocumentNode(
                id=f"line_{i}",
                node_type='line',
                text=line.get('text', ''),
                bbox=line.get('bbox', {}),
                features=self._extract_node_features(line, 'line'),
                confidence=line.get('confidence', 0.0)
            )
            self.nodes.append(node)
    
    def _add_block_nodes(self, blocks: List[Dict]):
        """Add block-level nodes"""
        for i, block in enumerate(blocks):
            node = DocumentNode(
                id=f"block_{i}",
                node_type='block',
                text=block.get('text', ''),
                bbox=block.get('bbox', {}),
                features=self._extract_node_features(block, 'block'),
                confidence=block.get('confidence', 0.0)
            )
            self.nodes.append(node)
    
    def _add_zone_nodes(self, zones: List):
        """Add zone nodes"""
        for i, zone in enumerate(zones):
            node = DocumentNode(
                id=f"zone_{i}",
                node_type='zone',
                text=zone.content if hasattr(zone, 'content') else '',
                bbox=zone.bbox if hasattr(zone, 'bbox') else {},
                features={
                    'zone_type': zone.zone_type if hasattr(zone, 'zone_type') else 'unknown',
                    'text_density': zone.text_density if hasattr(zone, 'text_density') else 0.0
                },
                confidence=zone.confidence if hasattr(zone, 'confidence') else 0.0
            )
            self.nodes.append(node)
    
    def _add_table_nodes(self, tables: List):
        """Add table and cell nodes"""
        for i, table in enumerate(tables):
            # Add table node
            table_bbox = table.bbox if hasattr(table, 'bbox') else table.get('bbox', {})
            
            table_node = DocumentNode(
                id=f"table_{i}",
                node_type='table',
                text='',
                bbox=table_bbox,
                features={
                    'rows': table.rows if hasattr(table, 'rows') else table.get('rows', 0),
                    'cols': table.cols if hasattr(table, 'cols') else table.get('cols', 0),
                    'has_borders': table.has_borders if hasattr(table, 'has_borders') else True
                },
                confidence=table.confidence if hasattr(table, 'confidence') else 0.0
            )
            self.nodes.append(table_node)
            
            # Add cell nodes
            cells = table.cells if hasattr(table, 'cells') else table.get('cells', [])
            for j, cell in enumerate(cells):
                cell_bbox = cell.bbox if hasattr(cell, 'bbox') else cell.get('bbox', {})
                cell_text = cell.text if hasattr(cell, 'text') else cell.get('text', '')
                
                cell_node = DocumentNode(
                    id=f"cell_{i}_{j}",
                    node_type='cell',
                    text=cell_text,
                    bbox=cell_bbox,
                    features={
                        'row': cell.row if hasattr(cell, 'row') else cell.get('row', 0),
                        'col': cell.col if hasattr(cell, 'col') else cell.get('col', 0),
                        'table_id': f"table_{i}"
                    },
                    confidence=cell.confidence if hasattr(cell, 'confidence') else 0.0
                )
                self.nodes.append(cell_node)
    
    def _extract_node_features(self, element: Dict, element_type: str) -> Dict:
        """Extract features for a node"""
        bbox = element.get('bbox', {})
        text = element.get('text', '')
        
        features = {
            'x': bbox.get('x', 0),
            'y': bbox.get('y', 0),
            'width': bbox.get('width', 0),
            'height': bbox.get('height', 0),
            'area': bbox.get('width', 0) * bbox.get('height', 0),
            'aspect_ratio': bbox.get('width', 0) / max(bbox.get('height', 1), 1),
            'text_length': len(text),
            'is_numeric': text.replace(',', '').replace('.', '').isdigit(),
            'is_uppercase': text.isupper(),
            'has_digits': any(c.isdigit() for c in text)
        }
        
        return features
    
    def _add_spatial_edges(self):
        """Add spatial relationship edges"""
        # Get nodes that have spatial relationships (words, lines, blocks)
        spatial_nodes = [
            n for n in self.nodes 
            if n.node_type in ['word', 'line', 'block']
        ]
        
        # Extract spatial relations
        for i, node1 in enumerate(spatial_nodes):
            for node2 in spatial_nodes[i+1:]:
                relations = self.spatial_extractor.extract_relations(
                    node1.bbox,
                    node2.bbox
                )
                
                for relation, confidence in relations.items():
                    if confidence > 0.5:  # Threshold
                        edge = DocumentEdge(
                            source=node1.id,
                            target=node2.id,
                            edge_type='spatial',
                            relation=relation,
                            weight=confidence
                        )
                        self.edges.append(edge)
    
    def _add_semantic_edges(self):
        """Add semantic relationship edges (e.g., key-value pairs)"""
        # Find potential key-value pairs
        text_nodes = [
            n for n in self.nodes
            if n.node_type in ['word', 'line', 'block'] and n.text
        ]
        
        for i, node1 in enumerate(text_nodes):
            for node2 in text_nodes[i+1:]:
                # Check if they could be key-value pair
                relation = self.semantic_extractor.detect_key_value_pair(
                    node1.text,
                    node2.text,
                    node1.bbox,
                    node2.bbox
                )
                
                if relation:
                    edge = DocumentEdge(
                        source=node1.id,
                        target=node2.id,
                        edge_type='semantic',
                        relation=relation['type'],
                        weight=relation['confidence']
                    )
                    self.edges.append(edge)
    
    def _add_hierarchical_edges(self):
        """Add hierarchical relationship edges (parent-child)"""
        # Connect zones to their containing elements
        zone_nodes = [n for n in self.nodes if n.node_type == 'zone']
        other_nodes = [n for n in self.nodes if n.node_type != 'zone']
        
        for zone_node in zone_nodes:
            for other_node in other_nodes:
                if self._bbox_contains(zone_node.bbox, other_node.bbox):
                    edge = DocumentEdge(
                        source=zone_node.id,
                        target=other_node.id,
                        edge_type='hierarchical',
                        relation='contains',
                        weight=1.0
                    )
                    self.edges.append(edge)
        
        # Connect tables to cells
        table_nodes = [n for n in self.nodes if n.node_type == 'table']
        cell_nodes = [n for n in self.nodes if n.node_type == 'cell']
        
        for cell_node in cell_nodes:
            table_id = cell_node.features.get('table_id')
            if table_id:
                edge = DocumentEdge(
                    source=table_id,
                    target=cell_node.id,
                    edge_type='hierarchical',
                    relation='contains',
                    weight=1.0
                )
                self.edges.append(edge)
    
    def _bbox_contains(self, parent_bbox: Dict, child_bbox: Dict) -> bool:
        """Check if parent bbox contains child bbox"""
        px1, py1 = parent_bbox.get('x', 0), parent_bbox.get('y', 0)
        px2 = px1 + parent_bbox.get('width', 0)
        py2 = py1 + parent_bbox.get('height', 0)
        
        cx1, cy1 = child_bbox.get('x', 0), child_bbox.get('y', 0)
        cx2 = cx1 + child_bbox.get('width', 0)
        cy2 = cy1 + child_bbox.get('height', 0)
        
        return px1 <= cx1 and py1 <= cy1 and px2 >= cx2 and py2 >= cy2
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node data by ID"""
        if self.graph and node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        return None
    
    def get_neighbors(self, node_id: str, edge_type: str = None) -> List[str]:
        """Get neighboring nodes"""
        if not self.graph or node_id not in self.graph.nodes:
            return []
        
        if edge_type:
            neighbors = [
                n for n in self.graph.neighbors(node_id)
                if self.graph[node_id][n].get('edge_type') == edge_type
            ]
        else:
            neighbors = list(self.graph.neighbors(node_id))
        
        return neighbors
    
    def export_graph(self, format: str = 'dict') -> Dict:
        """
        Export graph to various formats
        
        Args:
            format: 'dict', 'json', or 'adjacency'
        """
        if format == 'dict':
            return {
                'nodes': [
                    {
                        'id': n.id,
                        'type': n.node_type,
                        'text': n.text,
                        'bbox': n.bbox,
                        'features': n.features
                    }
                    for n in self.nodes
                ],
                'edges': [
                    {
                        'source': e.source,
                        'target': e.target,
                        'type': e.edge_type,
                        'relation': e.relation,
                        'weight': e.weight
                    }
                    for e in self.edges
                ]
            }
        elif format == 'adjacency':
            return nx.to_dict_of_dicts(self.graph)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def visualize_graph(self, output_path: str = None):
        """
        Visualize graph (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            
            pos = nx.spring_layout(self.graph)
            
            # Color nodes by type
            node_colors = []
            for node_id in self.graph.nodes():
                node_type = self.graph.nodes[node_id]['node_type']
                if node_type == 'word':
                    node_colors.append('lightblue')
                elif node_type == 'line':
                    node_colors.append('lightgreen')
                elif node_type == 'block':
                    node_colors.append('orange')
                elif node_type == 'zone':
                    node_colors.append('red')
                elif node_type == 'table':
                    node_colors.append('purple')
                else:
                    node_colors.append('gray')
            
            plt.figure(figsize=(15, 10))
            nx.draw(
                self.graph,
                pos,
                node_color=node_colors,
                with_labels=True,
                node_size=500,
                font_size=8,
                arrows=True
            )
            
            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib is required for visualization")