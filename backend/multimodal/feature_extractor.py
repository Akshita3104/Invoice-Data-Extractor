"""
Multimodal Feature Extractor
Orchestrates extraction of features from multiple modalities:
- Visual features (from image)
- Textual features (from OCR)
- Layout features (spatial positions)
- Graph features (from document graph)
"""

import numpy as np
from typing import Dict, List, Optional
import networkx as nx

from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder
from .layout_encoder import LayoutEncoder


class MultimodalFeatureExtractor:
    """
    Extracts and combines features from multiple modalities
    """
    
    def __init__(
        self,
        use_visual: bool = True,
        use_text: bool = True,
        use_layout: bool = True,
        use_graph: bool = True,
        use_gpu: bool = True
    ):
        """
        Initialize multimodal feature extractor
        
        Args:
            use_visual: Extract visual features
            use_text: Extract text features
            use_layout: Extract layout features
            use_graph: Extract graph features
            use_gpu: Use GPU if available
        """
        self.use_visual = use_visual
        self.use_text = use_text
        self.use_layout = use_layout
        self.use_graph = use_graph
        
        # Initialize encoders
        if use_visual:
            self.visual_encoder = VisualEncoder(use_gpu=use_gpu)
        
        if use_text:
            self.text_encoder = TextEncoder(use_gpu=use_gpu)
        
        if use_layout:
            self.layout_encoder = LayoutEncoder()
        
        print("Multimodal feature extractor initialized")
    
    def extract_features(
        self,
        image: np.ndarray = None,
        ocr_result: Dict = None,
        zones: List = None,
        graph: nx.DiGraph = None
    ) -> Dict:
        """
        Extract features from all modalities
        
        Args:
            image: Document image
            ocr_result: OCR result with text and bounding boxes
            zones: Layout zones
            graph: Document graph
            
        Returns:
            Dictionary containing features from all modalities
        """
        features = {}
        
        # Extract visual features
        if self.use_visual and image is not None:
            print("Extracting visual features...")
            features['visual'] = self.visual_encoder.encode(image)
        
        # Extract text features
        if self.use_text and ocr_result is not None:
            print("Extracting text features...")
            features['text'] = self.text_encoder.encode(ocr_result)
        
        # Extract layout features
        if self.use_layout and ocr_result is not None:
            print("Extracting layout features...")
            features['layout'] = self.layout_encoder.encode(ocr_result, zones)
        
        # Extract graph features
        if self.use_graph and graph is not None:
            print("Extracting graph features...")
            features['graph'] = self._extract_graph_features(graph)
        
        return features
    
    def _extract_graph_features(self, graph: nx.DiGraph) -> Dict:
        """
        Extract features from document graph
        """
        features = {
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'density': nx.density(graph),
            'node_features': [],
            'edge_features': []
        }
        
        # Extract node features
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_features = node_data.get('features', {})
            features['node_features'].append(node_features)
        
        # Extract edge features
        for source, target in graph.edges():
            edge_data = graph[source][target]
            edge_features = {
                'type': edge_data.get('edge_type', 'unknown'),
                'relation': edge_data.get('relation', 'unknown'),
                'weight': edge_data.get('weight', 1.0)
            }
            features['edge_features'].append(edge_features)
        
        return features
    
    def extract_element_features(
        self,
        element: Dict,
        image: np.ndarray = None,
        context: Dict = None
    ) -> Dict:
        """
        Extract features for a specific element (word, line, block)
        
        Args:
            element: Element data with text and bbox
            image: Original image (for visual features)
            context: Additional context information
            
        Returns:
            Combined features for the element
        """
        features = {}
        
        # Text features
        if self.use_text and 'text' in element:
            text_features = self.text_encoder.encode_text(element['text'])
            features['text'] = text_features
        
        # Visual features (crop element from image)
        if self.use_visual and image is not None and 'bbox' in element:
            bbox = element['bbox']
            x, y = bbox.get('x', 0), bbox.get('y', 0)
            w, h = bbox.get('width', 0), bbox.get('height', 0)
            
            # Crop element region
            element_region = image[y:y+h, x:x+w]
            
            if element_region.size > 0:
                visual_features = self.visual_encoder.encode_region(element_region)
                features['visual'] = visual_features
        
        # Layout features
        if self.use_layout and 'bbox' in element:
            layout_features = self.layout_encoder.encode_bbox(
                element['bbox'],
                image.shape if image is not None else None
            )
            features['layout'] = layout_features
        
        # Context features
        if context:
            features['context'] = context
        
        return features
    
    def batch_extract_elements(
        self,
        elements: List[Dict],
        image: np.ndarray = None
    ) -> List[Dict]:
        """
        Extract features for multiple elements
        
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for element in elements:
            features = self.extract_element_features(element, image)
            features_list.append(features)
        
        return features_list
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of features from each modality
        
        Returns:
            Dictionary mapping modality names to feature dimensions
        """
        dimensions = {}
        
        if self.use_visual:
            dimensions['visual'] = self.visual_encoder.get_feature_dim()
        
        if self.use_text:
            dimensions['text'] = self.text_encoder.get_feature_dim()
        
        if self.use_layout:
            dimensions['layout'] = self.layout_encoder.get_feature_dim()
        
        return dimensions
    
    def save_features(self, features: Dict, output_path: str):
        """
        Save extracted features to disk
        """
        import pickle
        
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        
        print(f"Features saved to {output_path}")
    
    def load_features(self, input_path: str) -> Dict:
        """
        Load features from disk
        """
        import pickle
        
        with open(input_path, 'rb') as f:
            features = pickle.load(f)
        
        print(f"Features loaded from {input_path}")
        return features