"""
Visual Encoder
Extracts visual features from document images using CNNs
"""

import numpy as np
import cv2
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VisualEncoder:
    """
    Encodes visual features from document images
    Uses pre-trained CNN (ResNet) for feature extraction
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        use_gpu: bool = True
    ):
        """
        Initialize visual encoder
        
        Args:
            model_name: CNN model to use ('resnet50', 'resnet18', 'vgg16')
            use_gpu: Use GPU if available
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Using handcrafted visual features.")
            self.use_deep_features = False
            return
        
        self.use_deep_features = True
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.feature_dim = 4096
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove final classification layer
        if 'resnet' in model_name:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif 'vgg' in model_name:
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Visual encoder initialized with {model_name} on {self.device}")
    
    def encode(self, image: np.ndarray) -> Dict:
        """
        Encode full document image
        
        Args:
            image: Document image (RGB)
            
        Returns:
            Dictionary with visual features
        """
        if self.use_deep_features:
            return self._encode_deep_features(image)
        else:
            return self._encode_handcrafted_features(image)
    
    def encode_region(self, region: np.ndarray) -> Dict:
        """
        Encode a specific region of the image
        
        Args:
            region: Image region (RGB)
            
        Returns:
            Visual features for the region
        """
        if region.size == 0:
            return self._get_zero_features()
        
        if self.use_deep_features:
            return self._encode_deep_features(region)
        else:
            return self._encode_handcrafted_features(region)
    
    def _encode_deep_features(self, image: np.ndarray) -> Dict:
        """
        Extract deep CNN features
        """
        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features.squeeze()
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        # Flatten if needed
        if len(features_np.shape) > 1:
            features_np = features_np.flatten()
        
        return {
            'deep_features': features_np,
            'feature_dim': len(features_np)
        }
    
    def _encode_handcrafted_features(self, image: np.ndarray) -> Dict:
        """
        Extract handcrafted visual features (fallback)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = {}
        
        # 1. Intensity statistics
        features['mean_intensity'] = float(np.mean(gray))
        features['std_intensity'] = float(np.std(gray))
        features['min_intensity'] = float(np.min(gray))
        features['max_intensity'] = float(np.max(gray))
        
        # 2. Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        # 3. Texture features (using histogram)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        features['texture_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # 4. HOG features (simplified)
        if gray.shape[0] > 16 and gray.shape[1] > 16:
            hog_features = self._extract_hog_features(gray)
            features['hog_mean'] = float(np.mean(hog_features))
            features['hog_std'] = float(np.std(hog_features))
        else:
            features['hog_mean'] = 0.0
            features['hog_std'] = 0.0
        
        # 5. Color features (if RGB)
        if len(image.shape) == 3:
            for i, channel in enumerate(['r', 'g', 'b']):
                features[f'{channel}_mean'] = float(np.mean(image[:, :, i]))
                features[f'{channel}_std'] = float(np.std(image[:, :, i]))
        
        # Create feature vector
        feature_vector = np.array([
            features['mean_intensity'],
            features['std_intensity'],
            features['edge_density'],
            features['texture_entropy'],
            features['hog_mean'],
            features['hog_std']
        ])
        
        return {
            'handcrafted_features': feature_vector,
            'feature_dict': features,
            'feature_dim': len(feature_vector)
        }
    
    def _extract_hog_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features
        """
        # Resize to fixed size for consistent features
        resized = cv2.resize(gray_image, (64, 64))
        
        # Calculate gradients
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
        
        # Magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Simplified histogram (9 bins)
        hist, _ = np.histogram(angle, bins=9, range=(-np.pi, np.pi), weights=magnitude)
        
        return hist / (np.sum(hist) + 1e-10)  # Normalize
    
    def _get_zero_features(self) -> Dict:
        """
        Return zero features for empty regions
        """
        if self.use_deep_features:
            return {
                'deep_features': np.zeros(self.feature_dim),
                'feature_dim': self.feature_dim
            }
        else:
            return {
                'handcrafted_features': np.zeros(6),
                'feature_dim': 6
            }
    
    def get_feature_dim(self) -> int:
        """
        Get dimension of feature vector
        """
        if self.use_deep_features:
            return self.feature_dim
        else:
            return 6  # Number of handcrafted features
    
    def encode_batch(self, images: list) -> list:
        """
        Encode multiple images in batch
        
        Args:
            images: List of images
            
        Returns:
            List of feature dictionaries
        """
        if not self.use_deep_features:
            return [self.encode(img) for img in images]
        
        # Batch processing for deep features
        batch_tensors = []
        
        for image in images:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            tensor = self.transform(image)
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        # Convert to list of feature dicts
        results = []
        for i in range(len(images)):
            feat = features[i].cpu().numpy().flatten()
            results.append({
                'deep_features': feat,
                'feature_dim': len(feat)
            })
        
        return results