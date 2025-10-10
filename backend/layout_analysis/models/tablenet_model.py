"""
TableNet Model
Deep learning model for table detection and structure recognition
Based on the TableNet paper: https://arxiv.org/abs/2001.01469

Architecture:
- Encoder: VGG-19 (pre-trained on ImageNet)
- Decoder: Two branches
  - Table Detection: Segments table regions
  - Column Detection: Detects table columns
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TableNetModel:
    """
    TableNet model for table detection
    """
    
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize TableNet model
        
        Args:
            model_path: Path to pre-trained weights (optional)
            use_gpu: Use GPU if available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TableNet. "
                "Install with: pip install torch torchvision"
            )
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path:
            self._load_weights(model_path)
        
        self.model.eval()
        
        print(f"TableNet model initialized on {self.device}")
    
    def _build_model(self):
        """
        Build TableNet architecture
        """
        return TableNet()
    
    def _load_weights(self, model_path: str):
        """
        Load pre-trained weights
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}: {e}")
    
    def detect_tables(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect tables in image
        
        Args:
            image: Input image (RGB)
            threshold: Detection threshold (0-1)
            
        Returns:
            List of detected table regions with bounding boxes
        """
        # Preprocess image
        input_tensor = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            table_mask, column_mask = self.model(input_tensor)
        
        # Post-process predictions
        table_mask = table_mask.cpu().numpy()[0, 0]
        column_mask = column_mask.cpu().numpy()[0, 0]
        
        # Threshold masks
        table_binary = (table_mask > threshold).astype(np.uint8) * 255
        column_binary = (column_mask > threshold).astype(np.uint8) * 255
        
        # Resize masks to original image size
        table_binary = cv2.resize(table_binary, (image.shape[1], image.shape[0]))
        column_binary = cv2.resize(column_binary, (image.shape[1], image.shape[0]))
        
        # Extract table regions
        tables = self._extract_table_regions(table_binary, column_binary, image.shape)
        
        return tables
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        """
        # Resize to model input size (typically 1024x1024)
        input_size = 1024
        resized = cv2.resize(image, (input_size, input_size))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor (CHW format)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _extract_table_regions(
        self,
        table_mask: np.ndarray,
        column_mask: np.ndarray,
        image_shape: Tuple
    ) -> List[Dict]:
        """
        Extract table regions from predicted masks
        """
        # Find contours in table mask
        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        tables = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 100 or h < 50:
                continue
            
            # Calculate confidence based on mask intensity
            roi_mask = table_mask[y:y+h, x:x+w]
            confidence = np.mean(roi_mask) / 255.0
            
            # Analyze columns in this region
            roi_columns = column_mask[y:y+h, x:x+w]
            num_columns = self._count_columns(roi_columns)
            
            tables.append({
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'confidence': float(confidence),
                'num_columns': num_columns,
                'table_mask': roi_mask,
                'column_mask': roi_columns
            })
        
        return tables
    
    def _count_columns(self, column_mask: np.ndarray) -> int:
        """
        Count number of columns from column mask
        """
        # Vertical projection
        projection = np.sum(column_mask, axis=0)
        
        # Smooth projection
        kernel_size = max(3, len(projection) // 50)
        projection = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find peaks (columns)
        threshold = np.max(projection) * 0.3
        in_peak = False
        peaks = 0
        
        for val in projection:
            if val > threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif val <= threshold:
                in_peak = False
        
        return max(peaks, 1)


class TableNet(nn.Module):
    """
    TableNet neural network architecture
    """
    
    def __init__(self, num_classes: int = 1):
        super(TableNet, self).__init__()
        
        # Encoder (VGG-19)
        vgg19 = models.vgg19(pretrained=True)
        self.encoder = vgg19.features
        
        # Freeze early layers
        for i, layer in enumerate(self.encoder):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Decoder for table detection
        self.table_decoder = self._build_decoder(num_classes)
        
        # Decoder for column detection
        self.column_decoder = self._build_decoder(num_classes)
    
    def _build_decoder(self, num_classes: int):
        """
        Build decoder branch
        """
        return nn.Sequential(
            # Upsample 1
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Upsample 2
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Upsample 3
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Upsample 4
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Upsample 5
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Final convolution
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            table_mask: Table detection mask
            column_mask: Column detection mask
        """
        # Encode
        features = self.encoder(x)
        
        # Decode
        table_mask = self.table_decoder(features)
        column_mask = self.column_decoder(features)
        
        return table_mask, column_mask


class TableNetTrainer:
    """
    Trainer for TableNet model
    (For fine-tuning on custom datasets)
    """
    
    def __init__(
        self,
        model: TableNet,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def train_step(
        self,
        images: torch.Tensor,
        table_masks: torch.Tensor,
        column_masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step
        
        Returns:
            Dictionary with losses
        """
        self.model.train()
        
        # Move to device
        images = images.to(self.device)
        table_masks = table_masks.to(self.device)
        column_masks = column_masks.to(self.device)
        
        # Forward pass
        pred_table, pred_column = self.model(images)
        
        # Calculate losses
        table_loss = self.criterion(pred_table, table_masks)
        column_loss = self.criterion(pred_column, column_masks)
        total_loss = table_loss + column_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'table_loss': table_loss.item(),
            'column_loss': column_loss.item()
        }
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")