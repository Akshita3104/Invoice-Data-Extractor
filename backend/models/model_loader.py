"""
Model loader utility
Loads pre-trained models with proper initialization and caching
"""

import torch
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from . import MODEL_REGISTRY, WEIGHTS_DIR, CONFIGS_DIR
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger
from utils.cache import LRUCache
from utils.gpu_utils import get_device

logger = get_logger(__name__)


class ModelLoader:
    """Load and manage pre-trained models"""
    
    def __init__(self, cache_size: int = 5):
        self.weights_dir = WEIGHTS_DIR
        self.configs_dir = CONFIGS_DIR
        self.device = get_device()
        self.model_cache = LRUCache(max_size=cache_size)
    
    def load_model(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Load model from weights
        
        Args:
            model_name: Name of the model
            device: Device to load model on
            use_cache: Use cached model if available
        
        Returns:
            Loaded model or None
        """
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        # Check cache first
        if use_cache:
            cached_model = self.model_cache.get(model_name)
            if cached_model is not None:
                logger.info(f"Using cached model: {model_name}")
                return cached_model
        
        device = device or self.device
        model_info = MODEL_REGISTRY[model_name]
        weight_file = self.weights_dir / model_info['weight_file']
        
        if not weight_file.exists():
            logger.error(f"Model weights not found: {weight_file}")
            logger.info(f"Run model_downloader.py to download {model_name}")
            return None
        
        try:
            logger.info(f"Loading {model_name} from {weight_file}")
            
            # Load based on file extension
            if weight_file.suffix in ['.pth', '.pt']:
                model = self._load_pytorch_model(model_name, weight_file, device)
            elif weight_file.suffix in ['.pkl', '.pickle']:
                model = self._load_pickle_model(weight_file)
            elif weight_file.suffix == '.bin':
                model = self._load_bin_model(model_name, weight_file, device)
            else:
                logger.error(f"Unsupported model format: {weight_file.suffix}")
                return None
            
            # Cache the model
            if use_cache and model is not None:
                self.model_cache.put(model_name, model)
            
            logger.info(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            return None
    
    def _load_pytorch_model(
        self,
        model_name: str,
        weight_file: Path,
        device: torch.device
    ) -> Optional[torch.nn.Module]:
        """Load PyTorch model"""
        try:
            # Load weights
            checkpoint = torch.load(weight_file, map_location=device)
            
            # Initialize model architecture based on model name
            if model_name == 'tablenet':
                from ..layout_analysis.models.tablenet_model import TableNet
                model = TableNet()
            elif model_name == 'yolov5':
                # YOLOv5 loads differently
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weight_file))
                return model
            elif model_name == 'doctr':
                from doctr.models import detection_predictor
                model = detection_predictor(arch='db_resnet50', pretrained=True)
                return model
            else:
                logger.error(f"No model architecture defined for {model_name}")
                return None
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return None
    
    def _load_pickle_model(self, weight_file: Path) -> Optional[Any]:
        """Load pickled model (scikit-learn, etc.)"""
        try:
            with open(weight_file, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading pickle model: {e}")
            return None
    
    def _load_bin_model(
        self,
        model_name: str,
        weight_file: Path,
        device: torch.device
    ) -> Optional[Any]:
        """Load .bin model (usually transformers)"""
        try:
            if model_name == 'bert_ner':
                from transformers import BertForTokenClassification
                model = BertForTokenClassification.from_pretrained(
                    'dslim/bert-base-NER',
                    cache_dir=str(self.weights_dir)
                )
            elif model_name == 'layoutlm':
                from transformers import LayoutLMForTokenClassification
                model = LayoutLMForTokenClassification.from_pretrained(
                    'microsoft/layoutlm-base-uncased',
                    cache_dir=str(self.weights_dir)
                )
            else:
                logger.error(f"No model architecture defined for {model_name}")
                return None
            
            model = model.to(device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            return None
    
    def load_config(self, model_name: str) -> Optional[Dict]:
        """Load model configuration"""
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        config_file = self.configs_dir / MODEL_REGISTRY[model_name]['config_file']
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return {}
        
        try:
            if config_file.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                import json
                with open(config_file) as f:
                    config = json.load(f)
            else:
                logger.error(f"Unsupported config format: {config_file.suffix}")
                return None
            
            logger.info(f"Loaded config for {model_name}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None
    
    def unload_model(self, model_name: str):
        """Remove model from cache"""
        if model_name in self.model_cache.cache:
            del self.model_cache.cache[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    def clear_cache(self):
        """Clear all cached models"""
        self.model_cache.clear()
        logger.info("Cleared model cache")
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get model metadata including size and info"""
        if model_name not in MODEL_REGISTRY:
            return None
        
        info = MODEL_REGISTRY[model_name].copy()
        weight_file = self.weights_dir / info['weight_file']
        
        metadata = {
            'name': model_name,
            'description': info['description'],
            'weight_file': str(weight_file),
            'exists': weight_file.exists(),
            'device': str(self.device)
        }
        
        if weight_file.exists():
            metadata['size_mb'] = weight_file.stat().st_size / (1024 * 1024)
        
        return metadata


class ModelRegistry:
    """Central registry for managing multiple models"""
    
    def __init__(self):
        self.loader = ModelLoader()
        self.loaded_models: Dict[str, Any] = {}
    
    def get_model(self, model_name: str, reload: bool = False) -> Optional[Any]:
        """
        Get model, loading if necessary
        
        Args:
            model_name: Name of the model
            reload: Force reload even if already loaded
        
        Returns:
            Loaded model
        """
        if model_name in self.loaded_models and not reload:
            return self.loaded_models[model_name]
        
        model = self.loader.load_model(model_name)
        if model is not None:
            self.loaded_models[model_name] = model
        
        return model
    
    def preload_models(self, model_names: list):
        """Preload multiple models"""
        for model_name in model_names:
            logger.info(f"Preloading {model_name}...")
            self.get_model(model_name)
    
    def unload_all(self):
        """Unload all models"""
        self.loaded_models.clear()
        self.loader.clear_cache()
        logger.info("Unloaded all models")
    
    def list_loaded_models(self) -> list:
        """List currently loaded models"""
        return list(self.loaded_models.keys())


# Singleton instance
_model_registry = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_model(model_name: str) -> Optional[Any]:
    """Convenience function to get a model"""
    return get_model_registry().get_model(model_name)