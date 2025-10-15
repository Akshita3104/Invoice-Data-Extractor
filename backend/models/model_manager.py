"""
High-level model management
Provides unified interface for model operations
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from .model_loader import ModelLoader, get_model_registry
from .model_downloader import ModelDownloader
from . import MODEL_REGISTRY, WEIGHTS_DIR
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger
from utils.gpu_utils import get_device, get_gpu_manager

logger = get_logger(__name__)


class ModelManager:
    """
    High-level model management interface
    Handles downloading, loading, and lifecycle of models
    """
    
    def __init__(self, auto_download: bool = False):
        """
        Initialize model manager
        
        Args:
            auto_download: Automatically download missing models
        """
        self.loader = ModelLoader()
        self.downloader = ModelDownloader()
        self.registry = get_model_registry()
        self.auto_download = auto_download
        self.device = get_device()
        
        logger.info(f"ModelManager initialized on device: {self.device}")
    
    def get_model(
        self,
        model_name: str,
        download_if_missing: Optional[bool] = None
    ) -> Optional[Any]:
        """
        Get a model, downloading if necessary
        
        Args:
            model_name: Name of the model
            download_if_missing: Download if not found (overrides auto_download)
        
        Returns:
            Loaded model or None
        """
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        # Check if model exists
        if not self.downloader.verify_model(model_name):
            should_download = (
                download_if_missing
                if download_if_missing is not None
                else self.auto_download
            )
            
            if should_download:
                logger.info(f"Downloading {model_name}...")
                if not self.downloader.download_model(model_name):
                    logger.error(f"Failed to download {model_name}")
                    return None
            else:
                logger.error(f"Model not found: {model_name}")
                logger.info(f"Run: python model_downloader.py download --model {model_name}")
                return None
        
        # Load model
        return self.registry.get_model(model_name)
    
    def preload_models(
        self,
        model_names: List[str],
        download_if_missing: bool = False
    ) -> Dict[str, bool]:
        """
        Preload multiple models
        
        Args:
            model_names: List of model names
            download_if_missing: Download missing models
        
        Returns:
            Dictionary of model_name: success
        """
        results = {}
        
        for model_name in model_names:
            logger.info(f"Preloading {model_name}...")
            model = self.get_model(model_name, download_if_missing)
            results[model_name] = model is not None
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Preloaded {successful}/{len(model_names)} models")
        
        return results
    
    def setup_pipeline(self, pipeline_type: str) -> Dict[str, Any]:
        """
        Setup a complete model pipeline
        
        Args:
            pipeline_type: Type of pipeline ('basic', 'advanced', 'full')
        
        Returns:
            Dictionary of loaded models
        """
        pipelines = {
            'basic': ['ocr_router'],
            'advanced': ['tablenet', 'yolov5', 'ocr_router'],
            'full': ['tablenet', 'yolov5', 'bert_ner', 'layoutlm', 'doctr', 'ocr_router']
        }
        
        if pipeline_type not in pipelines:
            logger.error(f"Unknown pipeline type: {pipeline_type}")
            return {}
        
        model_names = pipelines[pipeline_type]
        logger.info(f"Setting up {pipeline_type} pipeline with {len(model_names)} models")
        
        models = {}
        for model_name in model_names:
            model = self.get_model(model_name, download_if_missing=self.auto_download)
            if model is not None:
                models[model_name] = model
        
        logger.info(f"Pipeline setup complete: {len(models)}/{len(model_names)} models loaded")
        return models
    
    def optimize_models(self, models: List[str], optimization_level: int = 1):
        """
        Optimize models for inference
        
        Args:
            models: List of model names to optimize
            optimization_level: 1=basic, 2=aggressive
        """
        for model_name in models:
            model = self.registry.get_model(model_name)
            if model is None:
                continue
            
            logger.info(f"Optimizing {model_name}...")
            
            # Set to eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Disable gradients
            if isinstance(model, torch.nn.Module):
                for param in model.parameters():
                    param.requires_grad = False
            
            # Level 2: More aggressive optimizations
            if optimization_level >= 2 and torch.cuda.is_available():
                try:
                    # Try to use torch.compile (PyTorch 2.0+)
                    if hasattr(torch, 'compile'):
                        model = torch.compile(model)
                        logger.info(f"Compiled {model_name} with torch.compile")
                except Exception as e:
                    logger.warning(f"Could not compile {model_name}: {e}")
    
    def benchmark_models(self, models: Optional[List[str]] = None, iterations: int = 100):
        """
        Benchmark model inference times
        
        Args:
            models: List of models to benchmark (None = all loaded)
            iterations: Number of iterations
        
        Returns:
            Dictionary of benchmark results
        """
        import time
        
        if models is None:
            models = self.registry.list_loaded_models()
        
        results = {}
        
        for model_name in models:
            model = self.registry.get_model(model_name)
            if model is None:
                continue
            
            logger.info(f"Benchmarking {model_name}...")
            
            # Create dummy input
            if model_name == 'yolov5':
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            elif model_name == 'tablenet':
                dummy_input = torch.randn(1, 3, 1024, 768).to(self.device)
            else:
                # Skip non-torch models
                continue
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            avg_time = (elapsed / iterations) * 1000  # Convert to ms
            
            results[model_name] = {
                'avg_time_ms': avg_time,
                'throughput': 1000 / avg_time,
                'device': str(self.device)
            }
            
            logger.info(f"{model_name}: {avg_time:.2f}ms per inference")
        
        return results
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage of loaded models"""
        gpu_manager = get_gpu_manager()
        memory_info = gpu_manager.get_memory_info()
        
        loaded_models = self.registry.list_loaded_models()
        
        return {
            'loaded_models': loaded_models,
            'num_loaded': len(loaded_models),
            'gpu_memory': memory_info
        }
    
    def unload_models(self, model_names: Optional[List[str]] = None):
        """
        Unload models from memory
        
        Args:
            model_names: List of models to unload (None = all)
        """
        if model_names is None:
            self.registry.unload_all()
            logger.info("Unloaded all models")
        else:
            for model_name in model_names:
                self.loader.unload_model(model_name)
                logger.info(f"Unloaded {model_name}")
    
    def get_status(self) -> Dict:
        """Get overall status of model manager"""
        available_models = list(MODEL_REGISTRY.keys())
        downloaded_models = self.downloader.list_downloaded_models()
        loaded_models = self.registry.list_loaded_models()
        
        status = {
            'available_models': available_models,
            'downloaded_models': downloaded_models,
            'loaded_models': loaded_models,
            'num_available': len(available_models),
            'num_downloaded': len(downloaded_models),
            'num_loaded': len(loaded_models),
            'device': str(self.device),
            'auto_download': self.auto_download
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            gpu_manager = get_gpu_manager()
            status['gpu_info'] = gpu_manager.get_memory_info()
        
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        self.registry.unload_all()
        logger.info("ModelManager cleanup complete")


# Convenience functions
def setup_inference_pipeline(pipeline_type: str = 'basic', auto_download: bool = True) -> ModelManager:
    """
    Quick setup for inference pipeline
    
    Args:
        pipeline_type: 'basic', 'advanced', or 'full'
        auto_download: Automatically download missing models
    
    Returns:
        Configured ModelManager
    """
    manager = ModelManager(auto_download=auto_download)
    manager.setup_pipeline(pipeline_type)
    return manager


def get_model_status() -> Dict:
    """Get quick status of all models"""
    manager = ModelManager()
    return manager.get_status()


# CLI interface
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Model Manager CLI")
    parser.add_argument(
        'action',
        choices=['status', 'setup', 'benchmark', 'cleanup', 'memory'],
        help='Action to perform'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        choices=['basic', 'advanced', 'full'],
        default='basic',
        help='Pipeline type for setup'
    )
    parser.add_argument(
        '--auto-download',
        action='store_true',
        help='Automatically download missing models'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to operate on'
    )
    
    args = parser.parse_args()
    
    manager = ModelManager(auto_download=args.auto_download)
    
    if args.action == 'status':
        status = manager.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.action == 'setup':
        print(f"Setting up {args.pipeline} pipeline...")
        models = manager.setup_pipeline(args.pipeline)
        print(f"Loaded models: {list(models.keys())}")
    
    elif args.action == 'benchmark':
        print("Running benchmarks...")
        results = manager.benchmark_models(args.models)
        print("\nBenchmark Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Average time: {metrics['avg_time_ms']:.2f}ms")
            print(f"  Throughput: {metrics['throughput']:.2f} inferences/sec")
            print(f"  Device: {metrics['device']}")
    
    elif args.action == 'cleanup':
        print("Cleaning up...")
        manager.cleanup()
        print("Cleanup complete")
    
    elif args.action == 'memory':
        memory_info = manager.get_memory_usage()
        print(json.dumps(memory_info, indent=2))