"""
GPU acceleration utilities
Handles device selection, memory management, and batch processing
"""

import os
import torch
import numpy as np
from typing import Optional, List, Union
from .logger import get_logger

logger = get_logger(__name__)


class GPUManager:
    """Manage GPU resources and device selection"""
    
    def __init__(self, auto_select: bool = True):
        self.device = self._select_device(auto_select)
        self.device_name = self._get_device_name()
        self._log_device_info()
    
    def _select_device(self, auto_select: bool) -> torch.device:
        """Select the best available device"""
        if not auto_select:
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            # Select GPU with most free memory
            device_id = self._get_best_gpu()
            return torch.device(f'cuda:{device_id}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon GPU
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_best_gpu(self) -> int:
        """Get GPU with most free memory"""
        if not torch.cuda.is_available():
            return 0
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return 0
        
        # Find GPU with most free memory
        max_free_memory = 0
        best_gpu = 0
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_allocated(i)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        
        return best_gpu
    
    def _get_device_name(self) -> str:
        """Get device name"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(self.device.index or 0)
        elif self.device.type == 'mps':
            return 'Apple Silicon GPU'
        else:
            return 'CPU'
    
    def _log_device_info(self):
        """Log device information"""
        logger.info(f"Using device: {self.device} ({self.device_name})")
        
        if self.device.type == 'cuda':
            device_id = self.device.index or 0
            props = torch.cuda.get_device_properties(device_id)
            total_memory_gb = props.total_memory / (1024 ** 3)
            logger.info(f"GPU Memory: {total_memory_gb:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
    
    def get_device(self) -> torch.device:
        """Get current device"""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return self.device.type in ['cuda', 'mps']
    
    def get_memory_info(self) -> dict:
        """Get GPU memory information"""
        if self.device.type == 'cuda':
            device_id = self.device.index or 0
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
            free = total - allocated
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free,
                'total_gb': total,
                'utilization': (allocated / total * 100) if total > 0 else 0
            }
        else:
            return {
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0,
                'total_gb': 0,
                'utilization': 0
            }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def synchronize(self):
        """Synchronize GPU operations"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
    
    def set_memory_fraction(self, fraction: float = 0.9):
        """Set maximum memory fraction to use"""
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(fraction, self.device.index or 0)
            logger.info(f"GPU memory fraction set to {fraction}")


# Global GPU manager instance
_global_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager()
    return _global_gpu_manager


def get_device() -> torch.device:
    """Get current device"""
    return get_gpu_manager().get_device()


def to_device(
    data: Union[torch.Tensor, List, dict, np.ndarray],
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, List, dict]:
    """Move data to device"""
    if device is None:
        device = get_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    else:
        return data


def batch_process(
    data: List,
    process_fn,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    show_progress: bool = True
) -> List:
    """
    Process data in batches on GPU
    
    Args:
        data: List of data items to process
        process_fn: Function to process each batch
        batch_size: Number of items per batch
        device: Device to use
        show_progress: Show progress bar
    
    Returns:
        List of processed results
    """
    if device is None:
        device = get_device()
    
    results = []
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_device = to_device(batch, device)
        
        with torch.no_grad():
            batch_results = process_fn(batch_device)
        
        results.extend(batch_results)
        
        if show_progress:
            batch_num = i // batch_size + 1
            logger.info(f"Processed batch {batch_num}/{num_batches}")
    
    return results


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model for inference"""
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Try to use torch.jit if available
    try:
        if hasattr(torch, 'jit'):
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
            logger.info("Model optimized with TorchScript")
    except Exception as e:
        logger.warning(f"Could not optimize with TorchScript: {e}")
    
    return model


def mixed_precision_context():
    """Context manager for mixed precision training/inference"""
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    else:
        # No-op context manager for CPU
        from contextlib import nullcontext
        return nullcontext()


class MemoryMonitor:
    """Monitor GPU memory usage"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.peak_memory = 0
    
    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda':
            self.peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            logger.info(f"Peak GPU memory: {self.peak_memory:.2f} GB")
    
    def get_peak_memory_gb(self) -> float:
        """Get peak memory usage in GB"""
        return self.peak_memory


def enable_cudnn_benchmark():
    """Enable cuDNN autotuner for better performance"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled")


def set_deterministic(seed: int = 42):
    """Set deterministic behavior for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Deterministic mode enabled with seed {seed}")


def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    max_memory_fraction: float = 0.8
) -> int:
    """
    Estimate optimal batch size based on available memory
    
    Args:
        model: Model to test
        sample_input: Sample input tensor
        max_memory_fraction: Maximum fraction of memory to use
    
    Returns:
        Optimal batch size
    """
    device = get_device()
    
    if device.type != 'cuda':
        return 32  # Default for CPU
    
    # Get available memory
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    max_memory = total_memory * max_memory_fraction
    
    # Test with batch size 1
    model = model.to(device)
    sample_input = sample_input.unsqueeze(0).to(device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        _ = model(sample_input)
    
    memory_per_sample = torch.cuda.max_memory_allocated(device)
    
    # Calculate optimal batch size
    optimal_batch_size = int(max_memory / memory_per_sample)
    optimal_batch_size = max(1, min(optimal_batch_size, 128))
    
    logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size


def profile_gpu_usage(func):
    """Decorator to profile GPU memory usage"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryMonitor() as monitor:
            result = func(*args, **kwargs)
        logger.info(f"{func.__name__} peak memory: {monitor.get_peak_memory_gb():.2f} GB")
        return result
    return wrapper