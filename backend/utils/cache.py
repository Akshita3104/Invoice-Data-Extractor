"""
Caching utilities for intermediate results
Supports in-memory and disk-based caching with LRU eviction
"""

import os
import pickle
import hashlib
import functools
from pathlib import Path
from typing import Any, Optional, Callable
from collections import OrderedDict
from datetime import datetime, timedelta
from .logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        self.misses += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                removed_key = next(iter(self.cache))
                del self.cache[removed_key]
                logger.debug(f"Cache evicted: {removed_key}")
        
        self.cache[key] = value
        logger.debug(f"Cache stored: {key}")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def stats(self) -> dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%"
        }


class DiskCache:
    """Disk-based cache with expiration support"""
    
    def __init__(self, cache_dir: str = '.cache', max_age_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            logger.debug(f"Disk cache miss: {key}")
            return None
        
        # Check if cache is expired
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - modified_time > self.max_age:
            logger.debug(f"Disk cache expired: {key}")
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            logger.debug(f"Disk cache hit: {key}")
            return value
        except Exception as e:
            logger.error(f"Error reading disk cache: {e}")
            return None
    
    def put(self, key: str, value: Any):
        """Put value in disk cache"""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Disk cache stored: {key}")
        except Exception as e:
            logger.error(f"Error writing disk cache: {e}")
    
    def clear(self):
        """Clear all disk cache entries"""
        for cache_file in self.cache_dir.glob('*.cache'):
            cache_file.unlink()
        logger.info("Disk cache cleared")
    
    def cleanup(self):
        """Remove expired cache entries"""
        removed_count = 0
        for cache_file in self.cache_dir.glob('*.cache'):
            modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - modified_time > self.max_age:
                cache_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")
    
    def size_mb(self) -> float:
        """Get total cache size in MB"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.cache'))
        return total_size / (1024 * 1024)


class Cache:
    """Hybrid cache with memory and disk tiers"""
    
    def __init__(
        self,
        memory_size: int = 100,
        disk_enabled: bool = True,
        cache_dir: str = '.cache',
        max_age_days: int = 7
    ):
        self.memory_cache = LRUCache(max_size=memory_size)
        self.disk_cache = DiskCache(cache_dir, max_age_days) if disk_enabled else None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.put(key, value)
                return value
        
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache (both memory and disk)"""
        self.memory_cache.put(key, value)
        if self.disk_cache:
            self.disk_cache.put(key, value)
    
    def clear(self):
        """Clear all cache entries"""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()
    
    def stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            'memory': self.memory_cache.stats()
        }
        if self.disk_cache:
            stats['disk'] = {
                'size_mb': self.disk_cache.size_mb()
            }
        return stats


# Global cache instance
_global_cache = None


def get_cache() -> Cache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = Cache()
    return _global_cache


def cache_result(
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None,
    cache_instance: Optional[Cache] = None
):
    """
    Decorator to cache function results
    
    Args:
        key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds (not implemented yet)
        cache_instance: Cache instance to use (default: global cache)
    
    Example:
        @cache_result(key_func=lambda pdf_path: f"ocr_{pdf_path}")
        def extract_text(pdf_path):
            # expensive operation
            return text
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = cache_instance or get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name and arguments
                args_str = str(args) + str(kwargs)
                cache_key = f"{func.__name__}_{hashlib.md5(args_str.encode()).hexdigest()}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Using cached result for {func.__name__}")
                return result
            
            # Compute and cache result
            logger.debug(f"Computing result for {func.__name__}")
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        return wrapper
    return decorator


def hash_file(filepath: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments"""
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            key_parts.append(str(sorted(arg)))
        elif isinstance(arg, dict):
            key_parts.append(str(sorted(arg.items())))
        else:
            key_parts.append(str(type(arg)))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    key_string = '|'.join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()