"""
Configuration management for the Invoice Extractor
Supports YAML, JSON, and environment variables
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from .logger import get_logger

logger = get_logger(__name__)


class Config:
    """Configuration manager with hierarchical settings"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}
        self._defaults = self._get_defaults()
    
    def _get_defaults(self) -> Dict:
        """Get default configuration values"""
        return {
            # Application settings
            'app': {
                'name': 'Invoice Data Extractor',
                'version': '2.0.0',
                'debug': False,
            },
            
            # API settings
            'api': {
                'gemini': {
                    'model': 'gemini-1.5-flash',
                    'temperature': 0.1,
                    'max_tokens': 4096,
                    'timeout': 30,
                },
                'google_vision': {
                    'enabled': False,
                    'timeout': 30,
                },
                'aws_textract': {
                    'enabled': False,
                    'timeout': 30,
                }
            },
            
            # OCR settings
            'ocr': {
                'default_engine': 'tesseract',
                'fallback_enabled': True,
                'ensemble_enabled': False,
                'tesseract': {
                    'lang': 'eng',
                    'config': '--oem 3 --psm 6',
                    'dpi': 300,
                },
                'doctr': {
                    'model': 'db_resnet50',
                    'pretrained': True,
                },
                'confidence_threshold': 0.7,
            },
            
            # Preprocessing settings
            'preprocessing': {
                'enabled': True,
                'auto_rotate': True,
                'denoise': True,
                'enhance_contrast': True,
                'normalize_resolution': True,
                'target_dpi': 300,
                'min_quality_score': 0.5,
            },
            
            # Layout analysis settings
            'layout': {
                'enabled': True,
                'table_detection': True,
                'zone_segmentation': True,
                'reading_order': True,
            },
            
            # Graph settings
            'graph': {
                'enabled': False,
                'use_gnn': False,
                'spatial_relations': True,
                'semantic_relations': True,
            },
            
            # Validation settings
            'validation': {
                'enabled': True,
                'arithmetic_check': True,
                'format_check': True,
                'consistency_check': True,
                'plausibility_check': False,
                'min_confidence': 0.8,
            },
            
            # Export settings
            'export': {
                'formats': ['excel', 'json'],
                'excel': {
                    'engine': 'openpyxl',
                    'include_confidence': False,
                },
                'json': {
                    'indent': 4,
                    'ensure_ascii': False,
                }
            },
            
            # Performance settings
            'performance': {
                'use_gpu': True,
                'batch_size': 4,
                'num_workers': 4,
                'cache_enabled': True,
                'cache_size_mb': 500,
            },
            
            # Logging settings
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'max_file_size_mb': 10,
                'backup_count': 5,
                'console_output': True,
            },
            
            # Model paths
            'models': {
                'weights_dir': 'backend/models/weights',
                'configs_dir': 'backend/models/configs',
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('api.gemini.model')
        """
        keys = key.split('.')
        value = self._config
        
        # Try to get from config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Fall back to defaults
                value = self._defaults
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        Example: config.set('api.gemini.model', 'gemini-pro')
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Config updated: {key} = {value}")
    
    def update(self, config_dict: Dict):
        """Update configuration with a dictionary"""
        self._deep_update(self._config, config_dict)
        logger.info("Configuration updated")
    
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        return self._config.copy()
    
    def save(self, filepath: str):
        """Save configuration to file (YAML or JSON)"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
            with open(filepath, 'w') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
        elif filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=4)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    def __repr__(self):
        return f"Config({self._config})"


def load_config(filepath: Optional[str] = None, env_prefix: str = 'INVOICE_') -> Config:
    """
    Load configuration from file and environment variables
    
    Args:
        filepath: Path to config file (YAML or JSON)
        env_prefix: Prefix for environment variables
    
    Returns:
        Config instance
    """
    config = Config()
    
    # Load from file if provided
    if filepath:
        filepath = Path(filepath)
        if filepath.exists():
            try:
                if filepath.suffix in ['.yaml', '.yml']:
                    with open(filepath) as f:
                        file_config = yaml.safe_load(f)
                elif filepath.suffix == '.json':
                    with open(filepath) as f:
                        file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {filepath.suffix}")
                
                if file_config:
                    config.update(file_config)
                    logger.info(f"Configuration loaded from {filepath}")
            except Exception as e:
                logger.error(f"Error loading config from {filepath}: {e}")
        else:
            logger.warning(f"Config file not found: {filepath}")
    
    # Override with environment variables
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            config_key = key[len(env_prefix):].lower().replace('_', '.')
            
            # Try to parse as JSON for complex types
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value
            
            config.set(config_key, parsed_value)
            logger.debug(f"Config from env: {config_key} = {parsed_value}")
    
    return config


def get_default_config_path() -> Path:
    """Get default configuration file path"""
    return Path('config/default_config.yaml')


def create_default_config():
    """Create default configuration file"""
    config = Config()
    config_path = get_default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_path))
    logger.info(f"Default configuration created at {config_path}")