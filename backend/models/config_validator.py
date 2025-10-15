"""
Configuration validator for model configs
Ensures config files are valid and complete
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from . import CONFIGS_DIR
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """Validate model configuration files"""
    
    def __init__(self):
        self.configs_dir = CONFIGS_DIR
        self.errors = []
        self.warnings = []
    
    def validate_file(self, config_file: Path) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single config file
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Check file exists
        if not config_file.exists():
            self.errors.append(f"File not found: {config_file}")
            return False, self.errors, self.warnings
        
        # Load config
        try:
            config = self._load_config(config_file)
        except Exception as e:
            self.errors.append(f"Failed to parse config: {e}")
            return False, self.errors, self.warnings
        
        # Validate based on file name
        model_name = config_file.stem.replace('_config', '')
        
        if model_name == 'tablenet':
            self._validate_tablenet_config(config)
        elif model_name == 'yolo':
            self._validate_yolo_config(config)
        elif model_name == 'bert':
            self._validate_bert_config(config)
        elif model_name == 'ocr_router':
            self._validate_ocr_router_config(config)
        elif model_name == 'doctr':
            self._validate_doctr_config(config)
        elif model_name == 'layoutlm':
            self._validate_layoutlm_config(config)
        else:
            self.warnings.append(f"Unknown model type: {model_name}")
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _load_config(self, config_file: Path) -> Dict:
        """Load config file"""
        if config_file.suffix in ['.yaml', '.yml']:
            with open(config_file) as f:
                return yaml.safe_load(f)
        elif config_file.suffix == '.json':
            with open(config_file) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {config_file.suffix}")
    
    def _validate_tablenet_config(self, config: Dict):
        """Validate TableNet configuration"""
        required_keys = ['model', 'input', 'output']
        self._check_required_keys(config, required_keys, 'root')
        
        if 'model' in config:
            model_keys = ['name', 'architecture', 'backbone']
            self._check_required_keys(config['model'], model_keys, 'model')
        
        if 'input' in config:
            input_keys = ['image_size', 'channels']
            self._check_required_keys(config['input'], input_keys, 'input')
            
            # Validate image size
            if 'image_size' in config['input']:
                size = config['input']['image_size']
                if not isinstance(size, list) or len(size) != 2:
                    self.errors.append("input.image_size must be [height, width]")
    
    def _validate_yolo_config(self, config: Dict):
        """Validate YOLO configuration"""
        required_keys = ['model', 'input', 'classes', 'detection']
        self._check_required_keys(config, required_keys, 'root')
        
        if 'classes' in config:
            if 'names' not in config['classes']:
                self.errors.append("classes.names is required")
            if 'num_classes' not in config['classes']:
                self.errors.append("classes.num_classes is required")
            
            # Validate num_classes matches names
            if 'names' in config['classes'] and 'num_classes' in config['classes']:
                expected = len(config['classes']['names'])
                actual = config['classes']['num_classes']
                if expected != actual:
                    self.errors.append(
                        f"classes.num_classes ({actual}) doesn't match "
                        f"number of class names ({expected})"
                    )
    
    def _validate_bert_config(self, config: Dict):
        """Validate BERT configuration"""
        required_keys = ['model', 'labels', 'tokenizer']
        self._check_required_keys(config, required_keys, 'root')
        
        if 'model' in config:
            model_keys = ['name', 'architecture', 'hidden_size', 'num_hidden_layers']
            self._check_required_keys(config['model'], model_keys, 'model')
        
        if 'labels' in config:
            label_keys = ['id2label', 'label2id', 'num_labels']
            self._check_required_keys(config['labels'], label_keys, 'labels')
            
            # Validate label mappings
            if 'id2label' in config['labels'] and 'label2id' in config['labels']:
                id2label = config['labels']['id2label']
                label2id = config['labels']['label2id']
                
                # Check if mappings are inverse of each other
                for label_id, label_name in id2label.items():
                    if label_name not in label2id:
                        self.errors.append(f"Label '{label_name}' in id2label but not in label2id")
                    elif label2id[label_name] != int(label_id):
                        self.errors.append(f"Inconsistent mapping for label '{label_name}'")
    
    def _validate_ocr_router_config(self, config: Dict):
        """Validate OCR Router configuration"""
        required_keys = ['model', 'features', 'engines', 'routing']
        self._check_required_keys(config, required_keys, 'root')
        
        if 'features' in config:
            if not isinstance(config['features'], list):
                self.errors.append("features must be a list")
            elif len(config['features']) == 0:
                self.warnings.append("No features defined")
        
        if 'engines' in config:
            if not isinstance(config['engines'], dict):
                self.errors.append("engines must be a dictionary")
            elif len(config['engines']) == 0:
                self.errors.append("At least one OCR engine must be defined")
    
    def _validate_doctr_config(self, config: Dict):
        """Validate DocTR configuration"""
        required_keys = ['detection', 'recognition']
        self._check_required_keys(config, required_keys, 'root')
        
        if 'detection' in config:
            if 'model' not in config['detection']:
                self.errors.append("detection.model is required")
        
        if 'recognition' in config:
            if 'model' not in config['recognition']:
                self.errors.append("recognition.model is required")
    
    def _validate_layoutlm_config(self, config: Dict):
        """Validate LayoutLM configuration"""
        required_keys = ['model', 'labels', 'preprocessing']
        self._check_required_keys(config, required_keys, 'root')
        
        if 'model' in config:
            if 'max_2d_position_embeddings' not in config['model']:
                self.warnings.append(
                    "model.max_2d_position_embeddings not set, using default"
                )
        
        if 'preprocessing' in config:
            prep_keys = ['max_seq_length', 'include_bounding_boxes']
            self._check_required_keys(
                config['preprocessing'],
                prep_keys,
                'preprocessing'
            )
    
    def _check_required_keys(self, config: Dict, required: List[str], section: str):
        """Check if all required keys are present"""
        for key in required:
            if key not in config:
                self.errors.append(f"Missing required key: {section}.{key}")
    
    def validate_all_configs(self) -> Dict[str, Tuple[bool, List[str], List[str]]]:
        """
        Validate all configuration files
        
        Returns:
            Dictionary mapping config file names to (is_valid, errors, warnings)
        """
        results = {}
        
        for config_file in self.configs_dir.glob('*_config.*'):
            logger.info(f"Validating {config_file.name}...")
            is_valid, errors, warnings = self.validate_file(config_file)
            results[config_file.name] = (is_valid, errors, warnings)
        
        return results
    
    def print_validation_report(self, results: Dict):
        """Print a formatted validation report"""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION REPORT")
        print("="*60)
        
        total_files = len(results)
        valid_files = sum(1 for v, _, _ in results.values() if v)
        
        print(f"\nTotal configs: {total_files}")
        print(f"Valid configs: {valid_files}")
        print(f"Invalid configs: {total_files - valid_files}")
        
        for config_name, (is_valid, errors, warnings) in results.items():
            print(f"\n{config_name}:")
            
            if is_valid:
                print("  ✓ Valid")
            else:
                print("  ✗ Invalid")
            
            if errors:
                print(f"  Errors ({len(errors)}):")
                for error in errors:
                    print(f"    - {error}")
            
            if warnings:
                print(f"  Warnings ({len(warnings)}):")
                for warning in warnings:
                    print(f"    - {warning}")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate model configurations")
    parser.add_argument(
        '--file',
        type=str,
        help='Specific config file to validate'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all config files'
    )
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    if args.file:
        # Validate single file
        config_path = Path(args.file)
        is_valid, errors, warnings = validator.validate_file(config_path)
        
        print(f"\n{config_path.name}:")
        if is_valid:
            print("  ✓ Valid")
        else:
            print("  ✗ Invalid")
        
        if errors:
            print(f"\n  Errors:")
            for error in errors:
                print(f"    - {error}")
        
        if warnings:
            print(f"\n  Warnings:")
            for warning in warnings:
                print(f"    - {warning}")
    
    elif args.all:
        # Validate all configs
        results = validator.validate_all_configs()
        validator.print_validation_report(results)
    
    else:
        parser.print_help()