"""
Setup script for downloading and initializing models
Run this script once after installation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model_downloader import ModelDownloader
from models.model_manager import ModelManager
from utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_dir='logs', log_level='INFO')
logger = get_logger(__name__)


def create_directories():
    """Create necessary directories"""
    dirs = ['weights', 'configs', 'cache']
    base_dir = Path(__file__).parent
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'huggingface_hub',
        'pyyaml',
        'requests',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All required packages are installed")
    return True


def download_essential_models():
    """Download essential models for basic functionality"""
    downloader = ModelDownloader()
    
    # Essential models for basic operation
    essential_models = ['ocr_router']  # Lightweight, custom trained
    
    # Optional models (require more space)
    optional_models = ['yolov5', 'bert_ner', 'tablenet', 'doctr', 'layoutlm']
    
    logger.info("="*60)
    logger.info("DOWNLOADING ESSENTIAL MODELS")
    logger.info("="*60)
    
    for model_name in essential_models:
        if downloader.verify_model(model_name):
            logger.info(f"✓ {model_name} already exists")
        else:
            logger.info(f"Downloading {model_name}...")
            success = downloader.download_model(model_name)
            if success:
                logger.info(f"✓ {model_name} downloaded successfully")
            else:
                logger.error(f"✗ Failed to download {model_name}")
    
    logger.info("\n" + "="*60)
    logger.info("OPTIONAL MODELS")
    logger.info("="*60)
    
    print("\nThe following models are optional and require significant disk space:")
    for model_name in optional_models:
        info = downloader.get_model_info(model_name)
        print(f"  - {model_name}: {info.get('description', 'N/A')}")
    
    response = input("\nDownload all optional models? (y/N): ").strip().lower()
    
    if response == 'y':
        for model_name in optional_models:
            if downloader.verify_model(model_name):
                logger.info(f"✓ {model_name} already exists")
            else:
                logger.info(f"Downloading {model_name}...")
                success = downloader.download_model(model_name)
                if success:
                    logger.info(f"✓ {model_name} downloaded successfully")
                else:
                    logger.error(f"✗ Failed to download {model_name}")
    else:
        logger.info("Skipping optional models. You can download them later with:")
        logger.info("  python model_downloader.py download --model <model_name>")


def verify_installation():
    """Verify that models are properly installed"""
    manager = ModelManager(auto_download=False)
    status = manager.get_status()
    
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION VERIFICATION")
    logger.info("="*60)
    
    print(f"\nAvailable models: {status['num_available']}")
    print(f"Downloaded models: {status['num_downloaded']}")
    print(f"Device: {status['device']}")
    
    if status['num_downloaded'] > 0:
        print("\n✓ Installation successful!")
        print("\nDownloaded models:")
        for model in status['downloaded_models']:
            print(f"  - {model}")
    else:
        print("\n✗ No models downloaded")
        print("Please check the logs for errors")
    
    return status['num_downloaded'] > 0


def print_usage_instructions():
    """Print instructions for using the models"""
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    
    print("""
To use models in your code:

    from backend.models import get_model_registry
    
    # Get model registry
    registry = get_model_registry()
    
    # Load a model
    yolov5 = registry.get_model('yolov5')
    
    # Or use ModelManager for high-level operations
    from backend.models.model_manager import ModelManager
    
    manager = ModelManager(auto_download=True)
    models = manager.setup_pipeline('basic')

For more information, see models/README.md
    """)


def main():
    """Main setup function"""
    print("="*60)
    print("INVOICE EXTRACTOR - MODEL SETUP")
    print("="*60)
    
    # Step 1: Create directories
    logger.info("\nStep 1: Creating directories...")
    create_directories()
    
    # Step 2: Check dependencies
    logger.info("\nStep 2: Checking dependencies...")
    if not check_dependencies():
        logger.error("Missing required packages. Please install them first.")
        return False
    
    # Step 3: Download models
    logger.info("\nStep 3: Downloading models...")
    download_essential_models()
    
    # Step 4: Verify installation
    logger.info("\nStep 4: Verifying installation...")
    success = verify_installation()
    
    # Step 5: Print usage instructions
    if success:
        print_usage_instructions()
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)