"""
Model downloader utility
Downloads pre-trained model weights from various sources
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from . import MODEL_REGISTRY, WEIGHTS_DIR, CONFIGS_DIR
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelDownloader:
    """Download and manage model weights"""
    
    def __init__(self, weights_dir: Optional[Path] = None):
        self.weights_dir = weights_dir or WEIGHTS_DIR
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """
        Download file with progress bar
        
        Args:
            url: URL to download from
            destination: Local file path
            chunk_size: Download chunk size in bytes
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {destination.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if destination.exists():
                destination.unlink()
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        Download model weights
        
        Args:
            model_name: Name of the model (must be in MODEL_REGISTRY)
            force: Force re-download even if file exists
        
        Returns:
            True if successful, False otherwise
        """
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            logger.info(f"Available models: {list(MODEL_REGISTRY.keys())}")
            return False
        
        model_info = MODEL_REGISTRY[model_name]
        weight_file = model_info['weight_file']
        url = model_info['url']
        
        if not url:
            logger.warning(f"No download URL for {model_name}. This is a custom model.")
            return False
        
        destination = self.weights_dir / weight_file
        
        if destination.exists() and not force:
            logger.info(f"Model already exists: {weight_file}")
            return True
        
        logger.info(f"Downloading {model_name} from {url}")
        return self.download_file(url, destination)
    
    def download_all_models(self, force: bool = False) -> dict:
        """
        Download all available models
        
        Args:
            force: Force re-download even if files exist
        
        Returns:
            Dictionary with download results
        """
        results = {}
        
        for model_name in MODEL_REGISTRY:
            if MODEL_REGISTRY[model_name]['url']:
                logger.info(f"Processing {model_name}...")
                results[model_name] = self.download_model(model_name, force)
            else:
                logger.info(f"Skipping {model_name} (no URL)")
                results[model_name] = None
        
        return results
    
    def verify_model(self, model_name: str) -> bool:
        """
        Verify model file exists and is valid
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if model exists and is valid
        """
        if model_name not in MODEL_REGISTRY:
            return False
        
        weight_file = MODEL_REGISTRY[model_name]['weight_file']
        model_path = self.weights_dir / weight_file
        
        if not model_path.exists():
            logger.warning(f"Model not found: {weight_file}")
            return False
        
        # Check file is not empty
        if model_path.stat().st_size == 0:
            logger.error(f"Model file is empty: {weight_file}")
            return False
        
        logger.info(f"Model verified: {weight_file}")
        return True
    
    def list_available_models(self) -> list:
        """List all models in registry"""
        return list(MODEL_REGISTRY.keys())
    
    def list_downloaded_models(self) -> list:
        """List all downloaded models"""
        downloaded = []
        for model_name, info in MODEL_REGISTRY.items():
            weight_file = info['weight_file']
            if (self.weights_dir / weight_file).exists():
                downloaded.append(model_name)
        return downloaded
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model"""
        if model_name not in MODEL_REGISTRY:
            return {}
        
        info = MODEL_REGISTRY[model_name].copy()
        weight_path = self.weights_dir / info['weight_file']
        
        info['downloaded'] = weight_path.exists()
        if info['downloaded']:
            info['size_mb'] = weight_path.stat().st_size / (1024 * 1024)
        
        return info
    
    def remove_model(self, model_name: str) -> bool:
        """Remove downloaded model"""
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        weight_file = MODEL_REGISTRY[model_name]['weight_file']
        model_path = self.weights_dir / weight_file
        
        if not model_path.exists():
            logger.warning(f"Model not found: {weight_file}")
            return False
        
        try:
            model_path.unlink()
            logger.info(f"Removed model: {weight_file}")
            return True
        except Exception as e:
            logger.error(f"Error removing model: {e}")
            return False


def download_model_from_huggingface(
    model_id: str,
    filename: str,
    destination: Path,
    token: Optional[str] = None
) -> bool:
    """
    Download model from HuggingFace Hub
    
    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/layoutlm-base-uncased")
        filename: Specific file to download (e.g., "pytorch_model.bin")
        destination: Local destination path
        token: HuggingFace API token (optional)
    
    Returns:
        True if successful
    """
    try:
        from huggingface_hub import hf_hub_download
        
        logger.info(f"Downloading {filename} from {model_id}")
        
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            token=token,
            cache_dir=str(destination.parent)
        )
        
        # Move to destination if needed
        if Path(downloaded_path) != destination:
            import shutil
            shutil.move(downloaded_path, destination)
        
        logger.info(f"Successfully downloaded to {destination}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Error downloading from HuggingFace: {e}")
        return False


def download_model_from_google_drive(file_id: str, destination: Path) -> bool:
    """
    Download model from Google Drive
    
    Args:
        file_id: Google Drive file ID
        destination: Local destination path
    
    Returns:
        True if successful
    """
    try:
        import gdown
        
        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info(f"Downloading from Google Drive: {file_id}")
        
        gdown.download(url, str(destination), quiet=False)
        
        logger.info(f"Successfully downloaded to {destination}")
        return True
        
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        return False
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {e}")
        return False


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        'action',
        choices=['download', 'list', 'verify', 'info', 'remove', 'download-all'],
        help='Action to perform'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download'
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.action == 'list':
        print("\nAvailable models:")
        for model in downloader.list_available_models():
            print(f"  - {model}")
        
        print("\nDownloaded models:")
        for model in downloader.list_downloaded_models():
            print(f"  - {model}")
    
    elif args.action == 'download':
        if not args.model:
            print("Error: --model required for download")
        else:
            downloader.download_model(args.model, force=args.force)
    
    elif args.action == 'download-all':
        results = downloader.download_all_models(force=args.force)
        print("\nDownload results:")
        for model, success in results.items():
            status = "✓" if success else "✗" if success is False else "skipped"
            print(f"  {model}: {status}")
    
    elif args.action == 'verify':
        if not args.model:
            print("Error: --model required for verify")
        else:
            is_valid = downloader.verify_model(args.model)
            print(f"Model {args.model}: {'Valid' if is_valid else 'Invalid'}")
    
    elif args.action == 'info':
        if not args.model:
            print("Error: --model required for info")
        else:
            info = downloader.get_model_info(args.model)
            if info:
                print(f"\nModel: {args.model}")
                print(f"Description: {info['description']}")
                print(f"Weight file: {info['weight_file']}")
                print(f"Config file: {info['config_file']}")
                print(f"Downloaded: {info['downloaded']}")
                if info['downloaded']:
                    print(f"Size: {info['size_mb']:.2f} MB")
                if info['url']:
                    print(f"URL: {info['url']}")
            else:
                print(f"Model not found: {args.model}")
    
    elif args.action == 'remove':
        if not args.model:
            print("Error: --model required for remove")
        else:
            downloader.remove_model(args.model)