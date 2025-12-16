"""
NPC Report Generation System - Configuration
=============================================
Centralized configuration for the entire system.
Supports both U-Net and SwinUnet models.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Literal

# Check CUDA availability
def get_default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# SwinUnet default path (relative to Swin-Unet project)
SWINUNET_DIR = BASE_DIR.parent.parent.parent / "SFADA-GTV-Seg" / "New_Code" / "notebooks" / "outputs" / "experiments" / "swinunet_notebook"

@dataclass
class ModelConfig:
    """Model configuration - supports both U-Net and SwinUnet"""
    # Model selection: 'unet' or 'swinunet'
    model_type: Literal['unet', 'swinunet'] = 'unet'
    
    # U-Net model settings
    model_path: Path = OUTPUTS_DIR / "experiments" / "unet_notebook" / "unet_best_model.pth"
    in_channels: int = 1
    num_classes: int = 2
    base_width: int = 16
    patch_size: tuple = (256, 256)  # U-Net patch size
    
    # SwinUnet model settings
    swinunet_path: Path = OUTPUTS_DIR / "experiments" / "swin_unet_notebook" / "swin_best_model.pth"
    swinunet_img_size: int = 224
    swinunet_embed_dim: int = 96
    swinunet_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    swinunet_num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    swinunet_window_size: int = 7
    swinunet_patch_size: tuple = (224, 224)  # SwinUnet uses 224x224
    
    # Common settings
    device: str = field(default_factory=get_default_device)  # Auto-detect
    
    def get_current_model_path(self) -> Path:
        """Get the path of the currently selected model"""
        if self.model_type == 'swinunet':
            return self.swinunet_path
        return self.model_path
    
    def get_current_patch_size(self) -> tuple:
        """Get the patch size of the currently selected model"""
        if self.model_type == 'swinunet':
            return self.swinunet_patch_size
        return self.patch_size

@dataclass
class GeminiConfig:
    """Gemini API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_name: str = "gemini-2.5-flash"
    max_tokens: int = 8192
    temperature: float = 0.3
    
    def __post_init__(self):
        # Try to load from environment variable first
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")

@dataclass
class DataConfig:
    """Data paths configuration"""
    h5_dataset_dir: Path = OUTPUTS_DIR / "h5_dataset"
    test_set_dir: Path = OUTPUTS_DIR / "h5_dataset" / "test_set"
    val_set_dir: Path = OUTPUTS_DIR / "h5_dataset" / "val_set"
    reports_dir: Path = OUTPUTS_DIR / "reports"
    
    # Default voxel spacing (z, y, x) in mm
    # Note: Most NPC MRI scans have ~0.43mm in-plane and ~3-9mm slice thickness
    default_voxel_spacing: tuple = (9.0, 0.43, 0.43)

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    api_port: int = 8000
    gradio_port: int = 7860
    debug: bool = True
    
    # CORS settings
    cors_origins: list = field(default_factory=lambda: ["*"])

@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    data: DataConfig = field(default_factory=DataConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # System info
    system_name: str = "NPC Tumor Report Generation System"
    version: str = "1.0.0"

# Global config instance
config = SystemConfig()

def update_gemini_api_key(api_key: str):
    """Update Gemini API key"""
    config.gemini.api_key = api_key
    os.environ["GEMINI_API_KEY"] = api_key

def update_model_type(model_type: str):
    """
    Update the model type to use for segmentation.
    
    Args:
        model_type: Either 'unet' or 'swinunet'
    """
    if model_type not in ['unet', 'swinunet']:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'unet' or 'swinunet'")
    config.model.model_type = model_type

def update_swinunet_path(path: Path):
    """Update the path to SwinUnet checkpoint"""
    config.model.swinunet_path = Path(path)

def update_unet_path(path: Path):
    """Update the path to U-Net checkpoint"""
    config.model.model_path = Path(path)

def get_available_models() -> List[str]:
    """Get list of available model types"""
    return ['unet', 'swinunet']

def get_config() -> SystemConfig:
    """Get the global configuration"""
    return config
