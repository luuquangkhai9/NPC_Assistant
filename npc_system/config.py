"""
NPC Report Generation System - Configuration
=============================================
Centralized configuration for the entire system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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

@dataclass
class ModelConfig:
    """U-Net model configuration"""
    model_path: Path = OUTPUTS_DIR / "experiments" / "unet_notebook" / "unet_best_model.pth"
    in_channels: int = 1
    num_classes: int = 2
    base_width: int = 16
    device: str = field(default_factory=get_default_device)  # Auto-detect

@dataclass
class GeminiConfig:
    """Gemini API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_name: str = "gemini-2.0-flash"
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

def get_config() -> SystemConfig:
    """Get the global configuration"""
    return config
