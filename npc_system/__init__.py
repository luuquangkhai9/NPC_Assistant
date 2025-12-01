"""
NPC Report Generation System
=============================

A complete system for NPC (Nasopharyngeal Carcinoma) tumor 
segmentation and AI-powered report generation.

Components:
- config: System configuration
- models: U-Net model and tumor analysis
- gemini_service: Gemini API integration
- visualization: Tumor visualization
- pipeline: Main processing pipeline
- api: FastAPI backend
- gradio_ui: Gradio web interface
"""

from .config import get_config, update_gemini_api_key, SystemConfig
from .models import UNet, TumorSegmenter, TumorAnalyzer, TumorFeatures
from .gemini_service import GeminiReportGenerator
from .visualization import TumorVisualizer
from .pipeline import NPCReportPipeline

__version__ = "1.0.0"
__all__ = [
    "get_config",
    "update_gemini_api_key", 
    "SystemConfig",
    "UNet",
    "TumorSegmenter",
    "TumorAnalyzer",
    "TumorFeatures",
    "GeminiReportGenerator",
    "TumorVisualizer",
    "NPCReportPipeline"
]
