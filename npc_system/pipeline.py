"""
NPC Report Generation System - Pipeline
========================================
Main pipeline that orchestrates all components.
Supports both U-Net and SwinUnet models.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Generator, Union

from config import get_config, SystemConfig
from models import UNet, TumorSegmenter, TumorAnalyzer, TumorFeatures, SWIN_DEPS_AVAILABLE
if SWIN_DEPS_AVAILABLE:
    from models import SwinUnet, SwinUnetSegmenter
from gemini_service import GeminiReportGenerator
from visualization import TumorVisualizer


class NPCReportPipeline:
    """
    Complete pipeline for NPC tumor analysis and report generation.
    
    This pipeline orchestrates:
    1. Loading and preprocessing medical images
    2. Tumor segmentation using U-Net or SwinUnet
    3. Feature extraction and analysis
    4. Visualization generation
    5. AI-powered report generation
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        
        # Initialize components (lazy loading)
        self._segmenter: Optional[Union[TumorSegmenter, 'SwinUnetSegmenter']] = None
        self._analyzer: Optional[TumorAnalyzer] = None
        self._visualizer: Optional[TumorVisualizer] = None
        self._gemini: Optional[GeminiReportGenerator] = None
        
        # Current state
        self.current_case: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self._current_model_type: Optional[str] = None
    
    def initialize(self, model_type: Optional[str] = None) -> bool:
        """
        Initialize all components.
        
        Args:
            model_type: Override model type ('unet' or 'swinunet'). 
                       If None, uses config setting.
        """
        try:
            # Determine which model to use
            model_type = model_type or self.config.model.model_type
            
            # Load appropriate segmenter based on model type
            if model_type == 'swinunet':
                if not SWIN_DEPS_AVAILABLE:
                    raise ImportError("SwinUnet requires einops and timm. Install with: pip install einops timm")
                
                self._segmenter = SwinUnetSegmenter.load_from_checkpoint(
                    checkpoint_path=self.config.model.swinunet_path,
                    device=self.config.model.device,
                    num_classes=self.config.model.num_classes,
                    img_size=self.config.model.swinunet_img_size,
                    embed_dim=self.config.model.swinunet_embed_dim,
                    depths=self.config.model.swinunet_depths,
                    num_heads=self.config.model.swinunet_num_heads,
                    window_size=self.config.model.swinunet_window_size,
                    patch_size=self.config.model.swinunet_patch_size
                )
                print(f"✅ Loaded SwinUnet model from {self.config.model.swinunet_path}")
            else:
                # Default to U-Net
                self._segmenter = TumorSegmenter.load_from_checkpoint(
                    self.config.model.model_path,
                    device=self.config.model.device,
                    in_channels=self.config.model.in_channels,
                    num_classes=self.config.model.num_classes,
                    base_width=self.config.model.base_width,
                    patch_size=self.config.model.patch_size
                )
                print(f"✅ Loaded U-Net model from {self.config.model.model_path}")
            
            self._current_model_type = model_type
            
            # Initialize analyzer
            self._analyzer = TumorAnalyzer(
                voxel_spacing=self.config.data.default_voxel_spacing
            )
            
            # Initialize visualizer
            self._visualizer = TumorVisualizer()
            
            # Initialize Gemini (if API key available)
            if self.config.gemini.api_key:
                self._gemini = GeminiReportGenerator(
                    api_key=self.config.gemini.api_key,
                    model_name=self.config.gemini.model_name
                )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def switch_model(self, model_type: str) -> bool:
        """
        Switch to a different model type.
        
        Args:
            model_type: 'unet' or 'swinunet'
            
        Returns:
            True if switch was successful
        """
        if model_type not in ['unet', 'swinunet']:
            print(f"Invalid model type: {model_type}")
            return False
        
        if model_type == self._current_model_type:
            print(f"Already using {model_type}")
            return True
        
        # Re-initialize with new model type
        self.is_initialized = False
        self._segmenter = None
        return self.initialize(model_type=model_type)
    
    @property
    def current_model_type(self) -> Optional[str]:
        """Get the currently loaded model type"""
        return self._current_model_type
    
    def load_case(self, h5_path: Path) -> Dict[str, Any]:
        """
        Load a case from HDF5 file.
        
        Args:
            h5_path: Path to HDF5 file
            
        Returns:
            Dictionary with image, label, and metadata
        """
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]
            label = f['label'][:] if 'label' in f else None
            
            # Get voxel spacing if available
            # Note: H5 files store spacing as (x, y, z) but we use (z, y, x) internally
            if 'voxel_spacing' in f:
                vs = f['voxel_spacing'][:]
                # Convert from (x, y, z) to (z, y, x) format
                if len(vs) == 3:
                    # File format: [x_spacing, y_spacing, z_spacing]
                    # Our format: (z_spacing, y_spacing, x_spacing)
                    voxel_spacing = (float(vs[2]), float(vs[1]), float(vs[0]))
                else:
                    voxel_spacing = self.config.data.default_voxel_spacing
            else:
                voxel_spacing = self.config.data.default_voxel_spacing
        
        # Handle 2D vs 3D
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
            if label is not None:
                label = label[np.newaxis, ...]
        
        patient_id = h5_path.stem
        
        case_data = {
            'patient_id': patient_id,
            'image': image,
            'label': label,
            'voxel_spacing': voxel_spacing,
            'file_path': str(h5_path),
            'shape': tuple(int(s) for s in image.shape)
        }
        
        return case_data
    
    def process_case(self, h5_path: Path, generate_report: bool = True,
                    stream_report: bool = False) -> Dict[str, Any]:
        """
        Process a complete case through the pipeline.
        
        Args:
            h5_path: Path to HDF5 file
            generate_report: Whether to generate AI report
            stream_report: Whether to use streaming for report
            
        Returns:
            Dictionary with all results
        """
        if not self.is_initialized:
            self.initialize()
        
        results = {
            'patient_id': None,
            'status': 'processing',
            'timestamp': datetime.now().isoformat(),
            'features': None,
            'visualizations': {},
            'report': None,
            'error': None
        }
        
        try:
            # Step 1: Load case
            case_data = self.load_case(Path(h5_path))
            results['patient_id'] = case_data['patient_id']
            
            # Update analyzer with case-specific spacing
            self._analyzer.voxel_spacing = case_data['voxel_spacing']
            
            # Step 2: Segment
            segmentation = self._segmenter.segment_volume(case_data['image'])
            
            # Step 3: Extract features
            features = self._analyzer.analyze(segmentation)
            results['features'] = features.to_dict()
            
            # Step 4: Generate visualizations
            output_dir = self.config.data.reports_dir / case_data['patient_id']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results['visualizations']['multi_slice'] = self._visualizer.create_multi_slice_view(
                case_data['image'], segmentation,
                save_path=output_dir / 'multi_slice.png'
            )
            
            results['visualizations']['three_plane'] = self._visualizer.create_3plane_view(
                case_data['image'], segmentation,
                save_path=output_dir / 'three_plane.png'
            )
            
            results['visualizations']['summary'] = self._visualizer.create_summary_figure(
                case_data['image'], segmentation, results['features'],
                case_data['patient_id'],
                save_path=output_dir / 'summary.png',
                voxel_spacing=case_data['voxel_spacing']
            )
            
            # Step 5: Generate report (if enabled and Gemini available)
            if generate_report and self._gemini:
                if stream_report:
                    # Return generator for streaming
                    results['report_stream'] = self._gemini.generate_report_stream(
                        results['features'],
                        case_data['patient_id']
                    )
                else:
                    results['report'] = self._gemini.generate_report(
                        results['features'],
                        case_data['patient_id']
                    )
            
            # Store current case
            self.current_case = {
                'case_data': case_data,
                'segmentation': segmentation,
                'features': features,
                'results': results
            }
            
            results['status'] = 'completed'
            
            # Save results to JSON
            json_results = {k: v for k, v in results.items() 
                          if k != 'report_stream'}  # Exclude generator
            with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def process_case_stream(self, h5_path: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Process case with streaming updates.
        
        Yields:
            Status updates during processing
        """
        yield {'step': 'loading', 'message': 'Đang tải dữ liệu...', 'progress': 0}
        
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Load case
            case_data = self.load_case(Path(h5_path))
            yield {'step': 'loaded', 'message': f'Đã tải: {case_data["patient_id"]}', 'progress': 20}
            
            # Update analyzer
            self._analyzer.voxel_spacing = case_data['voxel_spacing']
            
            # Segment
            yield {'step': 'segmenting', 'message': 'Đang phân đoạn khối u...', 'progress': 30}
            segmentation = self._segmenter.segment_volume(case_data['image'])
            yield {'step': 'segmented', 'message': 'Hoàn thành phân đoạn', 'progress': 50}
            
            # Extract features
            yield {'step': 'analyzing', 'message': 'Đang phân tích đặc điểm...', 'progress': 60}
            features = self._analyzer.analyze(segmentation)
            yield {'step': 'analyzed', 'message': 'Hoàn thành phân tích', 'progress': 70,
                   'features': features.to_dict()}
            
            # Create output directory
            output_dir = self.config.data.reports_dir / case_data['patient_id']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            yield {'step': 'visualizing', 'message': 'Đang tạo hình ảnh...', 'progress': 80}
            
            visualizations = {}
            visualizations['multi_slice'] = self._visualizer.create_multi_slice_view(
                case_data['image'], segmentation,
                save_path=output_dir / 'multi_slice.png'
            )
            visualizations['three_plane'] = self._visualizer.create_3plane_view(
                case_data['image'], segmentation,
                save_path=output_dir / 'three_plane.png'
            )
            visualizations['summary'] = self._visualizer.create_summary_figure(
                case_data['image'], segmentation, features.to_dict(),
                case_data['patient_id'],
                save_path=output_dir / 'summary.png',
                voxel_spacing=case_data['voxel_spacing']
            )
            
            yield {'step': 'visualized', 'message': 'Hoàn thành hình ảnh', 'progress': 90,
                   'visualizations': visualizations}
            
            # Generate report (streaming)
            if self._gemini:
                yield {'step': 'reporting', 'message': 'Đang tạo báo cáo AI...', 'progress': 95}
                
                full_report = ""
                for chunk in self._gemini.generate_report_stream(features.to_dict(), case_data['patient_id']):
                    full_report += chunk
                    yield {'step': 'report_chunk', 'chunk': chunk}
                
                yield {'step': 'reported', 'message': 'Hoàn thành báo cáo', 'progress': 100,
                       'report': full_report}
            
            # Store current case
            self.current_case = {
                'case_data': case_data,
                'segmentation': segmentation,
                'features': features
            }
            
            yield {'step': 'completed', 'message': 'Hoàn thành!', 'progress': 100,
                   'patient_id': case_data['patient_id']}
            
        except Exception as e:
            yield {'step': 'error', 'message': f'Lỗi: {str(e)}', 'error': str(e)}
    
    def chat(self, message: str, stream: bool = False):
        """
        Chat about the current case.
        
        Args:
            message: User's question
            stream: Whether to stream response
            
        Returns:
            Response text or generator
        """
        if not self._gemini:
            return "Gemini API chưa được cấu hình."
        
        if stream:
            return self._gemini.chat_stream(message)
        else:
            return self._gemini.chat(message)
    
    def get_chat_history(self):
        """Get chat history"""
        if self._gemini:
            return self._gemini.get_chat_history()
        return []
    
    def reset_chat(self):
        """Reset chat session"""
        if self._gemini:
            self._gemini.reset_chat()
    
    def list_available_cases(self) -> Dict[str, list]:
        """List available cases in test and validation sets"""
        cases = {'test': [], 'val': []}
        
        if self.config.data.test_set_dir.exists():
            cases['test'] = sorted([f.name for f in self.config.data.test_set_dir.glob('*.h5')])
        
        if self.config.data.val_set_dir.exists():
            cases['val'] = sorted([f.name for f in self.config.data.val_set_dir.glob('*.h5')])
        
        return cases
    
    def get_case_path(self, filename: str, dataset: str = 'test') -> Optional[Path]:
        """Get full path for a case file"""
        if dataset == 'test':
            path = self.config.data.test_set_dir / filename
        else:
            path = self.config.data.val_set_dir / filename
        
        return path if path.exists() else None
