"""
NPC Report Generation System - Core Models
===========================================
Contains U-Net model architecture and tumor analysis classes.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
from pathlib import Path


# ============================================================
# U-Net Architecture (matching trained checkpoint)
# ============================================================

def conv_block(in_ch, out_ch, dropout_p=0.0):
    """Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU (matching training)"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(),
        nn.Dropout(dropout_p),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU()
    )


class UNet(nn.Module):
    """U-Net matching the trained checkpoint architecture"""
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_width: int = 16):
        super().__init__()
        
        # Dropout rates matching training
        dropouts = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        # Encoder
        self.encoder1 = conv_block(in_channels, base_width, dropouts[0])        # 1 -> 16
        self.encoder2 = conv_block(base_width, base_width * 2, dropouts[1])     # 16 -> 32
        self.encoder3 = conv_block(base_width * 2, base_width * 4, dropouts[2]) # 32 -> 64
        self.encoder4 = conv_block(base_width * 4, base_width * 8, dropouts[3]) # 64 -> 128
        
        # Bottleneck
        self.bottleneck = conv_block(base_width * 8, base_width * 16, dropouts[4])  # 128 -> 256
        
        # Upsampling (transposed conv)
        self.up4 = nn.ConvTranspose2d(base_width * 16, base_width * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_width * 8, base_width * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_width * 4, base_width * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_width * 2, base_width, kernel_size=2, stride=2)
        
        # Decoder (input channels = upsampled + skip connection, no dropout)
        self.decoder4 = conv_block(base_width * 16, base_width * 8, dropout_p=0.0)  # 256 -> 128
        self.decoder3 = conv_block(base_width * 8, base_width * 4, dropout_p=0.0)   # 128 -> 64
        self.decoder2 = conv_block(base_width * 4, base_width * 2, dropout_p=0.0)   # 64 -> 32
        self.decoder1 = conv_block(base_width * 2, base_width, dropout_p=0.0)       # 32 -> 16
        
        # Classifier
        self.classifier = nn.Conv2d(base_width, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))
        
        return self.classifier(d1)
    
    def _match_size(self, x, target):
        if x.size() != target.size():
            x = nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x


# ============================================================
# Tumor Features
# ============================================================

@dataclass
class TumorFeatures:
    """Data class for storing tumor features"""
    # Volume metrics
    volume_mm3: float = 0.0
    volume_ml: float = 0.0
    voxel_count: int = 0
    
    # Size metrics
    max_diameter_mm: float = 0.0
    dimensions_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # z, y, x
    
    # Shape metrics
    sphericity: float = 0.0
    elongation: float = 0.0
    surface_area_mm2: float = 0.0
    
    # Location
    centroid_voxel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box: Dict = field(default_factory=dict)
    
    # Status
    tumor_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with native Python types"""
        return {
            'volume_mm3': float(self.volume_mm3),
            'volume_ml': float(self.volume_ml),
            'voxel_count': int(self.voxel_count),
            'max_diameter_mm': float(self.max_diameter_mm),
            'dimensions_mm': tuple(float(d) for d in self.dimensions_mm),
            'sphericity': float(self.sphericity),
            'elongation': float(self.elongation),
            'surface_area_mm2': float(self.surface_area_mm2),
            'centroid_voxel': tuple(float(c) for c in self.centroid_voxel),
            'bounding_box': {k: int(v) if isinstance(v, (int, np.integer)) else float(v) 
                           for k, v in self.bounding_box.items()},
            'tumor_detected': bool(self.tumor_detected)
        }


# ============================================================
# Tumor Analyzer
# ============================================================

class TumorAnalyzer:
    """Analyzes tumor segmentation and extracts features"""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (3.0, 0.5, 0.5)):
        self.voxel_spacing = voxel_spacing  # (z, y, x) in mm
    
    def analyze(self, segmentation: np.ndarray, voxel_spacing: Tuple[float, float, float] = None) -> TumorFeatures:
        """
        Analyze tumor segmentation and extract features.
        
        Args:
            segmentation: Binary mask (D, H, W) where 1 = tumor
            voxel_spacing: Optional (z, y, x) spacing in mm. If None, uses default.
            
        Returns:
            TumorFeatures object
        """
        # Use provided voxel_spacing or default
        if voxel_spacing is not None:
            self.voxel_spacing = tuple(voxel_spacing)
            
        features = TumorFeatures()
        
        # Check if tumor exists
        tumor_mask = segmentation > 0
        features.voxel_count = int(np.sum(tumor_mask))
        
        if features.voxel_count == 0:
            features.tumor_detected = False
            return features
        
        features.tumor_detected = True
        
        # Volume calculation
        voxel_volume = np.prod(self.voxel_spacing)
        features.volume_mm3 = features.voxel_count * voxel_volume
        features.volume_ml = features.volume_mm3 / 1000.0
        
        # Get tumor region
        coords = np.where(tumor_mask)
        
        # Bounding box
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        features.bounding_box = {
            'z_min': int(z_min), 'z_max': int(z_max),
            'y_min': int(y_min), 'y_max': int(y_max),
            'x_min': int(x_min), 'x_max': int(x_max)
        }
        
        # Dimensions in mm
        z_dim = (z_max - z_min + 1) * self.voxel_spacing[0]
        y_dim = (y_max - y_min + 1) * self.voxel_spacing[1]
        x_dim = (x_max - x_min + 1) * self.voxel_spacing[2]
        features.dimensions_mm = (float(z_dim), float(y_dim), float(x_dim))
        
        # Max diameter
        features.max_diameter_mm = float(max(features.dimensions_mm))
        
        # Centroid
        centroid_z = float(np.mean(coords[0]))
        centroid_y = float(np.mean(coords[1]))
        centroid_x = float(np.mean(coords[2]))
        features.centroid_voxel = (centroid_z, centroid_y, centroid_x)
        
        # Shape metrics
        features.sphericity = self._calculate_sphericity(tumor_mask)
        features.elongation = self._calculate_elongation(features.dimensions_mm)
        features.surface_area_mm2 = self._calculate_surface_area(tumor_mask)
        
        return features
    
    def _calculate_sphericity(self, mask: np.ndarray) -> float:
        """Calculate sphericity (1 = perfect sphere)"""
        try:
            volume = np.sum(mask) * np.prod(self.voxel_spacing)
            surface_area = self._calculate_surface_area(mask)
            if surface_area > 0:
                sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
                return float(min(1.0, sphericity))
        except:
            pass
        return 0.0
    
    def _calculate_elongation(self, dimensions: Tuple[float, float, float]) -> float:
        """Calculate elongation (ratio of max to min dimension)"""
        dims = sorted(dimensions, reverse=True)
        if dims[-1] > 0:
            return float(dims[0] / dims[-1])
        return 1.0
    
    def _calculate_surface_area(self, mask: np.ndarray) -> float:
        """Estimate surface area using gradient method"""
        try:
            # Use erosion to find surface voxels
            eroded = ndimage.binary_erosion(mask)
            surface = mask & ~eroded
            
            # Count surface voxels and estimate area
            surface_count = np.sum(surface)
            avg_voxel_face = (self.voxel_spacing[0] * self.voxel_spacing[1] + 
                             self.voxel_spacing[1] * self.voxel_spacing[2] + 
                             self.voxel_spacing[0] * self.voxel_spacing[2]) / 3
            
            return float(surface_count * avg_voxel_face)
        except:
            return 0.0


# ============================================================
# Tumor Segmenter
# ============================================================

class TumorSegmenter:
    """Handles tumor segmentation using U-Net model"""
    
    def __init__(self, model: UNet, device: str = "cpu", patch_size: Tuple[int, int] = (256, 256)):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu", 
                            in_channels: int = 1, num_classes: int = 2, 
                            base_width: int = 16,
                            patch_size: Tuple[int, int] = (256, 256)) -> 'TumorSegmenter':
        """Load segmenter from checkpoint file"""
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_width=base_width)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # Format: {'epoch': ..., 'state_dict': {...}, ...}
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                # Format: {'model_state_dict': {...}, ...}
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the dict is the state_dict itself
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, device, patch_size)
    
    @torch.no_grad()
    def segment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment a 3D volume slice by slice.
        
        Args:
            volume: 3D numpy array (D, H, W) - assumed to be already z-score normalized
            
        Returns:
            Binary segmentation mask (D, H, W)
        """
        from scipy.ndimage import zoom
        
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        original_h, original_w = volume.shape[1], volume.shape[2]
        target_h, target_w = self.patch_size
        
        # Check if resize is needed
        need_resize = (original_h != target_h) or (original_w != target_w)
        
        for i in range(volume.shape[0]):
            slice_2d = volume[i]
            
            # Resize to patch_size if needed
            if need_resize:
                slice_resized = zoom(slice_2d, (target_h / original_h, target_w / original_w), order=1)
            else:
                slice_resized = slice_2d
            
            # To tensor
            tensor = torch.from_numpy(slice_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            # Predict
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize prediction back to original size
            if need_resize:
                pred = zoom(pred, (original_h / target_h, original_w / target_w), order=0)
            
            segmentation[i] = pred.astype(np.uint8)
        
        return segmentation
    
    def segment_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Segment a single 2D slice"""
        from scipy.ndimage import zoom
        
        original_h, original_w = slice_2d.shape
        target_h, target_w = self.patch_size
        need_resize = (original_h != target_h) or (original_w != target_w)
        
        with torch.no_grad():
            # Resize to patch_size if needed
            if need_resize:
                slice_resized = zoom(slice_2d, (target_h / original_h, target_w / original_w), order=1)
            else:
                slice_resized = slice_2d
            
            tensor = torch.from_numpy(slice_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize prediction back to original size
            if need_resize:
                pred = zoom(pred, (original_h / target_h, original_w / target_w), order=0)
            
            return pred.astype(np.uint8)
