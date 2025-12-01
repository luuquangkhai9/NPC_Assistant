"""
NPC Report Generation System - Visualization
=============================================
Handles tumor visualization generation.
"""

import io
import base64
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure matplotlib to use fonts that support Vietnamese characters
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class TumorVisualizer:
    """Generates visualizations for tumor segmentation results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 4)):
        self.figsize = figsize
    
    def create_multi_slice_view(self, volume: np.ndarray, segmentation: np.ndarray,
                                 num_slices: int = 5, save_path: Optional[Path] = None) -> str:
        """
        Create a multi-slice visualization with overlay.
        
        Args:
            volume: 3D image volume (D, H, W)
            segmentation: 3D segmentation mask (D, H, W)
            num_slices: Number of slices to display
            save_path: Optional path to save the image
            
        Returns:
            Base64 encoded image string
        """
        # Find slices with tumor
        tumor_slices = np.where(np.any(segmentation > 0, axis=(1, 2)))[0]
        
        if len(tumor_slices) == 0:
            # No tumor found, show middle slices
            middle = volume.shape[0] // 2
            indices = [max(0, middle - 2 + i) for i in range(num_slices)]
        else:
            # Select slices evenly from tumor region
            if len(tumor_slices) >= num_slices:
                step = len(tumor_slices) // num_slices
                indices = tumor_slices[::step][:num_slices]
            else:
                indices = tumor_slices
        
        fig, axes = plt.subplots(1, len(indices), figsize=self.figsize)
        if len(indices) == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, indices):
            ax.imshow(volume[idx], cmap='gray')
            
            # Create overlay and contour
            mask = segmentation[idx]
            if np.any(mask > 0):
                # Fill overlay with semi-transparent red
                overlay = np.ma.masked_where(mask == 0, mask)
                ax.imshow(overlay, cmap='Reds', alpha=0.4)
                # Draw contour for clear boundary
                ax.contour(mask, levels=[0.5], colors='red', linewidths=2)
            
            ax.set_title(f'Slice {idx}')
            ax.axis('off')
        
        plt.suptitle('Multi-Slice Tumor Visualization', fontsize=14)
        plt.tight_layout()
        
        # Save and return base64
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def create_3plane_view(self, volume: np.ndarray, segmentation: np.ndarray,
                           save_path: Optional[Path] = None) -> str:
        """
        Create axial, sagittal, and coronal views.
        
        Args:
            volume: 3D image volume (D, H, W)
            segmentation: 3D segmentation mask (D, H, W)
            save_path: Optional path to save the image
            
        Returns:
            Base64 encoded image string
        """
        # Find center of tumor or use volume center
        tumor_coords = np.where(segmentation > 0)
        if len(tumor_coords[0]) > 0:
            center_z = int(np.mean(tumor_coords[0]))
            center_y = int(np.mean(tumor_coords[1]))
            center_x = int(np.mean(tumor_coords[2]))
        else:
            center_z = volume.shape[0] // 2
            center_y = volume.shape[1] // 2
            center_x = volume.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial view (z-slice)
        axes[0].imshow(volume[center_z], cmap='gray')
        mask_axial = segmentation[center_z]
        if np.any(mask_axial > 0):
            overlay = np.ma.masked_where(mask_axial == 0, mask_axial)
            axes[0].imshow(overlay, cmap='Reds', alpha=0.4)
            axes[0].contour(mask_axial, levels=[0.5], colors='red', linewidths=2)
        axes[0].set_title(f'Axial (Z={center_z})')
        axes[0].axis('off')
        
        # Sagittal view (x-slice)
        sagittal = volume[:, :, center_x]
        axes[1].imshow(sagittal, cmap='gray', aspect='auto')
        mask_sagittal = segmentation[:, :, center_x]
        if np.any(mask_sagittal > 0):
            overlay = np.ma.masked_where(mask_sagittal == 0, mask_sagittal)
            axes[1].imshow(overlay, cmap='Reds', alpha=0.4, aspect='auto')
            axes[1].contour(mask_sagittal, levels=[0.5], colors='red', linewidths=2)
        axes[1].set_title(f'Sagittal (X={center_x})')
        axes[1].axis('off')
        
        # Coronal view (y-slice)
        coronal = volume[:, center_y, :]
        axes[2].imshow(coronal, cmap='gray', aspect='auto')
        mask_coronal = segmentation[:, center_y, :]
        if np.any(mask_coronal > 0):
            overlay = np.ma.masked_where(mask_coronal == 0, mask_coronal)
            axes[2].imshow(overlay, cmap='Reds', alpha=0.4, aspect='auto')
            axes[2].contour(mask_coronal, levels=[0.5], colors='red', linewidths=2)
        axes[2].set_title(f'Coronal (Y={center_y})')
        axes[2].axis('off')
        
        plt.suptitle('3-Plane Tumor Visualization', fontsize=14)
        plt.tight_layout()
        
        # Save and return base64
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def create_summary_figure(self, volume: np.ndarray, segmentation: np.ndarray,
                              features: dict, patient_id: str,
                              save_path: Optional[Path] = None,
                              voxel_spacing: Tuple[float, float, float] = None) -> str:
        """
        Create a comprehensive summary figure with stats.
        
        Args:
            voxel_spacing: (z, y, x) spacing in mm for correct aspect ratio
        
        Returns:
            Base64 encoded image string
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Calculate z_to_xy ratio for correct aspect ratio
        if voxel_spacing is not None:
            z_spacing, y_spacing, x_spacing = voxel_spacing
            z_to_xy_ratio = z_spacing / ((x_spacing + y_spacing) / 2)
        else:
            # Estimate from volume shape
            d, h, w = volume.shape
            z_to_xy_ratio = max(1.0, (h / d) * 0.5)
        
        # Find center slice
        tumor_coords = np.where(segmentation > 0)
        if len(tumor_coords[0]) > 0:
            center_z = int(np.mean(tumor_coords[0]))
        else:
            center_z = volume.shape[0] // 2
        
        # Main axial view
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(volume[center_z], cmap='gray')
        mask = segmentation[center_z]
        if np.any(mask > 0):
            overlay = np.ma.masked_where(mask == 0, mask)
            ax1.imshow(overlay, cmap='Reds', alpha=0.4)
            ax1.contour(mask, levels=[0.5], colors='red', linewidths=2)
        ax1.set_title(f'Axial View (Slice {center_z})')
        ax1.axis('off')
        
        # Multiple slices with contours
        ax2 = plt.subplot(2, 3, 2)
        tumor_slices = np.where(np.any(segmentation > 0, axis=(1, 2)))[0]
        if len(tumor_slices) >= 3:
            indices = [tumor_slices[0], tumor_slices[len(tumor_slices)//2], tumor_slices[-1]]
        else:
            indices = [max(0, center_z-1), center_z, min(volume.shape[0]-1, center_z+1)]
        
        # Create subplots for sequential slices with contours
        combined = np.hstack([volume[i] for i in indices])
        combined_mask = np.hstack([segmentation[i] for i in indices])
        ax2.imshow(combined, cmap='gray')
        if np.any(combined_mask > 0):
            overlay = np.ma.masked_where(combined_mask == 0, combined_mask)
            ax2.imshow(overlay, cmap='Reds', alpha=0.4)
            ax2.contour(combined_mask, levels=[0.5], colors='red', linewidths=1.5)
        ax2.set_title('Sequential Slices')
        ax2.axis('off')
        
        # 3D MIP - show all 3 projections (Axial, Coronal, Sagittal)
        ax3 = plt.subplot(2, 3, 3)
        
        # Create MIP for all 3 axes
        mip_axial_img = np.max(volume, axis=0)      # View from top (Z-axis), shape: (H, W)
        mip_axial_seg = np.max(segmentation, axis=0)
        
        mip_coronal_img = np.max(volume, axis=1)    # View from front (Y-axis), shape: (D, W)
        mip_coronal_seg = np.max(segmentation, axis=1)
        
        mip_sagittal_img = np.max(volume, axis=2)   # View from side (X-axis), shape: (D, H)
        mip_sagittal_seg = np.max(segmentation, axis=2)
        
        from scipy.ndimage import zoom
        
        # Use z_to_xy_ratio calculated earlier (from voxel_spacing or estimated)
        
        def correct_aspect_coronal(img, z_ratio):
            """Stretch Z dimension for Coronal view (D, W) -> correct aspect"""
            return zoom(img, (z_ratio, 1), order=1)
        
        def correct_aspect_sagittal(img, z_ratio):
            """Stretch Z dimension for Sagittal view (D, H) -> correct aspect"""
            return zoom(img, (z_ratio, 1), order=1)
        
        def correct_aspect_seg(img, z_ratio):
            """Same but for segmentation (nearest neighbor)"""
            return zoom(img, (z_ratio, 1), order=0)
        
        # Apply aspect ratio correction to Coronal and Sagittal
        mip_coronal_img_corrected = correct_aspect_coronal(mip_coronal_img, z_to_xy_ratio)
        mip_coronal_seg_corrected = correct_aspect_seg(mip_coronal_seg, z_to_xy_ratio)
        
        mip_sagittal_img_corrected = correct_aspect_sagittal(mip_sagittal_img, z_to_xy_ratio)
        mip_sagittal_seg_corrected = correct_aspect_seg(mip_sagittal_seg, z_to_xy_ratio)
        
        # Now resize all to same height for display
        target_h = mip_axial_img.shape[0]  # Use axial height as reference
        
        def resize_to_height(img, target_h):
            if img.shape[0] == target_h:
                return img
            scale_h = target_h / img.shape[0]
            scale_w = scale_h  # Keep aspect ratio
            return zoom(img, (scale_h, scale_w), order=1)
        
        def resize_to_height_seg(img, target_h):
            if img.shape[0] == target_h:
                return img
            scale_h = target_h / img.shape[0]
            scale_w = scale_h
            return zoom(img, (scale_h, scale_w), order=0)
        
        # Resize corrected images
        mip_coronal_resized = resize_to_height(mip_coronal_img_corrected, target_h)
        mip_coronal_seg_resized = resize_to_height_seg(mip_coronal_seg_corrected, target_h)
        
        mip_sagittal_resized = resize_to_height(mip_sagittal_img_corrected, target_h)
        mip_sagittal_seg_resized = resize_to_height_seg(mip_sagittal_seg_corrected, target_h)
        
        # Combine 3 MIP views horizontally with small gap
        gap = np.zeros((target_h, 10))  # Small black gap between views
        
        mip_combined_img = np.hstack([
            mip_axial_img,
            gap,
            mip_coronal_resized,
            gap,
            mip_sagittal_resized
        ])
        mip_combined_seg = np.hstack([
            mip_axial_seg,
            np.zeros((target_h, 10)),
            mip_coronal_seg_resized,
            np.zeros((target_h, 10)),
            mip_sagittal_seg_resized
        ])
        
        ax3.imshow(mip_combined_img, cmap='gray')
        if np.any(mip_combined_seg > 0):
            overlay = np.ma.masked_where(mip_combined_seg == 0, mip_combined_seg)
            ax3.imshow(overlay, cmap='Reds', alpha=0.4)
            ax3.contour(mip_combined_seg, levels=[0.5], colors='red', linewidths=2)
        ax3.set_title('MIP: Axial | Coronal | Sagittal')
        ax3.axis('off')
        
        # Statistics text
        ax4 = plt.subplot(2, 3, (4, 5))
        ax4.axis('off')
        
        stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║  PATIENT ID: {patient_id:<47}║
╠══════════════════════════════════════════════════════════════╣
║  TUMOR VOLUME                                                ║
║    • Volume: {features.get('volume_mm3', 0):.2f} mm³ ({features.get('volume_ml', 0):.3f} ml)         
║    • Voxel count: {features.get('voxel_count', 0):<10}                           ║
╠══════════════════════════════════════════════════════════════╣
║  DIMENSIONS                                                  ║
║    • Max diameter: {features.get('max_diameter_mm', 0):.2f} mm                     
║    • Size (Z×Y×X): {features.get('dimensions_mm', (0,0,0))[0]:.1f} × {features.get('dimensions_mm', (0,0,0))[1]:.1f} × {features.get('dimensions_mm', (0,0,0))[2]:.1f} mm
╠══════════════════════════════════════════════════════════════╣
║  MORPHOLOGY                                                  ║
║    • Sphericity: {features.get('sphericity', 0):.3f}                              
║    • Elongation: {features.get('elongation', 0):.3f}                              
║    • Surface area: {features.get('surface_area_mm2', 0):.2f} mm²                   
╚══════════════════════════════════════════════════════════════╝
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Histogram
        ax5 = plt.subplot(2, 3, 6)
        if np.any(segmentation > 0):
            tumor_values = volume[segmentation > 0]
            ax5.hist(tumor_values.flatten(), bins=50, color='red', alpha=0.7, label='Tumor')
        background_values = volume[segmentation == 0]
        ax5.hist(background_values.flatten(), bins=50, color='gray', alpha=0.5, label='Background')
        ax5.set_xlabel('Intensity')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Intensity Distribution')
        ax5.legend()
        
        plt.suptitle(f'NPC Tumor Analysis Report - {patient_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save and return base64
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
