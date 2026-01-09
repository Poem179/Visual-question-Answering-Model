"""
Attention Visualization Utilities for VQA Model
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


def visualize_self_attention(attention_weights, image, patch_size=32, num_heads_to_show=4):
    """
    Visualize self-attention weights on vision patches
    
    Args:
        attention_weights: [B, num_heads, num_patches, num_patches] or [num_heads, num_patches, num_patches]
        image: PIL Image or numpy array
        patch_size: Size of each patch (default: 32 for ViT-B/32)
        num_heads_to_show: Number of attention heads to visualize
        
    Returns:
        fig: Matplotlib figure with attention heatmaps
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Remove batch dimension if present
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # [num_heads, num_patches, num_patches]
    
    attention_weights = attention_weights.detach().cpu().numpy()
    num_heads = attention_weights.shape[0]
    num_patches = attention_weights.shape[1]
    
    # Calculate grid size (assuming square patches)
    grid_size = int(np.sqrt(num_patches))
    
    # Limit number of heads to show
    num_heads_to_show = min(num_heads_to_show, num_heads)
    
    # Create figure
    fig, axes = plt.subplots(2, num_heads_to_show, figsize=(4 * num_heads_to_show, 8))
    if num_heads_to_show == 1:
        axes = axes.reshape(2, 1)
    
    for head_idx in range(num_heads_to_show):
        # Get attention for this head (average across query patches)
        attn_head = attention_weights[head_idx]  # [num_patches, num_patches]
        attn_map = attn_head.mean(axis=0)  # Average across queries: [num_patches]
        
        # Reshape to 2D grid
        attn_map_2d = attn_map.reshape(grid_size, grid_size)
        
        # Resize to match image size
        h, w = image_np.shape[:2]
        attn_map_resized = cv2.resize(attn_map_2d, (w, h))
        
        # Show original image with attention overlay
        axes[0, head_idx].imshow(image_np)
        axes[0, head_idx].imshow(attn_map_resized, alpha=0.6, cmap='jet')
        axes[0, head_idx].set_title(f'Head {head_idx + 1} Overlay')
        axes[0, head_idx].axis('off')
        
        # Show attention heatmap alone
        sns.heatmap(attn_map_2d, ax=axes[1, head_idx], cmap='viridis', 
                   cbar=True, square=True, xticklabels=False, yticklabels=False)
        axes[1, head_idx].set_title(f'Head {head_idx + 1} Heatmap')
    
    plt.tight_layout()
    return fig


def visualize_cross_attention(attention_weights, image, patch_size=32):
    """
    Visualize cross-attention weights (text attending to vision)
    
    Args:
        attention_weights: [B, num_heads, 1, 1] - simplified cross attention
        image: PIL Image or numpy array
        patch_size: Size of each patch
        
    Returns:
        fig: Matplotlib figure with cross-attention visualization
    """
    # For this simplified cross-attention, we don't have spatial info
    # Just show the attention weight as a single value per head
    
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch: [num_heads, 1, 1]
    
    attention_weights = attention_weights.detach().cpu().numpy()
    attention_weights = attention_weights.squeeze()  # [num_heads]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(range(len(attention_weights)), attention_weights)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Cross-Modal Attention Weights (Text → Vision)')
    ax.set_xticks(range(len(attention_weights)))
    ax.set_xticklabels([f'Head {i+1}' for i in range(len(attention_weights))])
    
    # Color bars by intensity
    norm = plt.Normalize(attention_weights.min(), attention_weights.max())
    colors = plt.cm.viridis(norm(attention_weights))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    return fig


def visualize_attention_summary(self_attn, cross_attn, image, question_text):
    """
    Create a comprehensive attention visualization
    
    Args:
        self_attn: Self-attention weights [B, num_heads, num_patches, num_patches]
        cross_attn: Cross-attention weights [B, num_heads, seq_len, num_patches]
        image: PIL Image
        question_text: String with the question
        
    Returns:
        fig: Comprehensive matplotlib figure
    """
    try:
        # Create larger figure with clean grid layout
        fig = plt.figure(figsize=(24, 16))
        # 5 rows x 8 columns: row 0 for title/original, rows 1-4 for visualizations
        gs = fig.add_gridspec(5, 8, hspace=0.5, wspace=0.35,
                             height_ratios=[0.8, 1, 1, 1, 1])
        
        # Convert image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Handle self attention - [B, num_heads, 49, 49]
        if self_attn.dim() == 4:
            self_attn = self_attn[0]  # Remove batch -> [num_heads, 49, 49]
        self_attn_np = self_attn.detach().cpu().numpy()
        
        # Handle cross attention - [B, num_heads, seq_len, 49]
        if cross_attn.dim() == 4:
            cross_attn = cross_attn[0]  # Remove batch -> [num_heads, seq_len, 49]
        
        # Average over query tokens to get per-head attention to patches
        # cross_attn: [num_heads, seq_len, 49] -> [num_heads, 49]
        if cross_attn.dim() == 3:
            cross_attn = cross_attn.mean(dim=1)  # Average over text tokens
        
        cross_attn_np = cross_attn.detach().cpu().numpy()  # [num_heads, 49]
        
        num_heads = self_attn_np.shape[0]
        num_patches = self_attn_np.shape[1]
        grid_size = int(np.sqrt(num_patches))
        h, w = image_np.shape[:2]
        
        # === ROW 0: Title and Original Image ===
        # Add overall title
        fig.text(0.5, 0.98, f'Complete Attention Visualization (All {num_heads} Heads)', 
                ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.96, f'Question: "{question_text}"', 
                ha='center', fontsize=14, style='italic')
        
        # Original image in first row, centered, spanning 2 columns
        ax_img = fig.add_subplot(gs[0, 3:5])
        ax_img.imshow(image_np)
        ax_img.set_title('Original Image', fontsize=12, fontweight='bold', pad=10)
        ax_img.axis('off')
        
        # === ROW 1: Self-Attention Overlays (8 heads) ===
        for i in range(min(num_heads, 8)):
            ax = fig.add_subplot(gs[1, i])
            attn_map = self_attn_np[i].mean(axis=0).reshape(grid_size, grid_size)
            attn_map_resized = cv2.resize(attn_map, (w, h))
            
            ax.imshow(image_np)
            ax.imshow(attn_map_resized, alpha=0.6, cmap='jet')
            ax.set_title(f'Self-Attn Head {i + 1}', fontsize=10)
            ax.axis('off')
        
        # === ROW 2: Self-Attention Heatmaps (8 heads) ===
        for i in range(min(num_heads, 8)):
            ax = fig.add_subplot(gs[2, i])
            attn_map = self_attn_np[i].mean(axis=0).reshape(grid_size, grid_size)
            im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
            ax.set_title(f'Self H{i + 1} Map', fontsize=9)
            ax.axis('off')
            # Smaller colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
        
        # === ROW 3: Cross-Attention Overlays (8 heads) ===
        for i in range(min(num_heads, 8)):
            ax = fig.add_subplot(gs[3, i])
            # Get cross-attention for this head: [49] -> reshape to [7, 7]
            cross_head = cross_attn_np[i].reshape(grid_size, grid_size)
            cross_resized = cv2.resize(cross_head, (w, h))
            
            ax.imshow(image_np)
            ax.imshow(cross_resized, alpha=0.7, cmap='hot')
            ax.set_title(f'Cross-Attn Head {i + 1}', fontsize=10)
            ax.axis('off')
        
        # === ROW 4: Cross-Attention Heatmaps (8 heads) ===
        for i in range(min(num_heads, 8)):
            ax = fig.add_subplot(gs[4, i])
            cross_head = cross_attn_np[i].reshape(grid_size, grid_size)
            im = ax.imshow(cross_head, cmap='hot', aspect='auto')
            ax.set_title(f'Cross H{i + 1} Map', fontsize=9)
            ax.axis('off')
            # Smaller colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
        
        # Add row labels on the left
        fig.text(0.02, 0.75, 'Self-Attention\nOverlays', 
                rotation=90, va='center', fontsize=11, fontweight='bold')
        fig.text(0.02, 0.60, 'Self-Attention\nHeatmaps', 
                rotation=90, va='center', fontsize=11, fontweight='bold')
        fig.text(0.02, 0.40, 'Cross-Attention\nOverlays', 
                rotation=90, va='center', fontsize=11, fontweight='bold')
        fig.text(0.02, 0.25, 'Cross-Attention\nHeatmaps', 
                rotation=90, va='center', fontsize=11, fontweight='bold')
        
        return fig
    
    except Exception as e:
        # If visualization fails, create a simple error figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
