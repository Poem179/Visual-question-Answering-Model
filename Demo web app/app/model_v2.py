"""
Model definitions matching the EXACT architecture from max1000.ipynb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import models
from torchvision.models import ResNet18_Weights


class AdvancedCLIPVQAModel(nn.Module):
    """
    VQA Model matching EXACT architecture from max1000.ipynb
    
    Architecture:
    - CLIP Encoders (768-dim vision, 512-dim text)
    - Visual Self-Attention (8 heads) + LayerNorm
    - Textual Self-Attention (8 heads) + LayerNorm  
    - Projection layers (768->512, 512->512)
    - Cross-Attention with full Transformer block:
      - Multi-Head Attention
      - Residual + LayerNorm
      - Feed-Forward Network (512 -> 2048 -> 512) with GELU
      - Residual + LayerNorm
    - Fusion (1024 -> 512)
    - Classifier (512 -> 256 -> num_classes)
    """
    def __init__(self, num_classes=1000, clip_model_name="openai/clip-vit-base-patch32", 
                 unfreeze_layers=2, dropout_rate=0.4):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load CLIP
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze all first
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers (for training, not needed for inference)
        if unfreeze_layers > 0:
            vision_layers = list(self.clip_model.vision_model.encoder.layers)
            for layer in vision_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            text_layers = list(self.clip_model.text_model.encoder.layers)
            for layer in text_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Feature dimensions
        self.vision_dim = 768
        self.text_dim = 512
        
        # Visual Self-Attention
        self.vision_self_attn = nn.MultiheadAttention(
            embed_dim=self.vision_dim, num_heads=8, 
            dropout=dropout_rate, batch_first=True
        )
        self.vision_self_norm = nn.LayerNorm(self.vision_dim)
        
        # Textual Self-Attention
        self.text_self_attn = nn.MultiheadAttention(
            embed_dim=self.text_dim, num_heads=8, 
            dropout=dropout_rate, batch_first=True
        )
        self.text_self_norm = nn.LayerNorm(self.text_dim)
        
        # Projection layers
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-attention (with full Transformer block)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, 
            dropout=dropout_rate, batch_first=True
        )
        self.cross_attn_norm1 = nn.LayerNorm(512)
        self.cross_attn_norm2 = nn.LayerNorm(512)
        self.cross_attn_ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.vision_proj, self.text_proj, self.fusion_net, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None, questions=None, return_attention=False):
        """
        Forward pass matching max1000.ipynb architecture
        
        Args:
            pixel_values: [B, 3, 224, 224]
            input_ids: [B, seq_len] - tokenized question ids
            attention_mask: [B, seq_len] - attention mask
            questions: dict with 'input_ids' and 'attention_mask' (alternative input format)
            return_attention: whether to return attention weights
            
        Returns:
            logits: [B, num_classes]
            attn_weights: attention weights (if return_attention=True)
        """
        device = pixel_values.device
        
        # Handle both input formats
        if questions is not None:
            # Dict format (like in training)
            text_inputs = {k: v.to(device) for k, v in questions.items()}
        else:
            # Separate tensors format
            text_inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }
        
        # CLIP encoding
        vision_outputs = self.clip_model.vision_model(pixel_values)
        vision_embeddings = vision_outputs.last_hidden_state
        vision_patches = vision_embeddings[:, 1:, :]  # Remove CLS token -> [B, 49, 768]
        
        text_outputs = self.clip_model.text_model(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state  # [B, seq_len, 512]
        
        # Visual Self-Attention (capture attention weights for visualization)
        vision_self_attended, vision_self_attn_weights = self.vision_self_attn(
            vision_patches, vision_patches, vision_patches,
            need_weights=True, average_attn_weights=False
        )
        # vision_self_attn_weights: [B, num_heads, 49, 49]
        vision_refined = self.vision_self_norm(vision_patches + vision_self_attended)
        
        # Textual Self-Attention
        text_self_attended, _ = self.text_self_attn(
            text_embeddings, text_embeddings, text_embeddings
        )
        text_refined = self.text_self_norm(text_embeddings + text_self_attended)
        
        # Project to common space (512 dim)
        vision_proj = self.vision_proj(vision_refined)  # [B, 49, 512]
        text_proj = self.text_proj(text_refined)        # [B, seq_len, 512]
        
        # Cross-attention with Transformer block
        # Step 1: Multi-Head Cross-Attention (text queries vision)
        cross_attended, cross_attn_weights = self.cross_attn(
            text_proj, vision_proj, vision_proj,
            need_weights=True, average_attn_weights=False
        )
        # cross_attn_weights: [B, num_heads, seq_len, 49]
        # Step 2: Residual + LayerNorm
        text_proj = self.cross_attn_norm1(text_proj + cross_attended)
        # Step 3: Feed-Forward Network
        ffn_output = self.cross_attn_ffn(text_proj)
        # Step 4: Residual + LayerNorm
        text_proj = self.cross_attn_norm2(text_proj + ffn_output)
        
        # Fusion - Global average pooling
        text_global = text_proj.mean(dim=1)      # [B, 512]
        cross_global = text_proj.mean(dim=1)     # [B, 512] (same as text_global after FFN)
        fused = torch.cat([text_global, cross_global], dim=1)  # [B, 1024]
        fused = self.fusion_net(fused)           # [B, 512]
        
        # Classification
        logits = self.classifier(fused)          # [B, num_classes]
        
        if return_attention:
            # Return attention weights in dict format expected by app.py
            # self_attn: [B, num_heads, 49, 49] -> average query to get [B, num_heads, 49]
            # cross_attn: [B, num_heads, seq_len, 49] -> average query to get [B, num_heads, 49]
            return logits, {
                'self_attn': vision_self_attn_weights,   # [B, num_heads, 49, 49]
                'cross_attn': cross_attn_weights,        # [B, num_heads, seq_len, 49]
                'num_patches': 49
            }
        return logits


# Alias for backward compatibility
VQAModelFromCheckpoint = AdvancedCLIPVQAModel


class FlawClassifier(nn.Module):
    """
    Image quality flaw detection model
    Uses ResNet18 backbone for 8-class multi-label classification
    """
    def __init__(self, num_classes=8, use_dropout=False):
        super().__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace final layer for multi-label classification
        num_features = self.resnet.fc.in_features
        
        if use_dropout:
            self.resnet.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
