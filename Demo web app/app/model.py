"""
Model definitions for VQA application
"""
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPModel


class AdvancedCLIPVQAModel(nn.Module):
    """
    Advanced CLIP-based VQA Model
    Used for answering questions about images
    """
    def __init__(self, num_classes, clip_model_name="openai/clip-vit-base-patch32", 
                 unfreeze_layers=2, dropout_rate=0.3, custom_vision_dim=None, custom_text_dim=None):
        super().__init__()
        
        # Load pretrained CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Get dimensions - allow custom override for checkpoint compatibility
        if custom_vision_dim is not None:
            self.vision_dim = custom_vision_dim
        else:
            self.vision_dim = self.clip.vision_model.config.hidden_size
            
        if custom_text_dim is not None:
            self.text_dim = custom_text_dim
        else:
            self.text_dim = self.clip.text_model.config.hidden_size
        
        # Freeze CLIP parameters initially
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers of vision and text encoders
        if unfreeze_layers > 0:
            # Unfreeze vision encoder layers
            vision_layers = list(self.clip.vision_model.encoder.layers)
            for layer in vision_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Unfreeze text encoder layers
            text_layers = list(self.clip.text_model.encoder.layers)
            for layer in text_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.vision_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Fusion layer
        fusion_dim = self.vision_dim + self.text_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier head
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, images, questions):
        """
        Forward pass
        Args:
            images: tensor of shape (batch_size, 3, 224, 224)
            questions: dict with 'input_ids' and 'attention_mask'
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        # Get CLIP embeddings
        vision_outputs = self.clip.vision_model(pixel_values=images)
        text_outputs = self.clip.text_model(
            input_ids=questions['input_ids'].to(images.device),
            attention_mask=questions['attention_mask'].to(images.device)
        )
        
        # Get pooled features
        vision_features = vision_outputs.pooler_output  # (batch_size, vision_dim)
        text_features = text_outputs.pooler_output      # (batch_size, text_dim)
        
        # Cross-modal attention
        vision_features_expanded = vision_features.unsqueeze(1)  # (batch_size, 1, vision_dim)
        text_features_expanded = text_features.unsqueeze(1)      # (batch_size, 1, text_dim)
        
        # Project text features to vision dimension for attention
        text_proj = nn.Linear(self.text_dim, self.vision_dim).to(images.device)
        text_features_proj = text_proj(text_features_expanded)
        
        attended_features, attention_weights = self.cross_attention(
            vision_features_expanded,
            text_features_proj,
            text_features_proj
        )
        attended_features = attended_features.squeeze(1)  # (batch_size, vision_dim)
        
        # Concatenate features
        fused_features = torch.cat([attended_features, text_features], dim=1)
        
        # Pass through fusion layers
        fused_features = self.fusion(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights


class FlawClassifier(nn.Module):
    """
    Image Quality Flaw Classifier
    Used for detecting quality issues in unanswerable images
    """
    def __init__(self, num_classes=8, use_dropout=False):
        super().__init__()
        # Use weights parameter instead of deprecated pretrained
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = base.fc.in_features
        
        # Replace fc layer - support both simple and dropout versions
        if use_dropout:
            base.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            # Simple version (matching checkpoint structure)
            base.fc = nn.Linear(in_features, num_classes)
        
        self.model = base
    
    def forward(self, x):
        return self.model(x)
