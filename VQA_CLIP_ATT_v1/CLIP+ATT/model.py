import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np 


class CLIPVQAModel(nn.Module):
    def __init__(self, num_classes, clip_model_name="openai/clip-vit-base-patch32"): # type: ignore
        super(CLIPVQAModel, self).__init__()
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Get dimensions
        self.vision_dim = self.clip_model.config.vision_config.hidden_size  # 768
        self.text_dim = self.clip_model.config.text_config.hidden_size  # 512
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Attention mechanism
        self.attention_proj = nn.Linear(self.vision_dim + self.text_dim, self.vision_dim)
        self.attention_weights = nn.Linear(self.vision_dim, 1)
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.vision_dim + self.text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    def compute_similarity_attention(self, visual_features, text_features):
        """
        Compute similarity-based attention weights
        """
        # visual_features: [batch_size, num_patches, vision_dim]
        # text_features: [batch_size, text_dim]
        
        batch_size, num_patches, vision_dim = visual_features.shape
        text_dim = text_features.shape[1]
        
        # Expand text features to match visual patches
        text_expanded = text_features.unsqueeze(1).expand(-1, num_patches, -1)
        
        # Concatenate visual and text features
        combined_features = torch.cat([visual_features, text_expanded], dim=-1)
        
        # Project combined features
        projected = self.attention_proj(combined_features)  # [batch_size, num_patches, vision_dim]
        
        # Compute attention weights
        attention_scores = self.attention_weights(projected).squeeze(-1)  # [batch_size, num_patches]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights
    
    def apply_attention(self, visual_features, attention_weights):
        """
        Apply attention weights to visual features
        """
        # attention_weights: [batch_size, num_patches]
        # visual_features: [batch_size, num_patches, vision_dim]
        
        # Apply attention
        attended_features = visual_features * attention_weights.unsqueeze(-1)
        
        # Global average pooling with attention
        pooled_features = torch.sum(attended_features, dim=1)  # [batch_size, vision_dim]
        
        return pooled_features
    
    def forward(self, images, questions):
        """
        Forward pass
        images: [batch_size, 3, H, W]
        questions: list of strings
        """
        batch_size = images.shape[0]
        
        # Process inputs through CLIP
        with torch.no_grad():
            # Get visual features (patch-level)
            vision_outputs = self.clip_model.vision_model(pixel_values=images)
            visual_features = vision_outputs.last_hidden_state  # [batch_size, num_patches, vision_dim]
            
            # Get text features
            text_inputs = self.clip_processor(
                text=questions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=77
            )
            
            # Move text inputs to same device as images
            text_inputs = {k: v.to(images.device) for k, v in text_inputs.items()}
            
            text_outputs = self.clip_model.text_model(**text_inputs)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token, [batch_size, text_dim]
        
        # Compute similarity-guided attention
        attention_weights = self.compute_similarity_attention(visual_features, text_features)
        
        # Apply attention to visual features
        attended_visual = self.apply_attention(visual_features, attention_weights)
        
        # Combine visual and text features
        combined_features = torch.cat([attended_visual, text_features], dim=-1)
        
        # Feature fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights
