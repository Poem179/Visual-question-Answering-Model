"""
Model CLIP VQA v2 với Fine-tuning và các cải tiến
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import math


class AdvancedCLIPVQAModel(nn.Module):
    """
    Mô hình VQA nâng cấp với CLIP và các cải tiến:
    - Fine-tuning CLIP layers
    - Improved attention mechanism
    - Better fusion techniques
    """
    
    def __init__(self, num_classes, clip_model_name="openai/clip-vit-base-patch32", 
                 unfreeze_layers=2, dropout_rate=0.3):
        """
        Args:
            num_classes: Số lượng classes (câu trả lời)
            clip_model_name: Tên model CLIP pre-trained
            unfreeze_layers: Số lớp cuối của CLIP để fine-tune
            dropout_rate: Tỷ lệ dropout
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.unfreeze_layers = unfreeze_layers
        
        # Load CLIP model
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze tất cả parameters trước
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Unfreeze một số lớp cuối để fine-tuning
        self._unfreeze_clip_layers(unfreeze_layers)
        
        # Lấy feature dimensions
        self.vision_dim = self.clip_model.config.vision_config.hidden_size  # 768
        self.text_dim = self.clip_model.config.text_config.hidden_size      # 512
        
        # Improved Attention Mechanism
        self.attention_dim = 256
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.vision_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Cross-modal fusion layers
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
        
        # Attention untuk cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
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
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✓ Model initialized with {num_classes} classes")
        print(f"✓ Fine-tuning {unfreeze_layers} CLIP layers")
        self._print_trainable_params()
    
    def _unfreeze_clip_layers(self, num_layers):
        """Unfreeze một số lớp cuối của CLIP để fine-tuning"""
        if num_layers <= 0:
            return
        
        # Unfreeze vision encoder layers
        vision_layers = list(self.clip_model.vision_model.encoder.layers)
        for layer in vision_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Unfreeze text encoder layers  
        text_layers = list(self.clip_model.text_model.encoder.layers)
        for layer in text_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"✓ Unfrozen {num_layers} layers each from vision and text encoders")
    
    def _init_weights(self):
        """Initialize weights cho các layer mới"""
        for module in [self.vision_proj, self.text_proj, self.fusion_net, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    def _print_trainable_params(self):
        """In thông tin về parameters có thể train được"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def forward(self, pixel_values, questions):
        """
        Forward pass
        
        Args:
            pixel_values: Tensor [batch_size, 3, 224, 224]
            questions: List của strings
            
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_patches] 
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # CLIP Vision Encoding
        vision_outputs = self.clip_model.vision_model(pixel_values)
        
        # Lấy patch embeddings (không bao gồm CLS token)
        vision_embeddings = vision_outputs.last_hidden_state  # [batch, num_patches+1, dim]
        vision_patch_embeddings = vision_embeddings[:, 1:, :]  # Bỏ CLS token
        vision_cls = vision_embeddings[:, 0, :]  # CLS token
        
        # CLIP Text Encoding
        # Questions should be pre-tokenized or raw text
        if isinstance(questions, list):
            # If questions are raw text, we need to access processor differently
            # For now, assume questions are already tokenized input_ids
            # This should be handled by the caller
            raise ValueError("Questions should be pre-tokenized. Use processor.tokenizer() before calling model.")
        elif isinstance(questions, torch.Tensor):
            # Assume questions are tokenized input_ids
            text_inputs = {'input_ids': questions.to(device)}
        else:
            # Assume questions is a dict with input_ids, attention_mask, etc.
            text_inputs = {k: v.to(device) for k, v in questions.items()}
        
        text_outputs = self.clip_model.text_model(**text_inputs)
        text_embeddings = text_outputs.pooler_output  # [batch, text_dim]
        
        # Self-attention trên vision patches
        attended_vision, vision_attn_weights = self.multihead_attn(
            vision_patch_embeddings,
            vision_patch_embeddings, 
            vision_patch_embeddings
        )
        
        # Global average pooling cho vision features
        vision_global = attended_vision.mean(dim=1)  # [batch, vision_dim]
        
        # Project về cùng dimension space
        vision_proj = self.vision_proj(vision_global)  # [batch, 512]
        text_proj = self.text_proj(text_embeddings)    # [batch, 512]
        
        # Cross-modal attention
        # Text query, Vision key/value
        text_query = text_proj.unsqueeze(1)  # [batch, 1, 512]
        vision_kv = vision_proj.unsqueeze(1)  # [batch, 1, 512]
        
        cross_attended, cross_attn_weights = self.cross_attention(
            text_query, vision_kv, vision_kv
        )
        cross_attended = cross_attended.squeeze(1)  # [batch, 512]
        
        # Fusion
        # Kết hợp original features và cross-attended features
        fused_features = torch.cat([
            vision_proj + cross_attended,  # Residual connection
            text_proj
        ], dim=-1)  # [batch, 1024]
        
        # Fusion network
        fused_output = self.fusion_net(fused_features)  # [batch, 512]
        
        # Final classification
        logits = self.classifier(fused_output)  # [batch, num_classes]
        
        # Return attention weights (từ vision self-attention)
        # Average attention weights across heads và patches
        attention_weights = vision_attn_weights
        # vision_attn_weights: [batch, num_heads, num_patches, num_patches]
        # Lấy mean qua head và query patch (trục 1 và 2), giữ lại [batch, num_patches]
        if attention_weights is not None and isinstance(attention_weights, torch.Tensor):
            # Nếu shape đúng, lấy mean
            if attention_weights.dim() == 4:
                # [batch, num_heads, num_patches, num_patches] -> [batch, num_patches]
                attention_weights = attention_weights.mean(dim=1).mean(dim=1)
            elif attention_weights.dim() == 3:
                # [batch, num_patches, num_patches] -> [batch, num_patches]
                attention_weights = attention_weights.mean(dim=1)
            elif attention_weights.dim() == 2:
                # [batch, num_patches] -> giữ nguyên
                pass
            elif attention_weights.dim() == 1:
                # [num_patches] -> thêm batch dim
                attention_weights = attention_weights.unsqueeze(0)
            else:
                # Nếu là scalar hoặc shape lạ, tạo dummy attention
                attention_weights = torch.zeros((batch_size, vision_patch_embeddings.shape[1]), device=logits.device)
        else:
            # Nếu không phải tensor, tạo dummy attention
            attention_weights = torch.zeros((batch_size, vision_patch_embeddings.shape[1]), device=logits.device)

        return logits, attention_weights


def get_model_parameters(model, clip_lr=2e-6, head_lr=1e-4, weight_decay=1e-5):
    """
    Tạo parameter groups với learning rates khác nhau
    
    Args:
        model: Model instance
        clip_lr: Learning rate cho CLIP parameters
        head_lr: Learning rate cho head parameters  
        weight_decay: Weight decay
        
    Returns:
        List of parameter groups
    """
    
    # Phân chia parameters
    clip_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip_model' in name:
                clip_params.append(param)
            else:
                head_params.append(param)
    
    # Tạo parameter groups
    param_groups = [
        {
            'params': clip_params,
            'lr': clip_lr,
            'weight_decay': weight_decay * 0.1,  # Ít weight decay hơn cho pre-trained weights
            'name': 'clip_params'
        },
        {
            'params': head_params, 
            'lr': head_lr,
            'weight_decay': weight_decay,
            'name': 'head_params'
        }
    ]
    
    print(f"✓ Parameter groups created:")
    print(f"  - CLIP params: {len(clip_params)} params with LR {clip_lr}")
    print(f"  - Head params: {len(head_params)} params with LR {head_lr}")
    
    return param_groups


class FocalLoss(nn.Module):
    """
    Focal Loss để xử lý class imbalance
    """
    
    def __init__(self, alpha=1, gamma=2, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        # Mask out ignore_index
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            
        # Apply mask
        inputs = inputs[mask]
        targets = targets[mask]
        
        # Compute focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


def create_model_v2(num_classes, clip_model_name="openai/clip-vit-base-patch32",
                   unfreeze_layers=2, dropout_rate=0.3):
    """
    Factory function để tạo model
    
    Args:
        num_classes: Số lượng classes
        clip_model_name: Tên CLIP model
        unfreeze_layers: Số lớp CLIP để fine-tune
        dropout_rate: Dropout rate
        
    Returns:
        model: AdvancedCLIPVQAModel instance
    """
    
    model = AdvancedCLIPVQAModel(
        num_classes=num_classes,
        clip_model_name=clip_model_name,
        unfreeze_layers=unfreeze_layers,
        dropout_rate=dropout_rate
    )
    
    return model
