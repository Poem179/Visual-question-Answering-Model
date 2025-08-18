# VQA CLIP v2 - Advanced Visual Question Answering

Phiên bản nâng cấp của mô hình VQA sử dụng CLIP với các kỹ thuật training hiện đại.

## 🚀 Các Cải Tiến Chính

### 1. **Data Augmentation**
- RandomResizedCrop với scale (0.8-1.0)
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness, contrast, saturation: 0.3)
- RandomRotation (±10°)
- RandomGrayscale (p=0.1)

### 2. **Fine-Tuning CLIP**
- Unfreeze các lớp cuối của CLIP encoder
- Cho phép model học features phù hợp với VQA task
- Cân bằng giữa transfer learning và task-specific learning

### 3. **Differential Learning Rates**
- Learning rate thấp hơn cho CLIP parameters (2e-6)
- Learning rate cao hơn cho head parameters (1e-4)
- Tránh làm hỏng pre-trained features

### 4. **Advanced Scheduling**
- Cosine Annealing LR Scheduler
- Warm Restarts (optional)
- ReduceLROnPlateau (optional)

### 5. **Training Stability**
- Gradient Clipping (norm=1.0)
- Early Stopping (patience=8)
- Weight Decay regularization

### 6. **Enhanced Architecture**
- Multi-head Self-Attention trên vision features
- Cross-modal Attention giữa vision và text
- Improved Fusion Network với LayerNorm

## 📁 Cấu Trúc Thư Mục

```
CLIP+ATT_v2/
├── dataset.py              # Dataset với augmentation
├── model.py                # Advanced CLIP VQA model
├── training_utils.py       # Training utilities
├── visualization.py        # Visualization tools
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── vqa_clip_v2.ipynb    # Main notebook
```

## 🔧 Installation

```bash
# Clone repository
cd VQA_CLIP_Att/CLIP+ATT_v2

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Setup

Đảm bảo cấu trúc data như sau:
```
data/
├── Annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
│   └── *.jpg
├── val/
│   └── *.jpg
└── test/
    └── *.jpg
```

## 🚀 Quick Start

### Option 1: Sử dụng Notebook
```bash
jupyter notebook vqa_clip_v2.ipynb
```

### Option 2: Sử dụng Python Script
```python
from dataset import create_dataloaders_v2
from model import AdvancedCLIPVQAModel, get_model_parameters
from training_utils import train_model_v2, EarlyStopping
from transformers import CLIPProcessor

# Load data
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
train_loader, val_loader, test_loader, num_classes = create_dataloaders_v2(
    data_dir="data", processor=processor, batch_size=24
)

# Create model
model = AdvancedCLIPVQAModel(num_classes=num_classes, unfreeze_layers=2)

# Setup training
param_groups = get_model_parameters(model, clip_lr=2e-6, head_lr=1e-4)
optimizer = torch.optim.AdamW(param_groups)
early_stopping = EarlyStopping(patience=8)

# Train
history = train_model_v2(
    model, train_loader, val_loader, 
    criterion, optimizer, num_epochs=30,
    early_stopping=early_stopping
)
```

## ⚙️ Configuration

Chỉnh sửa `config.py` để thay đổi hyperparameters:

```python
# Model config
UNFREEZE_LAYERS = 2        # Số lớp CLIP để fine-tune
DROPOUT_RATE = 0.3         # Dropout rate

# Training config  
BATCH_SIZE = 24            # Batch size
NUM_EPOCHS = 30            # Số epochs
LEARNING_RATE_HEAD = 1e-4  # LR cho head layers
LEARNING_RATE_CLIP = 2e-6  # LR cho CLIP layers

# Training techniques
GRADIENT_CLIP_NORM = 1.0   # Gradient clipping
EARLY_STOPPING_PATIENCE = 8 # Early stopping patience
```

## 📈 Expected Results

Với các cải tiến này, mong đợi:
- **Validation Accuracy**: 56-60%
- **Test Accuracy**: 56-60%
- **Training Time**: ~6-8 hours (30 epochs)
- **Convergence**: Faster và stable hơn

## 🔍 Monitoring Training

Notebook tự động tạo các visualizations:
- Training/Validation curves
- Learning rate schedule
- Confidence analysis
- Prediction distribution
- Attention visualizations

## 📊 Evaluation Tools

### 1. Model Analysis
```python
from visualization import analyze_predictions_v2

results = analyze_predictions_v2(
    model, test_loader, idx_to_answer, device, num_samples=1000
)
```

### 2. Demo với Custom Images
```python
from visualization import demo_model_predictions

demo_model_predictions(
    model, processor, image_urls, questions, 
    idx_to_answer, device
)
```

### 3. Attention Visualization
```python
from visualization import visualize_attention_v2

visualize_attention_v2(
    image, attention_weights, question, 
    prediction, confidence
)
```

## 🎯 Key Features

### Advanced Architecture
- **Multi-head Attention**: 8 heads cho better feature extraction
- **Cross-modal Fusion**: Attention mechanism giữa vision và text
- **Residual Connections**: Improved gradient flow
- **Layer Normalization**: Training stability

### Smart Training
- **Differential LR**: Tối ưu cho pre-trained + new components
- **Gradient Clipping**: Tránh exploding gradients  
- **Early Stopping**: Tự động dừng khi overfitting
- **Cosine Scheduling**: Smooth LR decay

### Robust Data Pipeline
- **Heavy Augmentation**: Improved generalization
- **Efficient Loading**: Multi-worker DataLoader
- **Memory Optimization**: Gradient accumulation support

## 🚀 Advanced Usage

### Custom Loss Functions
```python
from model import FocalLoss

# Sử dụng Focal Loss cho class imbalance
criterion = FocalLoss(alpha=1.0, gamma=2.0, ignore_index=-1)
```

### Different Schedulers
```python
from training_utils import LearningRateScheduler

# Cosine with Warm Restarts
scheduler = LearningRateScheduler(
    optimizer, 'cosine_warm', T_0=10, T_mult=2
)

# ReduceLROnPlateau
scheduler = LearningRateScheduler(
    optimizer, 'reduce_plateau', patience=3, factor=0.5
)
```

### Model Checkpointing
```python
from training_utils import load_checkpoint

# Load best model
model, epoch, metrics = load_checkpoint(
    model, 'checkpoints/clip_v2_best_model.pth', device
)
```

## 📝 Tips for Best Results

1. **Memory Management**: Giảm batch_size nếu GPU memory không đủ
2. **Learning Rates**: Thử các LR combinations khác nhau
3. **Augmentation**: Điều chỉnh augmentation strength
4. **Early Stopping**: Tăng patience nếu training chậm
5. **Fine-tuning**: Thử unfreeze nhiều layers hơn nếu data đủ lớn

## 🐛 Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**
```python
# Giảm batch size
BATCH_SIZE = 16  # instead of 24

# Hoặc enable gradient accumulation
accumulation_steps = 2
```

**2. Training Không Converge**
```python
# Tăng learning rate
LEARNING_RATE_HEAD = 2e-4  # instead of 1e-4

# Hoặc giảm weight decay
WEIGHT_DECAY = 1e-6  # instead of 1e-5
```

**3. Overfitting**
```python
# Tăng dropout
DROPOUT_RATE = 0.4  # instead of 0.3

# Hoặc tăng weight decay
WEIGHT_DECAY = 1e-4  # instead of 1e-5
```

## 📚 References

1. [CLIP Paper](https://arxiv.org/abs/2103.00020)
2. [VQA Dataset](https://visualqa.org/)
3. [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
4. [Fine-tuning Best Practices](https://arxiv.org/abs/1801.06146)

## 🤝 Contributing

Contributions are welcome! Please check issues và submit PRs.

## 📄 License

MIT License - see LICENSE file for details.
