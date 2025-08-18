# VQA CLIP Attention Model

Dự án Visual Question Answering (VQA) sử dụng CLIP encoders với cơ chế attention dựa trên similarity.

## Kiến trúc mô hình

```
Raw Image + Question 
    ↓
CLIP Visual Encoder + CLIP Text Encoder  
    ↓
Compute Similarity Score
    ↓
Use Similarity to Guide Attention
    ↓
Apply Attention to Visual Features
    ↓
Combine with Question Features
    ↓
Final Classification
```

## Cấu trúc project

```
VQA_CLIP_Att/
├── data/
│   ├── Annotations/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── train/
│   ├── val/
│   └── test/
├── model.py              # Định nghĩa mô hình CLIP VQA
├── dataset.py            # Dataset và DataLoader
├── train.py              # Training functions
├── visualization.py      # Visualization functions
├── vqa_experiment.ipynb  # Notebook thực nghiệm chính
├── requirements.txt      # Thư viện cần thiết
└── README.md
```

## Cài đặt

1. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu VizWiz VQA dataset trong folder `data/`

## Sử dụng

### Training và Evaluation

Chạy notebook `vqa_experiment.ipynb` để thực hiện:
- Import thư viện
- Chuẩn bị dữ liệu
- Training mô hình (15 epochs)
- Evaluation trên validation và test sets
- Visualization kết quả
- Demo với ảnh từ URLs

### Các file Python

- `model.py`: CLIPVQAModel - mô hình chính với attention mechanism
- `dataset.py`: VQADataset, create_dataloaders - xử lý dữ liệu
- `train.py`: train_model, evaluate_model - training functions
- `visualization.py`: Các hàm visualization và analysis

## Đặc điểm chính

- **CLIP Encoders**: Sử dụng pre-trained CLIP để extract features
- **Attention Mechanism**: Similarity-guided attention giữa visual và text features
- **Feature Fusion**: Kết hợp attended visual features với text features
- **Multi-class Classification**: Dự đoán top-1000 answers phổ biến nhất

## Kết quả

- Model tự động lưu best checkpoint tại `checkpoints/best_model.pth`
- Training curves được lưu tại `training_history.png`
- Evaluation results tại `results/evaluation_results.pth`
- Attention visualizations tại `results/attention_sample_*.png`

## Demo

Notebook bao gồm demo với ảnh từ URLs và visualization attention weights để hiểu cách model focus vào các vùng quan trọng của ảnh khi trả lời câu hỏi.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.21+
- CUDA (recommended)
