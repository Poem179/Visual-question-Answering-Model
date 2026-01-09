# 🖼️ VQA Application - Visual Question Answering

Ứng dụng trả lời câu hỏi về ảnh bằng AI, kết hợp 2 models:
1. **VQA Model** - Trả lời câu hỏi về nội dung ảnh (500 classes)
2. **Flaw Detection** - Phát hiện lỗi chất lượng ảnh khi không trả lời được (8 classes)

---

## 🚀 Quickstart

### Cài đặt
```cmd
cd Docs
pip install -r requirements.txt
```

### Chạy ứng dụng
```cmd
cd app
streamlit run app.py
```

### Extract vocabulary (if needed)
```cmd
cd app
python extract_vocab.py
```

---

## 📁 Cấu trúc Project

```
app-vqa/
├── 📱 App Directory
│   ├── app.py                    # Production app
│   ├── model_v2.py              # Model architectures
│   ├── model.py                 # Alternative models
│   ├── extract_vocab.py         # Vocabulary extraction
│   └── idx_to_answer.json       # Answer vocabulary
│
├── 🤖 Models Directory
│   ├── vqa.pth                  # VQA checkpoint (500 classes)
│   ├── unanswerable.pth         # Flaw detection (8 classes)
│   ├── vqa_clip.ipynb           # VQA training notebook
│   └── unanswerable.ipynb       # Flaw training notebook
│
├── 📚 Documentation (Docs/)
│   ├── README.md                # This file
│   ├── PROJECT_SUMMARY.md       # Project overview
│   ├── requirements.txt         # Dependencies
│   └── THRESHOLD_GUIDE.md       # Threshold explanation
│
└── __pycache__/                 # Python cache
```

---

## 🎯 Features

### 1. Visual Question Answering
- Upload image + ask question
- AI answers based on image content
- Shows confidence score
- 500 possible answer types

### 2. Flaw Detection (Auto-triggered)
- Activates when VQA returns "unanswerable"
- Detects 8 quality issues:
  ```
  ✨ No_issue          - Image is fine
  🌫️ Blurry           - Out of focus
  🔆 Overexposure     - Too bright
  🌑 Underexposure    - Too dark
  📐 Bad_framing      - Poor composition
  📱 Screenshot       - Screen capture
  ⚡ Noisy            - Grainy/artifacts
  🎨 Other_issues     - Misc problems
  ```

### 3. Threshold Control
- Adjustable detection threshold (0.3-0.8)
- Color-coded confidence levels:
  - 🔴 ≥80% - Very confident
  - 🟡 60-79% - Confident
  - 🟢 <60% - Possible
- Visual progress bars

---

## 🎮 Usage

1. **Navigate to app directory**
   ```cmd
   cd app
   ```

2. **Start app**
   ```cmd
   streamlit run app.py
   ```

3. **Open browser** → `http://localhost:8501`

4. **Use interface**
   - Upload image (PNG/JPG/JPEG)
   - Enter question (English)
   - Click "� Predict"

5. **Adjust settings** (Sidebar)
   - Threshold slider
   - View model status

---

## ⚙️ Model Architectures

### VQA Model (VQAModelFromCheckpoint)
```
Image (224×224) + Question (text)
         ↓
CLIP (Vision: 768d, Text: 512d)
         ↓
Multihead Attention (768d)
         ↓
Vision Proj (768→512) + Text Proj (512→512)
         ↓
Cross Attention (512d)
         ↓
Fusion Network (1024→512)
         ↓
Classifier (512→256→500)
         ↓
500 possible answers
```

### Flaw Classifier (FlawClassifier)
```
Image (224×224)
         ↓
ResNet18 (pretrained)
         ↓
FC (512→8)
         ↓
Sigmoid
         ↓
8 flaw probabilities
```

---

## 🔧 Troubleshooting

### Model won't load
- Check if `vqa.pth` and `unanswerable.pth` exist in `models/` directory
- Verify paths in `app/app.py`

### Missing dependencies
```cmd
cd Docs
pip install -r requirements.txt
```

### Missing vocabulary file
```cmd
cd app
python extract_vocab.py
```

---

## 📊 Requirements

### System
- Python 3.8+
- 4GB+ RAM
- Optional: NVIDIA GPU (CUDA)

### Models
- **models/vqa.pth**: ~400MB, 500-class CLIP-based VQA
- **models/unanswerable.pth**: ~45MB, 8-class ResNet18 flaw detector
- **app/idx_to_answer.json**: Answer vocabulary mapping

### Dependencies
- PyTorch 2.0+
- Streamlit 1.28.0
- Transformers 4.30.0+
- torchvision, PIL

---

## 📚 Documentation

- **README.md** (this file) - Overview
- **QUICKSTART.md** - Step-by-step guide
- **THRESHOLD_GUIDE.md** - Detailed threshold explanation
- **CHANGELOG.md** - Version history

---

## �️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML Framework | PyTorch |
| Vision Models | CLIP, ResNet18 |
| Transformers | HuggingFace |
| Image Processing | PIL, torchvision |

---

## 📝 Version Info

- **Version**: 2.0
- **Last Updated**: October 30, 2025
- **Status**: ✅ Production Ready
- **License**: Educational/Research use

---

**Made with ❤️ using PyTorch & Streamlit**
