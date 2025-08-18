"""
Dataset với Data Augmentation cho VQA CLIP v2
Cải tiến từ phiên bản trước với data augmentation và xử lý dữ liệu tốt hơn
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from collections import Counter
import numpy as np
from torchvision import transforms


class VQADatasetV2(Dataset):
    """Dataset nâng cấp với data augmentation cho VQA"""
    
    def __init__(self, image_dir, annotations_file, processor, 
                 max_answers=1000, transform=None, mode='train'):
        """
        Args:
            image_dir: Thư mục chứa hình ảnh
            annotations_file: File JSON chứa annotations  
            processor: CLIP processor
            max_answers: Số lượng câu trả lời phổ biến nhất để giữ lại
            transform: Các phép biến đổi augmentation
            mode: 'train', 'val', hoặc 'test'
        """
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.mode = mode
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded {len(self.annotations)} samples from {annotations_file}")
        
        # Xây dựng vocabulary từ tập train
        if mode == 'train':
            self._build_answer_vocab(max_answers)
        else:
            # Sẽ được set từ train dataset
            self.answer_to_idx = None
            self.idx_to_answer = None
    
    def _build_answer_vocab(self, max_answers):
        """Xây dựng vocabulary câu trả lời từ tập train"""
        answer_counts = Counter()
        
        for item in self.annotations:
            if 'answers' in item:
                for answer_obj in item['answers']:
                    answer = answer_obj['answer'].lower().strip()
                    answer_counts[answer] += 1
        
        # Lấy top max_answers câu trả lời phổ biến nhất
        most_common = answer_counts.most_common(max_answers)
        
        # Tạo mapping
        self.answer_to_idx = {answer: idx for idx, (answer, _) in enumerate(most_common)}
        self.idx_to_answer = {idx: answer for answer, idx in self.answer_to_idx.items()}
        
        print(f"Built vocabulary with {len(self.answer_to_idx)} answers")
        print(f"Top 10 answers: {list(self.answer_to_idx.keys())[:10]}")
    
    def set_answer_vocab(self, answer_to_idx, idx_to_answer):
        """Set vocabulary từ train dataset cho val/test"""
        self.answer_to_idx = answer_to_idx
        self.idx_to_answer = idx_to_answer
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback: tạo image trắng
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
        # Apply custom transforms nếu có (cho data augmentation)
        if self.transform:
            image = self.transform(image)
            # Convert về PIL nếu transform trả về tensor
            if isinstance(image, torch.Tensor):
                # Convert tensor về PIL để processor xử lý
                image = transforms.ToPILImage()(image)
        
        # Process với CLIP processor
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)  # Remove batch dim
        
        # Process question
        question = item['question']
        
        # Process answers và tạo label
        label = -1  # Default: không hợp lệ
        
        if 'answers' in item and self.answer_to_idx is not None:
            # Lấy câu trả lời phổ biến nhất
            answer_counts = Counter()
            for answer_obj in item['answers']:
                answer = answer_obj['answer'].lower().strip()
                answer_counts[answer] += 1
            
            if answer_counts:
                most_common_answer = answer_counts.most_common(1)[0][0]
                if most_common_answer in self.answer_to_idx:
                    label = self.answer_to_idx[most_common_answer]
        
        return {
            'pixel_values': pixel_values,
            'question': question,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': item.get('image_id', idx)
        }


def get_data_transforms():
    """Tạo các phép biến đổi data augmentation"""
    
    # CLIP normalization values
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    
    data_transforms = {
        'train': transforms.Compose([
            # Data Augmentation cho training
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3, 
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomGrayscale(p=0.1),
            # Chuẩn hóa theo CLIP
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ]),
        
        'val': transforms.Compose([
            # Không augmentation cho validation
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ]),
        
        'test': transforms.Compose([
            # Không augmentation cho test
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    }
    
    return data_transforms


def create_dataloaders_v2(data_dir, processor, batch_size=32, max_answers=1000, num_workers=4):
    """
    Tạo DataLoader với data augmentation
    
    Args:
        data_dir: Thư mục gốc chứa data
        processor: CLIP processor
        batch_size: Batch size
        max_answers: Số câu trả lời để giữ lại
        num_workers: Số worker cho DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    # Đường dẫn
    train_json = os.path.join(data_dir, "Annotations", "train.json")
    val_json = os.path.join(data_dir, "Annotations", "val.json")
    test_json = os.path.join(data_dir, "Annotations", "test.json")
    
    train_img_dir = os.path.join(data_dir, "train")
    val_img_dir = os.path.join(data_dir, "val")
    test_img_dir = os.path.join(data_dir, "test")
    
    # Tạo transforms
    transforms_dict = get_data_transforms()
    
    # Tạo datasets
    print("Creating training dataset...")
    train_dataset = VQADatasetV2(
        image_dir=train_img_dir,
        annotations_file=train_json,
        processor=processor,
        max_answers=max_answers,
        transform=transforms_dict['train'],
        mode='train'
    )
    
    print("Creating validation dataset...")
    val_dataset = VQADatasetV2(
        image_dir=val_img_dir,
        annotations_file=val_json,
        processor=processor,
        transform=transforms_dict['val'],
        mode='val'
    )
    
    print("Creating test dataset...")
    test_dataset = VQADatasetV2(
        image_dir=test_img_dir,
        annotations_file=test_json,
        processor=processor,
        transform=transforms_dict['test'],
        mode='test'
    )
    
    # Set vocabulary cho val và test từ train
    val_dataset.set_answer_vocab(train_dataset.answer_to_idx, train_dataset.idx_to_answer)
    test_dataset.set_answer_vocab(train_dataset.answer_to_idx, train_dataset.idx_to_answer)
    
    num_classes = len(train_dataset.answer_to_idx)
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ DataLoaders created successfully!")
    print(f"  - Train: {len(train_loader)} batches")
    print(f"  - Val: {len(val_loader)} batches") 
    print(f"  - Test: {len(test_loader)} batches")
    print(f"  - Num classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


def collate_fn(batch):
    """Custom collate function để xử lý batch"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    questions = [item['question'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'questions': questions,
        'labels': labels,
        'image_ids': image_ids
    }
