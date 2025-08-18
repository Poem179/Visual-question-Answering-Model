"""
Training utilities với các kỹ thuật nâng cao
"""

import torch
import torch.nn as nn
import time
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor


class EarlyStopping:
    """Early Stopping để tránh overfitting"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: Số epochs chờ trước khi dừng
            min_delta: Độ cải thiện tối thiểu để coi là có ý nghĩa
            restore_best_weights: Có khôi phục weights tốt nhất không
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_score, model):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Lưu model checkpoint"""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())


class LearningRateScheduler:
    """Custom Learning Rate Scheduler"""
    
    def __init__(self, optimizer, scheduler_type='cosine', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_type == 'cosine_warm':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, **kwargs)
        elif scheduler_type == 'reduce_plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(optimizer, **kwargs)
        else:
            self.scheduler = None
            
    def step(self, val_loss=None):
        if self.scheduler is not None:
            if self.scheduler_type == 'reduce_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
    def get_last_lr(self):
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip_norm=None, processor=None):
    """
    Training cho một epoch
    
    Args:
        model: Model để train
        dataloader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        grad_clip_norm: Gradient clipping norm (None để tắt)
        
    Returns:
        avg_loss, avg_accuracy
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # Move data to device
        pixel_values = batch['pixel_values'].to(device)
        
        # Handle question data - get raw questions and tokenize them
        if 'question' in batch:
            raw_questions = batch['question']
        elif 'questions' in batch:
            raw_questions = batch['questions']
        else:
            raise KeyError("No question field found in batch")
        
        # Handle label data
        if 'label' in batch:
            labels = batch['label'].to(device)
        elif 'labels' in batch:
            labels = batch['labels'].to(device)
        else:
            raise KeyError("No label field found in batch")
        
        # Filter out invalid labels
        valid_mask = labels != -1
        if not valid_mask.any():
            continue
            
        # Zero gradients
        optimizer.zero_grad()
        
        # Tokenize questions using CLIP processor
        if processor is None:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        tokenized_questions = processor.tokenizer(
            raw_questions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        # Forward pass
        logits, attention_weights = model(pixel_values, tokenized_questions)
        
        # Compute loss only on valid samples
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        
        loss = criterion(valid_logits, valid_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        _, preds = torch.max(valid_logits, 1)
        running_corrects += torch.sum(preds == valid_labels).item()
        total_samples += valid_labels.size(0)
        
        # Update progress bar
        current_acc = 100.0 * running_corrects / total_samples if total_samples > 0 else 0
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * running_corrects / total_samples if total_samples > 0 else 0
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, processor=None):
    """
    Validation cho một epoch
    
    Args:
        model: Model để validate
        dataloader: DataLoader
        criterion: Loss function
        device: Device
        
    Returns:
        avg_loss, avg_accuracy
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        
        for batch in progress_bar:
            # Move data to device
            pixel_values = batch['pixel_values'].to(device)
            
            # Handle question data - get raw questions and tokenize them
            if 'question' in batch:
                raw_questions = batch['question']
            elif 'questions' in batch:
                raw_questions = batch['questions']
            else:
                raise KeyError("No question field found in batch")
            
            # Handle label data
            if 'label' in batch:
                labels = batch['label'].to(device)
            elif 'labels' in batch:
                labels = batch['labels'].to(device)
            else:
                raise KeyError("No label field found in batch")
            
            # Filter out invalid labels
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            
            # Tokenize questions using CLIP processor
            if processor is None:
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            tokenized_questions = processor.tokenizer(
                raw_questions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            # Forward pass
            logits, attention_weights = model(pixel_values, tokenized_questions)
            
            # Compute loss only on valid samples
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            
            loss = criterion(valid_logits, valid_labels)
            
            # Statistics
            running_loss += loss.item()
            
            _, preds = torch.max(valid_logits, 1)
            running_corrects += torch.sum(preds == valid_labels).item()
            total_samples += valid_labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * running_corrects / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * running_corrects / total_samples if total_samples > 0 else 0
    
    return epoch_loss, epoch_acc


def train_model_v2(model, train_loader, val_loader, criterion, optimizer, 
                  num_epochs=30, device='cuda', scheduler=None, 
                  early_stopping=None, grad_clip_norm=1.0, 
                  save_path='best_model.pth', log_interval=1, processor=None):
    """
    Training loop nâng cấp với tất cả các tính năng
    
    Args:
        model: Model để train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader  
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Số epochs
        device: Device
        scheduler: Learning rate scheduler
        early_stopping: EarlyStopping instance
        grad_clip_norm: Gradient clipping norm
        save_path: Đường dẫn lưu best model
        log_interval: Interval để log
        
    Returns:
        history: Dict chứa training history
    """
    
    print(f"🚀 Starting training for {num_epochs} epochs")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"🎯 Device: {device}")
    print("=" * 60)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip_norm, processor
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, processor
        )
        
        # Update learning rate
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if scheduler is not None:
            current_lrs = scheduler.get_last_lr()
            history['learning_rates'].append(current_lrs)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch results
        print(f"\n📈 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        if scheduler is not None:
            print(f"  Learning Rates: {current_lrs}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"🎯 New best validation accuracy: {val_acc:.2f}% (saved to {save_path})")
        
        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        print("=" * 60)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"🕐 Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    return history


def evaluate_model_v2(model, dataloader, device, return_predictions=False, processor=None):
    """
    Đánh giá model trên test set
    
    Args:
        model: Model đã train
        dataloader: Test DataLoader
        device: Device
        return_predictions: Có trả về predictions không
        processor: CLIP processor for tokenizing questions
        
    Returns:
        accuracy, (predictions, labels) nếu return_predictions=True
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            # Handle both 'question' and 'questions' keys
            if 'question' in batch:
                raw_questions = batch['question']
            elif 'questions' in batch:
                raw_questions = batch['questions']
            else:
                raise KeyError("Neither 'question' nor 'questions' key found in batch")
            questions = raw_questions
            
            # Handle both 'label' and 'labels' keys
            if 'label' in batch:
                labels = batch['label'].to(device)
            elif 'labels' in batch:
                labels = batch['labels'].to(device)
            else:
                raise KeyError("Neither 'label' nor 'labels' key found in batch")
            
            # Tokenize questions if processor is provided
            if processor is not None:
                tokenized_questions = processor.tokenizer(
                    questions,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to(device)
                questions_input = tokenized_questions
            else:
                questions_input = questions
            
            # Forward pass
            logits, attention_weights = model(pixel_values, questions_input)
            
            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Filter out invalid labels
    valid_mask = all_labels != -1
    if valid_mask.sum() == 0:
        print("⚠️ No valid labels found!")
        return 0.0
    
    valid_predictions = all_predictions[valid_mask]
    valid_labels = all_labels[valid_mask]
    valid_confidences = all_confidences[valid_mask]
    
    # Calculate accuracy
    accuracy = (valid_predictions == valid_labels).mean() * 100
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Valid samples: {valid_mask.sum()}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average confidence: {valid_confidences.mean():.3f}")
    
    if return_predictions:
        return accuracy, (valid_predictions, valid_labels, valid_confidences)
    else:
        return accuracy


def plot_training_history(history, save_path=None):
    """
    Vẽ biểu đồ training history
    
    Args:
        history: Training history dict
        save_path: Đường dẫn lưu plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy') 
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history and history['learning_rates']:
        lrs = [lr[0] if isinstance(lr, list) else lr for lr in history['learning_rates']]
        axes[1, 0].plot(epochs, lrs, 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Performance summary
    axes[1, 1].axis('off')
    
    # Find best metrics
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    final_val_acc = history['val_acc'][-1]
    
    summary_text = f"""
Training Summary:
────────────────────
Total Epochs: {len(epochs)}
Best Val Accuracy: {best_val_acc:.2f}%
Best Epoch: {best_val_epoch}
Final Val Accuracy: {final_val_acc:.2f}%

Final Losses:
Train: {history['train_loss'][-1]:.4f}
Val: {history['val_loss'][-1]:.4f}
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('VQA CLIP v2 Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to: {save_path}")
    
    plt.show()


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model từ checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Đường dẫn checkpoint
        device: Device
        
    Returns:
        model, epoch, metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0)
    val_loss = checkpoint.get('val_loss', 0)
    
    print(f"✅ Loaded checkpoint from epoch {epoch}")
    print(f"📊 Checkpoint metrics - Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")
    
    return model, epoch, {'val_acc': val_acc, 'val_loss': val_loss}
