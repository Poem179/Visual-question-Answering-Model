import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


def train_model(model, train_loader, val_loader, num_epochs=15, 
                learning_rate=1e-4, device='cuda', save_dir='checkpoints'):
    """
    Training function for VQA model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=3)
    
    # Setup tensorboard
    writer = SummaryWriter('runs/vqa_training')
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            questions = batch['questions']
            labels = batch['labels'].to(device)
            
            # Skip samples with invalid labels
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
                
            pixel_values = pixel_values[valid_mask]
            questions = [questions[i] for i in range(len(questions)) if valid_mask[i]]
            labels = labels[valid_mask]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(pixel_values, questions)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * train_correct / train_total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save metrics
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': epoch_train_acc
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"New best model saved with val acc: {val_acc:.2f}%")
    
    writer.close()
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluation function
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            questions = batch['questions']
            labels = batch['labels'].to(device)
            
            # Skip samples with invalid labels
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
                
            pixel_values = pixel_values[valid_mask]
            questions = [questions[i] for i in range(len(questions)) if valid_mask[i]]
            labels = labels[valid_mask]
            
            # Forward pass
            logits, _ = model(pixel_values, questions)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def plot_training_history(history):
    """
    Plot training curves
    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accs'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_accs'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def load_best_model(model, checkpoint_path):
    """
    Load best model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val acc: {checkpoint['val_acc']:.2f}%")
    return model
