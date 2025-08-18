import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import requests
from io import BytesIO


def visualize_attention(image, attention_weights, save_path=None):
    """
    Visualize attention weights on image
    """
    if isinstance(image, str):
        if image.startswith('http'):
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image).convert('RGB')
    
    # Resize attention weights to match image patches
    # Handle dynamic patch size based on actual attention shape
    total_patches = attention_weights.shape[0]
    patch_size = int(np.sqrt(total_patches))
    
    if total_patches == patch_size * patch_size:
        attention_map = attention_weights.reshape(patch_size, patch_size)
    else:
        # Fallback: pad or truncate to make it square
        print(f"Warning: {total_patches} patches doesn't form perfect square, using closest square")
        target_patches = 49  # 7x7 for ViT-B/32
        if total_patches > target_patches:
            attention_weights = attention_weights[:target_patches]
        elif total_patches < target_patches:
            attention_weights = np.pad(attention_weights, (0, target_patches - total_patches))
        attention_map = attention_weights.reshape(7, 7)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention heatmap
    im = ax2.imshow(attention_map, cmap='hot', interpolation='bilinear')
    ax2.set_title('Attention Weights')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # Overlay
    ax3.imshow(image, alpha=0.7)
    ax3.imshow(attention_map, cmap='hot', alpha=0.5, interpolation='bilinear')
    ax3.set_title('Attention Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    """
    # Limit to top classes for better visualization
    top_n = min(20, len(class_names))
    
    # Get top predicted classes
    unique_pred = np.unique(y_pred)
    pred_counts = [(cls, np.sum(y_pred == cls)) for cls in unique_pred]
    pred_counts.sort(key=lambda x: x[1], reverse=True)
    top_classes = [cls for cls, _ in pred_counts[:top_n]]
    
    # Filter data
    mask = np.isin(y_pred, top_classes) | np.isin(y_true, top_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names.get(i, f'Class_{i}') for i in top_classes],
                yticklabels=[class_names.get(i, f'Class_{i}') for i in top_classes])
    plt.title(f'Confusion Matrix (Top {top_n} Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_confidence(confidences, is_correct, save_path=None):
    """
    Plot prediction confidence distribution
    """
    correct_conf = confidences[is_correct]
    incorrect_conf = confidences[~is_correct]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_predictions(model, data_loader, idx_to_answer, device, num_samples=100):
    """
    Analyze model predictions and return statistics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_questions = []
    all_image_paths = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break
                
            pixel_values = batch['pixel_values'].to(device)
            questions = batch['questions']
            labels = batch['labels'].to(device)
            image_paths = batch['image_paths']
            
            # Skip invalid labels
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            
            pixel_values = pixel_values[valid_mask]
            questions = [questions[i] for i in range(len(questions)) if valid_mask[i]]
            labels = labels[valid_mask]
            image_paths = [image_paths[i] for i in range(len(image_paths)) if valid_mask[i]]
            
            # Forward pass
            logits, _ = model(pixel_values, questions)
            probs = torch.softmax(logits, dim=-1)
            
            # Get predictions and confidences
            max_probs, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())
            all_questions.extend(questions)
            all_image_paths.extend(image_paths)
            
            sample_count += len(labels)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Calculate accuracy
    accuracy = np.mean(all_predictions == all_labels)
    
    # Print analysis
    print(f"Analysis Results (n={len(all_predictions)}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {np.mean(all_confidences):.4f}")
    print(f"Confidence Std: {np.std(all_confidences):.4f}")
    
    # Show some examples
    print("\nSample Predictions:")
    for i in range(min(5, len(all_predictions))):
        pred_answer = idx_to_answer.get(all_predictions[i], "unknown")
        true_answer = idx_to_answer.get(all_labels[i], "unknown")
        correct = "✓" if all_predictions[i] == all_labels[i] else "✗"
        
        print(f"{correct} Q: {all_questions[i]}")
        print(f"   Predicted: {pred_answer} (conf: {all_confidences[i]:.3f})")
        print(f"   True: {true_answer}")
        print()
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'questions': all_questions,
        'image_paths': all_image_paths,
        'accuracy': accuracy
    }


def demo_predictions(model, processor, idx_to_answer, device, image_urls, questions):
    """
    Demo predictions on images from URLs
    """
    model.eval()
    
    print("Demo Predictions:")
    print("=" * 50)
    
    for i, (url, question) in enumerate(zip(image_urls, questions)):
        try:
            # Load image from URL
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Process image
            image_inputs = processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].to(device)
            
            # Predict
            with torch.no_grad():
                logits, attention_weights = model(pixel_values, [question])
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(logits, dim=-1).item()
                confidence = probs[0, pred_idx].item()
                
                pred_answer = idx_to_answer.get(pred_idx, "unknown")
            
            print(f"Example {i+1}:")
            print(f"Question: {question}")
            print(f"Predicted Answer: {pred_answer}")
            print(f"Confidence: {confidence:.3f}")
            
            # Visualize
            plt.figure(figsize=(12, 4))
            
            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Q: {question}\nA: {pred_answer} ({confidence:.3f})")
            plt.axis('off')
            
            # Show attention
            plt.subplot(1, 2, 2)
            # Fix: Get correct attention shape dynamically
            attention_flat = attention_weights[0].cpu().numpy()
            patch_size = int(np.sqrt(len(attention_flat)))
            
            if len(attention_flat) == patch_size * patch_size:
                attention_map = attention_flat.reshape(patch_size, patch_size)
            else:
                # Fallback: use available data
                print(f"Warning: attention shape {len(attention_flat)} doesn't form perfect square")
                # Pad or truncate to nearest square
                target_size = 7 * 7  # Default for ViT-B/32
                if len(attention_flat) > target_size:
                    attention_flat = attention_flat[:target_size]
                elif len(attention_flat) < target_size:
                    attention_flat = np.pad(attention_flat, (0, target_size - len(attention_flat)))
                attention_map = attention_flat.reshape(7, 7)
            
            plt.imshow(attention_map, cmap='hot', interpolation='bilinear')
            plt.title('Attention Weights')
            plt.axis('off')
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()
            
            print("-" * 30)
            
        except Exception as e:
            print(f"Error processing example {i+1}: {e}")
            continue
