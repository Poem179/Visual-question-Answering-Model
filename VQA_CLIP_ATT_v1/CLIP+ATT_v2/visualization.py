"""
Visualization utilities cho VQA CLIP v2
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from collections import Counter
import cv2


def visualize_attention_v2(image, attention_weights, question, prediction, confidence, 
                          save_path=None, title=None):
    """
    Trực quan hóa attention weights trên hình ảnh
    
    Args:
        image: PIL Image hoặc numpy array
        attention_weights: Tensor attention weights [num_patches]
        question: Câu hỏi
        prediction: Câu trả lời dự đoán
        confidence: Độ tin cậy
        save_path: Đường dẫn lưu
        title: Tiêu đề tùy chỉnh
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert image nếu cần
    if isinstance(image, torch.Tensor):
        # Unnormalize nếu đã được normalize
        image = image.cpu().numpy().transpose(1, 2, 0)
        # Clip giá trị
        image = np.clip(image, 0, 1)
    elif not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Đảm bảo image có giá trị từ 0-1
    if image.max() > 1:
        image = image / 255.0
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    attention_np = attention_weights.cpu().numpy() if isinstance(attention_weights, torch.Tensor) else attention_weights

    # Always force attention to 1D vector of length 49 (for ViT-B/32)
    target_size = 49
    if np.isscalar(attention_np) or np.ndim(attention_np) == 0:
        attention_np = np.ones(target_size) * float(attention_np)
    elif np.ndim(attention_np) == 1:
        if len(attention_np) == 1:
            attention_np = np.ones(target_size) * float(attention_np[0])
        elif len(attention_np) != target_size:
            # Pad or truncate
            if len(attention_np) > target_size:
                attention_np = attention_np[:target_size]
            else:
                attention_np = np.pad(attention_np, (0, target_size - len(attention_np)), constant_values=0)
    else:
        # Flatten any higher-dim to 1D and pad/truncate
        attention_np = np.array(attention_np).flatten()
        if len(attention_np) > target_size:
            attention_np = attention_np[:target_size]
        else:
            attention_np = np.pad(attention_np, (0, target_size - len(attention_np)), constant_values=0)

    grid_size = 7
    attention_map = attention_np.reshape(grid_size, grid_size)
    
    # Resize attention map để match image size
    attention_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
    
    axes[1].imshow(attention_resized, cmap='hot', alpha=0.8)
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_resized, cmap='hot', alpha=0.4)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    # Add text info
    info_text = f"Q: {question}\nA: {prediction}\nConfidence: {confidence:.3f}"
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def demo_model_predictions(model, processor, image_urls, questions, 
                          idx_to_answer, device, save_dir=None):
    """
    Demo model với các hình ảnh từ URLs
    
    Args:
        model: Trained model
        processor: CLIP processor
        image_urls: List URLs hình ảnh
        questions: List câu hỏi tương ứng
        idx_to_answer: Mapping từ index sang answer
        device: Device
        save_dir: Thư mục lưu kết quả
    """
    model.eval()
    
    print("🎯 Running Model Demo")
    print("=" * 50)
    
    for i, (url, question) in enumerate(zip(image_urls, questions)):
        try:
            print(f"\n📸 Example {i+1}")
            print(f"Question: {question}")
            
            # Load image from URL
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                continue
            
            # Process image
            image_inputs = processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].to(device)
            
            # Tokenize question
            tokenized_question = processor.tokenizer(
                [question],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)
            
            # Predict
            with torch.no_grad():
                logits, attention_weights = model(pixel_values, tokenized_question)
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(logits, dim=-1).item()
                confidence = probs[0, pred_idx].item()
                
                pred_answer = idx_to_answer.get(pred_idx, "unknown")
            
            print(f"Predicted Answer: {pred_answer}")
            print(f"Confidence: {confidence:.3f}")
            
            # Visualize
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/demo_example_{i+1}.png"
            
            visualize_attention_v2(
                image=image,
                attention_weights=attention_weights[0],
                question=question,
                prediction=pred_answer,
                confidence=confidence,
                save_path=save_path,
                title=f"Demo Example {i+1}"
            )
            
        except Exception as e:
            print(f"❌ Error processing example {i+1}: {e}")
            continue
    
    print("\n✅ Demo completed!")


def analyze_predictions_v2(model, dataloader, idx_to_answer, device, 
                          num_samples=500, save_dir=None, processor=None):
    """
    Phân tích chi tiết predictions của model
    
    Args:
        model: Trained model
        dataloader: DataLoader
        idx_to_answer: Mapping từ index sang answer
        device: Device
        num_samples: Number of samples to analyze
        save_dir: Directory to save results
        processor: CLIP processor for tokenizing questions
        num_samples: Số samples để phân tích
        save_dir: Thư mục lưu kết quả
        
    Returns:
        analysis_results: Dict chứa kết quả phân tích
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_questions = []
    sample_count = 0
    
    print(f"🔍 Analyzing {num_samples} predictions...")
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
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
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            # Store results
            batch_size = len(questions)
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                all_predictions.append(predictions[i].item())
                all_labels.append(labels[i].item())
                all_confidences.append(confidences[i].item())
                all_questions.append(questions[i])
                sample_count += 1
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Filter valid samples
    valid_mask = all_labels != -1
    valid_predictions = all_predictions[valid_mask]
    valid_labels = all_labels[valid_mask]
    valid_confidences = all_confidences[valid_mask]
    valid_questions = [q for i, q in enumerate(all_questions) if valid_mask[i]]
    
    # Calculate metrics
    correct_mask = valid_predictions == valid_labels
    accuracy = correct_mask.mean() * 100
    
    print(f"📊 Analysis Results:")
    print(f"  Samples analyzed: {len(valid_predictions)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average confidence: {valid_confidences.mean():.3f}")
    print(f"  Confidence std: {valid_confidences.std():.3f}")
    
    # Confidence analysis
    correct_confidences = valid_confidences[correct_mask]
    incorrect_confidences = valid_confidences[~correct_mask]
    
    print(f"  Correct predictions confidence: {correct_confidences.mean():.3f} ± {correct_confidences.std():.3f}")
    print(f"  Incorrect predictions confidence: {incorrect_confidences.mean():.3f} ± {incorrect_confidences.std():.3f}")
    
    # Visualization
    if save_dir:
        plot_confidence_analysis(correct_confidences, incorrect_confidences, 
                                f"{save_dir}/confidence_analysis.png")
        plot_prediction_distribution(valid_predictions, valid_labels, idx_to_answer,
                                   f"{save_dir}/prediction_distribution.png")
    
    # Top correct và incorrect predictions
    print(f"\n🎯 Sample Correct Predictions:")
    correct_indices = np.where(correct_mask)[0]
    if len(correct_indices) > 0:
        # Sort by confidence
        sorted_correct = sorted(correct_indices, 
                              key=lambda i: valid_confidences[i], reverse=True)
        
        for i in sorted_correct[:5]:
            pred_answer = idx_to_answer.get(valid_predictions[i], "unknown")
            print(f"  Q: {valid_questions[i][:50]}...")
            print(f"  A: {pred_answer} (confidence: {valid_confidences[i]:.3f})")
            print()
    
    print(f"❌ Sample Incorrect Predictions:")
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        # Sort by confidence (high confidence mistakes)
        sorted_incorrect = sorted(incorrect_indices,
                                key=lambda i: valid_confidences[i], reverse=True)
        
        for i in sorted_incorrect[:5]:
            pred_answer = idx_to_answer.get(valid_predictions[i], "unknown")
            true_answer = idx_to_answer.get(valid_labels[i], "unknown")
            print(f"  Q: {valid_questions[i][:50]}...")
            print(f"  Predicted: {pred_answer}, True: {true_answer} (confidence: {valid_confidences[i]:.3f})")
            print()
    
    results = {
        'predictions': valid_predictions,
        'labels': valid_labels,
        'confidences': valid_confidences,
        'questions': valid_questions,
        'accuracy': accuracy,
        'correct_mask': correct_mask
    }
    
    return results


def plot_confidence_analysis(correct_confidences, incorrect_confidences, save_path):
    """Plot confidence distribution analysis"""
    plt.figure(figsize=(12, 8))
    
    # Confidence histograms
    plt.subplot(2, 2, 1)
    plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    data = [correct_confidences, incorrect_confidences]
    labels = ['Correct', 'Incorrect']
    plt.boxplot(data, labels=labels)
    plt.ylabel('Confidence Score')
    plt.title('Confidence Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Accuracy vs confidence bins
    plt.subplot(2, 2, 3)
    all_confidences = np.concatenate([correct_confidences, incorrect_confidences])
    all_correct = np.concatenate([np.ones(len(correct_confidences)), 
                                 np.zeros(len(incorrect_confidences))])
    
    # Bin by confidence
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    
    for i in range(len(bins)-1):
        mask = (all_confidences >= bins[i]) & (all_confidences < bins[i+1])
        if mask.sum() > 0:
            acc = all_correct[mask].mean()
        else:
            acc = 0
        accuracies.append(acc)
    
    plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Confidence Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Confidence')
    plt.grid(True, alpha=0.3)
    
    # Statistics summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
Confidence Statistics:
────────────────────
Correct Predictions:
  Mean: {correct_confidences.mean():.3f}
  Std:  {correct_confidences.std():.3f}
  
Incorrect Predictions:
  Mean: {incorrect_confidences.mean():.3f}
  Std:  {incorrect_confidences.std():.3f}
  
Difference:
  Mean: {correct_confidences.mean() - incorrect_confidences.mean():.3f}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Confidence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_distribution(predictions, labels, idx_to_answer, save_path):
    """Plot prediction distribution"""
    # Top predicted classes
    pred_counter = Counter(predictions)
    top_preds = pred_counter.most_common(20)
    
    # Top true classes
    valid_labels = labels[labels != -1]
    label_counter = Counter(valid_labels)
    top_labels = label_counter.most_common(20)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top predictions
    pred_classes = [idx_to_answer.get(idx, f"Class_{idx}") for idx, _ in top_preds]
    pred_counts = [count for _, count in top_preds]
    
    axes[0].bar(range(len(pred_classes)), pred_counts)
    axes[0].set_title('Top 20 Predicted Classes')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(range(len(pred_classes)))
    axes[0].set_xticklabels(pred_classes, rotation=45, ha='right')
    
    # Top true labels
    label_classes = [idx_to_answer.get(idx, f"Class_{idx}") for idx, _ in top_labels]
    label_counts = [count for _, count in top_labels]
    
    axes[1].bar(range(len(label_classes)), label_counts, color='orange')
    axes[1].set_title('Top 20 True Classes')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(range(len(label_classes)))
    axes[1].set_xticklabels(label_classes, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_sample_demo_urls():
    """Tạo sample demo URLs và questions"""
    demo_urls = [
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",  # Cat
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",    # Dog
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=400",  # Food
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Mountain
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",  # City
    ]
    
    demo_questions = [
        "What animal is this?",
        "What color is the dog?",
        "What type of food is shown?", 
        "What is in the background?",
        "Is this a city or countryside?"
    ]
    
    return demo_urls, demo_questions
