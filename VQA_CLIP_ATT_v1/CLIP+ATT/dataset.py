import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
import requests
from io import BytesIO


def safe_load_json(json_file):
    """Safely load JSON file with multiple encoding attempts"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"  Trying encoding: {encoding}")
            with open(json_file, 'r', encoding=encoding) as f:
                data = json.load(f)
            print(f"  ✓ Successfully loaded with {encoding}")
            return data
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  Error with {encoding}: {e}")
            continue
    
    raise ValueError(f"Could not load {json_file} with any encoding")


class VQADataset(Dataset):
    def __init__(self, json_file, image_dir, processor, max_answers=1000):
        print(f"Loading dataset from: {json_file}")
        self.data = safe_load_json(json_file)
        
        self.image_dir = image_dir
        self.processor = processor
        
        # Build answer vocabulary
        self.answer_to_idx, self.idx_to_answer = self.build_answer_vocab(max_answers)
        self.num_classes = len(self.answer_to_idx)
        
    def build_answer_vocab(self, max_answers):
        """Build answer vocabulary from most frequent answers"""
        answer_counts = {}
        
        for item in self.data:
            if 'answers' in item:
                for answer_obj in item['answers']:
                    answer = answer_obj['answer'].lower().strip()
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Get most frequent answers
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        top_answers = [answer for answer, count in sorted_answers[:max_answers]]
        
        # Create mappings
        answer_to_idx = {answer: idx for idx, answer in enumerate(top_answers)}
        idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
        
        return answer_to_idx, idx_to_answer
    
    def get_most_frequent_answer(self, answers):
        """Get most frequent answer from answer list"""
        answer_counts = {}
        for answer_obj in answers:
            answer = answer_obj['answer'].lower().strip()
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        most_frequent = max(answer_counts.items(), key=lambda x: x[1])[0]
        return most_frequent
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Create dummy image if file not found
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Get question
        question = item['question']
        
        # Get answer label
        if 'answers' in item and len(item['answers']) > 0:
            answer = self.get_most_frequent_answer(item['answers'])
            label = self.answer_to_idx.get(answer, -1)  # -1 for unknown answers
        else:
            label = -1
        
        return {
            'pixel_values': pixel_values,
            'question': question,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': image_path
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    questions = [item['question'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'questions': questions,
        'labels': labels,
        'image_paths': image_paths
    }


def create_dataloaders(train_json, val_json, test_json, 
                      train_dir, val_dir, test_dir, 
                      processor, batch_size=16, max_answers=1000):
    """Create train, validation and test dataloaders"""
    
    try:
        # Create datasets
        print("Creating training dataset...")
        train_dataset = VQADataset(train_json, train_dir, processor, max_answers)
        
        print("Creating validation dataset...")
        val_dataset = VQADataset(val_json, val_dir, processor, max_answers)
        
        print("Creating test dataset...")
        test_dataset = VQADataset(test_json, test_dir, processor, max_answers)
        
        # Align vocabularies
        print("Aligning vocabularies...")
        val_dataset.answer_to_idx = train_dataset.answer_to_idx
        val_dataset.idx_to_answer = train_dataset.idx_to_answer
        val_dataset.num_classes = train_dataset.num_classes
        
        test_dataset.answer_to_idx = train_dataset.answer_to_idx
        test_dataset.idx_to_answer = train_dataset.idx_to_answer
        test_dataset.num_classes = train_dataset.num_classes
        
        # Create dataloaders
        print("Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        return train_loader, val_loader, test_loader, train_dataset.num_classes
        
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        print("Try opening the JSON file with a different text editor and save it as UTF-8.")
        raise
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise


def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except:
        return None


def predict_single_image(model, processor, image, question, answer_vocab, device):
    """Predict answer for single image and question"""
    model.eval()
    
    with torch.no_grad():
        # Process image
        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(device)
        
        # Forward pass
        logits, attention_weights = model(pixel_values, [question])
        
        # Get prediction
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred_idx].item()
        
        pred_answer = answer_vocab.get(pred_idx, "unknown")
        
        return pred_answer, confidence, attention_weights.cpu().numpy()
