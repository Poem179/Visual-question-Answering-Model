"""
Utility script to extract answer vocabulary from VQA model checkpoint
Run this once to create idx_to_answer.json file
"""
import torch
import json

def extract_vocab_from_checkpoint(checkpoint_path, output_path='idx_to_answer.json'):
    """
    Extract answer vocabulary from model checkpoint if available
    """
    try:
        # Load with weights_only=False for PyTorch 2.6+ compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to find vocabulary in checkpoint
        vocab = None
        
        if isinstance(checkpoint, dict):
            # Common keys where vocab might be stored
            possible_keys = [
                'idx_to_answer', 
                'answer_vocab', 
                'vocab',
                'answer_to_idx',
                'class_names'
            ]
            
            for key in possible_keys:
                if key in checkpoint:
                    vocab = checkpoint[key]
                    print(f"✅ Found vocabulary under key: {key}")
                    break
            
            # If answer_to_idx is found, invert it to get idx_to_answer
            if vocab and 'answer_to_idx' in str(possible_keys):
                vocab = {v: k for k, v in vocab.items()}
        
        if vocab:
            # Convert keys to strings for JSON serialization
            vocab_json = {str(k): str(v) for k, v in vocab.items()}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_json, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Vocabulary saved to: {output_path}")
            print(f"📊 Total answers: {len(vocab_json)}")
            
            # Show sample
            print("\n📋 Sample answers:")
            for i, (idx, answer) in enumerate(list(vocab_json.items())[:10]):
                print(f"   {idx}: {answer}")
            
            return vocab_json
        else:
            print("⚠️ Vocabulary not found in checkpoint")
            print("💡 Available keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "N/A")
            
            # Create default vocabulary
            print("\n🔄 Creating default vocabulary...")
            default_vocab = create_default_vocabulary()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(default_vocab, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Default vocabulary saved to: {output_path}")
            return default_vocab
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def create_default_vocabulary():
    """
    Create a default vocabulary based on common VizWiz answers
    This is a fallback if vocabulary is not found in checkpoint
    """
    # Common VizWiz VQA answers (you should replace this with actual vocabulary)
    common_answers = [
        "unanswerable",
        "yes",
        "no",
        "white",
        "black",
        "blue",
        "red",
        "green",
        "yellow",
        "brown",
        "orange",
        "purple",
        "pink",
        "gray",
        "grey",
        "silver",
        "gold",
        "tan",
        "beige",
        "1",
        "2",
        "3",
        "4",
        "5",
        "0",
        "6",
        "7",
        "8",
        "9",
        "10",
        "shirt",
        "can",
        "bottle",
        "box",
        "bag",
        "cup",
        "glass",
        "plate",
        "bowl",
        "phone",
        "remote",
        "book",
        "pen",
        "paper",
        "card",
        "label",
        "button",
        "door",
        "window",
        "table",
        "chair",
        "bed",
        "couch",
        "floor",
        "wall",
        "ceiling",
    ]
    
    # Create idx to answer mapping
    vocab = {str(i): answer for i, answer in enumerate(common_answers)}
    
    # Fill up to 1000 (approximate size used in training)
    for i in range(len(common_answers), 1000):
        vocab[str(i)] = f"answer_{i}"
    
    return vocab


def load_vocabulary(vocab_path='idx_to_answer.json'):
    """
    Load vocabulary from JSON file
    """
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Convert string keys back to integers
        vocab = {int(k): v for k, v in vocab.items()}
        
        print(f"✅ Vocabulary loaded: {len(vocab)} answers")
        return vocab
    except FileNotFoundError:
        print(f"⚠️ Vocabulary file not found: {vocab_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    print("🔍 VQA Vocabulary Extractor")
    print("=" * 50)
    
    # Check if checkpoint path is provided
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "../models/vqa.pth"  # Default path relative to app directory
    
    print(f"📂 Checkpoint: {checkpoint_path}")
    
    # Extract vocabulary
    vocab = extract_vocab_from_checkpoint(checkpoint_path)
    
    if vocab:
        print("\n✅ Success!")
        print("💡 You can now use this vocabulary in your app")
        print("💡 Update load_answer_vocab() in app.py to load from idx_to_answer.json")
    else:
        print("\n⚠️ Failed to extract vocabulary")
        print("💡 You may need to manually create the vocabulary mapping")
