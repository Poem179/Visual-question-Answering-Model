"""
VQA Application with Streamlit
Pipeline: VQA Model -> If unanswerable -> Flaw Detection Model -> Gemini Analysis
Features: Vietnamese/English input, Translation, Gemini Form
"""
import os
import warnings

# Suppress warnings before importing other libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor
import numpy as np
import matplotlib.pyplot as plt
from model_v2 import VQAModelFromCheckpoint, FlawClassifier
from attention_viz import visualize_attention_summary

# Gemini AI Analyzer
try:
    from gemini_analyzer import create_gemini_analyzer, parse_gemini_response
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Google Translate for Vietnamese to English
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    try:
        # Fallback to googletrans
        from googletrans import Translator as GoogleTranslator
        TRANSLATOR_AVAILABLE = True
    except ImportError:
        TRANSLATOR_AVAILABLE = False

# =====================
# CONFIGURATION
# =====================

# Fixed Gemini API Key (thay YOUR_API_KEY_HERE bằng API key thật của bạn)
GEMINI_API_KEY = ""

# Model paths
VQA_MODEL_PATH = "../models/vqa.pth"
FLAW_MODEL_PATH = "../models/unanswerable.pth"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Flaw labels
FLAW_LABELS = ["FRM", "BLR", "DRK", "BRT", "OBS", "OTH", "NON", "ROT"]
FLAW_NAME_MAP = {
    "FRM": "Object_out_of_frame", "BLR": "Blur", "DRK": "Too_dark",
    "BRT": "Too_bright", "OBS": "Obstruction", "OTH": "Other",
    "NON": "No_issue", "ROT": "Rotated"
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================
# MODEL LOADING (Silent)
# =====================

@st.cache_resource
def load_vqa_model():
    """Load VQA model silently"""
    try:
        checkpoint = torch.load(VQA_MODEL_PATH, map_location=device, weights_only=False)
        config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        
        # Get state dict first to detect num_classes
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Auto-detect num_classes from classifier layer shape
        num_classes = 500  # default
        for key in state_dict.keys():
            if 'classifier' in key and 'weight' in key:
                # Get the output dimension (first dim of weight matrix)
                if state_dict[key].dim() == 2:
                    num_classes = state_dict[key].shape[0]
                    break
        
        # Also check idx_to_answer if available
        if isinstance(checkpoint, dict) and 'idx_to_answer' in checkpoint:
            num_classes = max(num_classes, len(checkpoint['idx_to_answer']))
        
        model = VQAModelFromCheckpoint(
            num_classes=config.get('num_classes', num_classes),
            clip_model_name=config.get('clip_model_name', CLIP_MODEL_NAME),
            dropout_rate=config.get('dropout_rate', 0.3)
        ).to(device)
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        return None


@st.cache_resource
def load_flaw_model():
    """Load Flaw Detection model silently"""
    try:
        checkpoint = torch.load(FLAW_MODEL_PATH, map_location=device, weights_only=False)
        use_dropout = False
        if isinstance(checkpoint, dict):
            if 'model.fc.0.weight' in checkpoint or ('state_dict' in checkpoint and 'model.fc.0.weight' in checkpoint.get('state_dict', {})):
                use_dropout = True
        
        model = FlawClassifier(num_classes=8, use_dropout=use_dropout).to(device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        return model
    except:
        return None


@st.cache_resource
def load_processor():
    """Load CLIP processor silently"""
    try:
        return CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    except:
        return None


@st.cache_resource
def load_answer_vocab():
    """Load answer vocabulary"""
    import json
    vocab_path = 'idx_to_answer.json'
    try:
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            return {int(k): v for k, v in vocab.items()}
    except:
        pass
    return {0: "unanswerable", 1: "yes", 2: "no"}


@st.cache_resource
def load_gemini():
    """Load Gemini analyzer with fixed API key"""
    if GEMINI_AVAILABLE and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
        try:
            return create_gemini_analyzer(api_key=GEMINI_API_KEY)
        except:
            return None
    return None


# =====================
# TRANSLATION FUNCTIONS
# =====================

def translate_vi_to_en(text, gemini_analyzer=None):
    """
    Translate Vietnamese text to English
    Uses deep-translator or Gemini as fallback
    """
    if not text or text.strip() == "":
        return text
    
    # Check if text is already in English (simple heuristic)
    vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
    has_vietnamese = any(c.lower() in vietnamese_chars for c in text)
    
    if not has_vietnamese:
        return text  # Already in English or no Vietnamese characters
    
    # Try deep-translator first
    if TRANSLATOR_AVAILABLE:
        try:
            # deep-translator syntax
            translator = GoogleTranslator(source='vi', target='en')
            translated = translator.translate(text)
            if translated:
                return translated
        except Exception as e1:
            try:
                # googletrans syntax (fallback)
                translator = GoogleTranslator()
                result = translator.translate(text, src='vi', dest='en')
                if result and hasattr(result, 'text'):
                    return result.text
            except Exception as e2:
                pass
    
    # Use Gemini as final fallback
    if gemini_analyzer:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Translate the following Vietnamese question to English. Only return the translated text, nothing else:\n\n{text}"
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except:
            pass
    
    return text  # Return original if translation fails


def translate_answer_to_vi(answer, gemini_analyzer=None):
    """
    Translate English answer to Vietnamese for display
    """
    if not answer or answer.strip() == "":
        return answer
    
    # Common VQA answers translation dictionary
    common_translations = {
        'yes': 'có',
        'no': 'không', 
        'unanswerable': 'không thể trả lời',
        'unknown': 'không xác định',
        'none': 'không có',
        'left': 'trái',
        'right': 'phải',
        'up': 'trên',
        'down': 'dưới',
        'red': 'đỏ',
        'blue': 'xanh dương',
        'green': 'xanh lá',
        'yellow': 'vàng',
        'white': 'trắng',
        'black': 'đen',
        'orange': 'cam',
        'pink': 'hồng',
        'purple': 'tím',
        'brown': 'nâu',
        'gray': 'xám',
        'grey': 'xám',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'man': 'người đàn ông',
        'woman': 'người phụ nữ',
        'boy': 'con trai',
        'girl': 'con gái',
        'dog': 'con chó',
        'cat': 'con mèo',
        'car': 'xe hơi',
        'bus': 'xe buýt',
        'tree': 'cây',
        'water': 'nước',
        'food': 'thức ăn',
        'sky': 'bầu trời',
        'grass': 'cỏ',
        'building': 'tòa nhà',
        'street': 'đường phố',
        'beach': 'bãi biển',
        'mountain': 'núi',
        'river': 'sông',
        'sun': 'mặt trời',
        'moon': 'mặt trăng',
        'cloud': 'mây',
        'rain': 'mưa',
        'snow': 'tuyết',
    }
    
    answer_lower = answer.lower().strip()
    if answer_lower in common_translations:
        return common_translations[answer_lower]
    
    # Try number detection
    if answer_lower.isdigit():
        return answer
    
    # Use Gemini for complex answers
    if gemini_analyzer and len(answer) > 2:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Translate this English word/phrase to Vietnamese. Only return the translation, nothing else:\n\n{answer}"
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except:
            pass
    
    return answer  # Return original if translation fails


def ask_gemini_direct(image, question, language="vi"):
    """
    Ask Gemini directly about an image without using VQA model
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if language == "vi":
            prompt = f"""Hãy trả lời câu hỏi sau về hình ảnh được cung cấp bằng tiếng Việt.
Trả lời ngắn gọn, súc tích và chính xác.

Câu hỏi: {question}

Trả lời:"""
        else:
            prompt = f"""Please answer the following question about the provided image in English.
Be concise, accurate and to the point.

Question: {question}

Answer:"""
        
        response = model.generate_content([prompt, image])
        if response and response.text:
            return response.text.strip()
        return "Không thể trả lời" if language == "vi" else "Unable to answer"
    except Exception as e:
        return f"Lỗi: {str(e)}" if language == "vi" else f"Error: {str(e)}"


# =====================
# PREDICTION FUNCTIONS
# =====================

def preprocess_image_for_vqa(image, processor):
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(device)


def preprocess_image_for_flaw(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def predict_vqa(model, processor, image, question, return_attention=False):
    """Predict answer using VQA model"""
    try:
        image_tensor = preprocess_image_for_vqa(image, processor)
        question_tokens = processor.tokenizer(
            question, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        question_tokens = {k: v.to(device) for k, v in question_tokens.items()}
        
        with torch.no_grad():
            if return_attention:
                logits, attention_weights = model(
                    pixel_values=image_tensor,
                    input_ids=question_tokens['input_ids'],
                    attention_mask=question_tokens['attention_mask'],
                    return_attention=True
                )
            else:
                logits = model(
                    pixel_values=image_tensor,
                    input_ids=question_tokens['input_ids'],
                    attention_mask=question_tokens['attention_mask']
                )
            
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)
        
        idx_to_answer = load_answer_vocab()
        predicted_answer = idx_to_answer.get(predicted_idx.item(), "unknown")
        confidence_score = confidence.item()
        is_unanswerable = predicted_answer.lower() == "unanswerable"
        
        if return_attention:
            return predicted_answer, confidence_score, is_unanswerable, attention_weights
        return predicted_answer, confidence_score, is_unanswerable
    except Exception as e:
        if return_attention:
            return None, 0.0, False, None
        return None, 0.0, False


def predict_flaws(model, image, threshold=0.5):
    """Predict image quality flaws"""
    try:
        image_tensor = preprocess_image_for_flaw(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            preds = (probs > threshold).astype(int)
        
        detected_flaws = []
        for label, pred, prob in zip(FLAW_LABELS, preds, probs):
            if pred == 1:
                detected_flaws.append({
                    'label': label,
                    'name': FLAW_NAME_MAP[label],
                    'confidence': prob
                })
        return detected_flaws
    except:
        return []


# =====================
# MAIN APPLICATION
# =====================

def main():
    st.set_page_config(
        page_title="VQA Application",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Load all models silently
    vqa_model = load_vqa_model()
    flaw_model = load_flaw_model()
    processor = load_processor()
    gemini_analyzer = load_gemini()
    
    # =====================
    # HEADER
    # =====================
    st.title("🖼️ Visual Question Answering")
    
    # =====================
    # SIDEBAR - Settings
    # =====================
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Language settings
        st.subheader("🌐 Ngôn ngữ")
        input_language = st.selectbox(
            "Ngôn ngữ nhập câu hỏi",
            options=["vi", "en"],
            index=0,  # Default to Vietnamese
            format_func=lambda x: "🇻🇳 Tiếng Việt" if x == "vi" else "🇬🇧 English"
        )
        
        gemini_language = st.selectbox(
            "Ngôn ngữ phân tích",
            options=["vi", "en"],
            index=0,  # Default to Vietnamese
            format_func=lambda x: "🇻🇳 Tiếng Việt" if x == "vi" else "🇬🇧 English"
        )
        
        translate_answer = st.checkbox(
            "🔄 Dịch câu trả lời sang Tiếng Việt",
            value=True,
            help="Tự động dịch câu trả lời từ tiếng Anh sang tiếng Việt"
        )
        
        st.markdown("---")
        
        # Show attention heatmap
        show_attention = st.checkbox("🔍 Hiển thị Attention Heatmap", value=True)
        
        # Gemini toggle
        use_gemini = st.checkbox(
            "🤖 Phân tích bằng Gemini AI", 
            value=True,
            disabled=(gemini_analyzer is None)
        )
        
        if gemini_analyzer is None and GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            st.caption("⚠️ Chưa cấu hình Gemini API Key")
        
        # Threshold
        st.markdown("---")
        confidence_threshold = st.slider(
            "🎯 Ngưỡng phát hiện lỗi",
            min_value=0.3, max_value=0.8, value=0.5, step=0.05
        )
        
        # Model status
        st.markdown("---")
        st.caption("📦 **Trạng thái Model**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"VQA: {'✅' if vqa_model else '❌'}")
            st.caption(f"Flaw: {'✅' if flaw_model else '❌'}")
        with col2:
            st.caption(f"Gemini: {'✅' if gemini_analyzer else '❌'}")
            st.caption(f"Dịch: {'✅' if TRANSLATOR_AVAILABLE else '⚠️'}")
        st.caption(f"Device: `{device}`")
    
    # =====================
    # MAIN CONTENT
    # =====================
    
    # Check models
    if not vqa_model or not flaw_model or not processor:
        st.error("❌ Không thể tải model. Kiểm tra thư mục `models/`")
        return
    
    # Create tabs for different modes
    tab_vqa, tab_gemini = st.tabs(["🤖 VQA Model", "✨ Hỏi Gemini trực tiếp"])
    
    # =====================
    # TAB 1: VQA MODEL
    # =====================
    with tab_vqa:
        # Input section
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            st.subheader("📤 Đầu vào")
            uploaded_file = st.file_uploader(
                "Chọn ảnh", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed",
                key="vqa_uploader"
            )
            
            # Question input with language hint
            if input_language == "vi":
                question_placeholder = "Đây là cái gì?"
                question_label = "❓ Câu hỏi (Tiếng Việt)"
            else:
                question_placeholder = "What is this?"
                question_label = "❓ Question (English)"
            
            question = st.text_input(question_label, value=question_placeholder, key="vqa_question")
            predict_btn = st.button("🔮 Phân tích", type="primary", use_container_width=True, key="vqa_btn")
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        with col_result:
            st.subheader("📊 Kết quả")
            result_container = st.container()
        
        # =====================
        # PREDICTION LOGIC
        # =====================
        if uploaded_file and predict_btn:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Store original question
            original_question = question
            
            # Translate question if Vietnamese
            if input_language == "vi":
                with st.spinner("🔄 Đang dịch câu hỏi..."):
                    question_en = translate_vi_to_en(question, gemini_analyzer)
            else:
                question_en = question
            
            # Show translation info
            if input_language == "vi" and question_en != original_question:
                with col_input:
                    st.info(f"📝 Câu hỏi đã dịch: *\"{question_en}\"*")
            
            with st.spinner("🔄 Đang xử lý..."):
                # VQA Prediction
                if show_attention:
                    answer, confidence, is_unanswerable, attention_weights = predict_vqa(
                        vqa_model, processor, image, question_en, return_attention=True
                    )
                else:
                    answer, confidence, is_unanswerable = predict_vqa(
                        vqa_model, processor, image, question_en, return_attention=False
                    )
                    attention_weights = None
            
            # Translate answer to Vietnamese if needed
            answer_display = answer
            if translate_answer and input_language == "vi" and answer:
                answer_vi = translate_answer_to_vi(answer, gemini_analyzer)
                if answer_vi != answer:
                    answer_display = f"{answer_vi} ({answer})"
            
            # Display results
            with col_result:
                # Main answer
                if is_unanswerable:
                    st.error(f"**Câu trả lời:** `{answer_display}` ({confidence:.1%})")
                else:
                    st.success(f"**Câu trả lời:** `{answer_display}` ({confidence:.1%})")
                
                # Show both original and translated question
                if input_language == "vi" and question_en != original_question:
                    st.caption(f"❓ *{original_question}*")
                    st.caption(f"🔄 *→ {question_en}*")
                else:
                    st.caption(f"❓ *{original_question}*")
            
            # =====================
            # ATTENTION VISUALIZATION
            # =====================
            if show_attention and attention_weights is not None:
                with st.expander("🎯 Attention Heatmap", expanded=True):
                    try:
                        fig = visualize_attention_summary(
                            self_attn=attention_weights['self_attn'],
                            cross_attn=attention_weights['cross_attn'],
                            image=image,
                            question_text=question_en
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Không thể tạo heatmap: {e}")
            
            # =====================
            # UNANSWERABLE FLOW
            # =====================
            if is_unanswerable:
                st.markdown("---")
                st.subheader("🔍 Phân tích lỗi ảnh")
                
                # Detect flaws
                detected_flaws = predict_flaws(flaw_model, image, confidence_threshold)
                
                if detected_flaws:
                    # Show detected flaws
                    flaw_cols = st.columns(len(detected_flaws))
                    for i, flaw in enumerate(sorted(detected_flaws, key=lambda x: x['confidence'], reverse=True)):
                        with flaw_cols[i]:
                            emoji = "🔴" if flaw['confidence'] >= 0.7 else "🟡" if flaw['confidence'] >= 0.5 else "🟢"
                            st.metric(
                                label=f"{emoji} {flaw['name']}", 
                                value=f"{flaw['confidence']:.0%}"
                            )
                else:
                    st.info("✅ Không phát hiện lỗi chất lượng ảnh rõ ràng")
                
                # =====================
                # GEMINI ANALYSIS
                # =====================
                if use_gemini and gemini_analyzer:
                    st.markdown("---")
                    st.subheader("🤖 Phân tích từ Gemini AI")
                    
                    with st.spinner("🧠 Gemini đang phân tích..."):
                        try:
                            # Prepare attention figure
                            attention_fig = None
                            if show_attention and attention_weights is not None:
                                attention_fig = visualize_attention_summary(
                                    self_attn=attention_weights['self_attn'],
                                    cross_attn=attention_weights['cross_attn'],
                                    image=image,
                                    question_text=question_en
                                )
                            
                            # Call Gemini
                            gemini_response = gemini_analyzer.analyze_unanswerable_result(
                                original_image=image,
                                question=question_en,
                                detected_flaws=detected_flaws,
                                attention_heatmap_fig=attention_fig,
                                vqa_confidence=confidence,
                                language=gemini_language
                            )
                            
                            if attention_fig:
                                plt.close(attention_fig)
                            
                            # Parse response into 2 parts
                            parsed = parse_gemini_response(gemini_response)
                            
                            # PART 1: Kết luận & Hướng dẫn (always visible)
                            if parsed['part1']:
                                st.markdown(parsed['part1'])
                            
                            # PART 2: Phân tích chi tiết (in expander)
                            if parsed['part2']:
                                with st.expander("📖 Xem phân tích chi tiết", expanded=False):
                                    st.markdown(parsed['part2'])
                            elif not parsed['part1']:
                                # Fallback: show raw response if parsing failed
                                st.markdown(gemini_response)
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi Gemini: {str(e)}")
            
            # =====================
            # SUCCESSFUL ANSWER + GEMINI
            # =====================
            else:
                if use_gemini and gemini_analyzer and show_attention and attention_weights is not None:
                    with st.expander("🤖 Gemini: Giải thích cách model trả lời"):
                        with st.spinner("🧠 Đang phân tích..."):
                            try:
                                attention_fig = visualize_attention_summary(
                                    self_attn=attention_weights['self_attn'],
                                    cross_attn=attention_weights['cross_attn'],
                                    image=image,
                                    question_text=question_en
                                )
                                
                                gemini_response = gemini_analyzer.analyze_attention_patterns(
                                    original_image=image,
                                    attention_heatmap_fig=attention_fig,
                                    question=question_en,
                                    answer=answer,
                                    confidence=confidence,
                                    language=gemini_language
                                )
                                
                                plt.close(attention_fig)
                                st.markdown(gemini_response)
                            except Exception as e:
                                st.error(f"❌ Lỗi: {str(e)}")
    
    # =====================
    # TAB 2: GEMINI DIRECT
    # =====================
    with tab_gemini:
        st.subheader("✨ Hỏi Gemini AI trực tiếp về ảnh")
        st.caption("Sử dụng Gemini AI để trả lời câu hỏi mà không cần qua VQA Model")
        
        col_gemini_input, col_gemini_result = st.columns([1, 1])
        
        with col_gemini_input:
            gemini_uploaded = st.file_uploader(
                "Chọn ảnh", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed",
                key="gemini_uploader"
            )
            
            # Question input for Gemini
            if gemini_language == "vi":
                gemini_question_placeholder = "Bạn muốn hỏi gì về ảnh này?"
                gemini_question_label = "❓ Câu hỏi cho Gemini (Tiếng Việt)"
            else:
                gemini_question_placeholder = "What would you like to know about this image?"
                gemini_question_label = "❓ Question for Gemini (English)"
            
            gemini_question = st.text_area(
                gemini_question_label, 
                value=gemini_question_placeholder,
                height=100,
                key="gemini_question"
            )
            
            gemini_ask_btn = st.button("✨ Hỏi Gemini", type="primary", use_container_width=True, key="gemini_btn")
            
            if gemini_uploaded:
                gemini_image = Image.open(gemini_uploaded).convert('RGB')
                st.image(gemini_image, caption="Ảnh đã tải lên", use_container_width=True)
        
        with col_gemini_result:
            st.subheader("💬 Câu trả lời từ Gemini")
            gemini_result_container = st.container()
        
        # Gemini Direct Query
        if gemini_uploaded and gemini_ask_btn:
            gemini_image = Image.open(gemini_uploaded).convert('RGB')
            
            with col_gemini_result:
                with st.spinner("🧠 Gemini đang suy nghĩ..."):
                    gemini_direct_answer = ask_gemini_direct(
                        gemini_image, 
                        gemini_question, 
                        gemini_language
                    )
                
                st.markdown("---")
                st.markdown(f"**❓ Câu hỏi:** {gemini_question}")
                st.markdown("---")
                st.markdown("**💡 Trả lời:**")
                st.markdown(gemini_direct_answer)
                
                # Option to copy answer
                st.markdown("---")
                if st.button("📋 Sao chép câu trả lời", key="copy_gemini"):
                    st.code(gemini_direct_answer, language=None)


if __name__ == "__main__":
    main()
