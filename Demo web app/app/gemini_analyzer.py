"""
Gemini AI Analyzer for VQA Application
Provides intelligent analysis and suggestions based on attention heatmaps and flaw detection
"""
import os
import io
import base64
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class GeminiAnalyzer:
    """
    Uses Gemini API to analyze VQA results and provide intelligent suggestions
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini Analyzer
        
        Args:
            api_key: Gemini API key. If None, will look for GEMINI_API_KEY env variable
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.0 Flash - fast and has higher free tier quota
        # gemini-3-pro-preview has very limited free quota
        # Alternative models: 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-lite'
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    
    def analyze_unanswerable_result(
        self,
        original_image: Image.Image,
        question: str,
        detected_flaws: list,
        attention_heatmap_fig: plt.Figure = None,
        vqa_confidence: float = 0.0,
        language: str = "vi"
    ) -> str:
        """
        Analyze an unanswerable VQA result and provide intelligent suggestions
        
        Args:
            original_image: The original uploaded image
            question: The question asked
            detected_flaws: List of detected flaws from flaw detection model
            attention_heatmap_fig: Matplotlib figure with attention visualization
            vqa_confidence: Confidence score of VQA prediction
            language: Response language ("vi" for Vietnamese, "en" for English)
            
        Returns:
            Analysis and suggestions from Gemini
        """
        # Prepare images for Gemini
        images_to_send = [original_image]
        
        # Convert attention heatmap to image if available
        attention_image = None
        if attention_heatmap_fig is not None:
            attention_image = self._fig_to_image(attention_heatmap_fig)
            images_to_send.append(attention_image)
        
        # Format detected flaws
        flaws_text = ""
        if detected_flaws:
            flaws_list = [f"- {flaw['name']} (confidence: {flaw['confidence']:.1%})" 
                         for flaw in detected_flaws]
            flaws_text = "\n".join(flaws_list)
        else:
            flaws_text = "No specific flaws detected by the model"
        
        # Build prompt based on language
        if language == "vi":
            prompt = self._build_vietnamese_prompt(
                question, flaws_text, attention_heatmap_fig is not None, vqa_confidence
            )
        else:
            prompt = self._build_english_prompt(
                question, flaws_text, attention_heatmap_fig is not None, vqa_confidence
            )
        
        try:
            # Send to Gemini with images
            response = self.model.generate_content([prompt] + images_to_send)
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
    
    def _build_vietnamese_prompt(
        self, 
        question: str, 
        flaws_text: str, 
        has_attention: bool,
        confidence: float
    ) -> str:
        """Build Vietnamese prompt for Gemini"""
        
        attention_instruction = ""
        if has_attention:
            attention_instruction = """
**Hình thứ 2 là Attention Heatmap:**
- Hàng 1-2: Self-Attention - Model đang "nhìn" vào các vùng nào của ảnh
- Hàng 3-4: Cross-Attention - Câu hỏi đang liên kết với vùng nào của ảnh
- Vùng sáng (đỏ/vàng) = model chú ý nhiều
- Vùng tối (xanh/tím) = model ít chú ý

Hãy phân tích xem:
1. Model có đang nhìn đúng vào object cần trả lời không?
2. Attention có phân tán hay tập trung?
3. Cross-attention có match với câu hỏi không?
"""

        prompt = f"""Bạn là một chuyên gia phân tích AI trong lĩnh vực Visual Question Answering (VQA).

**BỐI CẢNH:**
- Người dùng upload một ảnh và hỏi câu hỏi
- Model VQA đã dự đoán là "UNANSWERABLE" (không thể trả lời)
- Confidence của dự đoán: {confidence:.1%}

**CÂU HỎI CỦA NGƯỜI DÙNG:** "{question}"

**KẾT QUẢ PHÁT HIỆN LỖI ẢNH:**
{flaws_text}

{attention_instruction}

**NHIỆM VỤ CỦA BẠN:**
Hãy phân tích và trả lời theo ĐÚNG FORMAT sau (bắt buộc giữ nguyên các marker):

---PART1_START---
## 🎯 Kết Luận & Hướng Dẫn

**Vấn đề chính:** [Mô tả ngắn gọn 1-2 câu vấn đề của ảnh]

**Hướng dẫn khắc phục:**
- [Gợi ý 1: cách chụp lại ảnh hoặc điều chỉnh]
- [Gợi ý 2: cách đặt câu hỏi khác nếu cần]
- [Gợi ý 3: các lưu ý khác nếu có]
---PART1_END---

---PART2_START---
## 🔬 Phân Tích Chi Tiết

### 1. Phân tích nguyên nhân
[Giải thích chi tiết tại sao model không thể trả lời - 2-3 câu]

### 2. Đánh giá chất lượng ảnh
[Phân tích các vấn đề: mờ, tối, sáng, che khuất, object có xuất hiện không - 2-3 câu]

### 3. Phân tích Attention (nếu có heatmap)
[Nhận xét về vùng model đang chú ý, có đúng object không]

### 4. Nhận xét thêm
[Bất kỳ quan sát hoặc đề xuất bổ sung nào]
---PART2_END---

Hãy trả lời bằng tiếng Việt, thân thiện và dễ hiểu. BẮT BUỘC giữ nguyên các marker ---PART1_START---, ---PART1_END---, ---PART2_START---, ---PART2_END---.
"""
        return prompt
    
    def _build_english_prompt(
        self, 
        question: str, 
        flaws_text: str, 
        has_attention: bool,
        confidence: float
    ) -> str:
        """Build English prompt for Gemini"""
        
        attention_instruction = ""
        if has_attention:
            attention_instruction = """
**Second image is the Attention Heatmap:**
- Rows 1-2: Self-Attention - Which regions the model is "looking" at
- Rows 3-4: Cross-Attention - How the question relates to image regions
- Bright areas (red/yellow) = high attention
- Dark areas (blue/purple) = low attention

Please analyze:
1. Is the model looking at the correct object to answer the question?
2. Is attention focused or scattered?
3. Does cross-attention align with the question semantics?
"""

        prompt = f"""You are an expert AI analyst specializing in Visual Question Answering (VQA).

**CONTEXT:**
- User uploaded an image and asked a question
- VQA model predicted "UNANSWERABLE"
- Prediction confidence: {confidence:.1%}

**USER'S QUESTION:** "{question}"

**DETECTED IMAGE FLAWS:**
{flaws_text}

{attention_instruction}

**YOUR TASK:**
Analyze and respond using EXACTLY this format (keep all markers):

---PART1_START---
## 🎯 Conclusion & Guidance

**Main Issue:** [Brief 1-2 sentence description of the image problem]

**How to fix:**
- [Suggestion 1: how to retake or adjust the photo]
- [Suggestion 2: alternative way to phrase the question if needed]
- [Suggestion 3: other tips if applicable]
---PART1_END---

---PART2_START---
## 🔬 Detailed Analysis

### 1. Root Cause Analysis
[Detailed explanation of why the model couldn't answer - 2-3 sentences]

### 2. Image Quality Assessment
[Analyze issues: blur, darkness, brightness, obstruction, is object present - 2-3 sentences]

### 3. Attention Analysis (if heatmap available)
[Comment on which regions the model is focusing on, is it correct]

### 4. Additional Observations
[Any other insights or suggestions]
---PART2_END---

Respond in a friendly, easy-to-understand manner. YOU MUST keep all markers ---PART1_START---, ---PART1_END---, ---PART2_START---, ---PART2_END---.
"""
        return prompt

    def analyze_attention_patterns(
        self,
        original_image: Image.Image,
        attention_heatmap_fig: plt.Figure,
        question: str,
        answer: str,
        confidence: float,
        language: str = "vi"
    ) -> str:
        """
        Analyze attention patterns for ANY VQA result (not just unanswerable)
        
        Args:
            original_image: The original image
            attention_heatmap_fig: Attention visualization
            question: The question asked
            answer: The predicted answer
            confidence: Confidence score
            language: Response language
            
        Returns:
            Analysis of attention patterns from Gemini
        """
        attention_image = self._fig_to_image(attention_heatmap_fig)
        
        if language == "vi":
            prompt = f"""Bạn là chuyên gia phân tích Attention trong mô hình VQA.

**CÂU HỎI:** "{question}"
**CÂU TRẢ LỜI:** "{answer}" (confidence: {confidence:.1%})

**HÌNH ẢNH:**
- Hình 1: Ảnh gốc
- Hình 2: Attention Heatmap (8 heads)
  + Row 1-2: Self-Attention (ảnh tự attend với chính nó)
  + Row 3-4: Cross-Attention (câu hỏi attend vào ảnh)

**HÃY PHÂN TÍCH:**
1. Model đang tập trung vào vùng nào của ảnh?
2. Attention có hợp lý với câu hỏi không?
3. Cross-attention có đang "nhìn" đúng object không?
4. Có head nào đặc biệt quan trọng không?

Trả lời ngắn gọn, dễ hiểu bằng tiếng Việt.
"""
        else:
            prompt = f"""You are an expert in analyzing Attention patterns in VQA models.

**QUESTION:** "{question}"
**ANSWER:** "{answer}" (confidence: {confidence:.1%})

**IMAGES:**
- Image 1: Original image
- Image 2: Attention Heatmap (8 heads)
  + Rows 1-2: Self-Attention (image attends to itself)
  + Rows 3-4: Cross-Attention (question attends to image)

**PLEASE ANALYZE:**
1. Which regions is the model focusing on?
2. Does the attention pattern make sense for the question?
3. Is cross-attention looking at the correct object?
4. Are any heads particularly important?

Respond concisely and clearly.
"""
        
        try:
            response = self.model.generate_content([prompt, original_image, attention_image])
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"


def parse_gemini_response(response: str) -> dict:
    """
    Parse Gemini response into two parts: conclusion/guidance and detailed analysis
    
    Args:
        response: Raw response from Gemini
        
    Returns:
        Dictionary with 'part1' (conclusion & guidance) and 'part2' (detailed analysis)
    """
    result = {
        'part1': '',  # Kết luận & Hướng dẫn
        'part2': '',  # Phân tích chi tiết
        'raw': response  # Backup nếu parse thất bại
    }
    
    try:
        # Extract Part 1
        if '---PART1_START---' in response and '---PART1_END---' in response:
            start = response.find('---PART1_START---') + len('---PART1_START---')
            end = response.find('---PART1_END---')
            result['part1'] = response[start:end].strip()
        
        # Extract Part 2
        if '---PART2_START---' in response and '---PART2_END---' in response:
            start = response.find('---PART2_START---') + len('---PART2_START---')
            end = response.find('---PART2_END---')
            result['part2'] = response[start:end].strip()
        
        # Fallback if markers not found
        if not result['part1'] and not result['part2']:
            result['part1'] = response
            result['part2'] = ''
            
    except Exception:
        result['part1'] = response
        result['part2'] = ''
    
    return result


def create_gemini_analyzer(api_key: str = None) -> GeminiAnalyzer:
    """
    Factory function to create GeminiAnalyzer instance
    
    Args:
        api_key: Optional API key. If not provided, uses GEMINI_API_KEY env variable
        
    Returns:
        GeminiAnalyzer instance
    """
    return GeminiAnalyzer(api_key=api_key)
