1. MULTI-HEAD ATTENTION 
Input → [Head 1] →         ↘
        [Head 2] →          → Concatenate → Linear → Output
        [Head 3] →         ↗
        ...

2. Multi-Head Cross-Modal Attention
 Ảnh (patches)      Câu hỏi (tokens)
   [A1]                [W1]
   [A2]                [W2]
   [A3]                [W3]
                       [W4]
                       [W5]

↓ Embedding (Linear projection)
─────────────────────────────────────────────

Multi-Head Cross-Modal Attention Layer
─────────────────────────────────────────────
    [Head 1]   [Head 2]   ...   [Head N]
      │           │               │
      └─────┬─────┴─────...───────┘
            │
      Concatenate (ghép lại)
            │
      Linear (tổng hợp)
            │
─────────────────────────────────────────────

→ Output:  
- Embedding của từng token câu hỏi đã được làm giàu thông tin từ các patch ảnh mà nó chú ý.
- (Có thể làm tương tự chiều ngược lại: patch ảnh chú ý vào token câu hỏi.)

─────────────────────────────────────────────

→ Classifier (dự đoán đáp án)