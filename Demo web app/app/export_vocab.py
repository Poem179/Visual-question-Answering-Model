"""
Export vocabulary from train.json
"""
import json
from collections import Counter

# Read train.json
print('Loading train.json...')
with open('../models/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total samples: {len(data)}')

# Count all answers (same logic as VQADatasetV2)
answer_counter = Counter()
min_answer_freq = 5  # Same as training

for item in data:
    answers = item.get('answers', [])
    # Get most common answer (majority voting)
    ans_list = [a['answer'].lower().strip() for a in answers]
    if ans_list:
        most_common = Counter(ans_list).most_common(1)[0][0]
        answer_counter[most_common] += 1

# Filter by frequency
print(f'Unique answers before filtering: {len(answer_counter)}')
filtered_answers = [ans for ans, count in answer_counter.items() if count >= min_answer_freq]
print(f'Answers with freq >= {min_answer_freq}: {len(filtered_answers)}')

# Build vocab (index 0 = unanswerable)
idx_to_answer = {0: 'unanswerable'}
answer_to_idx = {'unanswerable': 0}

# Sort by frequency
sorted_answers = sorted(filtered_answers, key=lambda x: answer_counter[x], reverse=True)

# Add to vocab
idx = 1
for ans in sorted_answers:
    if ans != 'unanswerable' and ans not in answer_to_idx:
        idx_to_answer[idx] = ans
        answer_to_idx[ans] = idx
        idx += 1

print(f'Final vocab size: {len(idx_to_answer)}')

# Save vocab
with open('idx_to_answer.json', 'w', encoding='utf-8') as f:
    json.dump({str(k): v for k, v in idx_to_answer.items()}, f, indent=2, ensure_ascii=False)

print('Saved to idx_to_answer.json')

# Show top 20 answers
print('\nTop 20 answers:')
for i in range(min(20, len(idx_to_answer))):
    print(f'  {i}: {idx_to_answer[i]}')
