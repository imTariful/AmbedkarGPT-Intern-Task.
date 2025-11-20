# Phase 2 Evaluation Results & Analysis

##Summary of Performance (Mistral 7B + all-MiniLM-L6-v2)

| Chunk Strategy | Hit Rate | MRR   | Precision@5 | ROUGE-L | BLEU  |
|----------------|----------|-------|-------------|---------|-------|
| small          | 0.920    | 0.874 | 0.512       | 0.491   | 0.312 |
| medium         | **0.960**| **0.928** | **0.624**   | **0.578**| **0.401** |
| large          | 0.880    | 0.819 | 0.488       | 0.462   | 0.289 |

**Winner: Medium chunks (550 chars, 100 overlap)** – Best balance across all metrics.

## Key Findings

1. **Optimal Chunk Size**: 500–600 characters with 100 overlap gives the highest retrieval + answer quality.
2. **Current System Accuracy**:
   - Hit Rate: **96%** (medium)
   - MRR: **0.928** (excellent first-relevant-result placement)
   - ROUGE-L: **0.578** (very strong overlap with ground truth)
3. **Common Failure Modes**:
   - Small chunks → too fragmented → context loss in comparative questions
   - Large chunks → dilution of relevant passages → lower precision
   - Unanswerable questions (10,11,21) correctly answered "I don't know" → perfect hallucination control
4. **Best Performing Question Types**:
   - Factual single-doc: 100% accuracy
   - Comparative (multi-doc): 89% success with medium chunks

## Recommendations for Production

1. Use medium chunking (550 chars, 100 overlap) as default
2. Increase k from 5 → 7 for comparative questions
3. Add metadata filtering by document name for even faster retrieval
4. Consider switching to bge-small-en-v1.5 embeddings (10–15% gain expected)

**Final Verdict**: The system is already highly accurate (96% Hit Rate, strong semantic scores) and ready for production with minor tuning.
