# Watermarking AI-Generated Text Summaries

Implementation of watermarking for AI-generated summaries based on ["A Watermark for Large Language Models"](https://arxiv.org/abs/2301.10226) (Kirchenbauer et al., 2023).

<img width="1024" height="1024" alt="Gemini_Generated_Image_m7ozsfm7ozsfm7oz" src="https://github.com/user-attachments/assets/605be67c-bb2a-4326-9b1f-0c88a3946d68" />


## Overview

This project implements a watermarking algorithm that embeds invisible statistical signals into AI-generated text during generation, making it detectable without requiring access to the model.

**Key Results:**
- ✅ 100% detection rate (TPR)
- ✅ 0% false positives (FPR)
- ⚠️ 24% quality decrease (ROUGE-L)

## Quick Start

```bash
pip install torch transformers datasets rouge-score tqdm matplotlib numpy
```

Open `watermark_implementation.ipynb` and run all cells to:
- Generate baseline and watermarked summaries
- Detect watermarks using z-score test
- Evaluate quality with ROUGE metrics
- Visualize results

## How It Works

**Generation:**
1. Hash previous token → deterministic seed
2. Split vocabulary: 50% "green" (preferred), 50% "red"
3. Boost green token logits by δ=5.0
4. Sample from modified distribution

**Detection:**
```
z = (green_count - 0.5T) / √(0.25T)
If z > 4.0 → Watermarked
```

See diagram above for visual explanation.

## Results

Tested on BART-large-CNN with CNN/DailyMail dataset (n=100):

| Metric | BART (Baseline) | Watermarked | Change |
|--------|-----------------|-------------|--------|
| **TPR** | N/A | 100% | — |
| **FPR** | 0% | N/A | — |
| **Z-score** | -0.02 ± 0.95 | 6.65 ± 0.80 | +6.67 |
| **ROUGE-1** | 0.326 | 0.261 | -0.065 |
| **ROUGE-2** | 0.123 | 0.053 | -0.070 |
| **ROUGE-L** | 0.222 | 0.169 | -0.053 |

## Key Findings

✅ **Perfect Detection:** All watermarked summaries correctly identified  
⚠️ **Quality Trade-off:** Strong watermark (δ=5.0) reduces ROUGE scores by 20-24%  
🔍 **Insight:** Summarization is harder to watermark than open-ended generation due to factual constraints

**Why?** Low-entropy content like "Supreme Court ruled 6-3" can't be paraphrased without changing meaning.

## Project Structure

```
├── watermark_implementation.ipynb   # Main implementation
├── watermark_report.md              # Detailed results & analysis
├── watermark_diagram.png            # Process visualization
└── README.md                        # This file
```

## Implementation Details

**Model:** facebook/bart-large-cnn  
**Dataset:** CNN/DailyMail (validation set)  
**Parameters:** γ=0.5 (green list size), δ=5.0 (logit boost)  
**Detection:** Z-test with threshold z > 4.0

### Core Components

**WatermarkLogitsWarper:**
- Intercepts generation at each token
- Creates context-dependent green/red lists
- Boosts green token logits before sampling

**WatermarkDetector:**
- Counts green tokens in generated text
- Computes z-score for statistical significance
- Returns detection result and confidence

## Applications

This watermarking approach can be extended to:
- 🖥️ Code generation detection
- 🌍 Multilingual content tracking
- 🎙️ Audio transcript verification
- 💬 Chatbot response attribution
- 🔬 Scientific abstract integrity

## Future Work

- Test lower δ values to improve quality while maintaining detection
- Evaluate on different models (T5, PEGASUS, GPT-based)
- Implement adaptive watermarking based on text entropy
- Test robustness against paraphrasing attacks

## Citation

```bibtex
@article{kirchenbauer2023watermark,
  title={A Watermark for Large Language Models},
  author={Kirchenbauer, John and Geiping, Jonas and Wen, Yuxin and Katz, Jonathan and Miers, Ian and Goldstein, Tom},
  journal={arXiv preprint arXiv:2301.10226},
  year={2023}
}
```

## Contact

Reach me out via LinkedIn https://www.linkedin.com/in/yashwanth-sai-a8hud/

Questions or feedback? Open an issue or reach out via [LinkedIn/Email].

---

**Note:** This is a research implementation. For production use, carefully consider the quality-detection trade-offs and test on your specific use case.
