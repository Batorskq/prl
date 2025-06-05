<div align="center">
<h1>PRL: Prompts from Reinforcement Learning</h1>

Paweł Batorski, Adrian Kosmala, Paul Swoboda


[![arXiv](https://img.shields.io/badge/arXiv-2505.14412-red)](https://arxiv.org/abs/2505.14412)  

</div>

<div align="center">
  <img src="imgs/prl.png" alt="PRL Figure">
</div>

---

This repository contains the official implementation of the paper  
**[PRL: Prompts from Reinforcement Learning](https://arxiv.org/abs/2505.14412)**

**Abstract:**  
Effective prompt engineering remains a central challenge in fully harnessing the capabilities of LLMs. While well-designed prompts can dramatically enhance performance, crafting them typically demands expert intuition and a nuanced understanding of the task. Moreover, the most impactful prompts often hinge on subtle semantic cues, ones that may elude human perception but are crucial for guiding LLM behavior.  
In this paper, we introduce **PRL (Prompts from Reinforcement Learning)**, a novel RL-based approach for automatic prompt generation. Unlike previous methods, PRL can produce novel few-shot examples that were not seen during training. Our approach achieves state-of-the-art performance across a range of benchmarks, including text classification, simplification, and summarization.

---

## ✨ Highlights

- 🚀 Outperforms prior prompt optimization methods like APE and EvoPrompt
- 🧠 Automatically generates **novel few-shot prompts** unseen during training
- 📈 Strong gains across multiple NLP tasks:
  - **+2.58%** over APE and **+1.00%** over EvoPrompt (Classification)
  - **+4.32 ROUGE** over APE and **+2.12** over EvoPrompt (Summarization)
  - **+6.93 SARI** over APE and **+6.01** over EvoPrompt (Simplification)

---

## 🛠️ Installation & Setup

PRL is based on the [**ms-swift**](https://github.com/modelscope/ms-swift) framework.  
Please follow the environment setup instructions provided in that repository.

You also need to download the benchmark datasets (Classification, Summarization, Simplification) from  
👉 [https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar)

---

## 🧪 Benchmarks

### 🔤 Classification

To train PRL on the MR dataset:
```bash
./scripts/mr/mr_qwen_qwen.sh
```

### 📚 Summarization
To run PRL on summarization tasks:
```bash
./scripts/sum/sum_qwen_qwen.sh
```

### ✏️ Simplification
To evaluate PRL on simplification:
```bash
./scripts/sim/sim_qwen_qwen.sh
```

## 📄 Citation

If you find our work useful, please consider citing:

```bibtex
@article{batorski2025prl,
  title     = {PRL: Prompts from Reinforcement Learning},
  author    = {Batorski, Pawe{\l} and Kosmala, Adrian and Swoboda, Paul},
  journal   = {arXiv preprint arXiv:2505.14412},
  year      = {2025}
}
```

## 🙏 Acknowledgments

[![ms-swift](https://img.shields.io/badge/Base%20Code-ms--swift-blue)](https://github.com/modelscope/ms-swift)
[![APE](https://img.shields.io/badge/Inspired%20by-APE-red)](https://arxiv.org/abs/2211.01910)
[![EvoPrompt](https://img.shields.io/badge/Inspired%20by-EvoPrompt-orange)](https://arxiv.org/abs/2309.08532)
[![APO](https://img.shields.io/badge/Inspired%20by-APO-yellow)](https://arxiv.org/abs/2305.03495)

This work builds on the [**ms-swift**](https://github.com/modelscope/ms-swift) framework.  
We thank the authors of [**APE**](https://arxiv.org/abs/2211.01910), [**EvoPrompt**](https://arxiv.org/abs/2309.08532), and [**APO**](https://arxiv.org/abs/2305.03495) for their inspiring contributions to the field of automated prompt generation.

