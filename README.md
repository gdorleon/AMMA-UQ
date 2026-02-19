# AMMA-UQ: Adaptive Multi-Modal Attention for Uncertainty Quantification in Black-Box LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the paper **"Beyond Fixed Aggregation: Learning to Aggregate Consistency Signals for Adaptive Uncertainty Quantification in Black-Box LLMs"**.


## Overview

AMMA-UQ is a framework for uncertainty quantification in black-box large language models that extends the consistency hypothesis work of Xiao et al. (UAI 2025). Our method introduces three key innovations:

1. **Adaptive Sampling**  
   Dynamically determines required samples based on entropy stabilization (48.7% fewer samples)

2. **Multi-Modal Similarity Fusion**  
   Combines lexical, semantic, and task-specific signals

3. **Attention-Based Aggregation**  
   Learns discriminative weightings for pairwise similarities

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AMMA-UQ.git
cd AMMA-UQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .


# Evaluate
from src.utils.metrics import compute_auroc, compute_ece
auroc = compute_auroc(labels, confidences)
ece = compute_ece(labels, confidences)
print(f"AUROC: {auroc:.3f}, ECE: {ece:.3f}")
```
## Quick Start

```python
from src.amma_uq import AMMAUQ
from src.utils.data_loader import load_dataset

# Load data
queries, generations, labels = load_dataset('coqa', split='validation')

# Initialize AMMA-UQ
model = AMMAUQ(
    similarity_metrics=['rouge_l', 'sbert', 'task_specific'],
    adaptive_sampling=True,
    attention_hidden_dim=64
)

# Compute confidence scores
confidences = model.estimate_confidence(queries, generations)

# Evaluate
from src.utils.metrics import compute_auroc, compute_ece
auroc = compute_auroc(labels, confidences)
ece = compute_ece(labels, confidences)

print(f"AUROC: {auroc:.3f}, ECE: {ece:.3f}")
```

---

## Repository Structure

```text
AMMA-UQ/
|---data/               # Dataset loading scripts
|---src/                # Core implementation
|---experiments/        # Experiment scripts
|---config/             # Configuration files
|---results/            # Output results
|---notebooks/          # Analysis notebooks
|---tests/              # Unit tests
|--- docs/               # Documentation
```

---

## Reproducing Results

To reproduce all experiments from the paper:

```bash
# Run all experiments
bash experiments/run_all_experiments.sh

# Or run individual experiments
python experiments/run_qa_experiments.py --config config/qa_config.yaml
python experiments/run_ablation_studies.py --dataset coqa
```

Results will be saved in the `results/` directory.  
See `docs/reproducing_results.md` for detailed instructions.

---

## Datasets

We evaluate on 8 benchmark datasets:

**QA**
- CoQA  
- TriviaQA  
- Natural Questions  

**Summarization**
- CNN/DailyMail  
- XSum  

**Text-to-SQL**
- Spider  
- Spider-Realistic  
- BIRD  

To download and prepare datasets:

```bash
python data/download_datasets.py --all
python data/download_datasets.py --dataset coqa
```

See `docs/dataset_preparation.md` for detailed instructions.

---

## Citation

```bibtex
@inproceedings{smith2026amma,
  title={AMMA-UQ: Adaptive Multi-Modal Attention for Uncertainty Quantification in Black-Box LLMs},
  author={....},
  booktitle={....},
  year={2026}
}
```

---

## License

MIT License
