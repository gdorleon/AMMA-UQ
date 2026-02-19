# AMMA-UQ: Adaptive Multi-Modal Attention for Uncertainty Quantification in Black-Box LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Beyond Fixed Aggregation: Learning to Aggregate Consistency Signals for Adaptive Uncertainty Quantification in Black-Box LLMs"**.

## Overview

AMMA-UQ is a framework for uncertainty quantification in black-box large language models that extends previous consistency hypothesis work. Our method introduces three key innovations:

1. **Adaptive Sampling**: Dynamically determines required samples based on entropy stabilization (48.7% fewer samples)
2. **Multi-Modal Similarity Fusion**: Combines lexical, semantic, and task-specific signals
3. **Attention-Based Aggregation**: Learns discriminative weightings for pairwise similarities

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
