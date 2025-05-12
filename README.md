# BERT NSP Preprocessing Notebook

This notebook prepares data for training a BERT model on a Next Sentence Prediction (NSP) task using titles and descriptions from the NYTimesEN dataset.

## Overview

The notebook includes:

- Environment and logging setup
- Dependencies and imports
- Data loading and filtering
- Tokenization using BERT
- Feature generation for NSP

## Setup

Before running the notebook, ensure:

1. You have a local BERT cache directory at `/mnt/Intel/bert_tmp`.
2. You have configured your proxy settings (if needed).
3. The NYTimesEN data file (`NYTimesEN.json`) is present in the same directory.

### Environment Variables

python
```PYTORCH_PRETRAINED_BERT_CACHE = "/mnt/Intel/bert_tmp"```

Key Steps
Logging Setup:
Uses Python's built-in logging module to track progress and debug tokenization and feature generation.

Data Inspection:
Filters entries based on description length:

Minimum: 10 characters

Long entries: marked for statistics

Tokenizer:
Uses pytorch_pretrained_bert.BertTokenizer from the original BERT implementation.

Feature Construction:
Converts sentence pairs to token IDs, attention masks, and segment IDs.
Pads/truncates to a maximum sequence length (default: 200).

Example Feature
Each feature contains:

input_ids

input_mask

segment_ids

target (default: 1 for correct sentence pairs)

Dependencies
Python 3.6+

PyTorch

pandas

numpy

matplotlib

scikit-learn

pytorch_pretrained_bert (older version of HuggingFace transformers)

fastprogress

Output
List of InputFeatures objects ready for training or evaluation.

Logged examples of tokenization and IDs for first 5 samples.

Notes
The notebook prints a warning if NVIDIA Apex is not installed for speed optimization.

Adjust max_seq_length as needed based on GPU memory and task requirements.
