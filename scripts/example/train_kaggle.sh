#!/bin/bash
# ============================================================
# Sinhala TTS — Kaggle Training Guide
#
# Free GPU training on Kaggle (Tesla T4, 16 GB VRAM).
# 12-hour session limit — use checkpointing to resume.
#
# Steps:
#   1. Create a new Kaggle Notebook
#   2. Enable GPU (Settings > Accelerator > GPU T4 x2)
#   3. Add dataset: keshan/multi-speaket-tts-dataset-sinhala
#   4. Copy notebooks/kaggle_train.ipynb cells into the notebook
#   5. Run all cells
#   6. Download checkpoints before session expires
#   7. Upload checkpoints to resume in next session
# ============================================================

echo "This is a guide script. Use notebooks/kaggle_train.ipynb for Kaggle training."
