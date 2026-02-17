#!/bin/bash
# ============================================================
# Sinhala TTS — Modal Training Script
#
# Train VITS model on Modal with A100-80GB GPU.
# All training parameters are in configs/modal_a100.py
# ============================================================

set -e

echo "============================================================"
echo "Sinhala TTS — Modal Training"
echo "============================================================"

# Fresh training (detached so it survives disconnects)
# modal run --detach scripts/train_modal.py

# Resume from latest checkpoint (detached)
# modal run --detach scripts/train_modal.py --restore

# Budget option: A10G GPU
# modal run --detach scripts/train_modal.py --config modal_a10g

# Default: A100-80GB, fresh start, detached
modal run --detach scripts/train_modal.py

echo ""
echo "============================================================"
echo "Training started! Monitor at https://modal.com/apps"
echo "============================================================"
