# ============================================================
# Google Colab T4 Training Configuration
#
# Free GPU training on Colab (Tesla T4, 15 GB VRAM).
# ~12-hour session limit on free tier.
# Batch size 8 to fit in 15 GB VRAM.
# ============================================================

gpu = "T4"
timeout_hours = 12

training = dict(
    batch_size=8,
    eval_batch_size=4,
    max_audio_len=500000,   # ~22.7s at 22050 Hz
    epochs=100,             # Per session; resume across sessions
    mixed_precision=True,
    num_loader_workers=2,
    num_eval_loader_workers=1,
    run_eval_steps=1000,
    save_step=1000,
    print_step=250,
)
