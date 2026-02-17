# ============================================================
# Modal A100-80GB Training Configuration
#
# Optimized for Modal's A100-80GB GPU.
# Batch size 64 fits comfortably in 80 GB VRAM.
# ============================================================

gpu = "A100-80GB"
timeout_hours = 24

training = dict(
    batch_size=64,
    eval_batch_size=16,
    max_audio_len=500000,   # ~22.7s at 22050 Hz
    epochs=300,
    mixed_precision=True,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval_steps=2000,
    save_step=2000,
    print_step=500,
)
