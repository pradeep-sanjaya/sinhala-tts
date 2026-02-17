# ============================================================
# Modal A10G Training Configuration
#
# Budget-friendly option on Modal's A10G (22 GB VRAM).
# Smaller batch size to fit in memory.
# ============================================================

gpu = "A10G"
timeout_hours = 24

training = dict(
    batch_size=16,
    eval_batch_size=8,
    max_audio_len=500000,   # ~22.7s at 22050 Hz
    epochs=300,
    mixed_precision=True,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval_steps=2000,
    save_step=2000,
    print_step=500,
)
