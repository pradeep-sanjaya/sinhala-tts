# ============================================================
# AWS SageMaker Training Configuration
#
# For SageMaker training jobs with ml.g5.xlarge (A10G, 24 GB)
# or ml.p3.2xlarge (V100, 16 GB).
# Adjust batch_size based on instance type.
# ============================================================

# SageMaker instance type (informational)
instance_type = "ml.g5.xlarge"
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
