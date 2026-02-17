# inference.py
# Shared inference utilities for synthesizing speech from checkpoints.

from pathlib import Path
from typing import Optional

from sinhala_tts.training import CoquiTrainer


def find_checkpoint(run_dir: Path, checkpoint: Optional[str] = None) -> Path:
    """Find a checkpoint to use for inference.

    Args:
        run_dir: Directory containing training runs.
        checkpoint: Optional explicit path to a checkpoint file.

    Returns:
        Path to the checkpoint file.
    """
    if checkpoint:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    ckpt = CoquiTrainer.find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return ckpt


def synthesize_to_wav(
    text: str,
    checkpoint: Path,
    config: Optional[Path] = None,
    use_cuda: bool = True,
) -> bytes:
    """Synthesize speech and return WAV bytes.

    Args:
        text: Text to synthesize.
        checkpoint: Path to .pth checkpoint.
        config: Path to config.json (defaults to same directory as checkpoint).
        use_cuda: Whether to use GPU.

    Returns:
        WAV file contents as bytes.
    """
    import io

    import numpy as np
    import soundfile as sf
    import torch
    from TTS.utils.synthesizer import Synthesizer

    if config is None:
        config = checkpoint.parent / "config.json"
    if not config.exists():
        raise FileNotFoundError(f"No config.json found at {config}")

    use_cuda = use_cuda and torch.cuda.is_available()

    synthesizer = Synthesizer(
        tts_checkpoint=str(checkpoint),
        tts_config_path=str(config),
        use_cuda=use_cuda,
    )

    outputs = synthesizer.tts(text)

    buf = io.BytesIO()
    sf.write(buf, np.array(outputs), synthesizer.tts_config.audio.sample_rate, format="WAV")
    return buf.getvalue()
