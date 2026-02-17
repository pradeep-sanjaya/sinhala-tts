# scripts/synthesize_modal.py
# Test inference from the latest checkpoint on Modal.
#
# Usage:
#   modal run scripts/synthesize_modal.py --text "ආයුබෝවන්"
#   modal run scripts/synthesize_modal.py --text "ආයුබෝවන්" --out /tmp/output.wav

import sys
from pathlib import Path

import modal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train_modal import app, vol, image, RUN_DIR
from sinhala_tts.training import CoquiTrainer


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=5 * 60,
)
def synthesize(text: str) -> bytes:
    """Synthesize speech from text using the latest checkpoint. Returns WAV bytes."""
    import torch
    from TTS.utils.synthesizer import Synthesizer

    # Find the latest checkpoint
    ckpt = CoquiTrainer.find_latest_checkpoint(RUN_DIR)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {RUN_DIR}")

    # The config.json lives next to the checkpoint
    config_path = ckpt.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")

    print(f"Using checkpoint: {ckpt}")
    print(f"Using config   : {config_path}")

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available : {use_cuda}")

    synthesizer = Synthesizer(
        tts_checkpoint=str(ckpt),
        tts_config_path=str(config_path),
        use_cuda=use_cuda,
    )

    outputs = synthesizer.tts(text)

    # Save to in-memory WAV bytes
    import io
    import soundfile as sf
    import numpy as np

    buf = io.BytesIO()
    sf.write(buf, np.array(outputs), synthesizer.tts_config.audio.sample_rate, format="WAV")
    return buf.getvalue()


@app.local_entrypoint()
def main(text: str = "ආයුබෝවන්", out: str = "output.wav"):
    print(f"Synthesizing: {text}")
    wav_bytes = synthesize.remote(text)
    Path(out).write_bytes(wav_bytes)
    print(f"Saved to: {out} ({len(wav_bytes)} bytes)")
