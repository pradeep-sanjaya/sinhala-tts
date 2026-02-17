#!/usr/bin/env python3
"""
Local inference script — synthesize Sinhala speech from a downloaded checkpoint.

Usage:
    1. Download checkpoint + config from Modal:
       modal volume get sinhala-tts-vol runs/vits_sinhala/ ./model_export/ --force

    2. Run inference:
       python scripts/synthesize_local.py \\
           --checkpoint model_export/vits_sinhala-*/best_model_*.pth \\
           --config model_export/vits_sinhala-*/config.json \\
           --text "ආයුබෝවන්" \\
           --out output.wav
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sinhala TTS local inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--text", type=str, default="ආයුබෝවන්", help="Text to synthesize")
    parser.add_argument("--out", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--use-cuda", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    import torch
    from TTS.utils.synthesizer import Synthesizer

    ckpt = Path(args.checkpoint)
    cfg = Path(args.config)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")

    use_cuda = args.use_cuda and torch.cuda.is_available()
    print(f"Checkpoint : {ckpt}")
    print(f"Config     : {cfg}")
    print(f"CUDA       : {use_cuda}")
    print(f"Text       : {args.text}")

    synthesizer = Synthesizer(
        tts_checkpoint=str(ckpt),
        tts_config_path=str(cfg),
        use_cuda=use_cuda,
    )

    outputs = synthesizer.tts(args.text)

    import numpy as np
    import soundfile as sf

    sf.write(args.out, np.array(outputs), synthesizer.tts_config.audio.sample_rate)
    print(f"Saved to   : {args.out}")


if __name__ == "__main__":
    main()
