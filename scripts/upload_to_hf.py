"""Upload trained VITS Sinhala model to Hugging Face Hub.

Usage:
    1. Download model from Modal:
       modal volume get sinhala-tts-vol runs/vits_sinhala/ ./model_export/ --force

    2. Find your best model and config:
       ls model_export/vits_sinhala-*/best_model_*.pth
       ls model_export/vits_sinhala-*/config.json

    3. Upload:
       python scripts/upload_to_hf.py \\
           --model_dir model_export/vits_sinhala-February-17-2026_12+31AM-0000000 \\
           --repo_id ngpsanjaya/vits-sinhala
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def find_best_model(model_dir: Path) -> Path:
    """Find the best model checkpoint in the directory."""
    best_models = sorted(model_dir.glob("best_model_*.pth"))
    if not best_models:
        raise FileNotFoundError(f"No best_model_*.pth found in {model_dir}")
    # Pick the one with the highest step number
    return max(best_models, key=lambda p: int(p.stem.split("_")[-1]))


def main():
    parser = argparse.ArgumentParser(description="Upload VITS model to Hugging Face")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory containing best_model_*.pth and config.json",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repo ID, e.g. 'ngpsanjaya/vits-sinhala'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Validate files exist
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    best_model = find_best_model(model_dir)
    print(f"Best model: {best_model}")
    print(f"Config    : {config_path}")

    # Create repo on HF
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repo created/exists: https://huggingface.co/{args.repo_id}")

    api = HfApi()

    # Upload config.json
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=args.repo_id,
    )
    print("Uploaded config.json")

    # Upload best model as model.pth (standard name)
    api.upload_file(
        path_or_fileobj=str(best_model),
        path_in_repo="model.pth",
        repo_id=args.repo_id,
    )
    print(f"Uploaded {best_model.name} as model.pth")

    # Create a model card
    model_card = f"""---
language:
  - si
tags:
  - tts
  - vits
  - sinhala
  - coqui-tts
  - speech-synthesis
library_name: coqui-tts
pipeline_tag: text-to-speech
---

# VITS Sinhala TTS

A VITS text-to-speech model trained on Sinhala speech data using [Coqui TTS](https://github.com/coqui-ai/TTS).

## Training Details

- **Model**: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
- **Language**: Sinhala (සිංහල)
- **Epochs**: 300
- **Final mel loss**: ~18.92
- **Dataset**: [Multi-speaker TTS Dataset Sinhala](https://www.kaggle.com/datasets/keshan/multi-speaket-tts-dataset-sinhala)
- **GPU**: NVIDIA A100-80GB (via Modal)

## Usage

### With Coqui TTS

```python
from TTS.utils.synthesizer import Synthesizer

synthesizer = Synthesizer(
    tts_checkpoint="model.pth",
    tts_config_path="config.json",
    use_cuda=True,
)

outputs = synthesizer.tts("ආයුබෝවන්")
```

### From Hugging Face

```python
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(repo_id="{args.repo_id}", filename="config.json")
model_path = hf_hub_download(repo_id="{args.repo_id}", filename="model.pth")

from TTS.utils.synthesizer import Synthesizer

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=True,
)

wav = synthesizer.tts("ආයුබෝවන්")
```

## License

Please check the dataset license for usage terms.
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )
    print("Uploaded README.md (model card)")

    print(f"\nDone! View your model at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
