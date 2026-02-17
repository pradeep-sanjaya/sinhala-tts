# scripts/upload_to_hf_modal.py
# Upload trained model directly from Modal volume to Hugging Face.
# No local download required!
#
# Usage:
#   modal run scripts/upload_to_hf_modal.py --repo-id ngpsanjaya/vits-sinhala
#   modal run scripts/upload_to_hf_modal.py --repo-id ngpsanjaya/vits-sinhala --private

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
    volumes={"/vol": vol},
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def upload_to_hf(repo_id: str, private: bool = False):
    """Upload best model + config from Modal volume directly to Hugging Face."""
    import os
    from huggingface_hub import HfApi, create_repo

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not found. Create a Modal secret named 'huggingface-secret' "
            "with key HF_TOKEN at https://modal.com/secrets"
        )

    # Find the latest checkpoint
    ckpt = CoquiTrainer.find_latest_checkpoint(RUN_DIR)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {RUN_DIR}")

    config_path = ckpt.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")

    print(f"Best model : {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")
    print(f"Config     : {config_path}")
    print(f"Repo       : {repo_id}")

    # Create repo
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=hf_token)
    print(f"Repo created/exists: https://huggingface.co/{repo_id}")

    api = HfApi(token=hf_token)

    # Upload model
    print("Uploading model.pth...")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="model.pth",
        repo_id=repo_id,
    )
    print("Uploaded model.pth")

    # Upload config
    print("Uploading config.json...")
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=repo_id,
    )
    print("Uploaded config.json")

    # Upload model card
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

A [VITS](https://arxiv.org/abs/2106.06103) text-to-speech model for **Sinhala (සිංහල)**, trained using [Coqui TTS](https://github.com/coqui-ai/TTS).

**GitHub**: [pradeep-sanjaya/sinhala-tts](https://github.com/pradeep-sanjaya/sinhala-tts)

## Training Details

| Detail | Value |
|--------|-------|
| **Model** | VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) |
| **Language** | Sinhala (සිංහල) |
| **Epochs** | 300 |
| **Final mel loss** | ~18.92 |
| **Dataset** | [Multi-speaker TTS Dataset Sinhala](https://www.kaggle.com/datasets/keshan/multi-speaket-tts-dataset-sinhala) |
| **GPU** | NVIDIA A100-80GB (via [Modal](https://modal.com)) |
| **Training time** | ~3.2 hours |
| **Framework** | [Coqui TTS](https://github.com/coqui-ai/TTS) 0.27.5 |

## Usage

### From Hugging Face

```python
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import Synthesizer

config_path = hf_hub_download(repo_id="{repo_id}", filename="config.json")
model_path = hf_hub_download(repo_id="{repo_id}", filename="model.pth")

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=True,
)

wav = synthesizer.tts("ආයුබෝවන්")
```

### Save to WAV

```python
import numpy as np
import soundfile as sf

sf.write("output.wav", np.array(wav), synthesizer.tts_config.audio.sample_rate)
```

## Training & Deployment

The full training pipeline supports **Modal**, **Kaggle**, **Google Colab**, and **AWS SageMaker**.

See the [GitHub repo](https://github.com/pradeep-sanjaya/sinhala-tts) for:
- Platform-specific configs and training scripts
- Kaggle and Colab notebooks for free GPU training
- Inference scripts (Modal and local)
- Checkpoint resume support

## License

MIT. Please check the [dataset license](https://www.kaggle.com/datasets/keshan/multi-speaket-tts-dataset-sinhala) for data usage terms.
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print("Uploaded README.md (model card)")

    print(f"\nDone! View your model at: https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main(repo_id: str = "ngpsanjaya/vits-sinhala", private: bool = False):
    upload_to_hf.remote(repo_id=repo_id, private=private)
