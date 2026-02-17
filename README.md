# Sinhala TTS

<div align="center">
    <a href="https://github.com/ngpsanjaya/sinhala-tts"><img src="https://img.shields.io/badge/Github-Star-yellow?logo=Github&amp"></a>
    <a href="https://huggingface.co/ngpsanjaya/vits-sinhala"><img src="https://img.shields.io/badge/Huggingface-Download-orange?logo=Huggingface&amp"></a>
</div>

A [VITS](https://arxiv.org/abs/2106.06103) text-to-speech model for **Sinhala (සිංහල)**, trainable on [Modal](https://modal.com), [Kaggle](https://www.kaggle.com), [Google Colab](https://colab.research.google.com), and [AWS SageMaker](https://aws.amazon.com/sagemaker/).

Built with [Coqui TTS](https://github.com/coqui-ai/TTS).

## Training Results

| Metric | Value |
|--------|-------|
| **Final mel loss** | 18.92 |
| **Epochs** | 300 |
| **GPU** | NVIDIA A100-80GB (Modal) |
| **Training time** | ~3.2 hours |
| **Dataset** | [Multi-speaker TTS Sinhala](https://www.kaggle.com/datasets/keshan/multi-speaket-tts-dataset-sinhala) |

## Quick Start

### Option 1: Modal (recommended for training)

```bash
pip install modal
modal setup

# Train on A100-80GB (detached so it survives disconnects)
modal run --detach scripts/train_modal.py

# Resume from checkpoint
modal run --detach scripts/train_modal.py --restore

# Test inference
modal run scripts/synthesize_modal.py --text "ආයුබෝවන්"
```

### Option 2: Kaggle (free T4 GPU)

Upload `notebooks/kaggle_train.ipynb` to Kaggle, enable GPU, and run all cells. See [Kaggle guide](#kaggle).

### Option 3: Google Colab (free T4 GPU)

Open `notebooks/colab_train.ipynb` in Colab, select T4 GPU runtime, and run all cells. Checkpoints save to Google Drive. See [Colab guide](#colab).

### Option 4: Local inference

```bash
pip install -e ".[inference]"

# Download model from Modal
modal volume get sinhala-tts-vol runs/vits_sinhala/ ./model_export/ --force

# Synthesize
python scripts/synthesize_local.py \
    --checkpoint model_export/vits_sinhala-*/best_model_*.pth \
    --config model_export/vits_sinhala-*/config.json \
    --text "ආයුබෝවන්"
```

### Option 5: From Hugging Face

```python
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import Synthesizer

config_path = hf_hub_download(repo_id="ngpsanjaya/vits-sinhala", filename="config.json")
model_path = hf_hub_download(repo_id="ngpsanjaya/vits-sinhala", filename="model.pth")

synthesizer = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path, use_cuda=True)
wav = synthesizer.tts("ආයුබෝවන්")
```

## Project Structure

```
sinhala-tts/
├── sinhala_tts/                # Core library package
│   ├── __init__.py
│   ├── domain.py               # ClipRecord dataclass
│   ├── dataset.py              # Kaggle fetching, LJSpeech building, character set extraction
│   ├── training.py             # VITS config writing + Coqui trainer execution
│   └── inference.py            # Shared inference utilities
├── configs/                    # Platform-specific training configs
│   ├── modal_a100.py           # Modal A100-80GB (fastest, ~$16)
│   ├── modal_a10g.py           # Modal A10G (budget, ~$10)
│   ├── kaggle_t4.py            # Kaggle free T4 GPU
│   ├── colab_t4.py             # Google Colab free T4 GPU
│   └── sagemaker.py            # AWS SageMaker
├── scripts/                    # Training, inference, and deployment scripts
│   ├── train_modal.py          # Modal training entrypoint
│   ├── synthesize_modal.py     # Modal inference
│   ├── synthesize_local.py     # Local inference from downloaded checkpoint
│   ├── upload_to_hf.py         # Upload model to Hugging Face
│   └── example/                # Shell script examples
│       ├── train_modal.sh
│       └── train_kaggle.sh
├── notebooks/                  # Jupyter notebooks for free platforms
│   ├── kaggle_train.ipynb      # Kaggle training notebook
│   └── colab_train.ipynb       # Google Colab training notebook
├── pyproject.toml              # Python packaging (pip install -e .)
├── .gitignore
└── README.md
```

## Platform Guides

### Modal

Modal provides on-demand GPUs with pay-per-second billing. Training runs are persisted to a Modal Volume.

```bash
# Install
pip install -e ".[modal]"

# Train (A100-80GB, ~3.2 hours, ~$16)
modal run --detach scripts/train_modal.py

# Train (A10G, ~13 hours, ~$10)
modal run --detach scripts/train_modal.py --config modal_a10g

# Resume from checkpoint
modal run --detach scripts/train_modal.py --restore

# Monitor at https://modal.com/apps
```

**Key flags:**
- `--detach` — keeps running after you disconnect
- `--restore` — resume from latest checkpoint
- `--config <name>` — use a specific config from `configs/`

### <a name="kaggle"></a>Kaggle

Free T4 GPU with 12-hour session limit.

1. Create a new Kaggle Notebook
2. Enable GPU: **Settings > Accelerator > GPU T4 x2**
3. Add dataset: `keshan/multi-speaket-tts-dataset-sinhala`
4. Upload `notebooks/kaggle_train.ipynb` or copy cells manually
5. Run all cells
6. Download checkpoints before session expires
7. Upload checkpoints to resume in next session

**Config:** `configs/kaggle_t4.py` — batch_size=8, epochs=100 per session.

### <a name="colab"></a>Google Colab

Free T4 GPU with ~12-hour session limit. Checkpoints save to Google Drive for persistence.

1. Open `notebooks/colab_train.ipynb` in Colab
2. **Runtime > Change runtime type > T4 GPU**
3. Run all cells (mounts Google Drive automatically)
4. Resume in new sessions — checkpoints persist on Drive

**Config:** `configs/colab_t4.py` — batch_size=8, epochs=100 per session.

### AWS SageMaker

For production training with `ml.g5.xlarge` (A10G) or `ml.p3.2xlarge` (V100).

**Config:** `configs/sagemaker.py` — batch_size=16, epochs=300.

## Key Dependencies

| Package        | Version  | Reason                                      |
|----------------|----------|---------------------------------------------|
| coqui-tts      | 0.27.5   | VITS model + training loop                  |
| transformers   | 4.57.6   | Required by TTS for pytorch_utils imports   |
| numpy          | 2.2.6    | Numba requires NumPy ≤ 2.2                  |
| numba          | 0.61.2   | Required by librosa                         |
| librosa        | 0.11.0   | Audio feature extraction                    |

## Training Metrics

During training, eval performance is printed periodically. Values in parentheses show the change from the previous eval.

| Metric | What it means | Good trend |
|---|---|---|
| `avg_loss_mel` | Mel spectrogram reconstruction — **most important** | ↓ Decreasing |
| `avg_loss_gen` | Generator adversarial loss | ↓ Decreasing |
| `avg_loss_disc` | Discriminator loss | Stable (~2.6) |
| `avg_loss_feat` | Feature matching loss | ↓ Decreasing |
| `avg_loss_kl` | KL divergence | ↓ Slowly decreasing |
| `avg_loss_duration` | Duration prediction | ↓ Decreasing |

### Expected quality by training stage

| Epochs | Expected quality |
|--------|-----------------|
| ~16 | Noisy/robotic, verifies model is learning |
| ~100 | Recognizable speech, some artifacts |
| ~200 | Decent quality, mostly intelligible |
| ~300 | Good quality for a small dataset (~3000 samples) |

## Publishing to Hugging Face

```bash
pip install -e ".[hf]"
huggingface-cli login

# Download model from Modal
modal volume get sinhala-tts-vol runs/vits_sinhala/ ./model_export/ --force

# Upload
python scripts/upload_to_hf.py \
    --model_dir model_export/vits_sinhala-February-17-2026_12+31AM-0000000 \
    --repo_id ngpsanjaya/vits-sinhala
```

## License

MIT. Please check the [dataset license](https://www.kaggle.com/datasets/keshan/multi-speaket-tts-dataset-sinhala) for data usage terms.
