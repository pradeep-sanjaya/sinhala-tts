# scripts/train_modal.py
# Modal + Coqui TTS (VITS) Sinhala training script
#
# Usage:
#   modal run scripts/train_modal.py
#   modal run scripts/train_modal.py --restore
#   modal run --detach scripts/train_modal.py --restore
#
# Outputs:
#   /vol/runs/vits_sinhala/ (checkpoints + best_model.pth + config.json)

import sys
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Ensure the project root is importable so `sinhala_tts` resolves correctly
# both locally (for Modal serialisation) and inside the container.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sinhala_tts.dataset import (
    KaggleDatasetFetcher,
    MappingLoader,
    AudioPathResolver,
    LJSpeechDatasetBuilder,
)
from sinhala_tts.training import VitsConfigWriter, CoquiTrainer

# =========================
# Modal Infrastructure
# =========================

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

app = modal.App("sinhala-tts-train")

vol = modal.Volume.from_name("sinhala-tts-vol", create_if_missing=True)

VOL_ROOT = Path("/vol")
DATA_DIR = VOL_ROOT / "data"
RUN_DIR = VOL_ROOT / "runs" / "vits_sinhala"

DATASET_SLUG = "keshan/multi-speaket-tts-dataset-sinhala"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "coqui-tts==0.27.5",
        "soundfile",
        "kagglehub[pandas-datasets]",
        "pandas",
        "sentencepiece",
        "tokenizers",
    )
    .run_commands(
        f"pip install --force-reinstall 'torch==2.6.0' 'torchaudio==2.6.0' --index-url {TORCH_INDEX}",
        "pip install -U --force-reinstall --no-deps "
        "'numpy==2.2.6' "
        "'numba==0.61.2' "
        "'llvmlite==0.44.0' "
        "'librosa==0.11.0' "
        "'transformers==4.57.6' && "
        "pip install 'huggingface-hub>=0.34.0,<1.0'",
        "python -c \"import numpy as np; import numba; import librosa; "
        "print('numpy', np.__version__); print('numba', numba.__version__); print('librosa', librosa.__version__)\"",
        "python -c \""
        "import torch; v = torch.__version__; "
        "print('torch', v, 'cuda', torch.version.cuda); "
        "assert '+cu124' in v or 'cu124' in v, "
        "f'Expected torch+cu124 build but got {v}. PyPI torch will crash on Modal GPUs.'\"",
    )
    .add_local_python_source("sinhala_tts")
)

# =========================
# Config loader
# =========================

def load_platform_config(config_name: str = "modal_a100") -> dict:
    """Load a platform config from configs/ directory."""
    cfg_path = _PROJECT_ROOT / "configs" / f"{config_name}.py"
    if not cfg_path.exists():
        print(f"Config {cfg_path} not found, using defaults.")
        return {}
    import importlib.util
    spec = importlib.util.spec_from_file_location("platform_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "training", {})


# =========================
# Modal Functions
# =========================

@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60)
def prepare_dataset(speaker_keep: Optional[str] = "01") -> str:
    out_dir = DATA_DIR / (f"sinhala_ljspeech_sin{speaker_keep}" if speaker_keep else "sinhala_ljspeech_all")
    wavs_dir = out_dir / "wavs"
    train_meta = out_dir / "metadata_train.txt"
    val_meta = out_dir / "metadata_val.txt"

    if wavs_dir.exists():
        wav_count = len(list(wavs_dir.glob("*.wav")))
        if wav_count > 0 and train_meta.exists() and val_meta.exists():
            print(f"Dataset already present: {out_dir} (wavs={wav_count}). Skipping download/build.")
            return str(out_dir)

    fetcher = KaggleDatasetFetcher(DATASET_SLUG)
    ds_root = fetcher.download_and_locate_root()

    clips = MappingLoader(ds_root).load()
    resolver = AudioPathResolver(ds_root)
    builder = LJSpeechDatasetBuilder(resolver)

    total, missing = builder.build(clips=clips, out_dir=out_dir, speaker_keep=speaker_keep)
    vol.commit()
    print(f"Prepared dataset at {out_dir} (total={total}, missing_audio={missing})")
    return str(out_dir)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/vol": vol},
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2),
)
def train(speaker_keep: Optional[str] = "01", restore: bool = False, config: str = "modal_a100") -> str:
    dataset_dir = Path(prepare_dataset.remote(speaker_keep=speaker_keep))

    platform_cfg = load_platform_config(config)
    cfg_path = VitsConfigWriter(cfg=platform_cfg).write(dataset_dir, RUN_DIR)

    restore_path = None
    if restore:
        restore_path = CoquiTrainer.find_latest_checkpoint(RUN_DIR)
        if restore_path:
            print(f"Resuming from checkpoint: {restore_path}")
        else:
            print("No checkpoint found, starting fresh.")

    CoquiTrainer().run(cfg_path, restore_path=restore_path)
    vol.commit()
    return str(RUN_DIR)


@app.local_entrypoint()
def main(restore: bool = False, config: str = "modal_a100"):
    out = train.remote(speaker_keep="01", restore=restore, config=config)
    print("Training outputs stored at:", out)
