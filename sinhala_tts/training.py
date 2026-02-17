# training.py
# VITS config writing and Coqui training execution.

import os
from pathlib import Path

from sinhala_tts.dataset import CharacterSetBuilder


class VitsConfigWriter:
    """Writes config.json using Coqui config classes."""

    def __init__(self, cfg: dict = None):
        """
        Args:
            cfg: Optional dict of overrides for VitsConfig fields.
                 Keys match VitsConfig constructor args, e.g.:
                 {"batch_size": 64, "epochs": 300, "mixed_precision": True}
        """
        self.overrides = cfg or {}

    def write(self, dataset_dir: Path, run_dir: Path) -> Path:
        import numpy as np
        import numba
        import librosa
        import transformers
        import transformers.pytorch_utils as pu
        import transformers.utils.import_utils as iu

        print("numpy:", np.__version__)
        print("numba:", numba.__version__)
        print("librosa:", librosa.__version__)
        print("transformers:", transformers.__version__)
        print("isin_mps_friendly:", hasattr(pu, "isin_mps_friendly"))
        print("is_torchcodec_available:", hasattr(iu, "is_torchcodec_available"))

        from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import VitsAudioConfig

        run_dir.mkdir(parents=True, exist_ok=True)

        dataset_cfg = BaseDatasetConfig(
            dataset_name="sinhala",
            path=str(dataset_dir),
            meta_file_train="metadata_train.txt",
            meta_file_val="metadata_val.txt",
            formatter="ljspeech",
        )

        chars = CharacterSetBuilder.from_metadata(dataset_dir / "metadata_train.txt")
        characters = CharactersConfig(
            characters_class="TTS.tts.utils.text.characters.Graphemes",
            characters=chars,
            punctuations=" .,!?:;\"'()-/",
            phonemes="",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
        )

        audio = VitsAudioConfig(sample_rate=22050)

        # Defaults — can be overridden via self.overrides
        defaults = dict(
            audio=audio,
            run_name="vits_sinhala",
            output_path=str(run_dir),
            datasets=[dataset_cfg],
            use_phonemes=False,
            text_cleaner="basic_cleaners",
            characters=characters,
            batch_size=64,
            eval_batch_size=16,
            max_audio_len=500000,
            epochs=300,
            mixed_precision=True,
            num_loader_workers=4,
            num_eval_loader_workers=2,
            run_eval_steps=2000,
            save_step=2000,
            print_step=500,
        )
        defaults.update(self.overrides)

        config = VitsConfig(**defaults)

        out = run_dir / "config.json"
        config.save_json(str(out))
        return out


class CoquiTrainer:
    """Runs Coqui training command."""

    @staticmethod
    def find_latest_checkpoint(run_dir: Path) -> Path | None:
        """Find the latest checkpoint or best_model in the run directory."""
        candidates = []
        for sub in sorted(run_dir.iterdir()) if run_dir.exists() else []:
            if not sub.is_dir():
                continue
            best = sub / "best_model.pth"
            if best.exists():
                candidates.append(best)
            for ckpt in sorted(sub.glob("checkpoint_*.pth")):
                candidates.append(ckpt)
        if not candidates:
            return None
        # Return the most recently modified file
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def run(self, config_path: Path, restore_path: Path | None = None) -> None:
        import sys
        import json
        import torch

        print(f"torch version : {torch.__version__}")
        print(f"torch CUDA    : {torch.version.cuda}")
        print(f"cuDNN version : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"cuDNN enabled : {torch.backends.cudnn.enabled}")
        if torch.cuda.is_available():
            print(f"GPU           : {torch.cuda.get_device_name(0)}")
            print(f"GPU cap       : {torch.cuda.get_device_capability(0)}")

        # --- Vocabulary sanity check ---
        cfg = json.loads(config_path.read_text())
        char_cfg = cfg.get("characters", {})
        vocab_chars = char_cfg.get("characters", "")
        vocab_puncts = char_cfg.get("punctuations", "")
        pad = char_cfg.get("pad", "<PAD>")
        bos = char_cfg.get("bos", "<BOS>")
        eos = char_cfg.get("eos", "<EOS>")
        blank = char_cfg.get("blank", "<BLNK>")
        # Total vocab = characters + punctuations + special tokens
        all_symbols = list(vocab_chars) + list(vocab_puncts)
        # Add special tokens
        specials = [pad, bos, eos, blank]
        total_vocab = len(all_symbols) + len(specials)
        print(f"Vocab chars   : {len(vocab_chars)} -> {repr(vocab_chars[:80])}")
        print(f"Vocab puncts  : {len(vocab_puncts)} -> {repr(vocab_puncts)}")
        print(f"Total vocab   : {total_vocab} (chars={len(vocab_chars)}, puncts={len(vocab_puncts)}, specials={len(specials)})")

        # Check dataset texts for out-of-vocab characters
        datasets_cfg = cfg.get("datasets", [])
        if datasets_cfg:
            ds = datasets_cfg[0]
            meta_path = Path(ds["path"]) / ds["meta_file_train"]
            if meta_path.exists():
                known = set(vocab_chars) | set(vocab_puncts)
                oov = set()
                for line in meta_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split("|")
                    if len(parts) >= 2:
                        text = parts[1]
                        for ch in text:
                            if ch not in known:
                                oov.add(ch)
                if oov:
                    print(f"WARNING: {len(oov)} out-of-vocab chars in training data: {sorted(oov)[:30]}")
                else:
                    print("All training text chars are in vocab. OK.")

        sys.argv = ["train_tts", "--config_path", str(config_path)]
        if restore_path:
            print(f"Restoring from: {restore_path}")
            sys.argv += ["--restore_path", str(restore_path)]
        from TTS.bin.train_tts import main
        main()
