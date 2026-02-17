# dataset.py
# Dataset fetching, loading, building, and character set extraction.

import json
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from sinhala_tts.domain import ClipRecord


class KaggleDatasetFetcher:
    """Downloads dataset and returns folder where file-mapping.json is located."""
    def __init__(self, dataset_slug: str):
        self.dataset_slug = dataset_slug

    def download_and_locate_root(self) -> Path:
        import kagglehub

        ds_root = Path(kagglehub.dataset_download(self.dataset_slug))

        direct = ds_root / "file-mapping.json"
        if direct.exists():
            return ds_root

        found = next(ds_root.rglob("file-mapping.json"), None)
        if not found:
            raise FileNotFoundError("Could not find file-mapping.json after download.")
        return found.parent


class MappingLoader:
    """Loads file-mapping.json into ClipRecord objects."""
    def __init__(self, dataset_root: Path):
        self.mapping_path = dataset_root / "file-mapping.json"

    def load(self) -> Dict[str, ClipRecord]:
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Missing {self.mapping_path}")

        raw = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        out: Dict[str, ClipRecord] = {}

        for key, rec in raw.items():
            newfn = rec.get("newfn")
            text = (rec.get("text") or "").strip()
            if not newfn or not text:
                continue
            out[str(key)] = ClipRecord(
                newfn=newfn,
                text=text,
                oldfn=rec.get("oldfn"),
                duration=rec.get("duration"),
            )
        return out


class SpeakerIdParser:
    """Parses speaker ID from dataset filenames."""
    @staticmethod
    def parse(filename: str) -> str:
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "sin":
            return parts[1]
        if len(parts) >= 4 and parts[0] == "pn" and parts[1] == "sin":
            return parts[2]
        return "unknown"


class AudioPathResolver:
    """Resolves actual audio file path on disk (supports pn_ prefix)."""
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root

    def resolve(self, filename: Optional[str]) -> Optional[Path]:
        if not filename:
            return None

        fn = Path(filename).name
        pn_fn = fn if fn.startswith("pn_") else f"pn_{fn}"

        p = self.dataset_root / fn
        if p.exists():
            return p
        p = self.dataset_root / pn_fn
        if p.exists():
            return p

        found = next(self.dataset_root.rglob(fn), None)
        if found:
            return found
        found = next(self.dataset_root.rglob(pn_fn), None)
        if found:
            return found

        return None


class LJSpeechDatasetBuilder:
    """
    Builds LJSpeech-style layout:
      out_dir/wavs/*.wav
      out_dir/metadata_train.txt
      out_dir/metadata_val.txt
    metadata format: file_id|text|text
    """
    def __init__(self, resolver: AudioPathResolver):
        self.resolver = resolver

    def build(
        self,
        clips: Dict[str, ClipRecord],
        out_dir: Path,
        speaker_keep: Optional[str] = "01",
        train_ratio: float = 0.95,
        seed: int = 42,
        min_samples: int = 200,
    ) -> Tuple[int, int]:
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = out_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        rows: list[str] = []
        missing_audio = 0

        for _, clip in clips.items():
            spk = SpeakerIdParser.parse(clip.newfn)
            if speaker_keep is not None and spk != speaker_keep:
                continue

            src = self.resolver.resolve(clip.newfn) or self.resolver.resolve(clip.oldfn)
            if src is None:
                missing_audio += 1
                continue

            target = wavs_dir / Path(clip.newfn).name
            if not target.exists():
                shutil.copyfile(src, target)

            file_id = target.stem
            rows.append(f"{file_id}|{clip.text}|{clip.text}")

        if len(rows) < min_samples:
            raise RuntimeError(
                f"Too few usable samples. Got {len(rows)} (missing audio={missing_audio}). "
                f"Try speaker_keep=None or choose another speaker id."
            )

        random.seed(seed)
        random.shuffle(rows)
        split = int(len(rows) * train_ratio)

        (out_dir / "metadata_train.txt").write_text("\n".join(rows[:split]), encoding="utf-8")
        (out_dir / "metadata_val.txt").write_text("\n".join(rows[split:]), encoding="utf-8")

        return len(rows), missing_audio


class CharacterSetBuilder:
    """Builds Sinhala grapheme set from metadata."""
    PUNCTUATIONS = set(" .,!?:;\"'()-/")

    @staticmethod
    def from_metadata(metadata_path: Path) -> str:
        chars = set()
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                chars.update(parts[1])
                chars.update(parts[2])

        chars -= CharacterSetBuilder.PUNCTUATIONS
        return "".join(sorted(chars))
