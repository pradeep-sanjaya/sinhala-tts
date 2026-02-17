# domain.py
# Domain model for clip records.

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClipRecord:
    newfn: str
    text: str
    oldfn: Optional[str] = None
    duration: Optional[float] = None
