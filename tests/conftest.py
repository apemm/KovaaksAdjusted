import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Real KovaaK's install, if present (skipped on CI / other machines).
_ROOT = os.environ.get(
    "KOVAAKS_ROOT",
    r"C:\Program Files (x86)\Steam\steamapps\common\FPSAimTrainer\FPSAimTrainer",
)


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def kovaaks_root() -> Path:
    p = Path(_ROOT)
    if not (p / "stats").is_dir():
        pytest.skip("KovaaK's install not available")
    return p
