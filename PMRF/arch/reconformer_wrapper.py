"""
Import wrapper for the cloned ReconFormer repository.

Temporarily adds ``ReconFormer/`` to ``sys.path`` so that the internal imports
inside ``Recurrent_Transformer.py`` (e.g. ``from models.RS_attention import ...``)
resolve correctly.  The ``from data import transforms`` import inside ReconFormer
resolves to PMRF's ``data/transforms.py`` which has a compatible API.
"""

import sys
from pathlib import Path

# Locate the cloned ReconFormer repo relative to this file:
#   PMRF/arch/reconformer_wrapper.py  ->  ../../ReconFormer
_RECONFORMER_DIR = str(Path(__file__).resolve().parent.parent.parent / "ReconFormer")

# Temporarily prepend so ``from models.RS_attention import ...`` finds
# ReconFormer/models/.  We insert *after* the project root (which is
# usually sys.path[0]) so that ``from data import transforms`` still
# resolves to PMRF's data module first.
_already_on_path = _RECONFORMER_DIR in sys.path
if not _already_on_path:
    sys.path.insert(1, _RECONFORMER_DIR)

from models.Recurrent_Transformer import ReconFormer  # noqa: E402

# Clean up: remove the path entry to avoid polluting imports elsewhere.
if not _already_on_path and _RECONFORMER_DIR in sys.path:
    sys.path.remove(_RECONFORMER_DIR)

__all__ = ["ReconFormer"]
