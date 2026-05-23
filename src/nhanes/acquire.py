"""STUB — owner: CS teammate. Download + cache NHANES source files.

NHANES continuous cycles (1999-2008) component files (.XPT) + the public-use
Linked Mortality File (.dat fixed-width). Cache under data/ (gitignored).

See data/README.md for the file list and CDC URLs. Idempotent: skip if already cached.
"""

from __future__ import annotations

from src import config


def acquire(cycles=("1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008")) -> None:
    """Download component + LMF files into config.DATA_DIR. TODO(cs): implement."""
    config.DATA_DIR.mkdir(exist_ok=True)
    raise NotImplementedError("CS teammate: fetch NHANES cycle files + LMF into data/. See data/README.md.")
