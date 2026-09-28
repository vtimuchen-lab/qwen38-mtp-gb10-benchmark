"""Entry point for ``python -m sparkbench``."""

from __future__ import annotations

import sys

from sparkbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
