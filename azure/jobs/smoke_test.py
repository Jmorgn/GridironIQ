"""First Azure Machine Learning smoke test for GridironIQ."""

from __future__ import annotations

from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn


def main() -> None:
    root = Path.cwd()
    required = [
        root / "README.md",
        root / "src",
        root / "azure",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Azure job did not receive the expected GridironIQ repo files: "
            + ", ".join(missing)
        )

    print("GRIDIRONIQ AZURE ML SMOKE TEST")
    print("=" * 72)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    print(f"Working dir:  {root}")
    print(f"pandas:       {pd.__version__}")
    print(f"numpy:        {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print("Repository payload: OK")
    print("Azure ML compute execution: OK")


if __name__ == "__main__":
    main()
