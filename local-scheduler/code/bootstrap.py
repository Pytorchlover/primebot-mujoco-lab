"""Keep all PrimeBuddy state in this experiment, never in ~/.primebuddy."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["PRIMEBUDDY_HOME"] = str(ROOT / ".runtime/home")
os.environ["PRIMEBUDDY_RUNTIME"] = str(ROOT / ".runtime/processes")
sys.path.insert(0, str(ROOT / "upstream/src"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT / "code"), str(ROOT / "upstream/src")]
)
