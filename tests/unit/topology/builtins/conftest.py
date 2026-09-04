from __future__ import annotations

import sys
from pathlib import Path

# The shared fakes live one level up and are imported by bare module name, as the
# sibling suites do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
