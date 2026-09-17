"""Run every test suite; non-zero exit code if any fails."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
failed = 0
for f in sorted(here.glob("test_*.py")) + [here / "smoke_ui.py"]:
    failed |= subprocess.call([sys.executable, str(f)], cwd=here)
print("ALL SUITES PASSED" if not failed else "SOME SUITES FAILED")
sys.exit(failed)
