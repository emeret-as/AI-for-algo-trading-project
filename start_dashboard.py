from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "start_dashboard.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
