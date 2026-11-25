import sys
from pathlib import Path

# make `environment` and `training` importable when running `uv run pytest`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
