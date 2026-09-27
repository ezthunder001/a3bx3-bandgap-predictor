import sys
from pathlib import Path

# Import the package from src/ without requiring `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
