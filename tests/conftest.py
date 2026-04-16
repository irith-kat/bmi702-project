import sys
from pathlib import Path

# m4-pheno has a hyphen so it can't be imported as a normal package.
# Add subdirectories directly to sys.path so test files can `import rollup`,
# `import preprocessing`, `import note_ner`, `import once`, etc.
_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_src / "preprocessing" / "structured"))
sys.path.insert(0, str(_src / "preprocessing" / "nlp"))
sys.path.insert(0, str(_src / "map"))
