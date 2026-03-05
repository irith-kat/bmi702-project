import sys
from pathlib import Path

# m4-pheno has a hyphen so it can't be imported as a normal package.
# Add it directly to sys.path so test files can `import note_ner`, `import once`, etc.
sys.path.insert(0, str(Path(__file__).parent.parent / "m4-pheno"))
