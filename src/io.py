import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_reference(path):
    return pd.read_csv(path,dtype=str,keep_default_na=False)

def read_aliases(path=None):
    return json.loads(Path(path or ROOT/'data/aliases.json').read_text(encoding='utf-8'))
