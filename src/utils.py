import json
import os
from typing import Optional


def check_folder(base_path: str, name: Optional[str] = None):
    if name is not None:
        out_path = os.path.join(base_path, str(name))
    else:
        out_path = base_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)


def load_settings(path: str):
    with open(path) as f:
        settings = json.load(f)
    return settings
