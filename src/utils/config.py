"""Configuration loader supporting YAML configs with default fallbacks."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    root = get_project_root()
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", str(root / "config" / "config.yaml"))
    
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
