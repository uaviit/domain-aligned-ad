"""Shared configuration, path, and protocol utilities."""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "p6_cal.json"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    return json.loads(path.read_text(encoding="utf-8"))


def supplied_path(value, env_name, required=True):
    candidate = value or os.environ.get(env_name)
    if not candidate:
        if required:
            raise ValueError(f"Supply --{env_name.lower().replace('_', '-')} or set {env_name}")
        return None
    return Path(candidate).expanduser().resolve()


def load_protocol(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol_image_paths(protocol, data_root):
    data_root = Path(data_root)
    train = [data_root / "good" / name for name in protocol["train_normal"]]
    normal = [data_root / "good" / name
              for name in protocol["calibration_normal"] + protocol["test_normal"]]
    defective = [data_root / "bad" / name for name in protocol["test_defective"]]
    return train, normal + defective


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
