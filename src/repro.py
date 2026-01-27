from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional


def _get_version(pkg_name: str) -> Optional[str]:
    try:
        mod = __import__(pkg_name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def collect_versions() -> Dict[str, Optional[str]]:
    return {
        "python": platform.python_version(),
        "torch": _get_version("torch"),
        "numpy": _get_version("numpy"),
        "pandas": _get_version("pandas"),
        "sklearn": _get_version("sklearn"),
        "matplotlib": _get_version("matplotlib"),
    }


def collect_env(keys: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for k in keys:
        out[k] = os.environ.get(k)
    return out


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_metadata(command: Iterable[str], extra: Optional[Dict] = None) -> Dict:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": list(command),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "versions": collect_versions(),
    }
    if extra:
        payload.update(extra)
    return payload
