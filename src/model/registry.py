from pathlib import Path
import joblib, pickle, os
from typing import Any, Dict

def _safe_name(x: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in x)

def model_dir(results_dir: Path, model_name: str, version: str) -> Path:
    d = Path(results_dir) / "models" / _safe_name(model_name) / _safe_name(version)
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_sklearn_like(model: Any, out_dir: Path, filename: str = "model.joblib") -> Path:
    out = Path(out_dir) / filename
    joblib.dump(model, out)
    return out

def save_pickle(model: Any, out_dir: Path, filename: str = "model.pkl") -> Path:
    out = Path(out_dir) / filename
    with open(out, "wb") as f:
        pickle.dump(model, f)
    return out

def save_keras(model: Any, out_dir: Path, filename: str = "model_keras") -> Path:
    out = Path(out_dir) / filename
    try:
        model.save(out)  # SavedModel o carpeta (Keras 3)
    except Exception:
        # fallback a H5
        out = out.with_suffix(".h5")
        model.save(out)
    return out

def write_manifest(out_dir: Path, meta: Dict) -> Path:
    import json
    mpath = Path(out_dir) / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return mpath
