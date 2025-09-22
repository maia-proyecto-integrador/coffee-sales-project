
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pickle
from datetime import datetime
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _read_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}

def _write_registry(registry: Dict[str, Any]) -> None:
    # Asegura POSIX y pretty print
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def save_artifact(model_obj: Any, name: str, ts: Optional[str] = None) -> Path:
    """
    Serializa un objeto (SARIMAXResults, Prophet, bundle LGBM, etc.) con joblib.
    Devuelve la ruta absoluta al .joblib.
    """
    ts = ts or _now_tag()
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model_obj, path)
    return path.resolve()

def update_best_model_entry(
    name: str,
    artifact_path: Path,
    metric: str,
    score: float,
    horizon: int,
    trained_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Actualiza (o crea) la entrada 'best_model' del registry con rutas POSIX relativas a BASE_DIR.
    """
    registry = _read_registry()

   
    try:
        rel = artifact_path.relative_to(BASE_DIR)
    except ValueError:
        
        rel = artifact_path

    entry = {
        "name": str(name),
        "artifact": rel.as_posix(),   
        "metric": str(metric),
        "score": float(score),
        "horizon": int(horizon),
        "trained_at": trained_at or _now_tag()
    }
    registry["best_model"] = entry
    _write_registry(registry)
    return entry

def register_best_model(
    model_obj: Any,
    name: str,
    metric: str,
    score: float,
    horizon: int,
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper todo-en-uno:
    1) Guarda el artefacto en results/models/<name>_<ts>.joblib
    2) Actualiza results/models/registry.json con ese artefacto (ruta POSIX relativa)
    3) Devuelve el dict 'best_model'
    """
    ts = ts or _now_tag()
    artifact_abspath = save_artifact(model_obj, name=name, ts=ts)
    return update_best_model_entry(
        name=name,
        artifact_path=artifact_abspath,
        metric=metric,
        score=score,
        horizon=horizon,
        trained_at=ts
    )

def load_pipeline(*, file_name: str) -> Any:
    """Load a persisted pipeline."""

    file_path = MODELS_DIR / f"{file_name}.joblib"
    trained_model = joblib.load(filename=file_path)
    return trained_model
