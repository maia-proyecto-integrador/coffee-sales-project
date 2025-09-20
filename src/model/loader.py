from pathlib import Path
import json
import joblib
import pickle

def read_manifest(path: str | Path) -> dict:
    path = Path(path)
    mpath = path / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"No se encontró manifest.json en {path}")
    return json.loads(mpath.read_text(encoding="utf-8"))

def load_lgbm_direct(version_dir: str | Path) -> dict:
    """Carga un conjunto de modelos LGBM direct (center/p10/p90 por horizonte).
    Retorna: { 'features': [...], 'models': {1:{'center':..., 'p10':..., 'p90':...}, ...} }"""
    vdir = Path(version_dir)
    mani = read_manifest(vdir)
    feats = mani.get("features", [])
    out = {"features": feats, "models": {}}
    for hdir in sorted([d for d in vdir.glob("h*") if d.is_dir()]):
        h = int(hdir.name.replace("h",""))
        out["models"][h] = {
            "center": joblib.load(hdir / "center.joblib"),
            "p10": joblib.load(hdir / "p10.joblib"),
            "p90": joblib.load(hdir / "p90.joblib"),
        }
    return out

def load_prophet(version_dir: str | Path) -> dict:
    vdir = Path(version_dir)
    mani = read_manifest(vdir)
    models = {}
    for pdir in vdir.glob("product=*"):
        prod = pdir.name.split("=", 1)[1]
        with open(pdir / "model.pkl", "rb") as f:
            models[prod] = pickle.load(f)
    return {"models": models, "manifest": mani}

def load_sarimax(version_dir: str | Path) -> dict:
    vdir = Path(version_dir)
    mani = read_manifest(vdir)
    models = {}
    for pdir in vdir.glob("product=*"):
        prod = pdir.name.split("=", 1)[1]
        with open(pdir / "model.pkl", "rb") as f:
            models[prod] = pickle.load(f)
    return {"models": models, "manifest": mani}

def load_lstm_direct(version_dir: str | Path) -> dict:
    from keras import models as kmodels
    import joblib as jl
    vdir = Path(version_dir)
    mani = read_manifest(vdir)
    model_path = vdir / "model_keras"
    if not model_path.exists():
        model_path = vdir / "model_keras.h5"
    model = kmodels.load_model(model_path)
    scaler = jl.load(vdir / "scaler.joblib")
    return {"model": model, "scaler": scaler, "manifest": mani}
