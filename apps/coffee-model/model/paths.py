from pathlib import Path

def find_data_dir(start: Path) -> Path:
    candidates = [start, start.parent, start.parent.parent, start.parent.parent.parent]
    for base in candidates:
        d = base / "apps" / "coffee-model" / "model" / "datasets"
        if d.exists() and d.is_dir():
            return d
    raise RuntimeError(
        f"No encuentro el directorio 'datasets' subiendo desde: {start}. "
        "Asegúrate de que la estructura sea <repo_root>/datasets y que este módulo se ejecute dentro del repo."
    )

def resolve_paths(cwd: Path):
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results" # 
    results_dir.mkdir(exist_ok=True, parents=True)
    data_dir = find_data_dir(cwd)
    data_path = data_dir / "coffee_ml_features.csv"
    index_path = data_dir / "index_1.csv"
    return data_dir, results_dir, data_path, index_path
