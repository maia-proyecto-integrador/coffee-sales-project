from pathlib import Path

def find_data_dir(start: Path) -> Path:
    candidates = [start, start.parent, start.parent.parent, start.parent.parent.parent]
    for base in candidates:
        d = base / "data"
        if d.exists() and d.is_dir():
            return d
    raise RuntimeError(
        f"No encuentro el directorio 'data' subiendo desde: {start}. "
        "Asegúrate de que la estructura sea <repo_root>/data y que este módulo se ejecute dentro del repo."
    )

def resolve_paths(cwd: Path):
    data_dir = find_data_dir(cwd)
    results_dir = data_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    data_path = data_dir / "processed" / "coffee_ml_features.csv"
    index_path = data_dir / "processed" / "index_1.csv"
    return data_dir, results_dir, data_path, index_path
