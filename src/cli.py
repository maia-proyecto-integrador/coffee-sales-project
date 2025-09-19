import argparse, sys, json
from pathlib import Path
from .model.main import run_all, train_and_version
from .model.paths import resolve_paths
from .model.loader import read_manifest
from .model.selector import select_best_model
from .model.pipelines import (
    data_pipeline,
    train_final_lgbm_direct,
    train_final_prophet,
    train_final_sarimax,
    train_final_lstm_direct,
)
import mlflow

def cmd_backtest(args):
    run_all()

def cmd_train_final(args):
    artifacts = train_and_version()
    print("\nArtefactos serializados:")
    for a in artifacts:
        if a:
            print(json.dumps({"name": a["name"], "version": a["version"], "path": str(a["path"])}, indent=2, ensure_ascii=False))

def cmd_log_final_to_mlflow(args):
    cwd = Path.cwd()
    _, results_dir, *_ = resolve_paths(cwd)
    versions = list((results_dir / "models").glob("*/*"))
    if not versions:
        print("No hay modelos versionados en data/results/models", file=sys.stderr)
        sys.exit(1)
    for v in versions:
        name = v.parent.name
        version = v.name
        with mlflow.start_run(run_name=f"{name}_final_{version}"):
            mlflow.log_param("version_id", version)
            mlflow.log_param("model_name", name)
            mlflow.log_artifacts(str(v))

def cmd_show_latest(args):
    cwd = Path.cwd()
    _, results_dir, *_ = resolve_paths(cwd)
    versions = sorted((results_dir / "models").glob("*/*"))
    for v in versions[-10:]:
        print(v)

def cmd_select_best(args):
    metric_priority = args.metrics.split(",") if args.metrics else None
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    cwd = Path.cwd()
    _, results_dir, *_ = resolve_paths(cwd)
    out = select_best_model(results_dir, metric_priority=metric_priority, horizon_weights=weights, prefer_stability=True)
    print(json.dumps(out, indent=2, ensure_ascii=False))

def cmd_train_selected_final(args):
    # Lee selección y entrena SOLO esa familia con versionado
    cwd = Path.cwd()
    df, data_dir, results_dir, *_ = data_pipeline(cwd)
    sel_path = results_dir / "selection" / "selected_model.json"
    if not sel_path.exists():
        raise SystemExit("No existe selection/selected_model.json. Ejecuta primero 'select-best'.")
    import json
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    fam = sel.get("selected_family")
    print(f"[train-selected-final] Seleccionado: {fam}")
    if fam == "lgbm_direct":
        a = train_final_lgbm_direct(df.copy(), results_dir)
    elif fam == "prophet":
        a = train_final_prophet(df.copy(), results_dir)
    elif fam == "sarimax":
        a = train_final_sarimax(df.copy(), results_dir)
    elif fam == "lstm_direct":
        a = train_final_lstm_direct(df.copy(), results_dir)
    else:
        raise SystemExit(f"Familia no soportada para entrenamiento final: {fam}")
    print(json.dumps({"trained": a["name"], "version": a["version"], "path": str(a["path"])}, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(prog="coffee-forecasting")
    sub = ap.add_subparsers(required=True)

    sp1 = sub.add_parser("backtest", help="Ejecuta backtesting + métricas + artefactos de evaluación")
    sp1.set_defaults(func=cmd_backtest)

    sp2 = sub.add_parser("train-final", help="Entrena y serializa modelos finales versionados")
    sp2.set_defaults(func=cmd_train_final)

    sp3 = sub.add_parser("log-final-to-mlflow", help="Sube los artefactos versionados a MLflow como runs separados")
    sp3.set_defaults(func=cmd_log_final_to_mlflow)

    sp4 = sub.add_parser("show-latest", help="Lista las últimas versiones entrenadas")
    sp4.set_defaults(func=cmd_show_latest)

    sp5 = sub.add_parser("select-best", help="Selecciona la mejor familia de modelo tras el backtesting")
    sp5.add_argument("--metrics", help="Prioridad de métricas, coma-separada. Ej: sMAPE,RMSE,MAE")
    sp5.add_argument("--weights", help="Pesos por horizonte h, coma-separados. Ej: 3,2,1,1,1,1,1")
    sp5.set_defaults(func=cmd_select_best)

    sp6 = sub.add_parser("train-selected-final", help="Entrena y versiona SOLO la familia seleccionada")
    sp6.set_defaults(func=cmd_train_selected_final)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

