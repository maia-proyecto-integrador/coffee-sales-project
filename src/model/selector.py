from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional

DEFAULT_PRIORITY = ["sMAPE", "RMSE", "MAE", "MAPE"]  # orden de preferencia de métricas
LOWER_IS_BETTER = {"sMAPE": True, "RMSE": True, "MAE": True, "MAPE": True}

def _load_metrics(results_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    results = {}
    for family in ["lgbm_direct", "prophet", "sarimax", "lstm_direct", "baselines"]:
        by_h = results_dir / f"{family}_metrics_by_h.csv"
        overall = results_dir / f"{family}_metrics_overall.csv"
        if by_h.exists() and overall.exists():
            try:
                results[family] = {
                    "by_h": pd.read_csv(by_h),
                    "overall": pd.read_csv(overall)
                }
            except Exception:
                continue
    return results

def _compute_weighted_score(by_h_df: pd.DataFrame, metric: str, weights: Optional[List[float]]) -> float:
    if by_h_df.empty or metric not in by_h_df.columns:
        return float("inf")
    df = by_h_df.copy()
    df = df.sort_values("h")
    vals = df[metric].values
    if weights is None or len(weights) != len(vals):
        # Peso uniforme si no se suministran pesos
        weights = [1.0] * len(vals)
    # normalizamos pesos
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]
    # suma ponderada, ignorando NaNs
    import math
    num, den = 0.0, 0.0
    for v, w in zip(vals, weights):
        if pd.notna(v):
            num += w * float(v)
            den += w
    return num / den if den > 0 else float("inf")

def select_best_model(results_dir: str | Path,
                      metric_priority: Optional[List[str]] = None,
                      horizon_weights: Optional[List[float]] = None,
                      prefer_stability: bool = True) -> Dict:
    """
    Lee métricas del backtesting y elige la familia de modelo con mejor desempeño.
    - metric_priority: orden de importancia de métricas (por defecto DEFAULT_PRIORITY).
    - horizon_weights: pesos por h para ponderar la métrica by_h (si None, uniforme).
    - prefer_stability: si True, desempata por menor varianza por h.
    Retorna y guarda en JSON: data/results/selection/selected_model.json
    """
    results_dir = Path(results_dir)
    metrics = _load_metrics(results_dir)
    if not metrics:
        raise FileNotFoundError("No se encontraron métricas en data/results/*.csv. Ejecuta el backtesting primero.")
    metric_priority = metric_priority or DEFAULT_PRIORITY

    candidates = []
    for family, blocks in metrics.items():
        by_h = blocks["by_h"]
        overall = blocks["overall"]

        # 1) Ranking por métrica prioritaria (ponderada por h)
        scores = {}
        for m in metric_priority:
            scores[m] = _compute_weighted_score(by_h, m, horizon_weights)

        # 2) Medida de estabilidad (varianza por h) de la métrica principal
        import numpy as np
        stab_metric = metric_priority[0]
        variance = float(np.nanvar(by_h[stab_metric])) if stab_metric in by_h.columns else float("inf")

        # 3) Guardamos resumen
        row = {
            "family": family,
            "scores": scores,
            "stability_var": variance,
            "overall": overall.to_dict(orient="records")[0] if not overall.empty else {}
        }
        candidates.append(row)

    # Ordenamiento: por la métrica prioritaria (menor es mejor), luego estabilidad, luego 2da métrica
    primary = metric_priority[0]
    def key_fn(x):
        return (
            x["scores"].get(primary, float("inf")),
            x["stability_var"] if prefer_stability else 0.0,
            x["scores"].get(metric_priority[1], float("inf")) if len(metric_priority) > 1 else 0.0
        )
    candidates.sort(key=key_fn)

    best = candidates[0]
    out_dir = results_dir / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "selected_family": best["family"],
        "metric_priority": metric_priority,
        "scores": best["scores"],
        "stability_var": best["stability_var"],
        "overall_selected": best["overall"],
        "ranking": candidates
    }
    (out_dir / "selected_model.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
