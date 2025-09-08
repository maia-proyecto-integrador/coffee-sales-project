# ===========================
# 📦 Importaciones
# ===========================
import dask.dataframe as dd
import pandas as pd
import panel as pn
import hvplot.pandas
import hvplot.dask
import numpy as np
from bokeh.models import WheelZoomTool
import holoviews as hv
import json
import joblib
from pathlib import Path

hv.extension("bokeh")

# ===========================
# ⚙️ Configuración global
# ===========================
DISABLE_SCROLL = True  # cambia a False si quieres habilitar scroll

# ===========================
# 🔒 Helper: inhabilitar scroll
# ===========================
def _no_scroll(plot, element):
    """Desactiva el scroll dentro de los plots."""
    try:
        plot.state.toolbar.active_scroll = None
        plot.state.tools = [
            t for t in plot.state.tools if not isinstance(t, WheelZoomTool)
        ]
    except Exception:
        pass


# ===========================
# 🎨 Función auxiliar para aplicar hooks
# ===========================
def _apply_hooks(base_hooks=None):
    """Devuelve hooks según configuración global."""
    hooks = base_hooks or []
    if DISABLE_SCROLL:
        hooks = hooks + [_no_scroll]
    return hooks


# ===========================
# 🎨 Estilos globales (CSS)
# ===========================
pn.extension(
    "tabulator",
    sizing_mode="stretch_width",
    raw_css=[
        """
/* ===== Fondo general ===== */
body {
    background-color: #f5f5f5 !important;
    font-family: 'Segoe UI', Tahoma, sans-serif !important;
    color: #333333 !important;
}

/* ===== Header ===== */
.bk-header {
    background-color: #ffffff !important;
}
.bk-header .bk-title {
    color: #333333 !important;
    font-weight: 600 !important;
    font-size: 20px !important;
}

/* ===== Sliders ===== */
.bk-slider-title, .bk .noUi-tooltip, .bk .noUi-value {
    font-size: 12px !important;
}
.bk-input {
    font-size: 13px !important;
}

/* ===== Tarjetas ===== */
.bk.card {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08) !important;
    padding: 16px !important;
    margin: 10px 0px !important;
}

/* ===== Títulos ===== */
h1, h2, h3, h4 {
    color: #333333 !important;
    font-weight: 600 !important;
}

/* ===== Widgets ===== */
.bk-input, .bk-slider-title, .bk-slider {
    background-color: #ffffff !important;
    color: #333333 !important;
    border-radius: 6px !important;
    border: 1px solid #cccccc !important;
    padding: 4px 8px !important;
}
.bk-root select {
    background-color: #ffffff !important;
    color: #333333 !important;
    border-radius: 6px !important;
    border: 1px solid #cccccc !important;
    font-size: 12px !important;
}

/* ===== KPIs ===== */
.pn-indicator .bk-input-group label {
    font-weight: 700 !important;
    font-size: 13px !important;
    color: #333333 !important;
}
"""
    ],
)

# ===========================
# 📂 Cargar datos
# ===========================
df = dd.read_csv(
    "data/processed/coffee_clean_dataset.csv",
    assume_missing=True,
)

# Normalizar columna datetime
df["datetime"] = dd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(
    None
)
df = df.dropna(subset=["datetime"])

# Extraer la hora
df["hour"] = df["datetime"].dt.hour

# Convertir money a numérico
df["money"] = dd.to_numeric(df["money"], errors="coerce").fillna(0)

# Asegurar coffee_name como string
df["coffee_name"] = df["coffee_name"].astype("object")

# Mapear día de la semana al español
dow_map = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}
df["dow"] = df["dow"].map(dow_map, meta=("dow", "object"))


# ===========================
# 🖼️ Template base
# ===========================
template = pn.template.FastListTemplate(
    title="☕ Coffee Machine AI Assistant",
    sidebar_width=280,
    theme_toggle=False,
    header_background="#ffffff",  # Fondo blanco
    header_color="#333333",       # Texto negro
    accent="#B8860B",
)

# ===========================
# 🎛️ Sidebar - Filtros
# ===========================

# Tipo de bebida
drink_options = ["All"] + df["coffee_name"].dropna().unique().compute().tolist()
drink_filter = pn.widgets.Select(
    name="☕ Tipo de bebida",
    options=drink_options,
    value="All",
    width=250,
)

# Período de fechas
date_min = df["datetime"].min().compute()
date_max = df["datetime"].max().compute()
date_filter = pn.widgets.DateRangeSlider(
    name="📅 Período",
    start=date_min,
    end=date_max,
    value=(date_min, date_max),
    width=250,
)

# Día de la semana
dow_options = ["All"] + df["dow"].dropna().unique().compute().tolist()
dow_filter = pn.widgets.MultiChoice(
    name="📆 Día de la semana",
    options=dow_options,
    value=["All"],
    solid=True,
    width=250,
)

# Hora del día
hour_filter = pn.widgets.IntRangeSlider(
    name="🕒 Hora del día",
    start=int(df["hour"].min().compute()),
    end=int(df["hour"].max().compute()),
    value=(
        int(df["hour"].min().compute()),
        int(df["hour"].max().compute()),
    ),
    step=1,
    width=240,
)

# Agrupar en Sidebar
sidebar_widgets = pn.WidgetBox(
    "### ⚙️ Filtros",
    drink_filter,
    date_filter,
    dow_filter,
    hour_filter,
    width=300,
    margin=(10, 10, 10, 10),
)

template.sidebar[:] = [sidebar_widgets]

# ===========================
# 📊 KPIs principales
# ===========================

# Definir indicadores
revenue_day_kpi = pn.indicators.Number(
    name="Ventas último día",
    value=0.0,
    format="₴ {value:,.0f}",
    font_size="28pt",
)

week_change_kpi = pn.indicators.Number(
    name="% cambio vs semana pasada",
    value=0.0,
    format="{value:.1f} %",
    font_size="28pt",
    colors=[(-100, "red"), (0, "orange"), (100, "green")],
)

hist_change_kpi = pn.indicators.Number(
    name="% cambio vs histórico del día",
    value=0.0,
    format="{value:.1f} %",
    font_size="28pt",
    colors=[(-100, "red"), (0, "orange"), (100, "green")],
)

revenue_total_kpi = pn.indicators.Number(
    name="Ventas total (rango)",
    value=0.0,
    format="₴ {value:,.0f}",
    font_size="28pt",
)

# Producto más vendido
top_product_kpi = pn.pane.Markdown("### Producto más vendido: N/A")


def update_kpis(drink, period, dow, hour_range):
    """Actualiza los KPIs en función de los filtros aplicados."""
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0]))
        & (df["datetime"] <= pd.to_datetime(period[1]))
    ]

    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]

    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]

    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]
    dff_pd = dff.compute()

    if dff_pd.empty:
        revenue_day_kpi.value = 0
        week_change_kpi.value = 0
        hist_change_kpi.value = 0
        revenue_total_kpi.value = 0
        top_product_kpi.object = "### Producto más vendido: N/A"
        return pn.Column(
            pn.Row(
                revenue_day_kpi,
                week_change_kpi,
                hist_change_kpi,
                revenue_total_kpi,
            ),
            top_product_kpi,
        )

    # Revenue último día
    last_day = dff_pd["datetime"].max().normalize()
    rev_last_day = float(
        dff_pd[dff_pd["datetime"].dt.normalize() == last_day]["money"].sum()
    )

    # Revenue semana anterior (mismo día)
    prev_day = last_day - pd.Timedelta(days=7)
    rev_prev = float(
        dff_pd[dff_pd["datetime"].dt.normalize() == prev_day]["money"].sum()
    )

    # Revenue histórico del mismo día de la semana
    tmp = dff_pd.assign(
        dow=dff_pd["datetime"].dt.day_name(),
        d=dff_pd["datetime"].dt.normalize(),
    )
    dow_day = last_day.day_name()
    rev_hist = float(
        tmp.loc[tmp["dow"] == dow_day].groupby("d")["money"].sum().mean() or 0.0
    )

    # Actualizar valores
    revenue_day_kpi.value = rev_last_day
    revenue_total_kpi.value = float(dff_pd["money"].sum())
    week_change_kpi.value = (
        ((rev_last_day - rev_prev) / rev_prev * 100.0) if rev_prev > 0 else 0.0
    )
    hist_change_kpi.value = (
        ((rev_last_day - rev_hist) / rev_hist * 100.0) if rev_hist > 0 else 0.0
    )

    # Producto más vendido
    prod = dff_pd.groupby("coffee_name")["money"].sum()
    top_product_kpi.object = (
        f"### Producto más vendido: {prod.idxmax() if not prod.empty else 'N/A'}"
    )

    return pn.Column(
        pn.Row(
            revenue_day_kpi,
            week_change_kpi,
            hist_change_kpi,
            revenue_total_kpi,
        ),
        top_product_kpi,
    )


# Vincular KPIs a los filtros
kpi_view = pn.bind(
    update_kpis,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
)

# Añadir al template
template.main.append(
    pn.Card(kpi_view, title="📊 KPIs principales", max_width=1300)
)

# ===========================
# 📈 Patrones de compra
# ===========================

def patrones_compra(drink, period, dow, hour_range):
    """Muestra la evolución temporal y la composición de ventas por bebida."""
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0]))
        & (df["datetime"] <= pd.to_datetime(period[1]))
    ]

    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]

    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]

    dff = dff[
        (dff["hour"] >= hour_range[0])
        & (dff["hour"] <= hour_range[1])
    ]
    dff_pd = dff.compute()

    if dff_pd.empty:
        return pn.pane.Markdown("⚠️ No hay datos para los filtros seleccionados.")

    # Serie temporal
    ts = (
        dff_pd.set_index("datetime")
        .resample("D")["money"].sum()
        .rename("Ventas")
        .to_frame()
        .reset_index()
    )
    ts_plot = ts.hvplot.line(
        x="datetime",
        y="Ventas",
        title="📈 Evolución de ventas en el rango",
        ylabel="Ventas (₴)",
        xlabel="Fecha",
        line_width=3,
        color="#B8860B",
        height=300,
        width=550,
    ).opts(responsive=True, hooks=_apply_hooks())

    # Composición por bebida
    comp = (
        dff_pd.groupby("coffee_name")["money"]
        .sum()
        .reset_index()
        .sort_values("money", ascending=True)
    )
    comp_plot = comp.hvplot.barh(
        x="coffee_name",
        y="money",
        title="🍹 Composición de ventas por bebida",
        xlabel="Ventas (₴)",
        ylabel="Tipo de bebida",
        color="#33030C",
        height=300,
        width=550,
    ).opts(responsive=True, hooks=_apply_hooks())

    return pn.Row(
        ts_plot,
        comp_plot,
        sizing_mode="stretch_width",
        align="center",
    )


# Vincular a los filtros
patterns_view = pn.bind(
    patrones_compra,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
)

# Añadir al template
template.main.append(
    pn.Card(
        pn.Column(patterns_view, sizing_mode="stretch_width"),
        title="📈 Patrones de compra",
        sizing_mode="stretch_width",
        max_width=1300,
        collapsible=True,
    )
)

# ===========================
# 📂 Cargar modelo ganador
# ===========================
import json
import joblib
from pathlib import Path

REGISTRY_PATH = Path("results/models/registry.json")

if not REGISTRY_PATH.exists():
    raise FileNotFoundError(
        "No existe registry.json, ejecuta primero step4_select_best.py"
    )

with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    registry = json.load(f)

best_model_info = registry["best_model"]
MODEL_NAME = best_model_info["name"]
ARTIFACT_PATH = Path(best_model_info["artifact"])

print(f"🏆 Modelo cargado desde registry: {MODEL_NAME}")
print(f"📦 Artefacto: {ARTIFACT_PATH}")

model = joblib.load(ARTIFACT_PATH)

# ===========================
# 🔮 Función de predicción
# ===========================
def generate_forecast(series, horizon=15):
    """
    Usa el modelo ganador (SARIMAX, Prophet o LGBM) cargado desde registry.json
    para generar un forecast.
    """
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series(
            [0.0] * horizon,
            index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D")
        )

    if MODEL_NAME.startswith("sarimax"):
        forecast = model.forecast(steps=horizon)
        idx = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )
        return pd.Series(forecast, index=idx)

    elif MODEL_NAME.startswith("prophet"):
        df = series.reset_index()
        df.columns = ["ds", "y"]
        future = model.make_future_dataframe(periods=horizon, freq="D")
        forecast_df = model.predict(future)
        return forecast_df.set_index("ds")["yhat"].iloc[-horizon:]

    elif MODEL_NAME.startswith("lgbm"):
        avg_val = series.tail(7).mean()
        idx = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )
        return pd.Series([avg_val] * horizon, index=idx)

    else:
        raise ValueError(f"Modelo {MODEL_NAME} no soportado en dashboard")


# ===========================
# 📈 Cálculo de métricas de error
# ===========================
def calculate_errors(real, fitted):
    """Calcula RMSE, MAE y MAPE de forma manual."""
    real = np.array(real, dtype=float)
    fitted = np.array(fitted, dtype=float)

    mask = ~np.isnan(real) & ~np.isnan(fitted)
    real = real[mask]
    fitted = fitted[mask]

    if len(real) == 0:
        return np.nan, np.nan, np.nan

    errors = real - fitted
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(
        np.abs(errors / np.where(real == 0, np.nan, real))
    ) * 100

    return rmse, mae, mape


# ===========================
# 🔮 Pronóstico con modelo ganador (total y por producto)
# ===========================
LOOKBACK = 7
def forecast_panel(drink, period, dow, hour_range, horizon=7, return_data=False):
    """Genera el pronóstico total a partir de la suma de pronósticos por producto."""
    # 📂 Filtros
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0]))
        & (df["datetime"] <= pd.to_datetime(period[1]))
    ]
    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]
    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]
    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]

    dff_pd = dff.compute()
    if dff_pd.empty:
        msg = "⚠️ No hay datos suficientes para generar pronósticos."
        empty_df = pd.DataFrame(columns=["date", "forecast", "coffee_name"])
        if return_data:
            return pn.pane.Markdown(msg), empty_df
        return pn.pane.Markdown(msg)

    # Normalizar nombres
    dff_pd["coffee_name"] = dff_pd["coffee_name"].astype(str).str.strip().str.title()

    # Serie real total
    ts_total = (
        dff_pd.set_index("datetime")
        .resample("D")["money"].sum()
        .rename("real")
    )

    # Forecast por producto
    prod_forecasts = []
    for prod, g in dff_pd.groupby("coffee_name"):
        ts_prod = g.set_index("datetime").resample("D")["money"].sum()
        fc_prod = generate_forecast(ts_prod, horizon)
        prod_forecasts.append(
            pd.DataFrame({
                "date": fc_prod.index,
                "forecast": fc_prod.values,
                "coffee_name": prod,
            })
        )
    prod_fc_df = pd.concat(prod_forecasts, ignore_index=True)

    # Forecast total = suma de productos
    forecast_total = (
        prod_fc_df.groupby("date")["forecast"].sum().rename("forecast")
    )

    # DataFrame para gráfico
    fc_df = pd.concat([ts_total, forecast_total], axis=1).reset_index().rename(columns={"index": "date"})

    # KPIs
    total_forecast = float(forecast_total.sum())
    prev_period = (
        ts_total.iloc[-horizon * 2 : -horizon]
        if len(ts_total) >= horizon * 2
        else ts_total.iloc[:-horizon]
    )
    prev_total = float(prev_period.sum()) if not prev_period.empty else np.nan
    pct_change = ((total_forecast - prev_total) / prev_total * 100) if prev_total > 0 else np.nan

    kpi_total = pn.indicators.Number(
        name=f"Ventas totales pronóstico ({horizon}d)",
        value=total_forecast,
        format="{value:,.1f}",
        font_size="28pt",
    )
    kpi_change = pn.indicators.Number(
        name="Cambio vs. periodo previo",
        value=pct_change,
        format="{value:+.1f}%",
        font_size="28pt",
        colors=[(-100, "red"), (0, "orange"), (100, "green")],
    )

    # Gráfico
    min_date = forecast_total.index.min() - pd.Timedelta(days=LOOKBACK)
    max_date = forecast_total.index.max()
    ymax_total = np.nanmax(fc_df[["real", "forecast"]].values) * 1.1

    total_plot = (
        fc_df.hvplot.line(
            x="date",
            y=["real", "forecast"],
            title=f"📊 Ventas totales ({MODEL_NAME})",
            ylabel="Ventas (₴)",
            xlabel="Fecha",
            line_width=2,
            height=350,
            width=1200,
            color=["#333333", "#B8860B"],
        )
        * hv.VLine(forecast_total.index.min()).opts(line_dash="dashed", color="red")
    ).opts(
        responsive=True,
        hooks=_apply_hooks(),
        xlim=(min_date, max_date),
        ylim=(0, ymax_total),
        legend_opts={"label_text_font_size": "8pt"},
    )

    panel_out = pn.Column(pn.Row(kpi_total, kpi_change), pn.pane.HoloViews(total_plot, linked_axes=False))

    if return_data:
        return panel_out, prod_fc_df
    return panel_out


def validation_panel(drink, period, dow, hour_range, horizon=7):
    """Valida el modelo contra la suma de pronósticos por producto."""
    # 📂 Filtros
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0]))
        & (df["datetime"] <= pd.to_datetime(period[1]))
    ]
    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]
    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]
    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]

    dff_pd = dff.compute()
    if dff_pd.empty:
        return pn.pane.Markdown("⚠️ No hay datos para validación.")

    # Serie real total
    ts = dff_pd.set_index("datetime").resample("D")["money"].sum().rename("real")
    if len(ts) <= horizon:
        return pn.pane.Markdown("⚠️ No hay suficientes datos para validación in-sample.")

    # División train/test
    train = ts.iloc[:-horizon]
    test = ts.iloc[-horizon:]

    # Forecast por producto en train
    # Forecast total = usar mismos pasos que forecast_panel
    prod_forecasts = []
    for prod, g in dff_pd.groupby("coffee_name"):
        ts_prod = g.set_index("datetime").resample("D")["money"].sum()
        fc_prod = generate_forecast(ts_prod, horizon)
        prod_forecasts.append(
            pd.Series(fc_prod.values, index=fc_prod.index)
        )
    forecast_total = pd.concat(prod_forecasts, axis=1).sum(axis=1).rename("forecast")

    # Cortar forecast al rango de test
    forecast_total = forecast_total.iloc[:len(test)]
    forecast_total.index = test.index



    # DataFrame validación
    val_df = pd.DataFrame({"date": test.index, "real": test.values, "forecast": forecast_total.values})

    # Métricas
    rmse, mae, mape = calculate_errors(val_df["real"].values, val_df["forecast"].values)

    kpi_mape = pn.indicators.Number(name="MAPE (%)", value=mape, format="{value:.1f}%", font_size="30pt",
                                    colors=[(-100, "green"), (30, "orange"), (100, "red")])
    kpi_rmse = pn.indicators.Number(name="RMSE", value=rmse, format="{value:,.0f}", font_size="18pt")
    kpi_mae = pn.indicators.Number(name="MAE", value=mae, format="{value:,.0f}", font_size="18pt")
    status = pn.pane.Markdown(f"### Validación con modelo: **{MODEL_NAME}**")

    # Gráfico
    ymax_val = np.nanpercentile(val_df[["real", "forecast"]].values, 95) * 1.2
    plot = (
        val_df.hvplot.line(
            x="date",
            y=["real", "forecast"],
            title=f"🔍 Validación in-sample ({MODEL_NAME})",
            ylabel="Ventas (₴)",
            xlabel="Fecha",
            line_width=2,
            height=300,
            width=1200,
            color=["#333333", "#FF6600"],
        )
        * hv.VLine(test.index[0]).opts(line_dash="dashed", color="red")
    ).opts(
        xlim=(test.index[0], test.index[-1]),
        ylim=(0, ymax_val),
        responsive=True,
        hooks=_apply_hooks(),
    )

    return pn.Column(pn.Row(kpi_mape, kpi_rmse, kpi_mae), status, pn.pane.HoloViews(plot, linked_axes=False))


# ===========================
# 🔍 Validación en dashboard
# ===========================
validation_view = pn.bind(
    validation_panel,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
)

template.main.append(
    pn.Card(
        pn.Column(validation_view, sizing_mode="stretch_width"),
        title="🔍 Validación del modelo ganador",
        max_width=1300,
        sizing_mode="stretch_width",
        collapsible=True,
    )
)

# ===========================
# 📦 Cálculo de insumos
# ===========================
RECIPES = {
    "Espresso": {"coffee_g": 18, "water_ml": 30, "sugar_g": 5},
    "Cappuccino": {"coffee_g": 18, "milk_ml": 150, "sugar_g": 10},
    "Latte": {"coffee_g": 18, "milk_ml": 200, "sugar_g": 10},
    "Americano": {"coffee_g": 18, "water_ml": 100, "sugar_g": 5},
    "Mocha": {"coffee_g": 18, "milk_ml": 150, "chocolate_g": 25, "sugar_g": 12},
    "Americano With Milk": {"coffee_g": 18, "water_ml": 80, "milk_ml": 50, "sugar_g": 8},
    "Cocoa": {"milk_ml": 200, "chocolate_g": 30, "sugar_g": 15},
    "Cortado": {"coffee_g": 18, "milk_ml": 50, "sugar_g": 5},
    "Hot Chocolate": {"milk_ml": 200, "chocolate_g": 25, "sugar_g": 15},
}


PRICE_REF = {
    "coffee_g": 2.0,       # ~2000 ₴/kg ≈ 2 ₴/g
    "milk_ml": 0.048,      # ~48 ₴/L ≈ 0.048 ₴/ml
    "water_ml": 0.001,     # simbólico
    "chocolate_g": 0.5,    # ~500 ₴/kg ≈ 0.5 ₴/g
    "sugar_g": 0.07        # ~70 ₴/kg ≈ 0.07 ₴/g
}

# ===========================
# 📦 Cálculo de insumos
# ===========================
def calculate_supply_needs(prod_fc_df, df_hist):
    """Calcula insumos y costos a partir de pronósticos y precios recientes."""
    if prod_fc_df.empty:
        print("DEBUG calculate_supply_needs → prod_fc_df está VACÍO")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0.0

    # 🔑 Normalizar nombres: quitar espacios y unificar mayúsculas
    prod_fc_df["coffee_name"] = (
        prod_fc_df["coffee_name"]
        .astype(str)
        .str.strip()           # quita espacios antes/después
        .str.replace(r"\s+", " ", regex=True)  # colapsa espacios múltiples
        .str.title()           # todo en Title Case
    )

    if not isinstance(df_hist, pd.DataFrame):
        df_hist = df_hist.compute()

    recent_prices = (
        df_hist.sort_values("datetime")
        .groupby("coffee_name")["money"]
        .last()
        .to_dict()
    )

    insumo_rows = []
    for prod, forecast_sum in prod_fc_df.groupby("coffee_name")["forecast"].sum().items():
        recipe = RECIPES.get(prod)
        if recipe is None:
            print(f"DEBUG calculate_supply_needs → producto sin receta: '{prod}'")
            continue
        for insumo, qty in recipe.items():
            cantidad = forecast_sum * qty
            costo_unitario = PRICE_REF.get(insumo, 0)
            costo_total = cantidad * costo_unitario
            insumo_rows.append({
                "Producto": prod,
                "Insumo": insumo,
                "Cantidad": cantidad,
                "Costo unitario (₴)": costo_unitario,
                "Costo total (₴)": costo_total,
            })

    if not insumo_rows:
        print("DEBUG calculate_supply_needs → no se generaron insumos válidos")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0.0

    supply_df = pd.DataFrame(insumo_rows)
    resumen = (
        supply_df.groupby("Insumo")[["Cantidad", "Costo total (₴)"]]
        .sum()
        .reset_index()
    )

    # Redondear
    supply_df["Cantidad"] = supply_df["Cantidad"].round(2)
    supply_df["Costo total (₴)"] = supply_df["Costo total (₴)"].round(2)
    resumen["Cantidad"] = resumen["Cantidad"].round(2)
    resumen["Costo total (₴)"] = resumen["Costo total (₴)"].round(2)

    total_cost = resumen["Costo total (₴)"].sum()

    ingresos = 0.0
    for prod, forecast_sum in prod_fc_df.groupby("coffee_name")["forecast"].sum().items():
        precio = recent_prices.get(prod, 0)
        ingresos += forecast_sum * precio

    profit = ingresos - total_cost
    return supply_df, resumen, total_cost, ingresos, profit


# ===========================
# 📦 Inventario y costos
# ===========================
def inventory_panel(prod_fc_df, df_hist):
    """Genera un panel con el inventario proyectado de insumos."""
    supply_df, resumen_df, _, _, _ = calculate_supply_needs(prod_fc_df, df_hist)

    if supply_df.empty:
        return pn.pane.Markdown("⚠️ No hay datos suficientes para calcular inventario.")

    detalle_table = pn.widgets.Tabulator(
        supply_df,
        formatters={"Cantidad": {"type": "number", "decimals": 0}},
        height=240,
        sizing_mode="stretch_width",
    )
    resumen_table = pn.widgets.Tabulator(
        resumen_df,
        formatters={"Cantidad": {"type": "number", "decimals": 0}},
        height=180,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        "### 📦 Detalle insumos por producto",
        detalle_table,
        "### 📊 Resumen total de insumos",
        resumen_table,
    )


# ===========================
# 💰 Finanzas (ingresos vs gastos)
# ===========================
def finances_panel(prod_fc_df, df_hist):
    """Genera un panel con el balance de ingresos, costos y ganancia neta."""
    _, _, total_cost, ingresos, profit = calculate_supply_needs(prod_fc_df, df_hist)

    kpi_cost = pn.indicators.Number(
        name="Costo total insumos",
        value=total_cost,
        format="{value:,.0f} ₴",
        font_size="22pt",
    )
    kpi_revenue = pn.indicators.Number(
        name="Ingresos proyectados",
        value=ingresos,
        format="{value:,.0f} ₴",
        font_size="22pt",
    )
    kpi_profit = pn.indicators.Number(
        name="Ganancia neta proyectada",
        value=profit,
        format="{value:,.0f} ₴",
        font_size="24pt",
        colors=[(-1e9, "red"), (0, "orange"), (1e9, "green")],
    )

    return pn.Row(kpi_cost, kpi_revenue, kpi_profit)


# ===========================
# 📦 Integrar en dashboard
# ===========================

forecast_view = pn.bind(
    forecast_panel,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
)


# Pronóstico dinámico con datos
def forecast_with_data(drink, period, dow, hour_range):
    return forecast_panel(
        drink=drink,
        period=period,
        dow=dow,
        hour_range=hour_range,
        return_data=True
    )

# Bind conjunto: devuelve (panel_out, prod_fc_df)
forecast_bound = pn.bind(
    forecast_with_data,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
)

# Vista del forecast (solo panel)
forecast_view = pn.bind(lambda x: x[0], forecast_bound)

# Datos del forecast (solo DataFrame)
forecast_data = pn.bind(lambda x: x[1], forecast_bound)

# Inventario ligado dinámicamente
inventory_view = pn.bind(lambda data: inventory_panel(data, df), forecast_data)

# Finanzas ligadas dinámicamente
finances_view = pn.bind(lambda data: finances_panel(data, df), forecast_data)

# Añadir al template
template.main.append(
    pn.Card(
        forecast_view,
        title="🔮 Pronóstico de ventas (modelo ganador)",
        max_width=1300,
        sizing_mode="stretch_width",
        collapsible=True,
    )
)

template.main.append(
    pn.Card(
        inventory_view,
        title="📦 Inventario proyectado",
        max_width=1300,
        sizing_mode="stretch_width",
        collapsible=True,
    )
)

template.main.append(
    pn.Card(
        finances_view,
        title="💰 Finanzas proyectadas",
        max_width=1300,
        sizing_mode="stretch_width",
        collapsible=True,
    )
)

# Cachear forecast para no recalcular en inventario/finanzas
def get_forecast_cached():
    """Obtiene forecast_data y lo cachea en pn.state.cache para no recalcularlo."""
    if "forecast_data" not in pn.state.cache:
        _, forecast_data = forecast_panel(
            drink=drink_filter.value,
            period=date_filter.value,
            dow=dow_filter.value,
            hour_range=hour_filter.value,
            return_data=True,
        )
        # ✅ Normalizar nombres
        if not forecast_data.empty:
            forecast_data["coffee_name"] = (
                forecast_data["coffee_name"].astype(str).str.strip().str.title()
            )
        # ⚠️ Guardar solo si hay detalle real
        pn.state.cache["forecast_data"] = forecast_data
    return pn.state.cache["forecast_data"]


# ===========================
# 🚀 Servir dashboard
# ===========================
template.servable()

