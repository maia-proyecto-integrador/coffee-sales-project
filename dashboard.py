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

hv.extension("bokeh")


# ===========================
# ⚙️ Configuración global
# ===========================
DISABLE_SCROLL = True  # cambia a False si quieres habilitar scroll

# ===========================
# 🔒 Helper: inhabilitar scroll
# ===========================
def _no_scroll(plot, element):
    try:
        plot.state.toolbar.active_scroll = None
        plot.state.tools = [t for t in plot.state.tools if not isinstance(t, WheelZoomTool)]
    except Exception:
        pass

# ===========================
# 🎨 Función auxiliar para aplicar hooks
# ===========================
def _apply_hooks(base_hooks=None):
    """Devuelve hooks según config global"""
    hooks = base_hooks or []
    if DISABLE_SCROLL:
        hooks = hooks + [_no_scroll]
    return hooks

# ===========================
# 🎨 Estilos globales (CSS)
# ===========================
pn.extension("tabulator", sizing_mode="stretch_width", raw_css=[ """
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
    font-weight: 700 !important; /* negrita */
    font-size: 13px !important;  /* más compacto */
    color: #333333 !important;
}
""" ])


# ===========================
# 📂 Cargar datos
# ===========================
df = dd.read_csv(
    "index_1.csv",
    dtype={"holiday_name": "object"},
    assume_missing=True
)

# Normalizar fecha y variables clave
df["datetime"] = dd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
df = df.dropna(subset=["datetime"])
df["money"] = dd.to_numeric(df["money"], errors="coerce").fillna(0)
df["coffee_name"] = df["coffee_name"].astype("object")

# Mapear día de la semana con meta explícito
dow_map = {
    0.0: "Lunes", 1.0: "Martes", 2.0: "Miércoles",
    3.0: "Jueves", 4.0: "Viernes", 5.0: "Sábado", 6.0: "Domingo"
}
df["dow"] = df["dow"].map(dow_map, meta=('dow', 'object'))

# ===========================
# 🖼️ Template base
# ===========================
template = pn.template.FastListTemplate(
    title="☕ Coffee Machine AI Assistant",
    sidebar_width=280,
    theme_toggle=False,
    header_background="#ffffff",  # fondo blanco
    header_color="#333333",       # <-- texto negro
    accent="#B8860B"
)


# ===========================
# 🎛️ Sidebar - Filtros (ajustado con ancho fijo)
# ===========================

# Tipo de bebida
drink_options = ["All"] + df["coffee_name"].dropna().unique().compute().tolist()
drink_filter = pn.widgets.Select(
    name="☕ Tipo de bebida",
    options=drink_options,
    value="All",
    width=250
)

# Período de fechas
date_min = df["datetime"].min().compute()
date_max = df["datetime"].max().compute()
date_filter = pn.widgets.DateRangeSlider(
    name="📅 Período",
    start=date_min, end=date_max,
    value=(date_min, date_max),
    width=250
)

# Día de la semana
dow_options = ["All"] + df["dow"].dropna().unique().compute().tolist()
dow_filter = pn.widgets.MultiChoice(
    name="📆 Día de la semana",
    options=dow_options,
    value=["All"],
    solid=True,
    width=250
)

# Hora del día
hour_filter = pn.widgets.IntRangeSlider(
    name="🕒 Hora del día",
    start=int(df["hour"].min().compute()),
    end=int(df["hour"].max().compute()),
    value=(int(df["hour"].min().compute()), int(df["hour"].max().compute())),
    step=1,
    width=240
)

# Festivo
holiday_filter = pn.widgets.Checkbox(
    name="🎉 Solo festivos",
    value=False,
    width=250
)

# Selección de modelo de forecast
model_filter = pn.widgets.Select(
    name="🔮 Modelo de forecast",
    options=["holt-winters", "arima", "prophet", "naive"],
    value="holt-winters",
    width=250
)

# Organizar en WidgetBox
sidebar_widgets = pn.WidgetBox(
    "### ⚙️ Filtros",
    drink_filter,
    date_filter,
    dow_filter,
    hour_filter,
    holiday_filter,
    model_filter,
    width=300,
    margin=(10, 10, 10, 10)
)

# Añadir al template
template.sidebar[:] = [sidebar_widgets]


# ===========================
# 📊 KPIs principales
# ===========================

# Definir indicadores
revenue_day_kpi = pn.indicators.Number(
    name="Ventas último día", value=0.0,
    format="₴ {value:,.0f}", font_size="28pt"
)
week_change_kpi = pn.indicators.Number(
    name="% cambio vs semana pasada", value=0.0,
    format="{value:.1f} %", font_size="28pt",
    colors=[(-100,"red"), (0,"orange"), (100,"green")]
)
hist_change_kpi = pn.indicators.Number(
    name="% cambio vs histórico del día", value=0.0,
    format="{value:.1f} %", font_size="28pt",
    colors=[(-100,"red"), (0,"orange"), (100,"green")]
)
revenue_total_kpi = pn.indicators.Number(
    name="Ventas total (rango)", value=0.0,
    format="₴ {value:,.0f}", font_size="28pt"
)

# Producto más vendido
top_product_kpi = pn.pane.Markdown("### Producto más vendido: N/A")

# Función para actualizar KPIs con filtros
def update_kpis(drink, period, dow, hour_range, holiday, model):
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0])) &
        (df["datetime"] <= pd.to_datetime(period[1]))
    ]

    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]
    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]
    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]
    if holiday:
        dff = dff[dff["is_holiday"] == True]

    dff_pd = dff.compute()
    if dff_pd.empty:
        revenue_day_kpi.value = 0
        week_change_kpi.value = 0
        hist_change_kpi.value = 0
        revenue_total_kpi.value = 0
        top_product_kpi.object = "### Producto más vendido: N/A"
        return pn.Column(
            pn.Row(revenue_day_kpi, week_change_kpi, hist_change_kpi, revenue_total_kpi),
            top_product_kpi
        )

    # Revenue último día
    last_day = dff_pd["datetime"].max().normalize()
    rev_last_day = float(dff_pd[dff_pd["datetime"].dt.normalize() == last_day]["money"].sum())

    # Revenue semana anterior (mismo día)
    prev_day = last_day - pd.Timedelta(days=7)
    rev_prev = float(dff_pd[dff_pd["datetime"].dt.normalize() == prev_day]["money"].sum())

    # Revenue histórico del mismo día de la semana
    tmp = dff_pd.assign(dow=dff_pd["datetime"].dt.day_name(), d=dff_pd["datetime"].dt.normalize())
    dow_day = last_day.day_name()
    rev_hist = float(tmp.loc[tmp["dow"] == dow_day].groupby("d")["money"].sum().mean() or 0.0)

    # Actualizar valores
    revenue_day_kpi.value = rev_last_day
    revenue_total_kpi.value = float(dff_pd["money"].sum())
    week_change_kpi.value = ((rev_last_day - rev_prev) / rev_prev * 100.0) if rev_prev > 0 else 0.0
    hist_change_kpi.value = ((rev_last_day - rev_hist) / rev_hist * 100.0) if rev_hist > 0 else 0.0

    # Producto más vendido
    prod = dff_pd.groupby("coffee_name")["money"].sum()
    top_product_kpi.object = f"### Producto más vendido: {prod.idxmax() if not prod.empty else 'N/A'}"

    return pn.Column(
        pn.Row(revenue_day_kpi, week_change_kpi, hist_change_kpi, revenue_total_kpi),
        top_product_kpi
    )

# Vincular KPIs a los filtros
kpi_view = pn.bind(
    update_kpis,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
    holiday=holiday_filter,
    model=model_filter
)

# Añadir al template
template.main.append(
    pn.Card(kpi_view, title="📊 KPIs principales", max_width=1300)
)

# ===========================
# 📈 Patrones de compra
# ===========================

def patrones_compra(drink, period, dow, hour_range, holiday, model):
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0])) &
        (df["datetime"] <= pd.to_datetime(period[1]))
    ]
    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]
    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]
    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]
    if holiday:
        dff = dff[dff["is_holiday"] == True]

    dff_pd = dff.compute()
    if dff_pd.empty:
        return pn.pane.Markdown("⚠️ No hay datos para los filtros seleccionados.")

    # Serie temporal
    ts = (
        dff_pd.set_index("datetime").resample("D")["money"].sum()
        .rename("Ventas").to_frame().reset_index()
    )
    ts_plot = ts.hvplot.line(
        x="datetime", y="Ventas",
        title="📈 Evolución de ventas en el rango",
        ylabel="Ventas (₴)", xlabel="Fecha",
        line_width=3, color="#B8860B", height=300, width=550
    ).opts(responsive=True, hooks=_apply_hooks())

    # Composición
    comp = (
        dff_pd.groupby("coffee_name")["money"].sum()
        .reset_index().sort_values("money", ascending=True)
    )
    comp_plot = comp.hvplot.barh(
        x="coffee_name", y="money",
        title="🍹 Composición de ventas por bebida",
        xlabel="Ventas (₴)", ylabel="Tipo de bebida",
        color="#33030C", height=300, width=550
    ).opts(responsive=True, hooks=_apply_hooks())

    # Ajuste en proporciones (fila con dos columnas iguales)
    return pn.Row(
        ts_plot, comp_plot,
        sizing_mode="stretch_width",
        align="center"
    )


# Vincular a los filtros
patterns_view = pn.bind(
    patrones_compra,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
    holiday=holiday_filter,
    model=model_filter
)

# Añadir al template
template.main.append(
    pn.Card(
        pn.Column(patterns_view, sizing_mode="stretch_width"),
        title="📈 Patrones de compra",
        sizing_mode="stretch_width",
        max_width=1300,
        collapsible=True
    )
)

# ===========================
# 📦 Modelos y Pronósticos
# ===========================

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_holt_winters(series, horizon):
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series([0.0] * horizon,
                         index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D"))
    model = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated"
    )
    fit = model.fit(optimized=True)
    return fit.forecast(horizon)

def forecast_arima(series, horizon):
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series([0.0] * horizon,
                         index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D"))
    model = ARIMA(series, order=(1,1,1))
    fit = model.fit()
    return fit.forecast(horizon)

def forecast_naive(series, horizon):
    last_val = series.iloc[-1] if not series.empty else 0.0
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                        periods=horizon, freq="D") if not series.empty else pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D")
    return pd.Series([last_val] * horizon, index=idx)

def forecast_prophet(series, horizon):
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series([0.0] * horizon,
                         index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D"))
    df = series.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)
    forecast = forecast.set_index("ds")["yhat"].iloc[-horizon:]
    return forecast

def forecast_ses(series, horizon):
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series([0.0] * horizon,
                         index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D"))
    model = SimpleExpSmoothing(series)
    fit = model.fit(optimized=True)
    return fit.forecast(horizon)

def forecast_sarima(series, horizon):
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series([0.0] * horizon,
                         index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D"))
    # Ejemplo: SARIMA(1,1,1)x(1,1,1,7) → estacionalidad semanal
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7))
    fit = model.fit(disp=False)
    return fit.forecast(horizon)

def forecast_ma(series, horizon, window=7):
    series = series.astype(float).dropna()
    if series.empty:
        return pd.Series([0.0] * horizon,
                         index=pd.date_range(pd.Timestamp.today(), periods=horizon, freq="D"))
    last_avg = series.tail(window).mean()
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                        periods=horizon, freq="D")
    return pd.Series([last_avg] * horizon, index=idx)


MODELS = {
    "holt-winters": forecast_holt_winters,
    "arima": forecast_arima,
    "sarima": forecast_sarima,
    "ses": forecast_ses,
    "ma": forecast_ma,
    "naive": forecast_naive,
    "prophet": forecast_prophet
}


import numpy as np

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
    rmse = np.sqrt(np.mean(errors ** 2))               # RMSE
    mae = np.mean(np.abs(errors))                      # MAE
    mape = np.mean(np.abs(errors / np.where(real == 0, np.nan, real))) * 100  # MAPE %

    return rmse, mae, mape

def run_forecast(series, model_name, horizon, fallback="naive"):
    """
    Ejecuta forecast con un modelo elegido.
    Si falla (error en ajuste o predicción), usa fallback.
    """
    try:
        model_func = MODELS[model_name]
        forecast = model_func(series, horizon)
    except Exception:
        forecast = MODELS[fallback](series, horizon)
        return forecast, {"status": f"❌ Modelo {model_name} falló → fallback {fallback}"}

    # Validación in-sample (sin usar fallback por métricas)
    try:
        if len(series) > horizon:
            train = series.iloc[:-horizon]
            test = series.iloc[-horizon:]
            fitted_forecast = model_func(train, horizon)
            rmse, mae, mape = calculate_errors(test.values, fitted_forecast.values)
        else:
            rmse, mae, mape = np.nan, np.nan, np.nan
    except Exception:
        rmse, mae, mape = np.nan, np.nan, np.nan

    return forecast, {
        "status": f"✅ {model_name} ejecutado correctamente",
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }


# ===========================
# 🔮 Pronóstico de ventas
# ===========================
LOOKBACK=7

def forecast_panel(drink, period, dow, hour_range, holiday, model, horizon=15, return_data=False):
    # 📂 Filtros
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0])) & 
        (df["datetime"] <= pd.to_datetime(period[1]))
    ]
    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]
    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]
    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]
    if holiday:
        dff = dff[dff["is_holiday"] == True]

    dff_pd = dff.compute()
    if dff_pd.empty:
        if return_data:
            return pn.pane.Markdown("⚠️ No hay datos suficientes para generar pronósticos."), pd.DataFrame()
        return pn.pane.Markdown("⚠️ No hay datos suficientes para generar pronósticos.")

    # 📈 Serie total
    ts = dff_pd.set_index("datetime").resample("D")["money"].sum().rename("Ventas")

    forecast, metrics = run_forecast(ts, model, horizon)
    fc_df = pd.concat([ts.rename("real"), forecast.rename("forecast")], axis=1).reset_index()
    fc_df = fc_df.rename(columns={"index": "date"})

    # ✅ Rango dinámico
    min_date = forecast.index.min() - pd.Timedelta(days=LOOKBACK)
    max_date = forecast.index.max()

    shown_vals_total = fc_df.loc[
        (fc_df["date"] >= min_date) & (fc_df["date"] <= max_date),
        ["real", "forecast"]
    ].values
    ymax_total = np.nanmax(shown_vals_total) * 1.1 if shown_vals_total.size > 0 else 1.5

    # 📊 KPIs de pronóstico
    total_forecast = float(forecast.sum())
    prev_period = ts.iloc[-horizon * 2:-horizon] if len(ts) >= horizon * 2 else ts.iloc[:-horizon]
    prev_total = float(prev_period.sum()) if not prev_period.empty else np.nan
    pct_change = ((total_forecast - prev_total) / prev_total * 100) if prev_total > 0 else np.nan

    prod_forecasts = []
    for prod, g in dff_pd.groupby("coffee_name"):
        series = g.set_index("datetime").resample("D")["money"].sum()
        fc, _ = run_forecast(series, model, horizon)
        prod_forecasts.append(pd.DataFrame({"coffee_name": prod, "forecast": fc}))

    prod_fc_df = pd.concat(prod_forecasts, ignore_index=True)

    top_prod = (
        prod_fc_df.groupby("coffee_name")["forecast"].sum()
        .idxmax() if not prod_fc_df.empty else "N/A"
    )

    kpi_total = pn.indicators.Number(
        name=f"Ventas totales pronóstico ({horizon}d)",
        value=total_forecast,
        format="{value:,.1f}", font_size="28pt"
    )
    kpi_change = pn.indicators.Number(
        name="Cambio vs. periodo previo",
        value=pct_change,
        format="{value:+.1f}%", font_size="28pt",
        colors=[(-100, "red"), (0, "orange"), (100, "green")]
    )
    kpi_top = pn.pane.Markdown(f"## Producto más demandado\n**{top_prod}**")

    # 📊 Gráfico total
    total_plot = (
        fc_df.hvplot.line(
            x="date", y=["real", "forecast"],
            title=f"📊 Total ventas (LOOKBACK + forecast) - {model}",
            ylabel="Ventas (₴)", xlabel="Fecha",
            line_width=2, height=350, width=1200,
            color=["#333333", "#B8860B"]
        )
        * hv.VLine(forecast.index.min()).opts(line_dash="dashed", color="red", apply_ranges=False)
    ).opts(
        responsive=True, hooks=_apply_hooks(),
        xlim=(min_date, max_date), ylim=(0, ymax_total),
        legend_opts={"label_text_font_size": "8pt"}
    )
    total_plot = pn.pane.HoloViews(total_plot, linked_axes=False)

    # ☕ Forecast por producto
    prod_forecasts = []
    for prod, g in dff_pd.groupby("coffee_name"):
        series = g.set_index("datetime").resample("D")["money"].sum().rename("real")
        fc, _ = run_forecast(series, model, horizon)

        df_fc = pd.concat([series.rename("real"), fc.rename("forecast")], axis=1).reset_index()
        if "datetime" in df_fc.columns:
            df_fc = df_fc.rename(columns={"datetime": "date"})
        elif "index" in df_fc.columns:
            df_fc = df_fc.rename(columns={"index": "date"})

        df_fc["coffee_name"] = prod
        prod_forecasts.append(df_fc)

    prod_fc_df = pd.concat(prod_forecasts, ignore_index=True)

    # ✅ Rango dinámico igual que en total
    min_date_prod = forecast.index.min() - pd.Timedelta(days=LOOKBACK)
    max_date_prod = forecast.index.max()

    ymax_prod = np.nanmax(
        pd.concat([prod_fc_df["real"], prod_fc_df["forecast"]], axis=0).values
    ) * 1.1

    prod_plot_real = prod_fc_df.hvplot.line(
        x="date", y="real", by="coffee_name",
        line_width=1.5, height=400, width=1200
    )
    prod_plot_fc = prod_fc_df.hvplot.line(
        x="date", y="forecast", by="coffee_name",
        line_dash="dotted", line_width=1.5, height=400, width=1200
    )

    prod_plot = (prod_plot_real * prod_plot_fc * hv.VLine(forecast.index.min()).opts(
        line_dash="dashed", color="red", apply_ranges=False
    )).opts(
        responsive=True, hooks=_apply_hooks(),
        xlim=(min_date_prod, max_date_prod),
        ylim=(0, ymax_prod),
        legend_opts={"label_text_font_size": "6pt", "ncols": 2},
        title=f"☕ Forecast de ventas por producto (LOOKBACK + forecast)",
        ylabel="Ventas (₴)", xlabel="Fecha"
    )
    prod_plot = pn.pane.HoloViews(prod_plot, linked_axes=False)

    # 📦 Ventas proyectadas (₴)
    supply_table = prod_fc_df.groupby("coffee_name")["forecast"].sum().reset_index()
    supply_table["forecast"] = supply_table["forecast"].round().astype(int)
    supply_table.columns = ["Producto", f"Forecast ({horizon}d)"]

    supply_widget = pn.widgets.Tabulator(
        supply_table,
        formatters={f"Forecast ({horizon}d)": {"type": "number", "decimals": 0}},
        height=220, sizing_mode="stretch_width"
    )

    panel_out = pn.Column(
        pn.Row(kpi_total, kpi_change, kpi_top),
        total_plot,
        prod_plot,
        "### 📦 Ventas proyectadas",
        supply_widget
    )

    if return_data:
        return panel_out, prod_fc_df
    return panel_out


# ===========================
# 📊 Validación de modelo
# ===========================
def validation_panel(drink, period, dow, hour_range, holiday, model, horizon=15):
    # 📂 Filtros
    dff = df[
        (df["datetime"] >= pd.to_datetime(period[0])) & 
        (df["datetime"] <= pd.to_datetime(period[1]))
    ]
    if drink != "All":
        dff = dff[dff["coffee_name"] == drink]
    if dow and "All" not in dow:
        dff = dff[dff["dow"].isin(dow)]
    dff = dff[(dff["hour"] >= hour_range[0]) & (dff["hour"] <= hour_range[1])]
    if holiday:
        dff = dff[dff["is_holiday"] == True]

    dff_pd = dff.compute()
    if dff_pd.empty:
        return pn.pane.Markdown("⚠️ No hay datos para validación.")

    # 📈 Serie diaria
    ts = dff_pd.set_index("datetime").resample("D")["money"].sum().rename("Ventas")
    if len(ts) <= horizon:
        return pn.pane.Markdown("⚠️ No hay suficientes datos para validación in-sample.")

    # 🔍 División train/test
    train = ts.iloc[:-horizon]
    test = ts.iloc[-horizon:]

    model_func = MODELS.get(model, forecast_naive)
    fitted_forecast = model_func(train, horizon)

    # 🔑 Asegurar alineación EXACTA
    fitted_forecast = fitted_forecast.iloc[:len(test)]
    fitted_forecast.index = test.index

    # DataFrame limpio SOLO de validación
    val_df = pd.DataFrame({
        "date": test.index,
        "real": test.values,
        "forecast": fitted_forecast.values
    }).dropna()

    if val_df.empty:
        return pn.pane.Markdown("⚠️ No se pudo generar validación por datos nulos.")

    # Rango SOLO de validación
    f0_val, f1_val = val_df["date"].iloc[0], val_df["date"].iloc[-1]
    ymax_val = np.nanpercentile(val_df[["real", "forecast"]].values, 95) * 1.2

    # 📊 Métricas
    rmse, mae, mape = calculate_errors(val_df["real"].values, val_df["forecast"].values)

    kpi_mape = pn.indicators.Number(
        name="MAPE (%)", value=mape,
        format="{value:.1f}%", font_size="30pt",
        colors=[(-100, "green"), (30, "orange"), (100, "red")]
    )
    kpi_rmse = pn.indicators.Number(name="RMSE", value=rmse,
                                    format="{value:,.0f}", font_size="18pt")
    kpi_mae = pn.indicators.Number(name="MAE", value=mae,
                                   format="{value:,.0f}", font_size="18pt")
    status = pn.pane.Markdown(f"### Estado del modelo elegido: {model}")

    # 📈 Gráfico SOLO en rango de validación
    base_plot = val_df.hvplot.line(
        x="date", y=["real", "forecast"],
        title=f"🔍 Validación in-sample ({model})",
        ylabel="Ventas (₴)", xlabel="Fecha",
        line_width=2, height=300, width=1200,
        color=["#333333", "#FF6600"]
    )

    vline = hv.VLine(f0_val).opts(line_dash="dashed", color="red", apply_ranges=False)

    plot = (base_plot * vline).opts(
        xlim=(f0_val, f1_val), ylim=(0, ymax_val),
        responsive=True, hooks=_apply_hooks(),
        legend_opts={"label_text_font_size": "8pt"}
    )

    # 🚨 Encapsular para que NO comparta ejes globales
    val_plot = pn.pane.HoloViews(plot, linked_axes=False)

    return pn.Column(
        pn.Row(kpi_mape, kpi_rmse, kpi_mae),
        status,
        val_plot
    )


# Pronóstico de ventas
forecast_view = pn.bind(
    forecast_panel,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
    holiday=holiday_filter,
    model=model_filter
)

template.main.append(
    pn.Card(
        pn.Column(forecast_view, sizing_mode="stretch_width"),
        title="🔮 Pronóstico de ventas (total y productos)",
        max_width=1300,
        sizing_mode="stretch_width",
        collapsible=True
    )
)

# Validación de modelo
validation_view = pn.bind(
    validation_panel,
    drink=drink_filter,
    period=date_filter,
    dow=dow_filter,
    hour_range=hour_filter,
    holiday=holiday_filter,
    model=model_filter
)

template.main.append(
    pn.Card(
        pn.Column(validation_view, sizing_mode="stretch_width"),
        title="🔍 Validación de modelo",
        max_width=1300,
        sizing_mode="stretch_width",
        collapsible=True
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
}

PRICE_REF = {
    "coffee_g": 2.0,        # ~2000 ₴/kg ≈ 2 ₴/g
    "milk_ml": 0.048,       # ~48 ₴/L ≈ 0.048 ₴/ml
    "water_ml": 0.001,      # simbólico
    "chocolate_g": 0.5,     # ~500 ₴/kg ≈ 0.5 ₴/g
    "sugar_g": 0.07         # ~70 ₴/kg ≈ 0.07 ₴/g
}

def calculate_supply_needs(prod_fc_df, df_hist):
    """
    Calcula insumos y costos a partir de pronósticos y precios reales recientes.
    - prod_fc_df: DataFrame con columnas ["date","real","forecast","coffee_name"]
    - df_hist: Dask o pandas DataFrame histórico para extraer precios de venta.
    """
    if prod_fc_df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0.0

    # Asegurar pandas
    if not isinstance(df_hist, pd.DataFrame):
        df_hist = df_hist.compute()

    # Precio de venta más reciente por producto
    recent_prices = (
        df_hist.sort_values("datetime")
        .groupby("coffee_name")["money"]
        .last()
        .to_dict()
    )

    insumo_rows = []
    for prod, forecast_sum in prod_fc_df.groupby("coffee_name")["forecast"].sum().items():
        units = forecast_sum
        recipe = RECIPES.get(prod, {})
        for insumo, qty in recipe.items():
            cantidad = units * qty
            costo_unitario = PRICE_REF.get(insumo, 0)
            costo_total = cantidad * costo_unitario
            insumo_rows.append({
                "Producto": prod,
                "Insumo": insumo,
                "Cantidad": cantidad,
                "Costo unitario (₴)": costo_unitario,
                "Costo total (₴)": costo_total
            })

    # DataFrame detallado
    supply_df = pd.DataFrame(insumo_rows)

    # Resumen por insumo
    resumen = (
        supply_df.groupby("Insumo")[["Cantidad","Costo total (₴)"]]
        .sum().reset_index()
    )

    # 🔑 Redondear a 2 decimales
    supply_df["Cantidad"] = supply_df["Cantidad"].round(2)
    supply_df["Costo total (₴)"] = supply_df["Costo total (₴)"].round(2)
    resumen["Cantidad"] = resumen["Cantidad"].round(2)
    resumen["Costo total (₴)"] = resumen["Costo total (₴)"].round(2)

    total_cost = resumen["Costo total (₴)"].sum()

    # Ingresos proyectados usando precios recientes
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
    supply_df, resumen_df, total_cost, ingresos, profit = calculate_supply_needs(prod_fc_df, df_hist)

    if supply_df.empty:
        return pn.pane.Markdown("⚠️ No hay datos suficientes para calcular inventario.")

    # Tablas
    detalle_table = pn.widgets.Tabulator(
        supply_df,
        formatters={"Cantidad": {"type":"number","decimals":0}},
        height=240, sizing_mode="stretch_width"
    )
    resumen_table = pn.widgets.Tabulator(
        resumen_df,
        formatters={"Cantidad": {"type":"number","decimals":0}},
        height=180, sizing_mode="stretch_width"
    )

    return pn.Column(
        "### 📦 Detalle insumos por producto",
        detalle_table,
        "### 📊 Resumen total de insumos",
        resumen_table
    )


# ===========================
# 💰 Finanzas (comparación ingresos vs gastos)
# ===========================
def finances_panel(prod_fc_df, df_hist):
    _, _, total_cost, ingresos, profit = calculate_supply_needs(prod_fc_df, df_hist)

    kpi_cost = pn.indicators.Number(
        name="Costo total insumos", value=total_cost,
        format="{value:,.0f} ₴", font_size="22pt"
    )
    kpi_revenue = pn.indicators.Number(
        name="Ingresos proyectados", value=ingresos,
        format="{value:,.0f} ₴", font_size="22pt"
    )
    kpi_profit = pn.indicators.Number(
        name="Ganancia neta proyectada", value=profit,
        format="{value:,.0f} ₴", font_size="24pt",
        colors=[(-1e9,"red"), (0,"orange"), (1e9,"green")]
    )

    return pn.Row(kpi_cost, kpi_revenue, kpi_profit)


# Forecast
forecast_view, forecast_data = pn.bind(
    lambda **kwargs: forecast_panel(**kwargs, return_data=True),
    drink=drink_filter, period=date_filter, dow=dow_filter,
    hour_range=hour_filter, holiday=holiday_filter, model=model_filter
)()

# Inventario
inventory_view = pn.bind(
    lambda **kwargs: inventory_panel(forecast_panel(**kwargs, return_data=True)[1], df),
    drink=drink_filter, period=date_filter, dow=dow_filter,
    hour_range=hour_filter, holiday=holiday_filter, model=model_filter
)

template.main.append(
    pn.Card(inventory_view, title="📦 Inventario", max_width=1300,
            sizing_mode="stretch_width", collapsible=True)
)

# Finanzas
finances_view = pn.bind(
    lambda **kwargs: finances_panel(forecast_panel(**kwargs, return_data=True)[1], df),
    drink=drink_filter, period=date_filter, dow=dow_filter,
    hour_range=hour_filter, holiday=holiday_filter, model=model_filter
)

template.main.append(
    pn.Card(finances_view, title="💰 Finanzas proyectadas", max_width=1300,
            sizing_mode="stretch_width", collapsible=True)
)

def select_best_model(series, horizon=15):
    results = {}
    for name, func in MODELS.items():
        try:
            if len(series) > horizon:
                train = series.iloc[:-horizon]
                test = series.iloc[-horizon:]
                forecast = func(train, horizon)
                forecast = forecast.iloc[:len(test)]
                forecast.index = test.index
                _, _, mape = calculate_errors(test.values, forecast.values)
                results[name] = mape
        except Exception:
            results[name] = np.inf
    # Devolver el modelo con menor MAPE
    best_model = min(results, key=results.get)
    return best_model, results

# Serie total nacional (ventas diarias agregadas, en pandas)
df_pd = df.compute().sort_values("datetime")
ts_total = df_pd.set_index("datetime")["money"].resample("D").sum()
best_model, model_scores = select_best_model(ts_total, horizon=15)

# Sidebar con modelo por defecto = mejor
model_filter = pn.widgets.Select(
    name="🔮 Modelo de forecast",
    options=list(MODELS.keys()),
    value=best_model,
    width=250
)


template.servable()
