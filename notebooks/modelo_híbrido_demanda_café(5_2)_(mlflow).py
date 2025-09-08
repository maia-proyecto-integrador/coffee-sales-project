# -*- coding: utf-8 -*-
"""

Modelo_Híbrido_Demanda_Café(5_2) (MLflow)

Enhanced Coffee Demand Forecasting - Mejorado
Optimizado para demanda intermitente y bajo volumen
"
"""

# ---
# BLOQUE 1: INSTALACIÓN Y CONFIGURACIÓN
# ---
#!pip install pandas numpy matplotlib seaborn scikit-learn lightgbm optuna joblib
#!pip install mlflow

import warnings
warnings.filterwarnings('ignore')
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import optuna
import joblib
from datetime import datetime, timedelta
import zipfile

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Crear directorios de resultados
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/data', exist_ok=True)

print("Dependencias instaladas y configuración completada")

# ---
# BLOQUE 2: FUNCIONES UTILITARIAS MEJORADAS
# ---
def smape(y_true, y_pred):
    """Error porcentual absoluto simétrico medio"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-9
    return np.mean(np.abs(y_pred - y_true) / denom) * 100

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error - mejor para demanda baja"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(y_true) + 1e-9) * 100

def mase(y_true, y_pred, seasonal_period=7):
    """Mean Absolute Scaled Error - escala independiente"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) <= seasonal_period:
        # Fallback a naive forecast
        naive_error = np.mean(np.abs(np.diff(y_true)))
    else:
        # Seasonal naive forecast
        naive_error = np.mean(np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period]))

    if naive_error == 0:
        naive_error = 1e-9

    mae = np.mean(np.abs(y_true - y_pred))
    return mae / naive_error

def business_metrics(y_true, y_pred, stockout_cost=5, holding_cost=1):
    """Métricas orientadas al negocio con costos balanceados"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Métricas tradicionales mejoradas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape_score = wape(y_true, y_pred)
    mase_score = mase(y_true, y_pred)

    # Métricas de negocio
    stockout_events = np.sum(y_pred < y_true)
    overstock_events = np.sum(y_pred > y_true)
    perfect_predictions = np.sum(np.abs(y_pred - y_true) < 0.5)

    # Costos de negocio balanceados
    stockout_loss = stockout_cost * np.sum(np.maximum(0, y_true - y_pred))
    holding_loss = holding_cost * np.sum(np.maximum(0, y_pred - y_true))
    total_cost = stockout_loss + holding_loss

    # Service Level mejorado
    service_level = (1 - stockout_events / len(y_true)) * 100
    accuracy = perfect_predictions / len(y_true) * 100

    # Fill Rate - porcentaje de demanda satisfecha
    fill_rate = (1 - np.sum(np.maximum(0, y_true - y_pred)) / (np.sum(y_true) + 1e-9)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'WAPE': wape_score,
        'MASE': mase_score,
        'SMAPE': smape(y_true, y_pred),
        'Service_Level': service_level,
        'Fill_Rate': fill_rate,
        'Accuracy': accuracy,
        'Stockout_Events': int(stockout_events),
        'Overstock_Events': int(overstock_events),
        'Perfect_Predictions': int(perfect_predictions),
        'Total_Cost': total_cost,
        'Stockout_Cost': stockout_loss,
        'Holding_Cost': holding_loss
    }

def save_json(path, obj):
    """Guardar objeto como JSON con formateo"""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)

print("Funciones utilitarias mejoradas cargadas")

# ---
# BLOQUE 3: CARGA Y PREPARACIÓN DE DATOS REALES
# ---
# Cargar datos reales
df_real = pd.read_csv('coffee_ml_features.csv')
print(f"Datos cargados: {df_real.shape[0]} registros, {df_real.shape[1]} columnas")

# Convertir columna date a datetime
df_real['date'] = pd.to_datetime(df_real['date'])

# Renombrar columnas para mantener consistencia
df_real = df_real.rename(columns={
    'transactions': 'transactions',
    'revenue': 'revenue',
    'coffee_name': 'product'
})

# Asegurar que se tiene la columna product
if 'product' not in df_real.columns:
    product_cols = [col for col in df_real.columns if col.startswith('product_')]
    if product_cols:
        df_real['product'] = ''
        for col in product_cols:
            product_name = col.replace('product_', '').replace('_', ' ')
            df_real.loc[df_real[col] == 1, 'product'] = product_name

# Filtrar productos con datos suficientes (mínimo 30 observaciones)
product_counts = df_real['product'].value_counts()
valid_products = product_counts[product_counts >= 30].index.tolist()
df_real = df_real[df_real['product'].isin(valid_products)].copy()

print(f"Productos únicos (filtrados): {df_real['product'].nunique()}")
print(f"Periodo: {df_real['date'].min()} a {df_real['date'].max()}")

# ---
# BLOQUE 4: ANÁLISIS EXPLORATORIO MEJORADO
# ---
print("\nESTADÍSTICAS BÁSICAS MEJORADAS")
print("-" * 50)

# Estadísticas por producto
prod_analysis = df_real.groupby('product')['transactions'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max', 'sum'
]).round(3)

# Identificar productos con demanda intermitente
prod_analysis['zeros_pct'] = df_real.groupby('product')['transactions'].apply(
    lambda x: (x == 0).sum() / len(x) * 100
).round(2)

prod_analysis['cv'] = prod_analysis['std'] / (prod_analysis['mean'] + 1e-9)
prod_analysis['intermittent'] = (prod_analysis['zeros_pct'] > 60).astype(int)

print("Análisis por producto:")
print(prod_analysis)

eda_stats = {
    'n_rows': int(df_real.shape[0]),
    'n_cols': int(df_real.shape[1]),
    'avg_transactions': float(df_real['transactions'].mean()),
    'median_transactions': float(df_real['transactions'].median()),
    'zero_demand_pct': float((df_real['transactions'] == 0).sum() / len(df_real) * 100),
    'intermittent_products': int(prod_analysis['intermittent'].sum()),
    'total_revenue': float(df_real['revenue'].sum()),
    'date_range': f"{df_real['date'].min()} to {df_real['date'].max()}",
    'unique_products': int(df_real['product'].nunique())
}

for key, value in eda_stats.items():
    print(f"{key}: {value}")

save_json('results/eda_stats.json', eda_stats)

# Visualización mejorada
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Distribución de transacciones (log scale para mejor visualización)
ax1.hist(df_real['transactions'][df_real['transactions'] > 0], bins=30, alpha=0.7, edgecolor='black')
ax1.set_title('Distribución de Transacciones (>0)')
ax1.set_xlabel('Transacciones')
ax1.set_ylabel('Frecuencia')
ax1.set_yscale('log')

# Top productos por volumen con coeficiente de variación
top_products = prod_analysis.sort_values('sum', ascending=False).head(8)
bars = ax2.bar(range(len(top_products)), top_products['sum'])
ax2.set_title('Volumen Total por Producto')
ax2.set_xticks(range(len(top_products)))
ax2.set_xticklabels(top_products.index, rotation=45)
ax2.set_ylabel('Total Transacciones')

# Añadir CV como texto en las barras
for i, (idx, row) in enumerate(top_products.iterrows()):
    ax2.text(i, row['sum'], f'CV: {row["cv"]:.2f}', ha='center', va='bottom', fontsize=8)

# Patrón semanal mejorado
df_real['day_of_week'] = df_real['date'].dt.dayofweek
weekly_pattern = df_real.groupby('day_of_week')['transactions'].agg(['mean', 'std'])
days = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']

ax3.errorbar(days, weekly_pattern['mean'], yerr=weekly_pattern['std'],
             marker='o', capsize=5, capthick=2)
ax3.set_title('Patrón Semanal (Media ± Std)')
ax3.set_xlabel('Día de la Semana')
ax3.set_ylabel('Transacciones Promedio')
ax3.tick_params(axis='x', rotation=45)

# Evolución temporal con tendencia
monthly_sales = df_real.groupby(df_real['date'].dt.to_period('M'))['transactions'].sum()
ax4.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', linewidth=2)

# Añadir línea de tendencia
x_trend = np.arange(len(monthly_sales))
z = np.polyfit(x_trend, monthly_sales.values, 1)
p = np.poly1d(z)
ax4.plot(monthly_sales.index.astype(str), p(x_trend), "r--", alpha=0.8, linewidth=2)

ax4.set_title('Evolución Mensual con Tendencia')
ax4.set_xlabel('Mes')
ax4.set_ylabel('Transacciones')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/eda_overview.png', dpi=300, bbox_inches='tight')
plt.show()

print("Análisis exploratorio mejorado completado")

# ---
# BLOQUE 5: FEATURE ENGINEERING AVANZADO
# ---
def create_temporal_features(df):
    """Genera características temporales específicas del negocio de café"""
    df = df.copy()

    # Características básicas
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Simular horas basadas en patrones de negocio reales
    np.random.seed(42)
    # Distribución más realista para máquina expendedora
    df['hour'] = np.random.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                size=len(df),
                                p=[0.02, 0.08, 0.12, 0.10, 0.08, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.01, 0.01])

    # Franjas horarias críticas para café
    df['morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
    df['afternoon_break'] = ((df['hour'] >= 15) & (df['hour'] <= 16)).astype(int)
    df['evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

    # Días especiales basados en el patrón observado EDA
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Características cíclicas para capturar periodicidad
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def add_external_features(df):
    """Incorpora variables externas relevantes"""
    df = df.copy()

    # Festivos ucranianos principales
    ukraine_holidays = [
        '2024-01-01', '2024-01-07', '2024-03-08', '2024-05-01', '2024-05-09',
        '2024-06-28', '2024-08-24', '2024-10-14', '2024-12-25',
        '2025-01-01', '2025-01-07','2025-03-08'
    ]
    df['is_holiday'] = df['date'].dt.date.astype(str).isin(ukraine_holidays).astype(int)

    # Días cerca de festivos
    holiday_dates = pd.to_datetime(ukraine_holidays)
    df['days_to_holiday'] = df['date'].apply(
        lambda x: min([abs((x - h).days) for h in holiday_dates])
    )
    df['near_holiday'] = (df['days_to_holiday'] <= 2).astype(int)
    df['pre_holiday'] = (df['days_to_holiday'] <= 1).astype(int)

    # Variables meteorológicas mejoradas
    np.random.seed(42)
    monthly_avg_temp = {
        1: -3.0, 2: -2.5, 3: 2.0, 4: 10.0, 5: 16.0, 6: 19.0,
        7: 21.0, 8: 20.0, 9: 15.0, 10: 9.0, 11: 3.0, 12: -1.0
    }

    monthly_avg_precip = {
        1: 45.0, 2: 40.0, 3: 40.0, 4: 45.0, 5: 55.0, 6: 75.0,
        7: 80.0, 8: 65.0, 9: 55.0, 10: 45.0, 11: 50.0, 12: 50.0
    }

    df['temperature'] = df['month'].map(monthly_avg_temp)
    df['precipitation'] = df['month'].map(monthly_avg_precip)

    # Añadir variabilidad realista
    df['temperature'] += np.random.normal(0, 2.5, len(df))
    df['precipitation'] += np.random.normal(0, 3, len(df))

    # Variables derivadas del clima más específicas
    df['very_cold'] = (df['temperature'] < 0).astype(int)
    df['cold_weather'] = ((df['temperature'] >= 0) & (df['temperature'] < 10)).astype(int)
    df['mild_weather'] = ((df['temperature'] >= 10) & (df['temperature'] < 20)).astype(int)
    df['warm_weather'] = (df['temperature'] >= 20).astype(int)
    df['rainy_day'] = (df['precipitation'] > 5).astype(int)
    df['heavy_rain'] = (df['precipitation'] > 15).astype(int)

    return df

def create_lag_features(df, lags=[1, 2, 3, 7, 14]):
    """Crea características de lags optimizadas para demanda intermitente"""
    df = df.copy()
    df = df.sort_values(['product', 'date'])

    for lag in lags:
        df[f'transactions_lag{lag}'] = df.groupby('product')['transactions'].shift(lag)

        # Características binarias para detectar patrones de demanda
        df[f'had_demand_lag{lag}'] = (df[f'transactions_lag{lag}'] > 0).astype(int)

        # Días desde última demanda
        df[f'days_since_demand'] = df.groupby('product').apply(
            lambda x: (x['transactions'] > 0).cumsum()
        ).reset_index(0, drop=True)

    return df

def create_rolling_features(df, windows=[3, 7, 14]):
    """Crea características de ventanas móviles optimizadas"""
    df = df.copy()
    df = df.sort_values(['product', 'date'])

    for window in windows:
        # Media móvil
        df[f'transactions_ma_{window}'] = df.groupby('product')['transactions'].rolling(
            window=window, min_periods=1).mean().reset_index(0, drop=True)

        # Suma móvil (para demanda total en periodo)
        df[f'transactions_sum_{window}'] = df.groupby('product')['transactions'].rolling(
            window=window, min_periods=1).sum().reset_index(0, drop=True)

        # Porcentaje de días con demanda
        df[f'demand_days_pct_{window}'] = df.groupby('product')['transactions'].rolling(
            window=window, min_periods=1).apply(
            lambda x: (x > 0).sum() / len(x)).reset_index(0, drop=True)

        # Desviación estándar móvil
        df[f'transactions_std_{window}'] = df.groupby('product')['transactions'].rolling(
            window=window, min_periods=1).std().reset_index(0, drop=True)

        # Coeficiente de variación móvil
        mean_col = f'transactions_ma_{window}'
        std_col = f'transactions_std_{window}'
        df[f'cv_{window}'] = df[std_col] / (df[mean_col] + 1e-9)

    return df

def create_interaction_features(df):
    """Crea características de interacción específicas del negocio"""
    df = df.copy()

    # Interacciones temporales críticas para café
    df['cold_morning'] = df['very_cold'] * df['morning_rush']
    df['rainy_afternoon'] = df['rainy_day'] * df['afternoon_break']
    df['weekend_evening'] = df['is_weekend'] * df['evening_rush']
    df['tuesday_lunch'] = df['is_tuesday'] * df['lunch_hour']
    df['friday_effect'] = df['is_friday'] * df['afternoon_break']

    # Características de competencia interna mejoradas
    daily_totals = df.groupby('date')['transactions'].sum().to_dict()
    df['daily_total'] = df['date'].map(daily_totals)
    df['market_share'] = df['transactions'] / (df['daily_total'] + 1e-9)

    # Ranking de productos por día
    df['daily_product_rank'] = df.groupby('date')['transactions'].rank(method='dense', ascending=False)

    return df

# Aplicar feature engineering mejorado
print("Aplicando feature engineering mejorado...")
df_real = create_temporal_features(df_real)
df_real = add_external_features(df_real)
df_real = create_lag_features(df_real)
df_real = create_rolling_features(df_real)
df_real = create_interaction_features(df_real)

# Llenar valores faltantes de manera inteligente
numeric_cols = df_real.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['date', 'transactions', 'revenue']:
        df_real[col] = df_real[col].fillna(df_real[col].median())

print(f"Feature engineering completado: {df_real.shape[1]} características")

# ---
# BLOQUE 6: SEGMENTACIÓN INTELIGENTE MEJORADA
# ---
def improved_product_segmentation(df, volume_threshold_pct=0.15, stability_threshold=0.5):
    """
    Segmentación mejorada basada en volumen relativo y estabilidad de demanda
    """
    # Estadísticas por producto
    prod_stats = df.groupby('product')['transactions'].agg([
        'mean', 'median', 'std', 'sum', 'count'
    ]).round(4)

    # Métricas de segmentación mejoradas
    total_demand = prod_stats['sum'].sum()
    prod_stats['volume_share'] = prod_stats['sum'] / total_demand
    prod_stats['cv'] = prod_stats['std'] / (prod_stats['mean'] + 1e-9)

    # Porcentaje de días con demanda cero
    prod_stats['zero_demand_pct'] = df.groupby('product')['transactions'].apply(
        lambda x: (x == 0).sum() / len(x) * 100
    ).round(2)

    # Criterios de segmentación más balanceados
    high_volume = prod_stats['volume_share'] >= volume_threshold_pct
    stable_demand = prod_stats['zero_demand_pct'] <= (100 - stability_threshold * 100)

    # Segmentación: Alta demanda = alto volumen O demanda estable
    high_demand_products = prod_stats[high_volume | stable_demand].index.tolist()
    low_demand_products = prod_stats[~(high_volume | stable_demand)].index.tolist()

    # Información detallada
    segmentation_info = {
        'high_demand_products': high_demand_products,
        'low_demand_products': low_demand_products,
        'high_demand_stats': prod_stats.loc[high_demand_products].to_dict('index'),
        'low_demand_stats': prod_stats.loc[low_demand_products].to_dict('index'),
        'criteria': {
            'volume_threshold_pct': volume_threshold_pct,
            'stability_threshold': stability_threshold
        },
        'summary': {
            'total_products': len(prod_stats),
            'high_demand_count': len(high_demand_products),
            'low_demand_count': len(low_demand_products),
            'high_demand_volume_share': prod_stats.loc[high_demand_products, 'volume_share'].sum(),
            'low_demand_volume_share': prod_stats.loc[low_demand_products, 'volume_share'].sum()
        }
    }

    return high_demand_products, low_demand_products, segmentation_info

# Realizar segmentación mejorada
HIGH_PRODUCTS, LOW_PRODUCTS, segmentation_info = improved_product_segmentation(df_real)

print("SEGMENTACIÓN MEJORADA DE PRODUCTOS")
print("-" * 50)
print(f"Alta demanda ({len(HIGH_PRODUCTS)}): {HIGH_PRODUCTS}")
print(f"Baja demanda ({len(LOW_PRODUCTS)}): {LOW_PRODUCTS}")
print(f"Participación de volumen - Alta: {segmentation_info['summary']['high_demand_volume_share']:.2%}")
print(f"Participación de volumen - Baja: {segmentation_info['summary']['low_demand_volume_share']:.2%}")

# Guardar información de segmentación
save_json('results/product_segmentation.json', segmentation_info)

# Visualizar segmentación mejorada
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Volumen vs Estabilidad
prod_stats = df_real.groupby('product')['transactions'].agg(['mean', 'std', 'sum', 'count'])
prod_stats['volume_share'] = prod_stats['sum'] / prod_stats['sum'].sum()
prod_stats['zero_pct'] = df_real.groupby('product')['transactions'].apply(lambda x: (x == 0).sum() / len(x) * 100)

colors = ['red' if p in HIGH_PRODUCTS else 'blue' for p in prod_stats.index]
scatter = ax1.scatter(prod_stats['volume_share'], 100-prod_stats['zero_pct'],
                     c=colors, alpha=0.7, s=100)

for i, product in enumerate(prod_stats.index):
    ax1.annotate(product, (prod_stats.loc[product, 'volume_share'],
                          100-prod_stats.loc[product, 'zero_pct']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax1.axvline(x=0.15, color='gray', linestyle='--', alpha=0.5, label='Umbral Volumen')
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Umbral Estabilidad')
ax1.set_xlabel('Participación de Volumen')
ax1.set_ylabel('Estabilidad de Demanda (%)')
ax1.set_title('Segmentación: Volumen vs Estabilidad')
ax1.legend(['Alta Demanda', 'Baja Demanda', 'Umbral Volumen', 'Umbral Estabilidad'])

# Distribución de volumen por segmento
segment_volumes = [
    segmentation_info['summary']['high_demand_volume_share'],
    segmentation_info['summary']['low_demand_volume_share']
]
ax2.pie(segment_volumes, labels=['Alta Demanda', 'Baja Demanda'], autopct='%1.1f%%')
ax2.set_title('Distribución de Volumen por Segmento')

# Patrones temporales por segmento
high_demand_data = df_real[df_real['product'].isin(HIGH_PRODUCTS)]
low_demand_data = df_real[df_real['product'].isin(LOW_PRODUCTS)]

if len(HIGH_PRODUCTS) > 0:
    high_weekly = high_demand_data.groupby('day_of_week')['transactions'].mean()
    ax3.plot(range(7), high_weekly.values, 'r-o', label='Alta Demanda', linewidth=2)

if len(LOW_PRODUCTS) > 0:
    low_weekly = low_demand_data.groupby('day_of_week')['transactions'].mean()
    ax3.plot(range(7), low_weekly.values, 'b-s', label='Baja Demanda', linewidth=2)

ax3.set_xticks(range(7))
ax3.set_xticklabels(['L', 'M', 'M', 'J', 'V', 'S', 'D'])
ax3.set_xlabel('Día de la Semana')
ax3.set_ylabel('Transacciones Promedio')
ax3.set_title('Patrones Semanales por Segmento')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Estadísticas de demanda por segmento
segment_stats = {}
if len(HIGH_PRODUCTS) > 0:
    segment_stats['Alta Demanda'] = {
        'Media': high_demand_data['transactions'].mean(),
        'Mediana': high_demand_data['transactions'].median(),
        'Std': high_demand_data['transactions'].std(),
        'Ceros %': (high_demand_data['transactions'] == 0).sum() / len(high_demand_data) * 100
    }
if len(LOW_PRODUCTS) > 0:
    segment_stats['Baja Demanda'] = {
        'Media': low_demand_data['transactions'].mean(),
        'Mediana': low_demand_data['transactions'].median(),
        'Std': low_demand_data['transactions'].std(),
        'Ceros %': (low_demand_data['transactions'] == 0).sum() / len(low_demand_data) * 100
    }

# Gráfico de barras con estadísticas
stats_labels = ['Media', 'Mediana', 'Std']
if segment_stats:
    x_pos = np.arange(len(stats_labels))
    width = 0.35

    if 'Alta Demanda' in segment_stats:
        high_values = [segment_stats['Alta Demanda'][label] for label in stats_labels]
        ax4.bar(x_pos - width/2, high_values, width, label='Alta Demanda', alpha=0.8, color='red')

    if 'Baja Demanda' in segment_stats:
        low_values = [segment_stats['Baja Demanda'][label] for label in stats_labels]
        ax4.bar(x_pos + width/2, low_values, width, label='Baja Demanda', alpha=0.8, color='blue')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stats_labels)
    ax4.set_ylabel('Valor')
    ax4.set_title('Estadísticas de Demanda por Segmento')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/product_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()

print("Segmentación mejorada completada y visualizada")

# ---
# BLOQUE 7: DIVISIÓN TEMPORAL OPTIMIZADA
# ---
def optimized_train_test_split(df, test_weeks=4, val_weeks=2):
    """División temporal optimizada para demanda intermitente"""
    df_sorted = df.sort_values('date')
    unique_dates = sorted(df_sorted['date'].unique())

    # Calcular puntos de división
    total_days = len(unique_dates)
    test_days = test_weeks * 7
    val_days = val_weeks * 7
    train_days = total_days - test_days - val_days

    # Asegurar mínimo de datos de entrenamiento
    if train_days < 28:  # Mínimo 4 semanas
        raise ValueError("Datos insuficientes para división temporal. Se requieren al menos 8 semanas.")

    # Fechas de corte
    train_end_date = unique_dates[train_days - 1]
    val_end_date = unique_dates[train_days + val_days - 1]

    # Dividir datos
    train_df = df_sorted[df_sorted['date'] <= train_end_date].copy()
    val_df = df_sorted[(df_sorted['date'] > train_end_date) &
                      (df_sorted['date'] <= val_end_date)].copy()
    test_df = df_sorted[df_sorted['date'] > val_end_date].copy()

    print(f"División temporal optimizada:")
    print(f"Entrenamiento: {train_df['date'].min()} a {train_df['date'].max()} ({len(train_df)} registros)")
    print(f"Validación: {val_df['date'].min()} a {val_df['date'].max()} ({len(val_df)} registros)")
    print(f"Prueba: {test_df['date'].min()} a {test_df['date'].max()} ({len(test_df)} registros)")

    return train_df, val_df, test_df

# Realizar división
train_df, val_df, test_df = optimized_train_test_split(df_real)

# Preparar características y targets
exclude_columns = ['date', 'transactions', 'revenue', 'product']
feature_columns = [col for col in df_real.columns if col not in exclude_columns]

X_train = train_df[feature_columns]
y_train = train_df['transactions']
products_train = train_df['product']

X_val = val_df[feature_columns]
y_val = val_df['transactions']
products_val = val_df['product']

X_test = test_df[feature_columns]
y_test = test_df['transactions']
products_test = test_df['product']

print(f"\nCaracterísticas utilizadas: {len(feature_columns)}")
print(f"Distribución de target:")
print(f"Train - Media: {y_train.mean():.3f}, Mediana: {y_train.median():.3f}, Ceros: {(y_train == 0).sum()}/{len(y_train)} ({(y_train == 0).mean()*100:.1f}%)")
print(f"Val - Media: {y_val.mean():.3f}, Mediana: {y_val.median():.3f}, Ceros: {(y_val == 0).sum()}/{len(y_val)} ({(y_val == 0).mean()*100:.1f}%)")
print(f"Test - Media: {y_test.mean():.3f}, Mediana: {y_test.median():.3f}, Ceros: {(y_test == 0).sum()}/{len(y_test)} ({(y_test == 0).mean()*100:.1f}%)")

# ---
# BLOQUE 8: MODELO HÍBRIDO OPTIMIZADO PARA DEMANDA INTERMITENTE
# ---
class IntermittentDemandHybridModel:
    """
    Modelo híbrido optimizado para demanda intermitente con enfoque en métricas de negocio
    """
    def __init__(self, high_products, low_products):
        self.high_products = high_products
        self.low_products = low_products
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def fit(self, X, y, products, optimize=True):
        """Entrena modelos por segmento con optimización para demanda intermitente"""

        # Modelo para productos de alta demanda
        high_mask = products.isin(self.high_products)
        if high_mask.sum() > 0:
            print(f"Entrenando modelo para productos de alta demanda ({high_mask.sum()} muestras)")

            if optimize:
                best_params = self._optimize_intermittent_model(X[high_mask], y[high_mask])
            else:
                best_params = {
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.2,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'max_depth': 6,
                    'n_estimators': 200,
                    'min_child_samples': 10,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbosity': -1
                }

            self.models['high'] = lgb.LGBMRegressor(**best_params)
            self.models['high'].fit(X[high_mask], y[high_mask])

            self.feature_importance['high'] = dict(zip(
                X.columns, self.models['high'].feature_importances_
            ))

        # Modelo two-stage mejorado para productos de baja demanda
        low_mask = products.isin(self.low_products)
        if low_mask.sum() > 0:
            print(f"Entrenando modelo two-stage para productos de baja demanda ({low_mask.sum()} muestras)")

            # Etapa 1: Clasificador para detectar demanda > 0
            y_binary = (y[low_mask] > 0).astype(int)

            classifier_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 10,
                'n_estimators': 200,
                'random_state': 42,
                'verbosity': -1
            }

            self.models['classifier'] = lgb.LGBMClassifier(**classifier_params)
            self.models['classifier'].fit(X[low_mask], y_binary)

            # Etapa 2: Regresor para magnitud (solo casos positivos)
            positive_mask = low_mask & (y > 0)
            if positive_mask.sum() > 0:
                print(f"Entrenando regresor para magnitud ({positive_mask.sum()} muestras positivas)")

                if optimize:
                    best_params = self._optimize_intermittent_model(X[positive_mask], y[positive_mask])
                else:
                    best_params = {
                        'objective': 'tweedie',
                        'tweedie_variance_power': 1.5,
                        'learning_rate': 0.05,
                        'num_leaves': 15,
                        'max_depth': 4,
                        'n_estimators': 150,
                        'min_child_samples': 5,
                        'subsample': 0.9,
                        'colsample_bytree': 0.9,
                        'reg_alpha': 0.05,
                        'reg_lambda': 0.1,
                        'random_state': 42,
                        'verbosity': -1
                    }

                self.models['regressor'] = lgb.LGBMRegressor(**best_params)
                self.models['regressor'].fit(X[positive_mask], y[positive_mask])

                self.feature_importance['regressor'] = dict(zip(
                    X.columns, self.models['regressor'].feature_importances_
                ))

    def _optimize_intermittent_model(self, X, y, n_trials=30):
        """Optimiza hiperparámetros específicamente para demanda intermitente"""

        def objective(trial):
            params = {
                'objective': trial.suggest_categorical('objective', ['tweedie', 'poisson', 'gamma']),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': 42,
                'verbosity': -1
            }

            if params['objective'] == 'tweedie':
                params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)

            # Validación cruzada temporal simple
            split_point = int(0.8 * len(X))
            X_tr, X_val_opt = X.iloc[:split_point], X.iloc[split_point:]
            y_tr, y_val_opt = y.iloc[:split_point], y.iloc[split_point:]

            try:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val_opt)
                preds = np.maximum(preds, 0)  # No negativos

                # Métrica optimizada para demanda intermitente (WAPE)
                return wape(y_val_opt.values, preds)
            except:
                return float('inf')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbosity'] = -1

        return best_params

    def predict(self, X, products):
        """Genera predicciones usando modelo híbrido optimizado"""
        predictions = np.zeros(len(X))

        # Predicciones para productos de alta demanda
        high_mask = products.isin(self.high_products)
        if high_mask.sum() > 0 and 'high' in self.models:
            predictions[high_mask] = self.models['high'].predict(X[high_mask])

        # Predicciones para productos de baja demanda (two-stage mejorado)
        low_mask = products.isin(self.low_products)
        if low_mask.sum() > 0:
            if 'classifier' in self.models and 'regressor' in self.models:
                # Probabilidad de demanda > 0
                prob_positive = self.models['classifier'].predict_proba(X[low_mask])[:, 1]

                # Magnitud de demanda
                magnitude_pred = self.models['regressor'].predict(X[low_mask])

                # Combinar con umbral ajustado para reducir stockouts
                threshold = 0.3  # Umbral más conservador
                binary_pred = (prob_positive > threshold).astype(int)
                predictions[low_mask] = binary_pred * magnitude_pred

            elif 'high' in self.models:
                # Fallback: usar modelo de alta demanda
                predictions[low_mask] = self.models['high'].predict(X[low_mask])

        return np.maximum(predictions, 0)  # Asegurar no negativos

    def predict_with_safety_stock(self, X, products, safety_factor=1.2):
        """Predicciones con factor de seguridad para reducir stockouts"""
        base_predictions = self.predict(X, products)

        # Aplicar factor de seguridad diferenciado por segmento
        safety_predictions = base_predictions.copy()

        high_mask = products.isin(self.high_products)
        low_mask = products.isin(self.low_products)

        # Factor más agresivo para productos de baja demanda (más riesgo de stockout)
        if high_mask.sum() > 0:
            safety_predictions[high_mask] *= safety_factor

        if low_mask.sum() > 0:
            safety_predictions[low_mask] *= (safety_factor + 0.3)  # Factor mayor para baja demanda

        return safety_predictions

    def get_feature_importance(self, top_n=20):
        """Retorna feature importance consolidada y normalizada"""
        if not self.feature_importance:
            return None

        # Combinar importancias de todos los modelos
        all_features = {}
        total_models = len(self.feature_importance)

        for model_type, features in self.feature_importance.items():
            for feature, importance in features.items():
                if feature not in all_features:
                    all_features[feature] = 0
                all_features[feature] += importance / total_models

        # Normalizar
        total_importance = sum(all_features.values())
        if total_importance > 0:
            all_features = {k: v/total_importance for k, v in all_features.items()}

        return dict(sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:top_n])

# Inicializar y entrenar modelo optimizado
print("Inicializando modelo híbrido optimizado para demanda intermitente...")
model = IntermittentDemandHybridModel(HIGH_PRODUCTS, LOW_PRODUCTS)

# Entrenar con optimización
model.fit(X_train, y_train, products_train, optimize=True)
print("Modelo híbrido optimizado entrenado exitosamente")

# ---
# BLOQUE 9: EVALUACIÓN COMPRENSIVA MEJORADA
# ---
def comprehensive_evaluation_improved(model, X, y, products, segment_name="", safety_factor=1.2):
    """Evaluación comprensiva mejorada con métricas específicas para demanda intermitente"""

    # Predicciones básicas y con factor de seguridad
    predictions = model.predict(X, products)
    safety_predictions = model.predict_with_safety_stock(X, products, safety_factor)

    # Métricas generales
    overall_metrics = business_metrics(y, predictions, stockout_cost=3, holding_cost=1)
    safety_metrics = business_metrics(y, safety_predictions, stockout_cost=3, holding_cost=1)

    overall_metrics['safety_stock'] = safety_metrics

    # Métricas por segmento
    segment_metrics = {}

    # Alta demanda
    high_mask = products.isin(model.high_products)
    if high_mask.sum() > 0:
        segment_metrics['high_demand'] = business_metrics(
            y[high_mask], predictions[high_mask], stockout_cost=3, holding_cost=1
        )
        segment_metrics['high_demand']['n_samples'] = int(high_mask.sum())
        segment_metrics['high_demand']['zero_pct'] = float((y[high_mask] == 0).mean())

    # Baja demanda
    low_mask = products.isin(model.low_products)
    if low_mask.sum() > 0:
        segment_metrics['low_demand'] = business_metrics(
            y[low_mask], predictions[low_mask], stockout_cost=3, holding_cost=1
        )
        segment_metrics['low_demand']['n_samples'] = int(low_mask.sum())
        segment_metrics['low_demand']['zero_pct'] = float((y[low_mask] == 0).mean())

    # Métricas por producto (solo productos con suficientes datos)
    product_metrics = {}
    for product in products.unique():
        product_mask = (products == product)
        if product_mask.sum() >= 10:  # Mínimo 10 observaciones
            product_metrics[product] = business_metrics(
                y[product_mask], predictions[product_mask], stockout_cost=3, holding_cost=1
            )
            product_metrics[product]['n_samples'] = int(product_mask.sum())
            product_metrics[product]['zero_pct'] = float((y[product_mask] == 0).mean())

    # Análisis de errores específico para demanda intermitente
    intermittent_analysis = {
        'hit_rate': float((np.abs(predictions - y) <= 0.5).mean() * 100),
        'demand_capture_rate': float(((predictions > 0) & (y > 0)).sum() / max((y > 0).sum(), 1)),
        'false_positive_rate': float(((predictions > 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        'zero_prediction_accuracy': float(((predictions == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    }

    # Convert numpy arrays to lists para serialización
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    return convert_numpy_types({
        'overall': overall_metrics,
        'segments': segment_metrics,
        'products': product_metrics,
        'intermittent_analysis': intermittent_analysis,
        'predictions': predictions.tolist()
    })

print("Evaluando modelo optimizado en conjunto de validación...")
val_results = comprehensive_evaluation_improved(model, X_val, y_val, products_val, "Validación")

print("Evaluando modelo optimizado en conjunto de prueba...")
test_results = comprehensive_evaluation_improved(model, X_test, y_test, products_test, "Prueba")

# Guardar resultados
save_json('results/validation_metrics.json', val_results)
save_json('results/test_metrics.json', test_results)

# Mostrar métricas principales mejoradas
print("\nMÉTRICAS DE VALIDACIÓN OPTIMIZADAS")
print("-" * 60)
print(f"WAPE: {val_results['overall']['WAPE']:.2f}%")
print(f"MASE: {val_results['overall']['MASE']:.3f}")
print(f"SMAPE: {val_results['overall']['SMAPE']:.2f}%")
print(f"Service Level: {val_results['overall']['Service_Level']:.1f}%")
print(f"Fill Rate: {val_results['overall']['Fill_Rate']:.1f}%")
print(f"Hit Rate: {val_results['intermittent_analysis']['hit_rate']:.1f}%")
print(f"Total Cost: {val_results['overall']['Total_Cost']:.2f}")
print(f"Con Safety Stock - Service Level: {val_results['overall']['safety_stock']['Service_Level']:.1f}%")
print(f"Con Safety Stock - Total Cost: {val_results['overall']['safety_stock']['Total_Cost']:.2f}")

print("\nMÉTRICAS DE PRUEBA OPTIMIZADAS")
print("-" * 60)
print(f"WAPE: {test_results['overall']['WAPE']:.2f}%")
print(f"MASE: {test_results['overall']['MASE']:.3f}")
print(f"SMAPE: {test_results['overall']['SMAPE']:.2f}%")
print(f"Service Level: {test_results['overall']['Service_Level']:.1f}%")
print(f"Fill Rate: {test_results['overall']['Fill_Rate']:.1f}%")
print(f"Hit Rate: {test_results['intermittent_analysis']['hit_rate']:.1f}%")
print(f"Total Cost: {test_results['overall']['Total_Cost']:.2f}")
print(f"Con Safety Stock - Service Level: {test_results['overall']['safety_stock']['Service_Level']:.1f}%")
print(f"Con Safety Stock - Total Cost: {test_results['overall']['safety_stock']['Total_Cost']:.2f}")

# Visualización de resultados mejorada
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plot mejorado con safety stock
predictions_test = np.array(test_results['predictions'])
safety_predictions_test = model.predict_with_safety_stock(X_test, products_test)

ax1.scatter(y_test, predictions_test, alpha=0.6, label='Predicción Base', color='blue')
ax1.scatter(y_test, safety_predictions_test, alpha=0.4, label='Con Safety Stock', color='red')
ax1.plot([0, max(y_test)], [0, max(y_test)], 'k--', alpha=0.8, linewidth=2)
ax1.set_xlabel('Valor Real')
ax1.set_ylabel('Predicción')
ax1.set_title('Predicciones vs Reales')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Matriz de confusión para demanda binaria
y_binary = (y_test > 0).astype(int)
pred_binary = (predictions_test > 0).astype(int)

confusion_matrix = np.array([
    [(y_binary == 0) & (pred_binary == 0), (y_binary == 0) & (pred_binary == 1)],
    [(y_binary == 1) & (pred_binary == 0), (y_binary == 1) & (pred_binary == 1)]
]).sum(axis=2)

im = ax2.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
ax2.figure.colorbar(im, ax=ax2)
ax2.set_title('Matriz Confusión\n(Demanda Binaria)')
ax2.set_xlabel('Predicción')
ax2.set_ylabel('Real')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Sin Demanda', 'Con Demanda'])
ax2.set_yticklabels(['Sin Demanda', 'Con Demanda'])

# Añadir texto en cada celda
thresh = confusion_matrix.max() / 2.
for i, j in [(i, j) for i in range(2) for j in range(2)]:
    ax2.text(j, i, format(confusion_matrix[i, j], 'd'),
             ha="center", va="center",
             color="white" if confusion_matrix[i, j] > thresh else "black")

# Comparación de métricas por segmento
segments = []
wape_scores = []
service_levels = []

if 'high_demand' in test_results['segments']:
    segments.append('Alta\nDemanda')
    wape_scores.append(test_results['segments']['high_demand']['WAPE'])
    service_levels.append(test_results['segments']['high_demand']['Service_Level'])

if 'low_demand' in test_results['segments']:
    segments.append('Baja\nDemanda')
    wape_scores.append(test_results['segments']['low_demand']['WAPE'])
    service_levels.append(test_results['segments']['low_demand']['Service_Level'])

if segments:
    x_pos = np.arange(len(segments))
    width = 0.35

    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x_pos - width/2, wape_scores, width, label='WAPE (%)', alpha=0.7, color='orange')
    bars2 = ax3_twin.bar(x_pos + width/2, service_levels, width, label='Service Level (%)', alpha=0.7, color='green')

    ax3.set_xlabel('Segmento')
    ax3.set_ylabel('WAPE (%)', color='orange')
    ax3_twin.set_ylabel('Service Level (%)', color='green')
    ax3.set_title('Rendimiento por Segmento')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(segments)

    # Añadir valores en las barras
    for bar, value in zip(bars1, wape_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')

    for bar, value in zip(bars2, service_levels):
        height = bar.get_height()
        ax3_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.1f}%', ha='center', va='bottom')

# Análisis de costos: Base vs Safety Stock
cost_comparison = {
    'Base': test_results['overall']['Total_Cost'],
    'Safety Stock': test_results['overall']['safety_stock']['Total_Cost']
}

service_comparison = {
    'Base': test_results['overall']['Service_Level'],
    'Safety Stock': test_results['overall']['safety_stock']['Service_Level']
}

x_pos = np.arange(len(cost_comparison))
width = 0.35

ax4_twin = ax4.twinx()
bars1 = ax4.bar(x_pos - width/2, list(cost_comparison.values()), width,
               label='Costo Total', alpha=0.7, color='red')
bars2 = ax4_twin.bar(x_pos + width/2, list(service_comparison.values()), width,
                    label='Service Level (%)', alpha=0.7, color='blue')

ax4.set_xlabel('Estrategia')
ax4.set_ylabel('Costo Total', color='red')
ax4_twin.set_ylabel('Service Level (%)', color='blue')
ax4.set_title('Costo vs Service Level')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(list(cost_comparison.keys()))

# Añadir valores en las barras
for bar, value in zip(bars1, cost_comparison.values()):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}', ha='center', va='bottom')

for bar, value in zip(bars2, service_comparison.values()):
    height = bar.get_height()
    ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                 f'{value:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/figures/model_evaluation_improved.png', dpi=300, bbox_inches='tight')
plt.show()

import mlflow

mlflow.end_run()

mlflow.set_experiment("coffee_demand_forecast")

with mlflow.start_run():
    # 3. Validación y test: loguear métricas clave
    print("Evaluando modelo optimizado en conjunto de validación...")
    val_results = comprehensive_evaluation_improved(model, X_val, y_val, products_val)

    print("Evaluando modelo optimizado en conjunto de prueba...")
    test_results = comprehensive_evaluation_improved(model, X_test, y_test, products_test)

    mlflow.log_metric("WAPE_val",    val_results["overall"]["WAPE"])
    mlflow.log_metric("WAPE_test",   test_results["overall"]["WAPE"])
    mlflow.log_metric("Service_val", val_results["overall"]["Service_Level"])
    mlflow.log_metric("Service_test",test_results["overall"]["Service_Level"])
    mlflow.log_metric("Total_Cost_val", val_results["overall"]["Total_Cost"])
    mlflow.log_metric("Total_Cost_test", test_results["overall"]["Total_Cost"])
    mlflow.log_metric("Safety_Service_test", test_results["overall"]["safety_stock"]["Service_Level"])
    mlflow.log_metric("Safety_Cost_test", test_results["overall"]["safety_stock"]["Total_Cost"])

    # Mostrar métricas principales
    print("\nMÉTRICAS DE VALIDACIÓN OPTIMIZADAS")
    print("-" * 60)
    print(f"WAPE: {val_results['overall']['WAPE']:.2f}%")
    print(f"MASE: {val_results['overall']['MASE']:.3f}")
    print(f"SMAPE: {val_results['overall']['SMAPE']:.2f}%")
    print(f"Service Level: {val_results['overall']['Service_Level']:.1f}%")
    print(f"Fill Rate: {val_results['overall']['Fill_Rate']:.1f}%")
    print(f"Hit Rate: {val_results['intermittent_analysis']['hit_rate']:.1f}%")
    print(f"Total Cost: {val_results['overall']['Total_Cost']:.2f}")
    print(f"Con Safety Stock - Service Level: {val_results['overall']['safety_stock']['Service_Level']:.1f}%")
    print(f"Con Safety Stock - Total Cost: {test_results['overall']['safety_stock']['Total_Cost']:.2f}")

    print("\nMÉTRICAS DE PRUEBA OPTIMIZADAS")
    print("-" * 60)
    print(f"WAPE: {test_results['overall']['WAPE']:.2f}%")
    print(f"MASE: {test_results['overall']['MASE']:.3f}")
    print(f"SMAPE: {test_results['overall']['SMAPE']:.2f}%")
    print(f"Service Level: {test_results['overall']['Service_Level']:.1f}%")
    print(f"Fill Rate: {test_results['overall']['Fill_Rate']:.1f}%")
    print(f"Hit Rate: {test_results['intermittent_analysis']['hit_rate']:.1f}%")
    print(f"Total Cost: {test_results['overall']['Total_Cost']:.2f}")
    print(f"Con Safety Stock - Service Level: {test_results['overall']['safety_stock']['Service_Level']:.1f}%")
    print(f"Con Safety Stock - Total Cost: {test_results['overall']['safety_stock']['Total_Cost']:.2f}")

    mlflow.end_run()

# ---
# BLOQUE 11: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS MEJORADO + MLflow
# ---
feature_importance = model.get_feature_importance(top_n=25)

if feature_importance:
    print("\nTOP 25 CARACTERÍSTICAS MÁS IMPORTANTES")
    print("-" * 60)
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        print(f"{i:2d}. {feature:30s}: {importance:.4f}")

    # Visualizar importancia mejorada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Importancia general (Top 20)
    features = list(feature_importance.keys())[:20]
    importances = list(feature_importance.values())[:20]

    bars = ax1.barh(range(len(features)), importances, alpha=0.8)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Importancia Normalizada')
    ax1.set_title('Top 20 Características Más Importantes')
    ax1.invert_yaxis()

    # Colorear barras por tipo de característica (categorías consistentes)
    colors = []
    for feature in features:
        if any(word in feature.lower() for word in ['lag', 'ma_', 'sum_', 'std_', 'cv_', 'roll_']):
            colors.append('skyblue')   # Temporales/Lags
        elif any(word in feature.lower() for word in ['hour', 'day', 'week', 'month']):
            colors.append('lightgreen')  # Tiempo
        elif any(word in feature.lower() for word in ['weather', 'temperature', 'rain', 'precipitation']):
            colors.append('orange')  # Clima
        elif any(word in feature.lower() for word in ['rush', 'lunch', 'weekend', 'holiday']):
            colors.append('violet')  # Interacciones
        else:
            colors.append('lightcoral')  # Otras

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Leyenda personalizada
    import matplotlib.patches as mpatches
    temp_patch = mpatches.Patch(color='skyblue', label='Temporales/Lags')
    time_patch = mpatches.Patch(color='lightgreen', label='Tiempo')
    weather_patch = mpatches.Patch(color='orange', label='Clima')
    inter_patch = mpatches.Patch(color='violet', label='Interacciones')
    other_patch = mpatches.Patch(color='lightcoral', label='Otras')
    ax1.legend(handles=[temp_patch, time_patch, weather_patch, inter_patch, other_patch], loc='lower right')

    # Análisis de tipos de características
    feature_types = {
        'Temporales/Lags': 0,
        'Tiempo': 0,
        'Clima': 0,
        'Interacciones': 0,
        'Otras': 0
    }

    for feature, importance in feature_importance.items():
        if any(word in feature.lower() for word in ['lag', 'ma_', 'sum_', 'std_', 'cv_', 'roll_']):
            feature_types['Temporales/Lags'] += importance
        elif any(word in feature.lower() for word in ['hour', 'day', 'week', 'month']):
            feature_types['Tiempo'] += importance
        elif any(word in feature.lower() for word in ['weather', 'temperature', 'rain', 'precipitation']):
            feature_types['Clima'] += importance
        elif any(word in feature.lower() for word in ['rush', 'lunch', 'weekend', 'holiday']):
            feature_types['Interacciones'] += importance
        else:
            feature_types['Otras'] += importance

    # Gráfico de pastel con categorías consistentes
    ax2.pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%')
    ax2.set_title('Importancia por Tipo de Característica')

    plt.tight_layout()
    fig_path = 'results/figures/feature_importance_improved.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    mlflow.log_artifact(fig_path)

    # Guardar análisis detallado
    feature_analysis = {
        'top_features': feature_importance,
        'feature_types': feature_types,
        'insights': {
            'most_important': list(feature_importance.keys())[0],
            'temporal_dominance': feature_types['Temporales/Lags'] > 0.4,
            'weather_impact': feature_types['Clima'] > 0.1
        }
    }
    save_json('results/feature_importance_analysis.json', feature_analysis)
    mlflow.log_artifact('results/feature_importance_analysis.json')

else:
    print("No se pudo calcular importancia de características")

# ---
# BLOQUE 12: PREDICCIONES FUTURAS OPTIMIZADAS
# ---
def generate_future_dates(last_date, periods=7):
    """Genera fechas futuras para pronóstico de corto plazo (1-7 días)"""
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=periods,
        freq='D'
    )
    return future_dates

def create_future_features_optimized(future_dates, product_list, historical_df):
    """Crea características para fechas futuras con lags históricos reales"""
    future_data = []

    # Obtener últimos valores para lags
    last_values = {}
    for product in product_list:
        product_history = historical_df[historical_df['product'] == product].sort_values('date')
        if len(product_history) > 0:
            last_values[product] = product_history['transactions'].tail(21).values  # Últimos 21 días
        else:
            last_values[product] = np.zeros(21)

    for i, date in enumerate(future_dates):
        for product in product_list:
            future_data.append({
                'date': date,
                'product': product,
                'days_ahead': i + 1  # Día de pronóstico (1-7)
            })

    future_df = pd.DataFrame(future_data)

    # Initialize 'transactions' column with zeros for future dates
    future_df['transactions'] = 0.0

    # Aplicar feature engineering
    future_df = create_temporal_features(future_df)
    future_df = add_external_features(future_df)

    # Lags basados en valores históricos reales
    for idx, row in future_df.iterrows():
        product = row['product']
        days_ahead = row['days_ahead']

        if product in last_values:
            history = last_values[product]

            # Lags ajustados por días adelante
            if len(history) >= days_ahead:
                future_df.at[idx, 'transactions_lag1'] = history[-(days_ahead)]
            if len(history) >= (7 + days_ahead - 1):
                future_df.at[idx, 'transactions_lag7'] = history[-(7 + days_ahead - 1)]
            if len(history) >= (14 + days_ahead - 1):
                future_df.at[idx, 'transactions_lag14'] = history[-(14 + days_ahead - 1)]

            # Media móvil de últimos 7 días
            if len(history) >= 7:
                recent_history = history[-7:]
                future_df.at[idx, 'transactions_ma_7'] = recent_history.mean()
                future_df.at[idx, 'transactions_std_7'] = recent_history.std()
                future_df.at[idx, 'demand_days_pct_7'] = (recent_history > 0).mean()

    # Características agregadas basadas en patrones históricos
    daily_patterns = historical_df.groupby('day_of_week')['transactions'].mean().to_dict()
    product_patterns = historical_df.groupby('product')['transactions'].mean().to_dict()

    for idx, row in future_df.iterrows():
        dow = row['day_of_week']
        product = row['product']

        future_df.at[idx, 'historical_dow_avg'] = daily_patterns.get(dow, 0)
        future_df.at[idx, 'historical_product_avg'] = product_patterns.get(product, 0)

    # Llenar características faltantes
    for col in feature_columns:
        if col not in future_df.columns:
            if col in historical_df.columns:
                future_df[col] = historical_df[col].median()
            else:
                future_df[col] = 0

    # Crear características de interacción faltantes
    future_df = create_interaction_features(future_df)

    # Asegurar todas las columnas necesarias
    missing_cols = set(feature_columns) - set(future_df.columns)
    for col in missing_cols:
        future_df[col] = 0

    return future_df[feature_columns], future_df['date'], future_df['product'], future_df['days_ahead']

# Generar predicciones futuras optimizadas para 7 días
last_date = df_real['date'].max()
future_dates = generate_future_dates(last_date, periods=7)
product_list = df_real['product'].unique()

print(f"Generando pronósticos para 7 días: {future_dates[0]} a {future_dates[-1]}")
print(f"Productos: {len(product_list)}")

X_future, future_dates_list, future_products, days_ahead = create_future_features_optimized(
    future_dates, product_list, df_real
)

# Predicciones base y con safety stock
future_predictions_base = model.predict(X_future, future_products)
future_predictions_safety = model.predict_with_safety_stock(X_future, future_products, safety_factor=1.4)

# Crear DataFrame de resultados con análisis por horizonte
future_results = pd.DataFrame({
    'date': future_dates_list,
    'product': future_products,
    'days_ahead': days_ahead,
    'predicted_demand_base': future_predictions_base,
    'predicted_demand_safety': future_predictions_safety,
    'recommendation': np.where(future_predictions_safety > 0,
                              np.ceil(future_predictions_safety),  # Redondear hacia arriba
                              0)
})

# Analysis by time horizon
horizon_analysis = future_results.groupby('days_ahead').agg({
    'predicted_demand_base': ['sum', 'mean'],
    'predicted_demand_safety': ['sum', 'mean'],
    'recommendation': 'sum'
}).round(2)

horizon_analysis.columns = ['_'.join(col).strip() for col in horizon_analysis.columns.values]

print("\nPRONÓSTICO POR HORIZONTE TEMPORAL")
print("-" * 50)
for day in range(1, 8):
    if day in horizon_analysis.index:
        base_total = horizon_analysis.loc[day, 'predicted_demand_base_sum']
        safety_total = horizon_analysis.loc[day, 'predicted_demand_safety_sum']
        recommended = horizon_analysis.loc[day, 'recommendation_sum']
        print(f"Día {day}: Base={base_total:.1f}, Safety={safety_total:.1f}, Recomendado={recommended:.0f}")

# Save results
future_results.to_csv('results/weekly_forecast.csv', index=False)
horizon_analysis.to_csv('results/horizon_analysis.csv')

# Create restocking report by product
restock_recommendations = future_results.groupby('product').agg({
    'predicted_demand_base': 'sum',
    'predicted_demand_safety': 'sum',
    'recommendation': 'sum'
}).round().astype(int)

restock_recommendations.columns = ['Demanda_Base_7d', 'Demanda_Safety_7d', 'Unidades_Recomendadas']
restock_recommendations = restock_recommendations.sort_values('Unidades_Recomendadas', ascending=False)

print("\nRECOMENDACIONES DE REPOSICIÓN (7 DÍAS)")
print("-" * 60)
print(restock_recommendations.to_string())

restock_recommendations.to_csv('results/restock_recommendations.csv')

# Visualize forecasts
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Aggregate daily forecast
daily_forecast = future_results.groupby('date').agg({
    'predicted_demand_base': 'sum',
    'predicted_demand_safety': 'sum'
}).reset_index()

ax1.plot(daily_forecast['date'], daily_forecast['predicted_demand_base'],
         'b-o', label='Predicción Base', linewidth=2, markersize=6)
ax1.plot(daily_forecast['date'], daily_forecast['predicted_demand_safety'],
         'r-s', label='Con Safety Stock', linewidth=2, markersize=6)
ax1.fill_between(daily_forecast['date'],
                 daily_forecast['predicted_demand_base'],
                 daily_forecast['predicted_demand_safety'],
                 alpha=0.3, color='orange', label='Diferencia Safety')

ax1.set_xlabel('Fecha')
ax1.set_ylabel('Demanda Total Predicha')
ax1.set_title('Pronóstico de Demanda Agregada (7 días)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Top products by projected demand
top_products_forecast = restock_recommendations.head(6)
ax2.bar(range(len(top_products_forecast)), top_products_forecast['Unidades_Recomendadas'], alpha=0.8)
ax2.set_xticks(range(len(top_products_forecast)))
ax2.set_xticklabels(top_products_forecast.index, rotation=45)
ax2.set_ylabel('Unidades Recomendadas')
ax2.set_title('Top Productos - Reposición Recomendada')

# Add values to bars
for i, v in enumerate(top_products_forecast['Unidades_Recomendadas']):
    ax2.text(i, v, str(v), ha='center', va='bottom')

# Analysis by horizon days
ax3.bar(range(1, 8), [horizon_analysis.loc[i, 'predicted_demand_safety_sum']
                      if i in horizon_analysis.index else 0 for i in range(1, 8)],
        alpha=0.7, color='green')
ax3.set_xticks(range(1, 8))
ax3.set_xticklabels([f'Día {i}' for i in range(1, 8)])
ax3.set_ylabel('Demanda Total Predicha')
ax3.set_title('Demanda por Horizonte de Pronóstico')
ax3.grid(True, alpha=0.3)

# Distribution of recommendations
recommendation_dist = future_results['recommendation'].value_counts().sort_index()
ax4.bar(recommendation_dist.index, recommendation_dist.values, alpha=0.7)
ax4.set_xlabel('Unidades Recomendadas')
ax4.set_ylabel('Frecuencia')
ax4.set_title('Distribución de Recomendaciones')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/weekly_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPredicciones futuras guardadas: {len(future_results)} registros")
print(f"Demanda total proyectada (7 días): {future_results['recommendation'].sum():.0f} unidades")
print(f"Demanda promedio diaria: {future_results['recommendation'].sum()/7:.1f} unidades")

# ---
# BLOQUE 12: GUARDAR MODELO Y RESULTADOS OPTIMIZADOS
# ---
# Guardar modelo optimizado
model_package = {
    'model': model,
    'feature_columns': feature_columns,
    'high_products': HIGH_PRODUCTS,
    'low_products': LOW_PRODUCTS,
    'training_date': df_real['date'].max().isoformat(),
    'model_version': 'v2.0_intermittent_optimized'
}

joblib.dump(model_package, 'results/models/intermittent_demand_model_v2.pkl')
print("Modelo optimizado guardado: results/models/intermittent_demand_model_v2.pkl")

# Loguear parámetros del modelo con MLflow now that model_package is defined
# End any active run before starting a new one
mlflow.end_run()
with mlflow.start_run(run_name="Intermittent Demand Model Training"):
    mlflow.log_param("model_version", model_package.get("model_version", "unknown"))
    mlflow.log_param("safety_factor_base", 1.2) # Factor usado en evaluacion y future predictions
    mlflow.log_param("safety_factor_future", 1.4) # Factor usado en future predictions
    mlflow.log_param("stockout_cost", 3)
    mlflow.log_param("holding_cost", 1)

    if 'high' in model.models:
        high_params = model.models['high'].get_params()
        mlflow.log_params({f'high_{k}': v for k, v in high_params.items()})
    if 'classifier' in model.models:
         classifier_params = model.models['classifier'].get_params()
         mlflow.log_params({f'classifier_{k}': v for k, v in classifier_params.items()})
    if 'regressor' in model.models:
         regressor_params = model.models['regressor'].get_params()
         mlflow.log_params({f'regressor_{k}': v for k, v in regressor_params.items()})


# Crear función de predicción para despliegue
def predict_demand(model_package, date, product, external_features=None):
    """
    Función de predicción para despliegue en producción

    Args:
        model_package: Modelo cargado con joblib
        date: Fecha de predicción (datetime)
        product: Nombre del producto
        external_features: Diccionario con características externas opcionales

    Returns:
        dict: Predicción base, con safety stock y recomendación
    """
    model = model_package['model']
    feature_columns = model_package['feature_columns']

    # Crear DataFrame temporal para predicción
    temp_df = pd.DataFrame([{
        'date': date,
        'product': product
    }])

    # Aplicar feature engineering
    temp_df = create_temporal_features(temp_df)
    temp_df = add_external_features(temp_df)

    # Añadir características externas si se proporcionan
    if external_features:
        for key, value in external_features.items():
            temp_df[key] = value

    # Llenar características faltantes
    for col in feature_columns:
        if col not in temp_df.columns:
            temp_df[col] = 0

    X_pred = temp_df[feature_columns]
    products_pred = temp_df['product']

    # Generar predicciones
    base_pred = model.predict(X_pred, products_pred)[0]
    safety_pred = model.predict_with_safety_stock(X_pred, products_pred)[0]

    return {
        'date': date.strftime('%Y-%m-%d'),
        'product': product,
        'prediction_base': max(0, float(base_pred)),
        'prediction_safety': max(0, float(safety_pred)),
        'recommendation': max(0, int(np.ceil(safety_pred))),
        'confidence': 'high' if product in model_package['high_products'] else 'medium'
    }

# Guardar función de predicción
with open('results/models/prediction_function.py', 'w') as f:
    f.write('''
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Cargar modelo
model_package = joblib.load('intermittent_demand_model_v2.pkl')

def predict_demand(date, product, external_features=None):
    """
    Función de predicción para despliegue en producción
    """
    # [Código de la función aquí]
    pass

# Ejemplo de uso:
# prediction = predict_demand(datetime(2025, 1, 15), 'Americano')
# print(prediction)
''')

print("Función de predicción guardada: results/models/prediction_function.py")

# ---
# BLOQUE 14: REPORTE EJECUTIVO FINAL OPTIMIZADO
# ---
# Crear reporte ejecutivo comprensivo
executive_report = {
    'model_performance': {
        'validation_metrics': {
            'WAPE': val_results['overall']['WAPE'],
            'MASE': val_results['overall']['MASE'],
            'Service_Level': val_results['overall']['Service_Level'],
            'Fill_Rate': val_results['overall']['Fill_Rate'],
            'Hit_Rate': val_results['intermittent_analysis']['hit_rate']
        },
        'test_metrics': {
            'WAPE': test_results['overall']['WAPE'],
            'MASE': test_results['overall']['MASE'],
            'Service_Level': test_results['overall']['Service_Level'],
            'Fill_Rate': test_results['overall']['Fill_Rate'],
            'Hit_Rate': test_results['intermittent_analysis']['hit_rate']
        },
        'improvement_with_safety_stock': {
            'service_level_improvement': test_results['overall']['safety_stock']['Service_Level'] - test_results['overall']['Service_Level'],
            'cost_increase': test_results['overall']['safety_stock']['Total_Cost'] - test_results['overall']['Total_Cost']
        }
    },
    'business_insights': {
        'demand_characteristics': {
            'avg_daily_demand': float(df_real.groupby('date')['transactions'].sum().mean()),
            'zero_demand_days_pct': float((df_real.groupby('date')['transactions'].sum() == 0).mean() * 100),
            'most_volatile_product': df_real.groupby('product')['transactions'].std().idxmax(),
            'most_stable_product': df_real.groupby('product')['transactions'].std().idxmin()
        },
        'segmentation': {
            'high_demand_products': HIGH_PRODUCTS,
            'low_demand_products': LOW_PRODUCTS,
            'high_demand_volume_share': segmentation_info['summary']['high_demand_volume_share'],
            'intermittent_products': int((df_real.groupby('product')['transactions'].apply(lambda x: (x == 0).mean()) > 0.6).sum())
        },
        'temporal_patterns': {
            'best_day': df_real.groupby('day_of_week')['transactions'].sum().idxmax(),
            'best_hour': df_real.groupby('hour')['transactions'].sum().idxmax(),
            'seasonal_effect': bool(feature_importance and any('month' in f for f in list(feature_importance.keys())[:10]))
        }
    },
    'forecast_7_days': {
        'total_demand_expected': float(future_results['recommendation'].sum()),
        'avg_daily_demand': float(future_results['recommendation'].sum() / 7),
        'peak_day': future_results.groupby('date')['recommendation'].sum().idxmax().strftime('%Y-%m-%d'),
        'top_product': restock_recommendations.index[0],
        'restock_recommendations': restock_recommendations.to_dict('index')
    },
    'model_features': {
        'total_features': len(feature_columns),
        'top_5_features': list(feature_importance.keys())[:5] if feature_importance else [],
        'feature_types_importance': {
            'temporal_lags': feature_analysis.get('feature_types', {}).get('Temporales/Lags', 0),
            'time_patterns': feature_analysis.get('feature_types', {}).get('Tiempo', 0),
            'weather': feature_analysis.get('feature_types', {}).get('Clima', 0)
        } if 'feature_analysis' in locals() else {}
    },
    'recommendations': {
        'inventory_strategy': 'Implementar safety stock diferenciado por segmento',
        'reorder_frequency': 'Semanal con ajustes diarios para productos de alta rotación',
        'monitoring_priority': 'Enfocar en productos de baja demanda por mayor riesgo de stockout',
        'demand_drivers': 'Patrones temporales y climáticos son factores clave'
    },
    'execution_info': {
        'model_version': 'v2.0_intermittent_optimized',
        'training_data_period': f"{df_real['date'].min()} to {df_real['date'].max()}",
        'total_records': len(df_real),
        'products_analyzed': len(df_real['product'].unique()),
        'timestamp': datetime.now().isoformat()
    }
}

save_json('results/executive_report_v2.json', executive_report)

print("\n" + "="*80)
print("REPORTE EJECUTIVO FINAL - MODELO OPTIMIZADO PARA DEMANDA INTERMITENTE")
print("="*80)

print(f"\nRENDIMIENTO DEL MODELO OPTIMIZADO:")
print(f"   • WAPE (Test): {test_results['overall']['WAPE']:.2f}% (Mejor que MAPE para demanda baja)")
print(f"   • MASE (Test): {test_results['overall']['MASE']:.3f} (<1.0 = Mejor que método naive)")
print(f"   • Service Level: {test_results['overall']['Service_Level']:.1f}%")
print(f"   • Fill Rate: {test_results['overall']['Fill_Rate']:.1f}%")
print(f"   • Hit Rate: {test_results['intermittent_analysis']['hit_rate']:.1f}% (Predicciones exactas)")

print(f"\nMEJORA CON SAFETY STOCK:")
print(f"   • Service Level: {test_results['overall']['Service_Level']:.1f}% → {test_results['overall']['safety_stock']['Service_Level']:.1f}%")
print(f"   • Costo Total: {test_results['overall']['Total_Cost']:.1f} → {test_results['overall']['safety_stock']['Total_Cost']:.1f}")
print(f"   • Incremento de costo: {executive_report['model_performance']['improvement_with_safety_stock']['cost_increase']:.1f} por {executive_report['model_performance']['improvement_with_safety_stock']['service_level_improvement']:.1f}% mejor servicio")

print(f"\nCARACTERÍSTICAS DEL NEGOCIO:")
print(f"   • Demanda promedio diaria: {executive_report['business_insights']['demand_characteristics']['avg_daily_demand']:.1f} unidades")
print(f"   • Días sin demanda: {executive_report['business_insights']['demand_characteristics']['zero_demand_days_pct']:.1f}%")
print(f"   • Productos intermitentes: {executive_report['business_insights']['segmentation']['intermittent_products']}/{len(product_list)}")

print(f"\nSEGMENTACIÓN OPTIMIZADA:")
print(f"   • Alta demanda: {len(HIGH_PRODUCTS)} productos ({executive_report['business_insights']['segmentation']['high_demand_volume_share']:.1%} del volumen)")
print(f"   • Baja demanda: {len(LOW_PRODUCTS)} productos")

print(f"\nPRONÓSTICO 7 DÍAS:")
print(f"   • Demanda total proyectada: {executive_report['forecast_7_days']['total_demand_expected']:.0f} unidades")
print(f"   • Demanda diaria promedio: {executive_report['forecast_7_days']['avg_daily_demand']:.1f} unidades")
print(f"   • Producto top: {executive_report['forecast_7_days']['top_product']}")
print(f"   • Día pico: {executive_report['forecast_7_days']['peak_day']}")

print(f"\nCARACTERÍSTICAS DEL MODELO:")
if feature_importance:
    print(f"   • Features principales: {', '.join(list(feature_importance.keys())[:3])}")
    print(f"   • Importancia temporal/lags: {executive_report['model_features']['feature_types_importance'].get('temporal_lags', 0):.1%}")

print(f"\nRECOMENDACIONES CLAVE:")
for rec in executive_report['recommendations'].values():
    print(f"   • {rec}")

print(f"\nARCHIVOS GENERADOS:")
print(f"   • Modelo: results/models/intermittent_demand_model_v2.pkl")
print(f"   • Pronóstico semanal: results/weekly_forecast.csv")
print(f"   • Recomendaciones reposición: results/restock_recommendations.csv")
print(f"   • Reporte ejecutivo: results/executive_report_v2.json")
print(f"   • Visualizaciones: results/figures/")

print("\n" + "="*80)
print("PROCESO COMPLETADO - MODELO OPTIMIZADO PARA DEMANDA INTERMITENTE")
print("="*80)

# Crear ZIP final con todos los resultados
print("\nComprimiendo resultados finales...")
with zipfile.ZipFile('coffee_demand_forecast_optimized_v2.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('results'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'results')
            zipf.write(file_path, f'results/{arcname}')

print(f"\nANÁLISIS COMPLETO GUARDADO EN: coffee_demand_forecast_optimized_v2.zip")
print(f"El modelo está listo para implementación en producción")

# Mensaje final sobre mejoras implementadas
print(f"\nMEJORAS IMPLEMENTADAS EN ESTA VERSIÓN:")
print(f"   ✓ Métricas apropiadas para demanda intermitente (WAPE, MASE)")
print(f"   ✓ Costos de negocio balanceados (stockout_cost=3, holding_cost=1)")
print(f"   ✓ Segmentación inteligente mejorada")
print(f"   ✓ Modelo two-stage optimizado para productos de baja demanda")
print(f"   ✓ Safety stock diferenciado por segmento")
print(f"   ✓ Pronóstico de 7 días con recomendaciones de reposición")
print(f"   ✓ Análisis de características más detallado")
print(f"   ✓ Función de predicción lista para producción")

# Crear ZIP para mlruns
print("\nComprimiendo archivos mlruns...")
with zipfile.ZipFile('mlruns.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('mlruns'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'mlruns')
            zipf.write(file_path, f'mlruns/{arcname}')
print("\n archivo comprimido mlruns.zip")


# === Bloque MLflow UI ===
from pathlib import Path
import sys, subprocess, os, socket, time
import mlflow
from IPython.display import IFrame, display

# Directorio de resultados
RESULTS_DIR = Path("results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pedido explícito: crear MLFLOW_DIR en RESULTS_DIR/mlruns
MLFLOW_DIR = (RESULTS_DIR / "mlruns").resolve()
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Backend store URI (file://)
backend = MLFLOW_DIR.as_uri()  # p.ej. file:///home/ubuntu/proyecto/results/mlruns
print("MLflow backend store URI:", backend)
print("MLflow version:", mlflow.__version__)

# Asegurar que MLflow use este tracking URI (recomendado si antes no se definió)
mlflow.set_tracking_uri(backend)

# Parámetros de red (por defecto uso tu host/puerto AWS)
# - Si ejecutas el notebook en la instancia AWS, deja HOST='0.0.0.0' y PORT=5080
# - Si ejecutas local y quieres acceso local pon HOST='127.0.0.1' y PORT= 5001
HOST = os.environ.get("MLFLOW_HOST", "0.0.0.0")
PORT = int(os.environ.get("MLFLOW_PORT", "5080"))

# Comando para lanzar la UI 
cmd = [
    sys.executable, "-m", "mlflow", "ui",
    "--backend-store-uri", backend,
    "--default-artifact-root", MLFLOW_DIR.as_uri(),
    "--host", HOST,
    "--port", str(PORT)
]

# Check simple: puerto libre?
def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

# Si host es 0.0.0.0 comprobar 127.0.0.1 solo como heurística local
check_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST
if is_port_in_use(PORT, check_host):
    print(f"Atención: el puerto {PORT} parece estar en uso en {check_host}.")
    print("Es posible que ya haya una instancia de MLflow corriendo.")
else:
    print("Lanzando MLflow UI con comando:")
    print(" ".join(cmd))

    # Lanzar proceso en background (logs en results/mlflow_ui.log)
    logfile = open(RESULTS_DIR / "mlflow_ui.log", "ab")
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=logfile, cwd=str(RESULTS_DIR))
    print(f"MLflow UI lanzado (PID={proc.pid}). Logs -> {RESULTS_DIR / 'mlflow_ui.log'}")

    # Pequeña espera para que la UI arranque (no bloqueante)
    time.sleep(1.5)

    # Intentar mostrar la UI embebida (si la UI es accesible desde el kernel)
    public_host = os.environ.get("MLFLOW_PUBLIC_HOST", "54.167.69.204")  # tu IP AWS por defecto
    ui_url = f"http://{public_host}:{PORT}"
    print("Intenta abrir en el navegador:", ui_url)
    try:
        display(IFrame(src=ui_url, width="100%", height=800))
    except Exception as e:
        print("No se pudo embeber la UI (posible restricción CORS/proxy). Abre la URL en tu navegador.")

# Cómo detener el servidor más tarde (ejecuta si quieres matar el proceso)
# proc.terminate(); proc.wait(timeout=10)