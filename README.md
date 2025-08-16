# Predicción de Demanda de Café - Sistema de Pronóstico para Máquina Expendedora

## Índice
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Objetivos](#objetivos)
   - [Objetivo General](#objetivo-general)
   - [Objetivos Específicos](#objetivos-específicos)
3. [Dataset](#dataset)
   - [Características de los Datos](#características-de-los-datos)
   - [Fuentes de Datos Adicionales](#fuentes-de-datos-adicionales)
4. [Arquitectura del Sistema](#arquitectura-del-sistema)
   - [Estructura del Proyecto](#estructura-del-proyecto)
5. [Tecnologías y Herramientas](#tecnologías-y-herramientas)
   - [Stack Tecnológico](#stack-tecnológico)
   - [MLOps y Gestión](#mlops-y-gestión)
6. [Metodología de Modelado](#metodología-de-modelado)
   - [Enfoque Incremental](#enfoque-incremental)
   - [Features Engineering](#features-engineering)
7. [Validación y Métricas](#validación-y-métricas)
   - [Estrategia de Validación](#estrategia-de-validación)
   - [Métricas de Evaluación](#métricas-de-evaluación)
8. [Instalación y Configuración](#instalación-y-configuración)
   - [Requisitos del Sistema](#requisitos-del-sistema)
   - [Configuración de DVC](#configuración-de-dvc)
   - [Variables de Entorno](#variables-de-entorno)
9. [Uso del Sistema](#uso-del-sistema)
   - [Pipeline Completo](#pipeline-completo)
   - [Dashboard](#dashboard)
   - [Experimentos MLflow](#experimentos-mlflow)
10. [Dashboard Features](#dashboard-features)
    - [Funcionalidades Principales](#funcionalidades-principales)
    - [Granularidades](#granularidades)
11. [Limitaciones y Consideraciones](#limitaciones-y-consideraciones)
    - [Riesgos Identificados](#riesgos-identificados)
    - [Mitigaciones](#mitigaciones)
12. [Privacidad y Ética](#privacidad-y-ética)
    - [Manejo de Datos](#manejo-de-datos)
    - [Cumplimiento](#cumplimiento)
13. [Equipo de Desarrollo](#equipo-de-desarrollo)
    - [Responsabilidades Compartidas](#responsabilidades-compartidas)
    - [Proceso de Desarrollo](#proceso-de-desarrollo)
14. [Cronograma del Proyecto](#cronograma-del-proyecto)
15. [Contribuciones](#contribuciones)
    - [Code Reviews](#code-reviews)
    - [Convenciones](#convenciones)
16. [Descarga de Documentación](#descarga-de-documentación)
17. [Licencia](#licencia)
18. [Soporte y Contacto](#soporte-y-contacto)

## Descripción del Proyecto

Sistema ligero de pronóstico de ventas para una máquina expendedora de café que utiliza análisis histórico transaccional para predecir la demanda a corto plazo (1-7 días). El sistema combina análisis descriptivo de patrones de consumo con modelos predictivos de aprendizaje supervisado, implementado a través de un dashboard interactivo.

## Objetivos

### Objetivo General
Desarrollar un sistema de pronóstico de ventas que permita anticipar la demanda y optimizar las decisiones de reposición y disponibilidad de productos.

### Objetivos Específicos
1. **Análisis de Patrones de Demanda**
   - Identificar productos de mayor/menor rotación por franjas horarias
   - Caracterizar comportamientos de compra por método de pago
   - Detectar estacionalidad y tendencias temporales

2. **Desarrollo del Modelo Predictivo**
   - Implementar modelos de ML para pronóstico de ventas diarias
   - Validar accuracy usando métricas MAE, MAPE y RMSE
   - Alcanzar error ≤ 15-20% sMAPE en horizonte semanal

3. **Sistema de Insights Operativos**
   - Generar recomendaciones para reposición de inventario
   - Optimizar ventas por franja horaria y día del año
   - Identificar horas pico y mix de bebidas

## Dataset

### Características de los Datos
- **Período**: Marzo 2024 - Marzo 2025
- **Registros**: 3,636 transacciones (~388 días)
- **Variables**: `date`, `datetime`, `cash_type`,`money` (UAH), `cofee_name`
- **Productos**: 10 tipos de bebidas (Americano, Cappuccino, Latte, etc.)
- **Métodos de pago**: Tarjeta (97.5%, 3,547), Efectivo (2.5%, 89)

### Fuentes de Datos Adicionales
- Calendario oficial de festivos de Ucrania
- Datos meteorológicos de Vinnytsia (opcional)
- Dataset público disponible en Kaggle.com

## Arquitectura del Sistema

### Estructura del Proyecto
```
coffee-demand-prediction/
├── data/
│   ├── raw/                    # Datos originales
│   ├── interim/                # Datos en procesamiento
│   ├── processed/              # Datos listos para ML
│   └── external/               # Datos externos (clima, festivos)
├── src/
│   ├── data/
│   │   ├── make_dataset.py     # Pipeline de datos
│   │   └── features.py         # Ingeniería de características
│   ├── models/
│   │   ├── train_model.py      # Entrenamiento
│   │   ├── predict_model.py    # Predicciones
│   │   └── baselines.py        # Modelos baseline
│   ├── visualization/
│   │   └── visualize.py        # Gráficos y análisis
│   └── dashboard/
│       └── streamlit_app.py    # Dashboard interactivo
├── models/                     # Modelos entrenados
├── notebooks/                  # Jupyter notebooks EDA
├── reports/                    # Reportes y documentación
├── requirements.txt            # Dependencias Python
├── pyproject.toml             # Configuración Poetry
├── dvc.yaml                   # Pipeline DVC
└── README.md                  # Este archivo
```

## Tecnologías y Herramientas

### Stack Tecnológico
- **Lenguaje**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, Prophet
- **Data Processing**: pandas, numpy, polars
- **Visualization**: plotly, matplotlib, seaborn
- **Dashboard**: Streamlit
- **Deep Learning**: TensorFlow/PyTorch (LSTM ligera)

### MLOps y Gestión
- **Version Control**: Git + GitHub/GitLab
- **Data Versioning**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **Environment**: Poetry/pip + requirements.txt
- **Deployment**: Streamlit Cloud / Heroku
- **Storage**: Google Drive / AWS S3

## Metodología de Modelado

### Enfoque Incremental
1. **Baselines Obligatorios**
   - Naive última observación
   - Naive estacional semanal (y(t-7))
   - Promedio móvil (ventanas 7/14/28 días)

2. **Modelos Candidatos**
   - **Prophet/SARIMAX**: Estacionalidad y efectos calendario
   - **XGBoost/LightGBM**: Regresión tabular con features engineering
   - **LSTM ligera**: Patrones no lineales (si el tamaño lo permite)

3. **Estrategia Jerárquica**
   - Pronóstico diario total + descomposición en patrón intra-día
   - Modelado directo de serie horaria (alternativa)

### Features Engineering
- **Calendario**: hora, día semana, fin de semana, mes, festivo
- **Lags**: ventanas móviles (rolling mean/median, sumas 7 y 28 días)
- **Estacionalidad**: perfil horario por día de semana
- **Variables externas**: clima (temperatura, precipitación)

## Validación y Métricas

### Estrategia de Validación
- **División temporal**: 80% entrenamiento, 10% validación, 10% prueba
- **Rolling-origin CV**: Ventanas expandibles
- **Horizonte**: 7 días (semana siguiente)

### Métricas de Evaluación
- **MAE & RMSE**: Sensibles a escala
- **sMAPE/MASE**: Comparables entre series, robustas con ceros
- **Cortes**: Día siguiente (h=1), semana siguiente (h=7)
- **Análisis**: Por día de semana, sensibilidad a outliers

## Instalación y Configuración

### Requisitos del Sistema
```bash
# Clonar el repositorio
git clone [repository-url]
cd coffee-demand-prediction

# Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
# o usando Poetry
poetry install
```

### Configuración de DVC
```bash
# Inicializar DVC
dvc init

# Configurar remote storage
dvc remote add -d storage s3://your-bucket/path
# o Google Drive
dvc remote add -d storage gdrive://your-folder-id

# Pull datos
dvc pull
```

### Variables de Entorno
```bash
# .env file
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
WEATHER_API_KEY=your_weather_api_key
```

## Uso del Sistema

### Pipeline Completo
```bash
# Ejecutar pipeline completo
dvc repro

# O paso a paso:
python src/data/make_dataset.py
python src/data/features.py
python src/models/train_model.py
python src/models/predict_model.py
```

### Dashboard
```bash
# Ejecutar dashboard localmente
streamlit run src/dashboard/streamlit_app.py

# URL: http://localhost:8501
```

### Experimentos MLflow
```bash
# Iniciar servidor MLflow
mlflow server --host 0.0.0.0 --port 5000

# Ver experimentos en: http://localhost:5000
```

## Dashboard Features

### Funcionalidades Principales
- **Histórico**: Filtros por bebida y método de pago
- **Pronósticos**: Predicciones diarias y horarias con bandas de incertidumbre
- **Analytics**: Top bebidas por franja, heatmap demanda hora/día
- **KPIs**: Métricas de modelo y negocio
- **Alertas**: Recomendaciones de reposición

### Granularidades
- **Diaria (D)**: Total bebidas por día
- **Horaria (H)**: Patrón intra-día por día de semana
- **Por producto**: Top-N bebidas más demandadas
- **Por método pago**: Card vs Cash

## Limitaciones y Consideraciones

### Riesgos Identificados
- **Historia limitada**: ~388 días de datos
- **Sparsidad por bebida**: Algunas categorías con pocos datos
- **Eventos especiales**: Feriados y cambios puntuales
- **Leakage**: Validación temporal estricta requerida

### Mitigaciones
- Baselines fuertes y regularización
- Empezar por demanda agregada, luego top-N bebidas
- Calendario externo y detección de outliers
- Pipeline a prueba de fugas de información

## Privacidad y Ética

### Manejo de Datos
- Datos transaccionales sin información personal identificable
- Análisis únicamente a nivel agregado
- Uso exclusivo para fines académicos
- Eliminación de datos post-proyecto

### Cumplimiento
- Respeto a términos de uso de Kaggle.com
- Licencia de código abierto
- Documentación transparente de limitaciones
- Evaluación de sesgos en modelos

## Equipo de Desarrollo

### Responsabilidades Compartidas
- **Data Manager** & **Principal Investigators**
  - Uldy Durlet Paloma Rozo
  - Jaime Andres Vera Jaramillo
  - Julian Francisco Cujabante Villamil
  - Rafael Andres Ortega Pabon

### Proceso de Desarrollo
- **Semanas 1-3**: Configuración y exploración inicial
- **Semanas 4-5**: Desarrollo paralelo (pipeline, modelos, dashboard)
- **Semanas 6-8**: Integración, pruebas y documentación

## Cronograma del Proyecto

- **Inicio**: 08-04-2025
- **Fin**: 09-23-2025
- **Última modificación**: 08-16-2025

## Contribuciones

### Code Reviews
- Rotación semanal de reviews cruzadas
- Branch protection con PR review requerido
- Commits individuales con documentación

### Convenciones
- Nomenclatura estandarizada para experimentos
- Estructura de carpetas consistente
- Documentación obligatoria en cada commit

## Descarga de Documentación

### Formatos Disponibles

Este README está disponible en múltiples formatos para facilitar su uso:

#### Markdown (Original)
- **Archivo**: `README.md`
- **Uso**: GitHub, GitLab, editores de texto
- **Características**: Formato nativo con enlaces y navegación

#### Documento Word (.docx)
Para descargar este README en formato Word:

```bash
# Opción 1: Usando pandoc (recomendado)
pandoc README.md -o README.docx

# Opción 2: Usando Python
python scripts/convert_readme.py --output docx

# Opción 3: Script automatizado
./scripts/generate_docs.sh
```

**Script de conversión** (`scripts/convert_readme.py`):
```python
import pypandoc
import sys
from pathlib import Path

def convert_readme_to_docx():
    """Convierte README.md a formato .docx"""
    try:
        # Leer el README.md
        readme_path = Path("README.md")
        output_path = Path("docs/README.docx")
        
        # Crear directorio docs si no existe
        output_path.parent.mkdir(exist_ok=True)
        
        # Convertir usando pandoc
        pypandoc.convert_file(
            str(readme_path),
            'docx',
            outputfile=str(output_path),
            extra_args=[
                '--toc',  # Tabla de contenidos
                '--toc-depth=3',  # Profundidad del índice
                '--highlight-style=github'  # Estilo de código
            ]
        )
        
        print(f"README convertido exitosamente: {output_path}")
        
    except Exception as e:
        print(f"Error en conversión: {e}")

if __name__ == "__main__":
    convert_readme_to_docx()
```

#### PDF
```bash
# Conversión a PDF
pandoc README.md -o docs/README.pdf --pdf-engine=xelatex
```

#### HTML
```bash
# Conversión a HTML con estilo
pandoc README.md -o docs/README.html --standalone --css=styles/github.css
```

### Paquete de Documentación Completo

Para descargar toda la documentación del proyecto:

```bash
# Generar paquete completo de documentación
python scripts/generate_documentation_package.py

# Contenido del paquete:
docs_package/
├── README.docx              # Este documento en Word
├── README.pdf               # Versión PDF
├── README.html              # Versión web
├── technical_specs.docx     # Especificaciones técnicas
├── data_dictionary.xlsx     # Diccionario de datos
├── model_documentation.pdf  # Documentación de modelos
└── user_manual.docx         # Manual de usuario
```

### Dependencias para Conversión

```bash
# Instalar pandoc (sistema)
# Ubuntu/Debian:
sudo apt-get install pandoc

# macOS:
brew install pandoc

# Windows: Descargar desde https://pandoc.org/installing.html

# Instalar dependencias Python
pip install pypandoc python-docx markdown beautifulsoup4
```

## Licencia

Código bajo licencia de código abierto para revisión académica. Los autores permiten uso y personalización citando como fuente.

## Soporte y Contacto

Agradecemos tu interés y confianza en coffee-demand-prediction. Si tienes alguna pregunta, sugerencia o necesitas ayuda, por favor, ponte en contacto con nosotros. Estamos aquí para ayudarte.
---

**Nota**: Este proyecto es parte de un trabajo académico. Los datos y modelos se eliminarán al finalizar el proyecto según las políticas de privacidad establecidas.
