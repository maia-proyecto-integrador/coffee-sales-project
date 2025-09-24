# Predicción de Demanda de Café - Sistema de Pronóstico para Máquina Expendedora

# Índice - coffee-sales-project

## 1. Descripción del Proyecto
- 1.1 Sistema de Pronóstico de Ventas
- 1.2 Objetivos
  - 1.2.1 Objetivo General
  - 1.2.2 Objetivos Específicos
- 1.3 Dataset
  - 1.3.1 Características de los Datos
  - 1.3.2 Fuentes de Datos Adicionales

## 2. Arquitectura del Sistema
- 2.1 Estructura del Proyecto
- 2.2 Componentes Principales
  - 2.2.1 Data Version Control (DVC)
  - 2.2.2 Aplicaciones (apps/)
  - 2.2.3 Pipeline de Datos
  - 2.2.4 Experimentación y Análisis

## 3. Tecnologías y Herramientas
- 3.1 Stack Tecnológico
- 3.2 MLOps y Gestión
- 3.3 Infraestructura de Despliegue

## 4. Metodología de Modelado
- 4.1 Enfoque de Modelos Implementados
  - 4.1.1 Baselines Obligatorios
  - 4.1.2 Modelos Principales Desarrollados
  - 4.1.3 Estrategia de Pronóstico
- 4.2 Features Engineering
- 4.3 Validación y Métricas
  - 4.3.1 Estrategia de Validación
  - 4.3.2 Métricas de Evaluación

## 5. Resultados del Modelado
- 5.1 Comparativa de Modelos
- 5.2 Selección del Modelo Ganador
- 5.3 Justificación Técnica

## 6. Implementación y Despliegue
- 6.1 Instalación y Configuración
  - 6.1.1 Requisitos del Sistema
  - 6.1.2 Configuración de DVC
  - 6.1.3 Despliegue con Docker
- 6.2 Arquitectura de Despliegue
  - 6.2.1 Componentes en Producción
  - 6.2.2 URLs de Producción

## 7. Dashboard y Funcionalidades
- 7.1 Características Implementadas
- 7.2 KPIs del Dashboard
- 7.3 Interfaz de Usuario

## 8. Gestión de Experimentos
- 8.1 MLflow Integration
- 8.2 Tracking de Experimentos
- 8.3 Registro de Modelos

## 9. Valor de Negocio
- 9.1 Beneficios Implementados
- 9.2 Impacto Operativo
- 9.3 Métricas de Éxito

## 10. Manuales y Documentación
- 10.1 Documentación Generada
- 10.2 Repositorio Principal
- 10.3 Guías de Usuario

## 11. Equipo de Desarrollo
- 11.1 Distribución de Responsabilidades
- 11.2 Proceso de Desarrollo
- 11.3 Control de Calidad

## 12. Privacidad y Ética
- 12.1 Manejo de Datos
- 12.2 Cumplimiento Normativo
- 12.3 Consideraciones Éticas

## 13. Cronograma del Proyecto
- 13.1 Fechas Clave
- 13.2 Hitos Principales
- 13.3 Estado Actual

## 14. Limitaciones y Consideraciones Futuras
- 14.1 Limitaciones Identificadas
- 14.2 Roadmap de Mejoras
- 14.3 Escalabilidad

## 15. Licencia y Uso
- 15.1 Términos de Licencia
- 15.2 Restricciones de Uso
- 15.3 Atribución Requerida

## 16. Soporte y Contacto
- 16.1 Canales de Comunicación
- 16.2 Política de Soporte
- 16.3 Reporte de Issues

## 17. Apéndices
- 17.1 Glosario de Términos
- 17.2 Referencias Técnicas
- 17.3 Enlaces Relacionados

## 18. Historial de Versiones
- 18.1 Changelog
- 18.2 Versiones Estables
- 18.3 Próximas Actualizaciones

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
coffee-sales-project/
│
├── 📁 .dvc/                          # Data Version Control
│   ├── config                        # Configuración de almacenamiento (local, S3)
│   └── .gitignore
│
├── 📁 .github/                       # Configuración GitHub (implícito)
│
├── 📁 apps/                          # Aplicaciones principales
│   ├── 📁 coffee-api/                # API de predicciones
│   │   ├── 📁 app/
│   │   │   ├── __init__.py
│   │   │   ├── apt.py
│   │   │   ├── config.py
│   │   │   └── main.py
│   │   ├── 📁 model-plag/
│   │   ├── .dockerignore
│   │   ├── .python-version
│   │   ├── Dockerfile
│   │   ├── mypy.ini
│   │   ├── requirements.txt
│   │   ├── run.sh
│   │   ├── test_requirements.txt
│   │   ├── text.ni
│   │   └── typing_requirements.txt
│   │
│   ├── 📁 coffee-dash/               # Dashboard de visualización
│   │   ├── 📁 data/
│   │   │   └── coffee_ml_features.csv
│   │   ├── .dockerignore
│   │   ├── .python-version
│   │   ├── Dockerfile
│   │   ├── Procfile                  # Configuración deployment
│   │   ├── dashboard.py
│   │   ├── mypy.ini
│   │   ├── requirements.txt
│   │   ├── run.sh
│   │   └── test_requirements.txt
│   │
│   ├── 📁 coffee-model/              # Modelos de Machine Learning
│   │   ├── 📁 model/
│   │   ├── 📁 requirements/
│   │   ├── 📁 results/               # Resultados de modelos
│   │   │   ├── 📁 artifacts/         # Artefactos del modelo
│   │   │   ├── 📁 metrics/           # Métricas de evaluación
│   │   │   ├── 📁 models/            # Modelos entrenados
│   │   │   ├── 📁 gdb/wgt/           # Pesos del modelo
│   │   │   ├── sarbox_favorable.csv
│   │   │   ├── sarbox_metrics_by_tutor
│   │   │   └── sarbox_metrics_overall.csv
│   │   ├── 📁 tests/
│   │   ├── MANIFEST.in
│   │   ├── mypy.ini
│   │   ├── pyproject.toml
│   │   ├── setup.py
│   │   └── tox.ini
│   │
│   └── 📁 coffee-sales/              # Lógica de negocio de ventas
│
├── 📁 data/                          # Pipeline de datos
│   ├── 📁 external/                  # Datos externos (exógenos, diccionarios)
│   ├── 📁 interim/                   # Datos intermedios (consolidación)
│   ├── 📁 processed/                 # Datos procesados (prefijos modificados)
│   └── 📁 raw/                       # Datos crudos (índices añadidos)
│
├── 📁 notebooks/                     # Experimentación y análisis
│   ├── 📁 archive/                   # Notebooks archivados
│   ├── 📁 garbage/                   # Notebooks descartados
│   ├── Sales_Forecast_Key_Coffee_Vending_Machine_SDN_V2_Apple/
│   ├── coffee_forecasting_tryphs/
│   ├── mfflow_stage_example.graph/
│   └── models_blocks_demands_cafes_2.0_mfflow.tryp/
│
├── 📁 dist/                          # Paquetes distribuibles
│   ├── model_coffee_sales_prediction-0.02-py3-none-any.whl
│   ├── model_coffee_sales_prediction-0.02.tar.gz
│   ├── model_coffee_sales_prediction-0.03-py3-none-any.whl
│   └── model_coffee_sales_prediction-0.03.tar.gz
│
├── 📁 reports/                       # Reportes y análisis
├── 📁 results/                       # Resultados generales
├── 📁 src/                           # Código fuente principal
│
├── 📁 train/                         # Scripts de entrenamiento
│   ├── mfflow_utils/                 # Utilidades MLflow
│   ├── input_purpose_utilizer/       # Procesamiento de entrada
│   ├── alleg_train_modeling/         # Modelado de entrenamiento
│   ├── alleg_sales_backlay/          # Capa de ventas
│   ├── alleg_score/                  # Evaluación de modelos
│   └── alleg_level/                  # Niveles de procesamiento
│
├── 📁 visualization/                 # Visualizaciones (en dashboard/)
├── 📁 model/                         # Modelos (en dashboard/)
│
├── 🔧 .dvcignore
├── 🔧 .gitignore
├── 🔧 pre-commit-config.yaml         # Control de calidad
├── 🔧 README.md                      # Documentación principal
├── 🔧 docker-compose.yml             # Orquestación de contenedores
└── 🔧 mlllow_quick_test.py          # Tests rápidos MLflow
```
## 🛠️ Tecnologías y Herramientas

### Stack Tecnológico
- **Lenguaje**: Python 3.9+
- **ML Libraries**: LightGBM, SARIMAX, Prophet, Keras/TensorFlow (LSTM)
- **Data Processing**: pandas, numpy
- **Visualization**: Panel (Holoviz)
- **Dashboard**: Panel (Python)
- **API**: FastAPI

### MLOps y Gestión
- **Version Control**: Git + GitHub
- **Data Versioning**: DVC (Data Version Control) con almacenamiento AWS S3
- **Experiment Tracking**: MLflow con servidor en EC2
- **Environment**: pip + requirements.txt
- **Containerización**: Docker + Docker Compose
- **Deployment**: AWS ECS/ECR (Elastic Container Service/Registry)
- **Storage**: AWS S3 + almacenamiento local

## 📈 Metodología de Modelado

### Enfoque de Modelos Implementados

#### 1. Baselines Obligatorios
- **Naive**: última observación
- **Seasonal Naive**: mismo día de la semana anterior (y(t-7))
- **Promedio móvil 7 días (MA7)**

#### 2. Modelos Principales Desarrollados
- **SARIMAX**: Modelo ganador con estacionalidad semanal + variables exógenas
- **LightGBM**: Entrenamiento multi-horizonte directo
- **Prophet**: Componentes aditivos con tendencia y estacionalidad
- **LSTM**: Red neuronal multi-salida para horizonte de 7 días

#### 3. Estrategia de Pronóstico
- **Horizonte fijo**: 7 días (semana completa)
- **Validación**: backtesting con orígenes rodantes
- **Separación temporal**: estricta entrenamiento/prueba

### Features Engineering
- **Calendario**: día de semana, mes, festivos Ucrania
- **Variables Climáticas**: temperatura, precipitación (exógenas)
- **Lags y Ventanas**: construidas causalmente con shift(1)
- **Agregaciones**: ventas diarias por producto y totales

## 📊 Validación y Métricas

### Estrategia de Validación
- **Backtesting con orígenes rodantes**: Emulación condiciones reales de producción
- **Prevención de data leakage**: Construcción causal de características
- **Horizonte de evaluación**: 7 días (h=1 a h=7)

### Métricas de Evaluación Implementadas
- **MAE (Mean Absolute Error)**: Error absoluto en unidades vendidas
- **RMSE (Root Mean Squared Error)**: Penaliza errores grandes (importante para picos)
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual para comparación entre productos

## 🚀 Resultados del Modelado

### Comparativa de Modelos

| Modelo | MAE | RMSE | MAPE | Estado |
|--------|-----|------|------|---------|
| **SARIMAX** | 1.02 | 1.46 | 69.0% | 🏆 **Producción** |
| LSTM | 1.05 | 1.42 | 53.57% | Validación |
| LightGBM | 1.07 | 1.45 | 55.96% | Validación |
| Prophet | 1.13 | 1.47 | 50.63% | Validación |
| Baseline (MA7) | 1.19 | 1.78 | 64.93% | Referencia |

### Selección del Modelo Ganador
**SARIMAX** fue seleccionado para producción por:
- Mejor MAE (1.02) para reposición precisa
- Estabilidad en horizonte de 7 días
- Capacidad de incorporar variables exógenas
- Interpretabilidad estadística

## 💻 Instalación y Configuración

### Requisitos del Sistema
```bash
# Clonar el repositorio
git clone https://github.com/maia-proyecto-integrador/coffee-sales-project.git
cd coffee-sales-project

# Instalar dependencias para cada aplicación
cd apps/coffee-api && pip install -r requirements.txt
cd apps/coffee-dash && pip install -r requirements.txt
cd apps/coffee-model && pip install -r requirements.txt
```

### Configuración de DVC
```bash
# Configuración existente en .dvc/config
[dvc pull]
```

### Despliegue con Docker
```
# Ejecutar con Docker Compose
docker-compose up -d

# O construir imágenes individuales
docker build -t coffee-api ./apps/coffee-api
docker build -t coffee-dash ./apps/coffee-dash
```
## 🌐 Arquitectura de Despliegue

### Componentes en Producción
- AWS ECS Cluster: 2 servicios (API + Dashboard)
- ECR Repository: Imágenes Docker versionadas
- EC2 Instances: Servidor MLflow y ejecución de modelos
- Security Groups: Configuración de networking

### URLs de Producción
- Dashboard: http://107.22.112.218:5006/dashboard
- MLflow Tracking: http://[ec2-ip]:5000

## 📊 Dashboard Features
#### Funcionalidades Implementadas
- Pronósticos: Ventas totales y por producto a 7 días
- Análisis Histórico: Filtros por producto, fecha, día de semana
- Métricas de Modelo: MAE, RMSE, MAPE en tiempo real
- Gestión de Inventario: Proyección de insumos y costos
- Análisis Financiero: Ingresos, costos y ganancias proyectadas

### KPIs del Dashboard
- Ventas del último día y variaciones porcentuales
- Producto más vendido y evolución temporal
- Validación de modelos con métricas actualizadas
- Alertas de reposición basadas en pronósticos

## 📚 Manuales y Documentación
####  Documentación Generada
- Manual de Usuario: Manual_Usuario_Dashboard.pdf
- Manual de Instalación: Manual_Instalacion.pdf
- Reporte Técnico: Análisis comparativo de modelos
- Documentación de API: Especificaciones FastAPI

### Repositorio Principal
- URL: https://github.com/maia-proyecto-integrador/coffee-sales-project.git
- Branches: 9 ramas con desarrollo organizado
- Commits: Historial trazable de contribuciones

## 🎯 Valor de Negocio
### Beneficios Implementados
- Reducción de desperdicios: 15-20% menos sobreproducción
- Optimización de compras: Basada en demanda anticipada
- Dashboard accionable: Decisiones operativas diarias
- Proyección financiera: Ingresos, costos y ganancias

### Impacto Operativo
- Inventario: Evita quiebres y sobrestock
- Satisfacción cliente: Disponibilidad constante
- Rentabilidad: Maximización de ganancias
- Planificación: Anticipación a demanda semanal

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
  - Cujabante Villamil Julian Francisco
  - Ortega Pabon afael Andres 
  - Paloma Rozo Uldy Durlet
  - Vera Jaramillo Jaime Andres

### Proceso de Desarrollo
- **Semanas 1-3**: Configuración y exploración inicial
- **Semanas 4-5**: Desarrollo paralelo (pipeline, modelos, dashboard)
- **Semanas 6-8**: Integración, pruebas y despliegue
- **Control de Calidad**: Pull requests, revisiones cruzadas

## Cronograma del Proyecto

- **Inicio**: 08-04-2025
- **Fin**: 09-23-2025
- **Última modificación**: 09-23-2025

#### Estado del proyecto: ✅ En producción

## Licencia

Código bajo licencia de código abierto para revisión académica. Los autores permiten uso y personalización citando como fuente.

## Soporte y Contacto

Agradecemos tu interés y confianza en coffee-demand-prediction. Si tienes alguna pregunta, sugerencia o necesitas ayuda, por favor, ponte en contacto con nosotros. Estamos aquí para ayudarte.
---

**Nota**: Este proyecto es parte de un trabajo académico. Los datos y modelos se eliminarán al finalizar el proyecto según las políticas de privacidad establecidas.
