# ☕ Manual de Usuario - Coffee Machine AI Assistant

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Acceso al Dashboard](#acceso-al-dashboard)
3. [Interfaz Principal](#interfaz-principal)
4. [Panel de Filtros](#panel-de-filtros)
5. [KPIs Principales](#kpis-principales)
6. [Análisis de Patrones](#análisis-de-patrones)
7. [Pronósticos de Ventas](#pronósticos-de-ventas)
8. [Validación del Modelo](#validación-del-modelo)
9. [Inventario Proyectado](#inventario-proyectado)
10. [Finanzas Proyectadas](#finanzas-proyectadas)
11. [Guía de Navegación](#guía-de-navegación)
12. [Solución de Problemas](#solución-de-problemas)

---

## 🎯 Introducción

El **Coffee Machine AI Assistant** es un dashboard interactivo que te permite:

- 📊 **Monitorear** las ventas de tu máquina de café en tiempo real
- 🔮 **Predecir** las ventas futuras usando inteligencia artificial
- 📦 **Planificar** el inventario de insumos necesarios
- 💰 **Proyectar** ingresos y rentabilidad
- 📈 **Analizar** patrones de consumo por tipo de bebida, día y hora

### ✨ Características principales:
- **Interfaz intuitiva** con filtros dinámicos
- **Predicciones con IA** usando modelo SARIMAX
- **Visualizaciones interactivas** en tiempo real
- **Cálculos automáticos** de inventario y finanzas
- **Responsive design** para cualquier dispositivo

---

## 🌐 Acceso al Dashboard

### 1. URL de acceso:
```
http://localhost:5006/dashboard
```

### 2. Requisitos del navegador:
- **Chrome** 90+ (Recomendado)
- **Firefox** 88+
- **Safari** 14+
- **Edge** 90+

### 3. Verificar que el servicio esté ejecutándose:
```bash
# Verificar estado del contenedor
docker ps | grep coffee-dashboard

# Ver logs si hay problemas
docker logs coffee-dashboard
```

---

## 🖥️ Interfaz Principal

### Estructura del Dashboard:

```
┌─────────────────────────────────────────────────────────────┐
│  ☕ Coffee Machine AI Assistant                              │
├─────────────┬───────────────────────────────────────────────┤
│             │  📊 KPIs Principales                          │
│   ⚙️ Filtros │  ├─ Ventas último día                        │
│             │  ├─ % cambio vs semana pasada                 │
│   ☕ Bebida  │  ├─ % cambio vs histórico                     │
│   📅 Período│  └─ Producto más vendido                       │
│   📆 Días    │                                               │
│   🕒 Horas   │  📈 Análisis de Patrones                      │
│             │  ├─ Evolución temporal                        │
│             │  └─ Composición por bebida                    │
│             │                                               │
│             │  🔮 Pronóstico de Ventas                      │
│             │  🔍 Validación del Modelo                     │
│             │  📦 Inventario Proyectado                     │
│             │  💰 Finanzas Proyectadas                      │
└─────────────┴───────────────────────────────────────────────┘
```

---

## ⚙️ Panel de Filtros

El panel lateral izquierdo contiene todos los filtros para personalizar tu análisis:

### 1. ☕ **Tipo de bebida**
- **Función**: Filtra los datos por tipo de café
- **Opciones**: All, Espresso, Cappuccino, Latte, Americano, Mocha, etc.
- **Uso**: Selecciona "All" para ver todas las bebidas o una específica

### 2. 📅 **Período**
- **Función**: Define el rango de fechas para el análisis
- **Control**: Deslizador de rango de fechas
- **Uso**: 
  - Arrastra los extremos para ajustar el período
  - El rango se actualiza automáticamente en todos los gráficos

### 3. 📆 **Día de la semana**
- **Función**: Filtra por días específicos de la semana
- **Opciones**: All, Lunes, Martes, Miércoles, Jueves, Viernes, Sábado, Domingo
- **Uso**: 
  - Selecciona múltiples días manteniendo Ctrl/Cmd
  - Útil para analizar patrones de fin de semana vs días laborables

### 4. 🕒 **Hora del día**
- **Función**: Filtra por rango horario
- **Control**: Deslizador de rango numérico
- **Uso**: 
  - Ajusta para analizar horarios específicos (ej: 7-10 AM para desayuno)
  - Útil para identificar horas pico

### 💡 **Consejos de uso de filtros:**
- Los filtros se aplican **automáticamente** al cambiarlos
- Todos los gráficos se actualizan **en tiempo real**
- Usa **combinaciones** de filtros para análisis específicos
- El botón **"Reset"** (si está disponible) restaura los valores por defecto

---

## 📊 KPIs Principales

Esta sección muestra los indicadores clave de rendimiento:

### 1. **Ventas último día**
- **Qué muestra**: Ingresos del último día en el rango seleccionado
- **Formato**: ₴ X,XXX (Hryvnia ucraniana)
- **Color**: Azul neutral

### 2. **% cambio vs semana pasada**
- **Qué muestra**: Variación porcentual respecto a la semana anterior
- **Colores**:
  - 🟢 **Verde**: Crecimiento positivo (>0%)
  - 🟡 **Naranja**: Sin cambios (0%)
  - 🔴 **Rojo**: Decrecimiento (<0%)

### 3. **% cambio vs histórico del día**
- **Qué muestra**: Comparación con el promedio histórico del mismo día de la semana
- **Utilidad**: Identifica si el rendimiento es normal para ese día específico

### 4. **Ventas total (rango)**
- **Qué muestra**: Suma total de ventas en el período seleccionado
- **Utilidad**: Vista general del volumen de negocio

### 5. **Producto más vendido**
- **Qué muestra**: La bebida con mayor volumen de ventas
- **Formato**: Texto descriptivo

### 📈 **Interpretación de KPIs:**
- **Verde** = Tendencia positiva, buen rendimiento
- **Naranja** = Rendimiento estable, monitorear
- **Rojo** = Tendencia negativa, requiere atención

---

## 📈 Análisis de Patrones

### **Evolución de ventas en el rango**
- **Tipo**: Gráfico de líneas temporal
- **Eje X**: Fechas
- **Eje Y**: Ventas en ₴
- **Color**: Dorado (#B8860B)

**Cómo interpretar:**
- **Línea ascendente**: Crecimiento en ventas
- **Picos**: Días de alta demanda
- **Valles**: Días de baja demanda
- **Tendencia general**: Dirección del negocio

### **Composición de ventas por bebida**
- **Tipo**: Gráfico de barras horizontales
- **Eje X**: Ventas en ₴
- **Eje Y**: Tipos de bebida
- **Orden**: De menor a mayor ventas

**Cómo interpretar:**
- **Barras más largas**: Bebidas más populares
- **Distribución**: Diversidad del portafolio
- **Oportunidades**: Bebidas con potencial de crecimiento

### 💡 **Casos de uso típicos:**
1. **Análisis semanal**: Filtra por semana para ver patrones diarios
2. **Comparación de productos**: Usa "All" en bebidas para ver el mix completo
3. **Análisis de horarios**: Filtra por horas pico (7-9 AM, 12-2 PM, 3-5 PM)

---

## 🔮 Pronósticos de Ventas

Esta sección utiliza **inteligencia artificial** para predecir ventas futuras.

### **Modelo utilizado: SARIMAX**
- **Tipo**: Modelo estadístico avanzado para series temporales
- **Ventajas**: Considera estacionalidad, tendencias y variables externas
- **Horizonte**: 7 días por defecto

### **Visualización del pronóstico:**
- **Línea negra**: Datos históricos reales
- **Línea dorada**: Predicciones futuras
- **Línea roja punteada**: Separación entre histórico y pronóstico

### **KPIs del pronóstico:**

#### 1. **Ventas totales pronóstico (7d)**
- **Qué muestra**: Suma estimada de ventas para los próximos 7 días
- **Utilidad**: Planificación de ingresos

#### 2. **Cambio vs. período previo**
- **Qué muestra**: Variación esperada respecto al período anterior
- **Colores**: Verde (↗️), Naranja (➡️), Rojo (↘️)

### **Factores considerados por el modelo:**
- 📅 **Estacionalidad**: Patrones semanales y mensuales
- 🌡️ **Clima**: Temperatura, precipitación, nubosidad
- 🎉 **Festividades**: Días festivos y eventos especiales
- 📈 **Tendencias históricas**: Comportamiento pasado

### 🎯 **Cómo usar los pronósticos:**
1. **Planificación de inventario**: Asegurar stock suficiente
2. **Gestión de personal**: Programar turnos según demanda esperada
3. **Estrategias de marketing**: Promociones en días de baja demanda
4. **Análisis financiero**: Proyecciones de ingresos

---

## 🔍 Validación del Modelo

Esta sección evalúa la **precisión** del modelo de predicción.

### **Métricas de evaluación:**

#### 1. **MAPE (Mean Absolute Percentage Error)**
- **Qué es**: Error porcentual promedio
- **Interpretación**: 
  - < 10% = Excelente precisión
  - 10-20% = Buena precisión  
  - 20-50% = Precisión razonable
  - > 50% = Baja precisión

#### 2. **RMSE (Root Mean Square Error)**
- **Qué es**: Error cuadrático medio
- **Utilidad**: Penaliza errores grandes más severamente

#### 3. **MAE (Mean Absolute Error)**
- **Qué es**: Error absoluto promedio
- **Utilidad**: Error promedio en las mismas unidades que los datos

### **Gráfico de validación:**
- **Línea azul**: Valores reales
- **Línea naranja**: Predicciones del modelo
- **Proximidad**: Qué tan cerca están las líneas indica la precisión

### **Estado del modelo:**
- ✅ **"Modelo funcionando correctamente"**: Alta precisión
- ⚠️ **"Precisión moderada"**: Usar con precaución
- ❌ **"Baja precisión"**: Revisar datos o modelo

---

## 📦 Inventario Proyectado

Calcula automáticamente los **insumos necesarios** basándose en los pronósticos.

### **Insumos tracked:**
- ☕ **Café molido** (gramos)
- 🥛 **Leche** (mililitros)  
- 💧 **Agua** (mililitros)
- 🍫 **Chocolate** (gramos)
- 🍯 **Azúcar** (gramos)

### **Recetas consideradas:**
```
Espresso:     18g café + 30ml agua + 5g azúcar
Cappuccino:   18g café + 150ml leche + 10g azúcar
Latte:        18g café + 200ml leche + 10g azúcar
Americano:    18g café + 100ml agua + 5g azúcar
Mocha:        18g café + 150ml leche + 25g chocolate + 12g azúcar
```

### **Visualización:**
- **Gráfico de barras**: Cantidad necesaria por insumo
- **Tabla detallada**: Desglose por tipo de bebida
- **Alertas**: Insumos con alta demanda proyectada

### 📋 **Cómo usar esta información:**
1. **Compras**: Lista de compras automática
2. **Stock**: Verificar inventario actual vs necesidades
3. **Proveedores**: Coordinar entregas según demanda
4. **Costos**: Estimar gastos en insumos

---

## 💰 Finanzas Proyectadas

Proyecta los **aspectos financieros** del negocio.

### **Métricas calculadas:**

#### 1. **Ingresos proyectados**
- **Fuente**: Pronósticos de ventas × precios
- **Período**: Basado en horizonte de predicción
- **Moneda**: Hryvnia ucraniana (₴)

#### 2. **Costos de insumos**
- **Cálculo**: Cantidad necesaria × precio unitario
- **Incluye**: Todos los ingredientes por bebida
- **Actualizable**: Precios configurables

#### 3. **Margen bruto proyectado**
- **Fórmula**: (Ingresos - Costos) / Ingresos × 100
- **Interpretación**: 
  - > 70% = Excelente rentabilidad
  - 50-70% = Buena rentabilidad
  - 30-50% = Rentabilidad moderada
  - < 30% = Revisar precios/costos

#### 4. **Punto de equilibrio**
- **Qué es**: Ventas mínimas para cubrir costos
- **Utilidad**: Objetivo diario/semanal mínimo

### **Visualizaciones:**
- **Gráfico de ingresos vs costos**: Comparación temporal
- **Gráfico de márgenes**: Evolución de rentabilidad
- **Tabla de desglose**: Detalle por tipo de bebida

### 💡 **Decisiones basadas en finanzas:**
1. **Ajuste de precios**: Si los márgenes son bajos
2. **Optimización de recetas**: Reducir costos de insumos
3. **Promociones estratégicas**: En productos de alto margen
4. **Expansión**: Si la rentabilidad es consistentemente alta

---

## 🧭 Guía de Navegación

### **Controles básicos:**

#### **Zoom en gráficos:**
- **Rueda del mouse**: Zoom in/out
- **Click + arrastrar**: Seleccionar área para zoom
- **Doble click**: Reset zoom

#### **Interactividad:**
- **Hover**: Información detallada al pasar el mouse
- **Click en leyenda**: Ocultar/mostrar series
- **Arrastrar**: Mover gráficos (si está habilitado)

#### **Filtros dinámicos:**
- **Cambios automáticos**: Los gráficos se actualizan al modificar filtros
- **Múltiple selección**: Mantén Ctrl/Cmd para seleccionar varios elementos
- **Reset**: Algunos filtros tienen botón de reinicio

### **Navegación por secciones:**

#### **Secciones colapsables:**
- **Click en título**: Expandir/contraer sección
- **Útil para**: Enfocar en análisis específicos
- **Performance**: Secciones contraídas cargan más rápido

### **Atajos de teclado:**
- **F5**: Refrescar dashboard
- **Ctrl + R**: Recargar página
- **F11**: Pantalla completa
- **Esc**: Salir de pantalla completa

---

## 🔧 Solución de Problemas

### **Problemas comunes y soluciones:**

#### 1. **El dashboard no carga**
```
Síntomas: Página en blanco o error de conexión
Soluciones:
- Verificar que el servicio esté ejecutándose: docker ps
- Revisar logs: docker logs coffee-dashboard
- Verificar la URL: http://localhost:5006/dashboard
- Probar en modo incógnito del navegador
```

#### 2. **Los gráficos no se muestran**
```
Síntomas: Espacios vacíos donde deberían estar los gráficos
Soluciones:
- Desactivar bloqueadores de anuncios
- Habilitar JavaScript en el navegador
- Actualizar el navegador a la última versión
- Limpiar caché: Ctrl+Shift+Del
```

#### 3. **Los filtros no funcionan**
```
Síntomas: Los gráficos no cambian al ajustar filtros
Soluciones:
- Esperar unos segundos (procesamiento)
- Refrescar la página (F5)
- Verificar que hay datos en el rango seleccionado
- Revisar conexión con la API
```

#### 4. **Datos faltantes o incorrectos**
```
Síntomas: Gráficos vacíos o valores extraños
Soluciones:
- Verificar rango de fechas (puede estar muy limitado)
- Revisar filtros (pueden ser muy restrictivos)
- Comprobar estado de la API: http://localhost:8001/api/v1/health
- Verificar logs del contenedor
```

#### 5. **Rendimiento lento**
```
Síntomas: Dashboard responde lentamente
Soluciones:
- Reducir rango de fechas para menos datos
- Cerrar otras pestañas del navegador
- Verificar recursos del sistema: docker stats
- Reiniciar el contenedor si es necesario
```

### **Verificaciones de estado:**

#### **1. Salud del sistema:**
```bash
# Estado de contenedores
docker ps

# Uso de recursos
docker stats coffee-dashboard

# Logs recientes
docker logs --tail 50 coffee-dashboard
```

#### **2. Conectividad con API:**
```bash
# Verificar API
curl http://localhost:8001/api/v1/health

# Test de predicción
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 3}'
```

#### **3. Navegador:**
```
1. Abrir herramientas de desarrollador (F12)
2. Ir a la pestaña "Console"
3. Buscar errores en rojo
4. Ir a "Network" y verificar requests fallidos
```

### **Contacto para soporte:**
Si los problemas persisten, contacta al equipo técnico con:
- **URL** que estás intentando acceder
- **Navegador** y versión
- **Mensaje de error** exacto
- **Capturas de pantalla** si es posible
- **Logs** del contenedor

---

## 📱 Compatibilidad y Rendimiento

### **Dispositivos soportados:**
- 💻 **Desktop**: Experiencia completa
- 📱 **Tablet**: Funcionalidad adaptada
- 📱 **Mobile**: Vista básica (no recomendado para análisis detallado)

### **Resoluciones recomendadas:**
- **Mínima**: 1024x768
- **Recomendada**: 1920x1080 o superior
- **4K**: Soporte completo

### **Rendimiento óptimo:**
- **RAM**: 4GB+ disponible
- **Conexión**: Banda ancha estable
- **Navegador**: Actualizado y con JavaScript habilitado

---

## 🎯 Casos de Uso Típicos

### **1. Análisis diario matutino (9:00 AM)**
```
Objetivo: Revisar rendimiento del día anterior y planificar el día
Pasos:
1. Abrir dashboard
2. Revisar KPIs principales
3. Verificar pronóstico del día
4. Ajustar inventario si es necesario
Tiempo estimado: 5 minutos
```

### **2. Planificación semanal (Lunes)**
```
Objetivo: Preparar la semana basándose en pronósticos
Pasos:
1. Configurar filtros para la semana actual
2. Revisar pronóstico de 7 días
3. Analizar inventario proyectado
4. Planificar compras y personal
Tiempo estimado: 15 minutos
```

### **3. Análisis de producto específico**
```
Objetivo: Evaluar rendimiento de una bebida particular
Pasos:
1. Filtrar por tipo de bebida específico
2. Analizar patrones temporales
3. Revisar márgenes de rentabilidad
4. Identificar oportunidades de mejora
Tiempo estimado: 10 minutos
```

### **4. Revisión mensual**
```
Objetivo: Análisis profundo de tendencias y rendimiento
Pasos:
1. Configurar período de 30 días
2. Revisar todos los KPIs
3. Analizar validación del modelo
4. Evaluar finanzas proyectadas
5. Documentar insights y decisiones
Tiempo estimado: 30 minutos
```

---

## 📚 Glosario de Términos

- **SARIMAX**: Seasonal AutoRegressive Integrated Moving Average with eXogenous variables
- **KPI**: Key Performance Indicator (Indicador Clave de Rendimiento)
- **MAPE**: Mean Absolute Percentage Error (Error Porcentual Absoluto Medio)
- **RMSE**: Root Mean Square Error (Raíz del Error Cuadrático Medio)
- **MAE**: Mean Absolute Error (Error Absoluto Medio)
- **Horizonte**: Período hacia el futuro para el cual se hacen predicciones
- **Estacionalidad**: Patrones que se repiten en períodos regulares
- **Exógenas**: Variables externas que influyen en el modelo (clima, festividades)

---

## 🔄 Actualizaciones y Versiones

### **Versión actual**: 1.0
### **Fecha**: Septiembre 2024

### **Historial de cambios:**
- **v1.0**: Versión inicial con funcionalidad completa
  - Dashboard interactivo
  - Integración con API de predicciones
  - Cálculos de inventario y finanzas
  - Validación de modelo

### **Próximas características (roadmap):**
- 📊 **Alertas automáticas** por email/SMS
- 🎯 **Recomendaciones inteligentes** de precios
- 📱 **App móvil** nativa
- 🔗 **Integración** con sistemas POS
- 🤖 **Chatbot** para consultas rápidas

---

¡Gracias por usar Coffee Machine AI Assistant! ☕✨

**Para soporte técnico**: Contacta al equipo de desarrollo
**Documentación técnica**: Ver MANUAL_INSTALACION.md
**Código fuente**: Disponible en el repositorio del proyecto

---

*Manual generado: Septiembre 2024*  
*Versión del documento: 1.0*
