# Coffee Sales - Predicción de Demanda de Café

Una aplicación web moderna desarrollada con Angular 20 para la predicción de demanda de café con visualización de datos interactiva.

## 🚀 Características

- **Dashboard Interactivo**: Interfaz limpia y moderna para visualizar datos de ventas
- **Predicción de Demanda**: Algoritmos para predecir ventas futuras de café
- **Gráficos Dinámicos**: Visualización de datos históricos y predicciones usando Chart.js
- **Filtros Interactivos**: Selección por tipo de café y rango de fechas
- **Responsive Design**: Optimizado para dispositivos móviles y desktop
- **Métricas en Tiempo Real**: Cards con métricas clave como ventas totales, próximo período y porcentaje de cambio

## 🛠️ Tecnologías Utilizadas

- **Angular 20**: Framework principal
- **TypeScript**: Lenguaje de programación
- **SCSS**: Preprocesador de CSS
- **Chart.js**: Librería para gráficos
- **Angular CLI**: Herramientas de desarrollo

## 📦 Instalación

### Prerrequisitos

- Node.js (v18 o superior)
- npm (v8 o superior)
- Angular CLI

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd coffee-sales
   ```

2. **Instalar dependencias**
   ```bash
   npm install
   ```

3. **Ejecutar la aplicación**
   ```bash
   ng serve
   ```

4. **Abrir en el navegador**
   Navega a `http://localhost:4200/`

## 🎯 Funcionalidades

### Dashboard Principal
- **Título**: "COFFEE DEMAND PREDICTION"
- **Selectores**: 
  - Tipo de café (Espresso, Americano, Cappuccino)
  - Rango de fechas
- **Métricas**:
  - Total de ventas
  - Predicción próximo período
  - Porcentaje de cambio

### Gráfico Interactivo
- Línea histórica de ventas
- Línea punteada para predicciones
- Datos actualizables según selecciones

### Tabla de Pronósticos
- Comparación por tipo de café
- Ventas actuales vs predicciones
- Datos numéricos formateados

## 🎨 Diseño

La interfaz está diseñada con:
- **Colores**: Paleta neutra con acentos en dorado (#c8860d)
- **Tipografía**: Sistema de fuentes nativo (-apple-system, BlinkMacSystemFont)
- **Layout**: Grid responsivo con cards y componentes modulares
- **Espaciado**: Consistente y aireado para mejor legibilidad

## 📱 Responsive

La aplicación es completamente responsive:
- **Desktop**: Layout de 3 columnas para métricas
- **Tablet**: Adaptación de espacios y tamaños
- **Mobile**: Layout de una columna con navegación optimizada

## 🔧 Comandos Disponibles

```bash
# Desarrollo
ng serve                    # Servidor de desarrollo
ng serve --open            # Servidor + abrir navegador

# Build
ng build                    # Build para producción
ng build --watch           # Build con watch mode

# Testing
ng test                     # Tests unitarios
ng e2e                      # Tests end-to-end

# Linting
ng lint                     # Verificar código
```

## 📊 Estructura de Datos

```typescript
interface ForecastData {
  type: string;      // Tipo de café
  sales: number;     // Ventas actuales
  forecast: number;  // Predicción
}
```

## 🚀 Próximas Mejoras

- [ ] Integración con API real de datos
- [ ] Más tipos de gráficos (barras, pie)
- [ ] Exportación de reportes (PDF, Excel)
- [ ] Filtros avanzados por fecha
- [ ] Modo oscuro
- [ ] Notificaciones push
- [ ] Dashboard personalizable

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 👥 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Si tienes alguna pregunta o problema, por favor abre un issue en el repositorio.

---

Desarrollado con ❤️ usando Angular 20