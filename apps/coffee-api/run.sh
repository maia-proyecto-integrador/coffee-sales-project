#!/bin/bash
# Script para ejecutar la API Coffee Sales

# Usar puerto 8001 por defecto si no se especifica PORT
PORT=${PORT:-8001}

echo "🚀 Iniciando Coffee Sales API en puerto $PORT..."

# Ejecutar la aplicación
uvicorn app.main:app --host 0.0.0.0 --port $PORT --log-level info