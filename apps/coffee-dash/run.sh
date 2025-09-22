#!/bin/bash
# Script para ejecutar el Dashboard de Coffee Sales con Panel

# Usar puerto 5006 por defecto si no se especifica PORT
PORT=${PORT:-5006}

echo "🚀 Iniciando Coffee Sales Dashboard en puerto $PORT..."

# Ejecutar el dashboard con Panel
panel serve dashboard.py --port $PORT --allow-websocket-origin=* --show