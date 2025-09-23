FROM python:3.12

# Crear usuario que ejecuta el dashboard
RUN adduser --disabled-password --gecos '' dashboard-user

# Definir directorio de trabajo 
WORKDIR /opt/coffee-dashboard

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements primero para aprovechar cache de Docker
COPY requirements.txt /opt/coffee-dashboard/
COPY test_requirements.txt /opt/coffee-dashboard/

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r /opt/coffee-dashboard/requirements.txt

# Copiar el código de la aplicación
COPY ./ /opt/coffee-dashboard/

# Crear directorios necesarios para los datos
RUN mkdir -p /opt/coffee-dashboard/data/processed

# Copiar solo los datos CSV necesarios para el dashboard
COPY data/coffee_ml_features.csv /opt/coffee-dashboard/data/processed/

# Configurar variables de entorno para la API
ENV COFFEE_API_URL=http://coffee-api:8001

# Hacer el directorio de trabajo ejecutable 
RUN chmod +x /opt/coffee-dashboard/run.sh
# Cambiar propiedad de la carpeta a dashboard-user 
RUN chown -R dashboard-user:dashboard-user ./

USER dashboard-user
# Puerto a exponer para el dashboard (Panel usa 5006 por defecto)
EXPOSE 5006

# Comandos a ejecutar al correr el contenedor 
CMD ["bash", "./run.sh"]