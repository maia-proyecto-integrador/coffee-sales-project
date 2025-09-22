FROM python:3.12

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Definir directorio de trabajo 
WORKDIR /opt/coffee-api

# Copiar archivos de requirements primero para aprovechar cache de Docker
COPY requirements.txt /opt/coffee-api/
COPY test_requirements.txt /opt/coffee-api/

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r /opt/coffee-api/requirements.txt
RUN pip install pydantic-settings

# Copiar el paquete del modelo y instalarlo
COPY model-pkg/*.whl /tmp/
RUN pip install /tmp/*.whl

# Copiar el código de la aplicación
COPY ./ /opt/coffee-api/

# Hacer el directorio de trabajo ejecutable 
RUN chmod +x /opt/coffee-api/run.sh
# Cambiar propiedad de la carpeta a api-user 
RUN chown -R api-user:api-user ./

USER api-user
# Puerto a exponer para la api 
EXPOSE 8001

# Comandos a ejecutar al correr el contenedor 
CMD ["bash", "./run.sh"]
