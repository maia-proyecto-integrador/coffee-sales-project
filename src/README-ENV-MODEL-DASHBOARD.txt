1. Creación de ambiente para el proyecto  usando anaconda

conda create -n coffee310 python=3.10 -y

1.1 Activación de ambiente recién creado

conda activate coffee310


2. Instalación de librerías y dependencias en el ambiente.


conda install -y -c conda-forge numpy=1.26 pandas=2.1.4 pyarrow mlflow=2.14.1

python -m pip install --upgrade pip
pip install packaging==24.1 scikit-learn==1.3.2 lightgbm==4.3.0 statsmodels==0.14.3 joblib==1.3.2 tqdm==4.66.5 matplotlib==3.8.4

pip install prophet==1.1.5 cmdstanpy==1.2.4

# Keras 3 con backend PyTorch (CPU)
pip install torch==2.2.2 torchvision==0.17.2
pip install keras==3.4.1

# Jupyter kernel para este env
pip install ipykernel==6.29.5

#Librerias Dashboard
conda install dask[complete] panel hvplot holoviews bokeh pandas numpy joblib


2.1 Creación de kernel de Jupyter con las dependencias del ambiente. Esto si se quiere correr un notebook con estas especificaciones 

python -m ipykernel install --user --name coffee-310 --display-name "Coffee (Py3.10)"

2.2. Inicialización de Jupyter

jupyter lab


3. Revisión de importación de librerías clave en el código.

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras, torch
print("Keras:", keras.__version__, "| Backend:", keras.config.backend(), "| Torch:", torch.__version__)


4. Correr el código, debes estar parado en la raíz del repo (puedes clonarlo local)
Estando ahí se activa el ambiente activate conda coffee310
luego se declaran la URI user y password para mlflow (La máquina EC2 que está en Databricks) Recuerden usar las credenciales que Rafa les pasó.

Para declararlas en CMD (windows)
set MLFLOW_TRACKING_URI=http://54.163.40.154:5000
set MLFLOW_TRACKING_USERNAME=julian
set MLFLOW_TRACKING_PASSWORD=3pJbG7gCZ6w6

Para declararlas en powershell/terminal (windows)
$env:MLFLOW_TRACKING_URI = "http://54.163.40.154:5000"
$env:MLFLOW_TRACKING_USERNAME = "julian"
$env:MLFLOW_TRACKING_PASSWORD = "3pJbG7gCZ6w6"

Para declararlas en mac/Linux
export MLFLOW_TRACKING_URI="http://54.163.40.154:5000"
export MLFLOW_TRACKING_USERNAME="julian"
export MLFLOW_TRACKING_PASSWORD="3pJbG7gCZ6w6"


Siempre se deben declarar antes

Luego se corre python -m src.model.main (por eso es importante estar parado en la raíz del repo o ajustar el comando de corrida de acuerdo a donde estén parados en el repo)

Si quisieran correr todo ósea declarar las variables de entorno y el código:

Para cmd:
set MLFLOW_TRACKING_URI=http://54.163.40.154:5000 && set MLFLOW_TRACKING_USERNAME=julian && set MLFLOW_TRACKING_PASSWORD=3pJbG7gCZ6w6 && python -m src.model.main

Para powershell:
$env:MLFLOW_TRACKING_URI = "http://54.163.40.154:5000"; $env:MLFLOW_TRACKING_USERNAME = "julian"; $env:MLFLOW_TRACKING_PASSWORD = "3pJbG7gCZ6w6"; python -m src.model.main

Para mac/Linux:
export MLFLOW_TRACKING_URI="http://54.163.40.154:5000" && export MLFLOW_TRACKING_USERNAME="julian" && export MLFLOW_TRACKING_PASSWORD="3pJbG7gCZ6w6" && python -m src.model.main


Una vez esto termine de correr podrán ver los experimentos en mlflow en la url: http://54.163.40.154:5000

Si quieren experimentar pueden cambiar configuraciones de los modelos en el .py de config dentro de src/model

Una vez terminado esto pueden correr igualmente el dashboard:

Para cmd:
set MLFLOW_TRACKING_URI=http://54.163.40.154:5000 && set MLFLOW_TRACKING_USERNAME=julian && set MLFLOW_TRACKING_PASSWORD=3pJbG7gCZ6w6 && python dashboard/dashboard.py

Para powershell:$env:MLFLOW_TRACKING_URI = "http://54.163.40.154:5000"; $env:MLFLOW_TRACKING_USERNAME = "julian"; $env:MLFLOW_TRACKING_PASSWORD = "3pJbG7gCZ6w6"; python dashboard/dashboard.py


Para mac:


export MLFLOW_TRACKING_URI="http://54.163.40.154:5000" && export MLFLOW_TRACKING_USERNAME="julian" && export MLFLOW_TRACKING_PASSWORD="3pJbG7gCZ6w6" && python dashboard/dashboard.py


Para ver el dashboard basta con poner http://localhost:5006 en el browser.


################A TENER EN CUENTA

No siempre que se ejecute el ambiente se deben instalar las dependencias y librerías, esto sólo se hace una vez. Para correr el código una vez instaladas estas, basta con activar el ambiente, lanzar jupyter, elegir el kernel correspondiente y tras la finalización de la corrida ejecutar código para mlflow.