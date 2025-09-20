import os, mlflow

exp = os.getenv("MLFLOW_EXPERIMENT") or "Default"
mlflow.set_experiment(exp)

with mlflow.start_run(run_name="ps-quick-test"):
    mlflow.log_param("who", "julian")
    mlflow.log_metric("m", 0.123)

print("done")
