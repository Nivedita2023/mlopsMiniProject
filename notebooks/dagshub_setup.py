import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/niveditaranjan223883/mlopsMiniProject.mlflow')
dagshub.init(repo_owner='niveditaranjan223883', repo_name='mlopsMiniProject', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)