import json
import mlflow
import logging
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/niveditaranjan223883/mlopsMiniProject.mlflow')
dagshub.init(repo_owner='niveditaranjan223883', repo_name='mlopsMiniProject', mlflow=True)

logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
logger.addHandler(console_handler)

def transition_to_staging(model_name: str):
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name)
    if latest_versions:
        latest_version = latest_versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        logger.debug(f'Model {model_name} version {latest_version} transitioned to Staging.')
    else:
        logger.error(f'No registered versions found for model {model_name}')

def main():
    try:
        transition_to_staging("my_model")
    except Exception as e:
        logger.error('Failed stage transition: %s', e)

if __name__ == '__main__':
    main()