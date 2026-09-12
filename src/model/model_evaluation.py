import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub
import os

# Set up DagsHub tracking
mlflow.set_tracking_uri('https://dagshub.com/niveditaranjan223883/mlopsMiniProject.mlflow')
dagshub.init(repo_owner='niveditaranjan223883', repo_name='mlopsMiniProject', mlflow=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            # Log metrics & params
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            if hasattr(clf, 'get_params'):
                for k, v in clf.get_params().items():
                    mlflow.log_param(k, v)
            
            # Infer signature
            signature = infer_signature(X_test, clf.predict(X_test))
            
            # LOG AND REGISTER IN THE SAME ACTIVE STEP
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                signature=signature,
                registered_model_name="my_model"
            )
            
            # Save info locally for downstream tracking
            os.makedirs('reports', exist_ok=True)
            with open('reports/model_info.json', 'w') as f:
                json.dump({'run_id': run.info.run_id, 'model_path': 'model'}, f, indent=4)
            with open('reports/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)

            logger.debug(f"Model successfully logged and registered in run {run.info.run_id}")
        except Exception as e:
            logger.error('Error during evaluation: %s', e)
            raise

if __name__ == '__main__':
    main()