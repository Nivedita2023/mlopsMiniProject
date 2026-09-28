import os
import json
import pickle
import unittest
import numpy as np
import pandas as pd
import dagshub
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestMLOpsPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Setup MLflow registry connection via DagsHub and load test context.
        Fetches the registered model directly from the MLflow Model Registry.
        """
        cls.repo_owner = "niveditaranjan223883"
        cls.repo_name = "mlopsMiniProject"
        cls.registered_model_name = "my-model"
        cls.model_alias_or_stage = "Version 6"  # Alias as seen in MLflow UI (@staging)

        cls.local_model_path = "./models/model.pkl"
        cls.vectorizer_path = "./models/vectorizer.pkl"
        cls.test_data_path = "./data/processed/test_bow.csv"
        cls.metrics_json_path = "./reports/metrics.json"
        cls.exp_info_json_path = "./reports/experiment_info.json"

        # --------------------------------------------------
        # 1. DagsHub & MLflow Authentication
        # --------------------------------------------------
        # Check both common environment variable names for CI/CD compatibility
        dagshub_token = os.getenv("DAGSHUB_TOKEN") or os.getenv("DAGSHUB_PAT")

        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set MLflow Tracking URI to DagsHub
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{cls.repo_owner}/{cls.repo_name}.mlflow"
        )

        # --------------------------------------------------
        # 2. Fetch Model from MLflow Model Registry
        # --------------------------------------------------
        cls.model = None
        try:
            # Try loading via MLflow Model Registry using model alias
            model_uri = f"models:/{cls.registered_model_name}@{cls.model_alias_or_stage}"
            print(f"Loading model from MLflow Registry: {model_uri}")
            cls.model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"Registry load by alias failed ({e}). Trying latest registered version...")
            try:
                client = mlflow.MlflowClient()
                versions = client.search_model_versions(f"name='{cls.registered_model_name}'")
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version)).version
                    model_uri = f"models:/{cls.registered_model_name}/{latest_version}"
                    print(f"Loading latest version from Registry: {model_uri}")
                    cls.model = mlflow.pyfunc.load_model(model_uri)
            except Exception as registry_err:
                print(f"Could not load model from MLflow Registry: {registry_err}")

        # Fallback to local pickle file if MLflow Registry is unavailable/offline
        if cls.model is None and os.path.exists(cls.local_model_path):
            print(f"Fallback: Loading local model from {cls.local_model_path}")
            with open(cls.local_model_path, "rb") as f:
                cls.model = pickle.load(f)

        # --------------------------------------------------
        # 3. Load Vectorizer & Test Data
        # --------------------------------------------------
        if os.path.exists(cls.vectorizer_path):
            with open(cls.vectorizer_path, "rb") as f:
                cls.vectorizer = pickle.load(f)
        else:
            cls.vectorizer = None

        if os.path.exists(cls.test_data_path):
            cls.test_data = pd.read_csv(cls.test_data_path)
        else:
            cls.test_data = None

    # ==========================================================
    # 1. Pipeline Output & File Existence Tests
    # ==========================================================

    def test_pipeline_artifact_existence(self):
        """Verify that essential local pipeline artifacts exist."""
        self.assertTrue(
            os.path.exists(self.vectorizer_path),
            f"Vectorizer artifact missing at {self.vectorizer_path}"
        )
        self.assertTrue(
            os.path.exists(self.test_data_path),
            f"Processed test dataset missing at {self.test_data_path}"
        )

    def test_report_metrics_existence(self):
        """Verify evaluation metrics and experiment info reports exist."""
        self.assertTrue(
            os.path.exists(self.metrics_json_path),
            f"Metrics report missing at {self.metrics_json_path}"
        )
        self.assertTrue(
            os.path.exists(self.exp_info_json_path),
            f"Experiment info report missing at {self.exp_info_json_path}"
        )

    # ==========================================================
    # 2. Model Integrity & Input Signature Tests
    # ==========================================================

    def test_model_loaded_properly(self):
        """Verify that the model was successfully loaded from MLflow or local fallback."""
        self.assertIsNotNone(
            self.model, "Failed to load model object from MLflow Registry or local path."
        )

    def test_model_feature_dimension_alignment(self):
        """
        Verify that the feature vector length generated by the vectorizer matches
        the exact number of input features expected by the trained model.
        """
        self.assertIsNotNone(self.model, "Model not loaded.")
        self.assertIsNotNone(self.vectorizer, "Vectorizer not loaded.")

        dummy_text = "This is a great product with awesome performance"
        transformed_data = self.vectorizer.transform([dummy_text])

        # Get expected feature count from sklearn model directly or via PyFunc wrapper
        if hasattr(self.model, "n_features_in_"):
            expected_features = self.model.n_features_in_
        elif hasattr(self.model, "_model_impl") and hasattr(self.model._model_impl, "n_features_in_"):
            expected_features = self.model._model_impl.n_features_in_
        else:
            # Fallback check against test_bow.csv feature count
            expected_features = self.test_data.shape[1] - 1

        self.assertEqual(
            transformed_data.shape[1],
            expected_features,
            f"Dimension mismatch! Vectorizer outputs {transformed_data.shape[1]} features, "
            f"but model expects {expected_features} features."
        )

    # ==========================================================
    # 3. Sanity / End-to-End Prediction Tests
    # ==========================================================

    def test_prediction_sanity_check(self):
        """Smoke test ensuring end-to-end inference from raw text to class label."""
        self.assertIsNotNone(self.model, "Model not loaded.")
        self.assertIsNotNone(self.vectorizer, "Vectorizer not loaded.")

        text_sample = ["I am very happy today with this result"]
        transformed_sample = self.vectorizer.transform(text_sample)

        # Predict using MLflow PyFunc or native model interface
        prediction = self.model.predict(transformed_sample.toarray())

        self.assertEqual(len(prediction), 1, "Prediction output batch size should be 1.")
        self.assertIn(
            int(prediction[0]),
            [0, 1],
            f"Unexpected class label: {prediction[0]}. Must be 0 or 1."
        )

    # ==========================================================
    # 4. Performance Threshold Tests
    # ==========================================================

    def test_model_performance(self):
        """Evaluate model performance on holdout data against quality standards."""
        self.assertIsNotNone(self.model, "Model not loaded.")
        self.assertIsNotNone(self.test_data, "Test data not loaded.")

        X_test = self.test_data.iloc[:, :-1].values
        y_test = self.test_data.iloc[:, -1].values

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Baseline thresholds
        min_threshold = 0.70

        self.assertGreaterEqual(
            acc, min_threshold, f"Accuracy below threshold: {acc:.4f} < {min_threshold}"
        )
        self.assertGreaterEqual(
            prec, min_threshold, f"Precision below threshold: {prec:.4f} < {min_threshold}"
        )
        self.assertGreaterEqual(
            rec, min_threshold, f"Recall below threshold: {rec:.4f} < {min_threshold}"
        )
        self.assertGreaterEqual(
            f1, min_threshold, f"F1 score below threshold: {f1:.4f} < {min_threshold}"
        )


if __name__ == "__main__":
    unittest.main()