# ==========================================================
# Flask Web Application Integration Tests Suite
# ==========================================================

import os
import sys
import unittest

# ==========================================================
# Add Project Root to Python Path
# ==========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ==========================================================
# Import Flask Application
# ==========================================================

from flask_app.app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configure Flask application for testing.
        """
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()

    # ==========================================================
    # 1. Home Page Tests
    # ==========================================================

    def test_home_page_loads_successfully(self):
        """
        Verify that the home page loads successfully and
        contains the expected HTML title.
        """
        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200,
            "Home page failed to return HTTP 200."
        )

        self.assertIn(
            b"<title>Sentiment Analysis AI</title>",
            response.data,
            "Title element missing from rendered HTML."
        )

    # ==========================================================
    # 2. Prediction Endpoint Tests
    # ==========================================================

    def test_predict_positive_sentiment(self):
        """
        Verify that a positive sentence returns
        a Positive prediction.
        """
        response = self.client.post(
            "/predict",
            data={
                "text": (
                    "I absolutely love this product! "
                    "It works so well and brings me huge joy."
                )
            },
            follow_redirects=True,
        )

        self.assertEqual(
            response.status_code,
            200,
            "Prediction endpoint failed with non-200 status."
        )

        self.assertIn(
            b"Positive",
            response.data,
            "Expected 'Positive' prediction result in response HTML."
        )

    def test_predict_negative_sentiment(self):
        """
        Verify that a negative sentence returns
        a Negative prediction.
        """
        response = self.client.post(
            "/predict",
            data={
                "text": (
                    "This is terrible and the worst experience ever. "
                    "Completely broken and bad."
                )
            },
            follow_redirects=True,
        )

        self.assertEqual(
            response.status_code,
            200,
            "Prediction endpoint failed with non-200 status."
        )

        self.assertIn(
            b"Negative",
            response.data,
            "Expected 'Negative' prediction result in response HTML."
        )

    def test_predict_empty_input_handling(self):
        """
        Verify that submitting blank input displays
        a validation message.
        """
        response = self.client.post(
            "/predict",
            data={"text": "   "},
            follow_redirects=True,
        )

        self.assertEqual(
            response.status_code,
            200,
            "Prediction endpoint failed with non-200 status."
        )

        self.assertIn(
            b"Please enter text.",
            response.data,
            "Expected validation warning message on empty input."
        )


# ==========================================================
# Test Runner
# ==========================================================

if __name__ == "__main__":
    unittest.main()