import unittest
from fastapi.testclient import TestClient
from app.main import app, load_artifacts

class TestEVBatteryAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_artifacts()
        cls.client = TestClient(app)

    def test_health_check(self):
        """GET / should return 200 with status ok."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_list_batteries(self):
        """GET /api/batteries should return list containing standard NASA battery IDs."""
        response = self.client.get("/api/batteries")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertIn("B0005", data)

    def test_get_metrics(self):
        """GET /api/metrics should return latest recorded RMSE and MAE."""
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rmse", data)
        self.assertIn("mae", data)
        self.assertIn("timestamp", data)

    def test_predict_existing_battery(self):
        """GET /api/predict/B0005 should return actual and predicted degradation trajectories."""
        response = self.client.get("/api/predict/B0005")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["battery_id"], "B0005")
        self.assertGreater(len(data["actual"]), 0)
        self.assertEqual(len(data["actual"]), len(data["predicted"]))
        self.assertIn("rmse", data)
        self.assertIn("mae", data)

    def test_predict_unknown_battery_returns_404(self):
        """GET /api/predict/B9999 should return 404 Not Found."""
        response = self.client.get("/api/predict/B9999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json().get("detail", "").lower())

if __name__ == "__main__":
    unittest.main()
