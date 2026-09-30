# API Contract Specification

This document defines the strict interface contract between the FastAPI inference backend and the React frontend dashboard for the EV Battery Remaining Useful Life (RUL) platform.

---

## Endpoints

### 1. Health Check
* **Method & Path:** `GET /`
* **Description:** Verifies service availability and operational status.
* **Request:** None
* **Response (200 OK):**
```json
{
  "status": "ok"
}
```
* **Error Cases:** None.

---

### 2. List Available Batteries
* **Method & Path:** `GET /api/batteries`
* **Description:** Returns an array of battery identifiers currently present and valid in the dataset directory.
* **Request:** None
* **Response (200 OK):**
```json
[
  "B0005",
  "B0006",
  "B0007",
  "B0018"
]
```
* **Error Cases:** None (returns empty list `[]` if no dataset files are found).

---

### 3. Battery RUL Prediction
* **Method & Path:** `GET /api/predict/{battery_id}`
* **Description:** Runs inference using the trained CNN-LSTM model on the specified battery cycles and returns actual vs predicted RUL sequences along with performance metrics.
* **Path Parameters:**
  * `battery_id` (string, required): e.g. `"B0005"`
* **Response (200 OK):**
```json
{
  "battery_id": "B0005",
  "actual": [124.0, 123.0, 122.0, 121.0],
  "predicted": [122.4, 121.8, 120.9, 119.5],
  "rmse": 6.42,
  "mae": 4.85
}
```
* **Error Cases:**
  * `404 Not Found`:
    ```json
    {
      "detail": "Battery ID 'B9999' not found. Available batteries: ['B0005', 'B0006', 'B0007', 'B0018']"
    }
    ```
  * `500 Internal Server Error`:
    ```json
    {
      "detail": "Model artifacts not loaded. Ensure models/cnn_lstm.pth exists."
    }
    ```

---

### 4. Latest Global Metrics
* **Method & Path:** `GET /api/metrics`
* **Description:** Returns the latest recorded evaluation metrics generated during training/evaluation from `results/metrics.json`.
* **Request:** None
* **Response (200 OK):**
```json
{
  "timestamp": "2026-10-01T04:30:00Z",
  "git_commit": "168f464",
  "rmse": 6.42,
  "mae": 4.85
}
```
* **Error Cases:**
  * `404 Not Found`:
    ```json
    {
      "detail": "Evaluation metrics not found. Run training/evaluation pipeline first."
    }
    ```

---

## Data Types & Constraints
* All RUL cycle predictions are positive floating point numbers representing estimated remaining discharge cycles.
* Output arrays `actual` and `predicted` must have identical lengths corresponding to the evaluated sliding windows.
