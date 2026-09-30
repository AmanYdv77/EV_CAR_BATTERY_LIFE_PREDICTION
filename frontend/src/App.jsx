import React, { useState, useEffect } from 'react';
import BatterySelector from './components/BatterySelector';
import RulChart from './components/RulChart';
import MetricsCard from './components/MetricsCard';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export default function App() {
  const [batteries, setBatteries] = useState([]);
  const [selectedBattery, setSelectedBattery] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [globalMetrics, setGlobalMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 1. Fetch available batteries and global metrics on mount
  useEffect(() => {
    async function init() {
      try {
        setError(null);
        const [batRes, metRes] = await Promise.all([
          fetch(`${API_BASE}/api/batteries`),
          fetch(`${API_BASE}/api/metrics`)
        ]);

        if (batRes.ok) {
          const batList = await batRes.json();
          setBatteries(batList);
          if (batList.length > 0) {
            setSelectedBattery(batList[0]);
          }
        }

        if (metRes.ok) {
          const metData = await metRes.json();
          setGlobalMetrics(metData);
        }
      } catch (err) {
        setError(`Failed to connect to backend service: ${err.message}`);
      }
    }
    init();
  }, []);

  // 2. Fetch prediction when selected battery changes
  useEffect(() => {
    if (!selectedBattery) return;

    let isCancelled = false;
    async function fetchPrediction() {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`${API_BASE}/api/predict/${selectedBattery}`);
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error (${res.status})`);
        }
        const data = await res.json();
        if (!isCancelled) {
          setPredictionData(data);
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err.message);
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    }

    fetchPrediction();
    return () => {
      isCancelled = true;
    };
  }, [selectedBattery]);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>EV Battery RUL Predictor</h1>
        <p className="subtitle">
          Deep Learning Degradation Analytics using Hybrid 1D-CNN + Dual-Layer LSTM Architecture
        </p>
      </header>

      <main className="dashboard-content">
        <div className="control-bar">
          <BatterySelector
            batteries={batteries}
            selectedBattery={selectedBattery}
            onSelect={setSelectedBattery}
            disabled={loading}
          />
        </div>

        {error && (
          <div className="error-banner">
            <strong>Error:</strong> {error}
          </div>
        )}

        <MetricsCard
          metrics={globalMetrics}
          batteryMetrics={
            predictionData
              ? { rmse: predictionData.rmse, mae: predictionData.mae }
              : null
          }
        />

        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Running CNN-LSTM temporal inference on {selectedBattery}...</p>
          </div>
        ) : (
          predictionData && (
            <RulChart
              actual={predictionData.actual}
              predicted={predictionData.predicted}
              batteryId={predictionData.battery_id}
            />
          )
        )}
      </main>

      <footer className="app-footer">
        <p>NASA Li-Ion Battery Aging Dataset &bull; PyTorch Neural Network Pipeline &bull; MIT License</p>
      </footer>
    </div>
  );
}
