export default function MetricsCard({ metrics, batteryMetrics }) {
  return (
    <div className="metrics-grid">
      <div className="metric-box">
        <span className="metric-title">Model Global RMSE</span>
        <span className="metric-value">
          {metrics && typeof metrics.rmse === 'number' ? `${metrics.rmse.toFixed(2)} cycles` : 'N/A'}
        </span>
      </div>
      <div className="metric-box">
        <span className="metric-title">Model Global MAE</span>
        <span className="metric-value">
          {metrics && typeof metrics.mae === 'number' ? `${metrics.mae.toFixed(2)} cycles` : 'N/A'}
        </span>
      </div>
      {batteryMetrics && (
        <>
          <div className="metric-box highlight">
            <span className="metric-title">Selected Cell RMSE</span>
            <span className="metric-value">{batteryMetrics.rmse.toFixed(2)} cycles</span>
          </div>
          <div className="metric-box highlight">
            <span className="metric-title">Selected Cell MAE</span>
            <span className="metric-value">{batteryMetrics.mae.toFixed(2)} cycles</span>
          </div>
        </>
      )}
    </div>
  );
}
