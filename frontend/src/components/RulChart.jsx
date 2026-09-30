import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';

export default function RulChart({ actual, predicted, batteryId }) {
  if (!actual || actual.length === 0) {
    return <div className="no-data">No trajectory data available.</div>;
  }

  const chartData = actual.map((actVal, idx) => ({
    cycle: idx + 1,
    Actual: actVal,
    Predicted: predicted[idx] !== undefined ? predicted[idx] : null
  }));

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>Degradation Trajectory &amp; Remaining Useful Life ({batteryId})</h3>
      </div>
      <div style={{ width: '100%', height: 380 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 15, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
            <XAxis
              dataKey="cycle"
              label={{ value: 'Discharge Cycle Window Index', position: 'insideBottom', offset: -10 }}
              stroke="#64748B"
            />
            <YAxis
              label={{ value: 'RUL (Cycles)', angle: -90, position: 'insideLeft', offset: 0 }}
              stroke="#64748B"
            />
            <Tooltip
              formatter={(val) => [`${Number(val).toFixed(2)} cycles`]}
              labelFormatter={(label) => `Cycle Index: ${label}`}
            />
            <Legend verticalAlign="top" height={36} />
            <Line
              type="monotone"
              dataKey="Actual"
              stroke="#2563EB"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="Predicted"
              stroke="#DC2626"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
