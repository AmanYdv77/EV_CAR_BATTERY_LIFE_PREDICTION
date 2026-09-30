export default function BatterySelector({ batteries, selectedBattery, onSelect, disabled }) {
  return (
    <div className="selector-container">
      <label htmlFor="battery-select">Select Battery Cell:</label>
      <select
        id="battery-select"
        value={selectedBattery}
        onChange={(e) => onSelect(e.target.value)}
        disabled={disabled}
      >
        {batteries.map((b) => (
          <option key={b} value={b}>Cell {b}</option>
        ))}
      </select>
    </div>
  );
}
