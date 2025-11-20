import React, { useEffect, useState } from "react";
import "./App.css";
import IndiaMap from "./IndiaMap";

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Forecast timeline (0 = current, 1-3 = future months)
  const [forecastMonth, setForecastMonth] = useState(0);

  useEffect(() => {
    setLoading(true);

    // You can later extend backend to accept ?month=
    fetch("http://127.0.0.1:5000/api/predictions")
      .then((res) => res.json())
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching data:", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="container">
      {/* Header */}
      <h1 className="title">AquaAlert</h1>
      <p className="subtitle">
        Early Water Scarcity Forecast • India
      </p>

      {/* Forecast Timeline */}
      <div style={{ marginBottom: "28px" }}>
        <input
          type="range"
          min="0"
          max="3"
          step="1"
          value={forecastMonth}
          onChange={(e) => setForecastMonth(Number(e.target.value))}
          style={{ width: "280px" }}
        />
        <div style={{ marginTop: "8px", color: "#94a3b8" }}>
          {forecastMonth === 0
            ? "Current Conditions"
            : `+${forecastMonth} Month Forecast`}
        </div>
      </div>

      {/* Map Card */}
      <div className="map-container">
        {loading ? (
          <p style={{ color: "#94a3b8" }}>Loading predictions…</p>
        ) : (
          <IndiaMap data={data} forecastMonth={forecastMonth} />
        )}

        {/* Legend */}
        <div className="legend">
          <span><i className="low" /> Low</span>
          <span><i className="mid" /> Moderate</span>
          <span><i className="high" /> High</span>
        </div>
      </div>
    </div>
  );
}

export default App;
