import { useState } from 'react'
import './App.css'

const AIRLINES = [
  { code: 'AA', name: 'American Airlines' },
  { code: 'AS', name: 'Alaska Airlines' },
  { code: 'B6', name: 'JetBlue Airways' },
  { code: 'DL', name: 'Delta Air Lines' },
  { code: 'F9', name: 'Frontier Airlines' },
  { code: 'G4', name: 'Allegiant Air' },
  { code: 'HA', name: 'Hawaiian Airlines' },
  { code: 'MQ', name: 'Envoy Air' },
  { code: 'NK', name: 'Spirit Airlines' },
  { code: 'OH', name: 'PSA Airlines' },
  { code: 'OO', name: 'SkyWest Airlines' },
  { code: 'UA', name: 'United Airlines' },
  { code: 'VX', name: 'Virgin America' },
  { code: 'WN', name: 'Southwest Airlines' },
  { code: 'YV', name: 'Mesa Airlines' },
  { code: 'YX', name: 'Republic Airways' },
]

const AIRPORTS = [
  { code: 'ATL', name: 'Atlanta', lat: 33.64, lon: -84.43 },
  { code: 'LAX', name: 'Los Angeles', lat: 33.94, lon: -118.40 },
  { code: 'ORD', name: "Chicago O'Hare", lat: 41.97, lon: -87.90 },
  { code: 'DFW', name: 'Dallas/Fort Worth', lat: 32.89, lon: -97.04 },
  { code: 'DEN', name: 'Denver', lat: 39.85, lon: -104.67 },
  { code: 'JFK', name: 'New York (JFK)', lat: 40.63, lon: -73.77 },
  { code: 'SFO', name: 'San Francisco', lat: 37.62, lon: -122.38 },
  { code: 'SEA', name: 'Seattle', lat: 47.44, lon: -122.31 },
  { code: 'LAS', name: 'Las Vegas', lat: 36.08, lon: -115.15 },
  { code: 'MCO', name: 'Orlando', lat: 28.43, lon: -81.31 },
  { code: 'EWR', name: 'Newark', lat: 40.69, lon: -74.17 },
  { code: 'CLT', name: 'Charlotte', lat: 35.21, lon: -80.94 },
  { code: 'PHX', name: 'Phoenix', lat: 33.43, lon: -112.01 },
  { code: 'MIA', name: 'Miami', lat: 25.79, lon: -80.29 },
  { code: 'IAH', name: 'Houston', lat: 29.98, lon: -95.33 },
  { code: 'BOS', name: 'Boston', lat: 42.36, lon: -71.01 },
  { code: 'MSP', name: 'Minneapolis', lat: 44.88, lon: -93.22 },
  { code: 'FLL', name: 'Fort Lauderdale', lat: 26.07, lon: -80.15 },
  { code: 'DTW', name: 'Detroit', lat: 42.21, lon: -83.35 },
  { code: 'PHL', name: 'Philadelphia', lat: 39.87, lon: -75.24 },
  { code: 'LGA', name: 'New York (LaGuardia)', lat: 40.77, lon: -73.87 },
  { code: 'BWI', name: 'Baltimore', lat: 39.17, lon: -76.67 },
  { code: 'SLC', name: 'Salt Lake City', lat: 40.78, lon: -111.97 },
  { code: 'SAN', name: 'San Diego', lat: 32.73, lon: -117.19 },
  { code: 'DCA', name: 'Washington D.C.', lat: 38.85, lon: -77.04 },
  { code: 'MDW', name: 'Chicago Midway', lat: 41.78, lon: -87.74 },
  { code: 'HNL', name: 'Honolulu', lat: 21.32, lon: -157.92 },
  { code: 'TPA', name: 'Tampa', lat: 27.97, lon: -82.53 },
  { code: 'PDX', name: 'Portland', lat: 45.58, lon: -122.59 },
  { code: 'STL', name: 'St. Louis', lat: 38.74, lon: -90.37 },
]

function calcDistance(originCode, destCode) {
  const o = AIRPORTS.find(a => a.code === originCode)
  const d = AIRPORTS.find(a => a.code === destCode)
  if (!o || !d) return 500
  const R = 3958.8
  const dLat = (d.lat - o.lat) * Math.PI / 180
  const dLon = (d.lon - o.lon) * Math.PI / 180
  const a = Math.sin(dLat/2)**2 + Math.cos(o.lat*Math.PI/180)*Math.cos(d.lat*Math.PI/180)*Math.sin(dLon/2)**2
  return Math.round(R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)))
}

function estimateDuration(miles) {
  return Math.round(miles / 7.5 + 30)
}

const today = new Date().toISOString().split('T')[0]

export default function App() {
  const [origin, setOrigin] = useState('JFK')
  const [destination, setDestination] = useState('LAX')
  const [airline, setAirline] = useState('AA')
  const [date, setDate] = useState(today)
  const [depTime, setDepTime] = useState('08:00')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const distance = calcDistance(origin, destination)
  const duration = estimateDuration(distance)

  const handleSubmit = async () => {
    if (origin === destination) { setError('Origin and destination cannot be the same.'); return }
    setLoading(true); setError(null); setResult(null)

    const d = new Date(date)
    const [depH] = depTime.split(':').map(Number)
    const arrHour = (depH + Math.floor(duration / 60)) % 24
    const originAirport = AIRPORTS.find(a => a.code === origin)

    const payload = {
      MONTH: d.getMonth() + 1,
      DAY: d.getDate(),
      DAY_OF_WEEK: d.getDay() === 0 ? 7 : d.getDay(),
      SCHEDULED_TIME: duration,
      AIRLINE_CODE: airline,
      ORIGIN_AIRPORT: origin,
      DESTINATION_AIRPORT: destination,
      DISTANCE: distance,
      ORIGIN_LAT: originAirport.lat,
      ORIGIN_LON: originAirport.lon,
      DIVERTED: 0,
      DEP_HOUR: depH,
      ARR_HOUR: arrHour,
    }

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }
  const riskColor = result
    ? result.risk_label === 'High risk' ? '#ff4d4d'
    : result.risk_label === 'Moderate risk' ? '#ffaa00'
    : '#00c97a'
    : '#00c97a'

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="logo">✈ FlightX</span>
          <span className="tagline">AI-Powered Delay Prediction</span>
        </div>
      </header>

      <main className="main">
        <div className="card form-card">
          <h2 className="card-title">Where are you flying?</h2>

          <div className="route-row">
            <div className="field">
              <label className="field-label">Flying from</label>
              <select value={origin} onChange={e => setOrigin(e.target.value)}>
                {AIRPORTS.map(a => <option key={a.code} value={a.code}>{a.name}</option>)}
              </select>
            </div>
            <div className="swap-icon">→</div>
            <div className="field">
              <label className="field-label">Flying to</label>
              <select value={destination} onChange={e => setDestination(e.target.value)}>
                {AIRPORTS.map(a => <option key={a.code} value={a.code}>{a.name}</option>)}
              </select>
            </div>
          </div>

          {origin !== destination && (
            <div className="info-pill">
              ✈ {distance.toLocaleString()} miles &nbsp;·&nbsp; ~{Math.floor(duration/60)}h {duration%60}m estimated flight time
            </div>
          )}

          <div className="field" style={{ marginBottom: '1rem' }}>
            <label className="field-label">Airline</label>
            <select value={airline} onChange={e => setAirline(e.target.value)}>
              {AIRLINES.map(a => <option key={a.code} value={a.code}>{a.name}</option>)}
            </select>
          </div>

          <div className="grid-2">
            <div className="field">
              <label className="field-label">Travel Date</label>
              <input type="date" value={date} onChange={e => setDate(e.target.value)} />
            </div>
            <div className="field">
              <label className="field-label">Departure Time</label>
              <input type="time" value={depTime} onChange={e => setDepTime(e.target.value)} />
            </div>
          </div>

          {error && <div className="error-box">⚠ {error}</div>}

          <button className="predict-btn" onClick={handleSubmit} disabled={loading}>
            {loading ? <span className="spinner" /> : '🔍 Check for Delays'}
          </button>
        </div>

        {result && (
          <div className="card result-card" style={{ '--risk-color': riskColor }}>
            <h2 className="card-title">Prediction Result</h2>

            <div className="route-summary">
              {AIRPORTS.find(a=>a.code===origin)?.name} → {AIRPORTS.find(a=>a.code===destination)?.name}
              &nbsp;·&nbsp; {AIRLINES.find(a=>a.code===airline)?.name}
            </div>

            <div className="verdict" style={{ color: riskColor }}>
              {result.prediction === 'Delayed' ? '⚠ Likely Delayed' : '✓ Likely On-Time'}
            </div>

            <div className="risk-badge" style={{ background: riskColor }}>
              {result.risk_label}
            </div>

            <div className="confidence-bar-wrap">
              <div className="confidence-label">Delay probability: {result.confidence_pct}%</div>
              <div className="confidence-bar">
                <div className="confidence-fill" style={{ width: `${result.confidence_pct}%`, background: riskColor }} />
              </div>
            </div>

            <div className="model-section">
              <div className="model-label">Model breakdown</div>
              <div className="prob-grid">
                <ProbCard label="LSTM Model" value={result.lstm_probability} />
                <ProbCard label="Transformer" value={result.transformer_probability} />
                <ProbCard label="Combined" value={result.ensemble_probability} highlight />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

function ProbCard({ label, value, highlight }) {
  return (
    <div className={`prob-card ${highlight ? 'highlight' : ''}`}>
      <div className="prob-label">{label}</div>
      <div className="prob-value">{(value * 100).toFixed(1)}%</div>
    </div>
  )
}
