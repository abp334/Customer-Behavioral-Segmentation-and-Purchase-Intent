import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'

export default function Traffic() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .forecast()
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  const chartRows = useMemo(() => {
    if (!data) return []
    const hist = (data.history || []).map((h) => ({
      date: h.date,
      label: h.day_name.slice(0, 3),
      history_visitors: h.active_users,
      forecast_visitors: null,
    }))
    const fut = (data.forecast || []).map((f) => ({
      date: f.date,
      label: f.day_name.slice(0, 3),
      history_visitors: null,
      forecast_visitors: f.predicted_visitors,
    }))
    return [...hist, ...fut]
  }, [data])

  return (
    <div className="page-enter">
      <p className="eyebrow">Traffic timing</p>
      <h1 className="page-title">When should we push?</h1>
      <p className="page-lead">
        A seven-day visitor outlook so you can align campaigns with higher-traffic windows.
      </p>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <div className="action-box" style={{ marginBottom: '1.25rem' }}>
            <strong>This week</strong>
            {data.headline}
          </div>
          <p className="muted" style={{ marginBottom: '1rem' }}>
            {data.subtitle}
          </p>

          <div className="panel">
            <h2 className="panel-title">Active visitors — recent history & outlook</h2>
            <div className="chart-box">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartRows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#e5e7eb" />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} width={48} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="history_visitors"
                    name="Recent active users"
                    stroke="#3d4450"
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast_visitors"
                    name="Forecast visitors"
                    stroke="#1f4b3a"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    connectNulls={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel" style={{ marginTop: '1rem' }}>
            <h2 className="panel-title">7-day schedule</h2>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Day</th>
                    <th>Predicted visitors</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {data.forecast.map((f) => (
                    <tr key={f.date}>
                      <td>{f.date}</td>
                      <td>{f.day_name}</td>
                      <td>{Math.round(f.predicted_visitors).toLocaleString()}</td>
                      <td>{f.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="disclaimer">
              Forecast accuracy and model details:{' '}
              <Link to="/methodology">Methodology → Forecast</Link>.
            </p>
          </div>
        </>
      )}
    </div>
  )
}
