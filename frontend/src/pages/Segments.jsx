import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, segmentChipClass } from '../api'

const COLORS = {
  0: '#8b2e2e',
  1: '#8a5a00',
  2: '#1f4b3a',
  3: '#2a4a6b',
}

export default function Segments() {
  const [personas, setPersonas] = useState([])
  const [points, setPoints] = useState([])
  const [primary, setPrimary] = useState(2)
  const [compare, setCompare] = useState(3)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([api.personas(), api.segmentMap()])
      .then(([p, m]) => {
        setPersonas(p.personas || [])
        setPoints(m.points || [])
      })
      .catch((e) => setError(e.message))
  }, [])

  const byId = useMemo(() => {
    const map = {}
    personas.forEach((p) => {
      map[p.cluster_id] = p
    })
    return map
  }, [personas])

  const a = byId[primary]
  const b = byId[compare]

  return (
    <div className="page-enter">
      <p className="eyebrow">Segments</p>
      <h1 className="page-title">Who are our customers?</h1>
      <p className="page-lead">
        Four behavioral segments from purchase and browsing patterns. Compare size, conversion, and
        the action each group warrants.
      </p>

      {error && <div className="error">{error}</div>}

      <div className="grid-2" style={{ marginBottom: '1rem' }}>
        <div className="field">
          <label htmlFor="primary">Primary segment</label>
          <select id="primary" value={primary} onChange={(e) => setPrimary(Number(e.target.value))}>
            {personas.map((p) => (
              <option key={p.cluster_id} value={p.cluster_id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="compare">Compare with</label>
          <select id="compare" value={compare} onChange={(e) => setCompare(Number(e.target.value))}>
            {personas.map((p) => (
              <option key={p.cluster_id} value={p.cluster_id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {a && b && (
        <div className="grid-2">
          {[a, b].map((p) => (
            <div className="panel" key={p.cluster_id}>
              <h2 className="panel-title">
                <span className={segmentChipClass(p.name)}>{p.name}</span>
              </h2>
              <div className="metric-row">
                <div className="metric">
                  <div className="label">Customers</div>
                  <div className="value" style={{ fontSize: '1.2rem' }}>
                    {p.customer_count.toLocaleString()}
                  </div>
                  <div className="sub">{p.pct_of_total}% of base</div>
                </div>
                <div className="metric">
                  <div className="label">Conversion</div>
                  <div className="value" style={{ fontSize: '1.2rem' }}>
                    {p.conversion_rate_pct}%
                  </div>
                </div>
                <div className="metric">
                  <div className="label">Avg spend</div>
                  <div className="value" style={{ fontSize: '1.2rem' }}>
                    ${Number(p.mean_spend).toFixed(0)}
                  </div>
                </div>
              </div>
              <div className="action-box">
                <strong>Recommended action</strong>
                {p.recommended_action}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="panel" style={{ marginTop: '1rem' }}>
        <h2 className="panel-title">Behavioral map</h2>
        <p className="muted" style={{ marginBottom: '0.5rem' }}>
          Each point is a customer. Nearby points behave similarly.
        </p>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid stroke="#e5e7eb" />
              <XAxis type="number" dataKey="x" name="Dim 1" tick={{ fontSize: 11 }} />
              <YAxis type="number" dataKey="y" name="Dim 2" tick={{ fontSize: 11 }} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ payload }) => {
                  const d = payload?.[0]?.payload
                  if (!d) return null
                  return (
                    <div className="panel" style={{ padding: '0.5rem 0.7rem' }}>
                      <div>{d.segment}</div>
                      <div className="muted">{d.purchase_likelihood}% likelihood</div>
                    </div>
                  )
                }}
              />
              <Scatter data={points} fill="#1f4b3a">
                {points.map((p, i) => (
                  <Cell key={i} fill={COLORS[p.cluster_id] || '#6b7280'} fillOpacity={0.55} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
