import { useEffect, useState } from 'react'
import { api } from '../api'

const TABS = [
  { id: 'models', label: 'Models', load: () => api.methodology.models() },
  { id: 'calibration', label: 'Calibration', load: () => api.methodology.calibration() },
  { id: 'segments', label: 'Segments', load: () => api.methodology.segments() },
  { id: 'optimization', label: 'Optimization', load: () => api.methodology.optimization() },
  { id: 'forecast', label: 'Forecast', load: () => api.methodology.forecast() },
  { id: 'pipeline', label: 'Pipeline', load: () => api.methodology.pipeline() },
]

function Notes({ notes }) {
  if (!notes?.length) return null
  return (
    <ul className="why-list">
      {notes.map((n) => (
        <li key={n}>{n}</li>
      ))}
    </ul>
  )
}

function Table({ rows }) {
  if (!rows?.length) return <p className="muted">No rows.</p>
  const cols = Object.keys(rows[0])
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 40).map((row, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>{String(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function allocationRows(allocations) {
  return Object.values(allocations || {}).map((a) => ({
    persona: a.persona_name,
    budget: a.budget_allocated,
    share_pct: a.budget_share_pct,
    expected_revenue: a.expected_segment_revenue,
    marginal: a.marginal_return_per_dollar,
  }))
}

export default function Methodology() {
  const [overview, setOverview] = useState(null)
  const [tab, setTab] = useState('models')
  const [payload, setPayload] = useState(null)
  const [payloadFor, setPayloadFor] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.methodology.overview().then(setOverview).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    const meta = TABS.find((t) => t.id === tab)
    if (!meta) return
    let cancelled = false
    setLoading(true)
    setPayload(null)
    setPayloadFor(null)
    setError('')
    meta
      .load()
      .then((data) => {
        if (cancelled) return
        setPayload(data)
        setPayloadFor(tab)
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [tab])

  const ready = !loading && payload && payloadFor === tab

  return (
    <div className="page-enter">
      <p className="eyebrow">For instructors</p>
      <h1 className="page-title">Methodology</h1>
      <p className="page-lead">
        Technical evidence for the models, statistics, optimization study, and data pipeline. The
        product pages hide this detail so operators can act without reading the syllabus.
      </p>

      {overview && (
        <div className="panel" style={{ marginBottom: '1rem' }}>
          <div className="metric-row">
            <div className="metric">
              <div className="label">Events</div>
              <div className="value" style={{ fontSize: '1.15rem' }}>
                {overview.dataset.events.toLocaleString()}
              </div>
            </div>
            <div className="metric">
              <div className="label">Feature mart</div>
              <div className="value" style={{ fontSize: '1.15rem' }}>
                {overview.dataset.feature_mart_customers.toLocaleString()}
              </div>
            </div>
            <div className="metric">
              <div className="label">App warehouse</div>
              <div className="value" style={{ fontSize: '1.15rem' }}>
                {overview.dataset.app_warehouse_sample.toLocaleString()}
              </div>
            </div>
            <div className="metric">
              <div className="label">Period</div>
              <div className="value" style={{ fontSize: '1.15rem' }}>
                {overview.dataset.period}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={tab === t.id ? 'active' : ''}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <p className="muted">Loading…</p>}

      {ready && tab === 'models' && (
        <div className="panel">
          <h2 className="panel-title">Intent model competition</h2>
          <Notes notes={payload.notes} />
          <Table rows={payload.comparison} />
          <h3 style={{ marginTop: '1rem', fontSize: '0.95rem' }}>Champion coefficients (excerpt)</h3>
          <pre className="method-pre">
            {JSON.stringify(
              {
                intercept: payload.coefficients?.intercept,
                feature_names: payload.coefficients?.feature_names?.slice(0, 8),
              },
              null,
              2,
            )}
          </pre>
        </div>
      )}

      {ready && tab === 'calibration' && (
        <div className="panel">
          <h2 className="panel-title">Calibration & threshold</h2>
          <Notes notes={payload.notes} />
          <h3 style={{ fontSize: '0.9rem' }}>Optimal threshold analysis</h3>
          <Table rows={payload.threshold_analysis} />
          <h3 style={{ marginTop: '1rem', fontSize: '0.9rem' }}>Optimization convergence (sample)</h3>
          <Table rows={(payload.convergence || []).slice(0, 15)} />
        </div>
      )}

      {ready && tab === 'segments' && (
        <div className="panel">
          <h2 className="panel-title">Segmentation statistics</h2>
          <Notes notes={payload.notes} />
          <h3 style={{ fontSize: '0.9rem' }}>Clustering metrics</h3>
          <Table rows={payload.clustering_metrics} />
          <h3 style={{ marginTop: '1rem', fontSize: '0.9rem' }}>Hypothesis tests</h3>
          <Table rows={payload.hypothesis_tests} />
        </div>
      )}

      {ready && tab === 'optimization' && (
        <div className="panel">
          <h2 className="panel-title">Lagrange budget study</h2>
          <Notes notes={payload.notes} />
          <div className="metric-row" style={{ margin: '0.75rem 0' }}>
            <div className="metric">
              <div className="label">Budget</div>
              <div className="value">${Number(payload.budget ?? 0).toLocaleString()}</div>
            </div>
            <div className="metric">
              <div className="label">Shadow price (lambda*)</div>
              <div className="value">{payload.lambda_star}</div>
            </div>
          </div>
          <Table rows={allocationRows(payload.allocations)} />
          {payload.report_table?.length > 0 && (
            <>
              <h3 style={{ marginTop: '1rem', fontSize: '0.9rem' }}>Benchmark table</h3>
              <Table rows={payload.report_table} />
            </>
          )}
        </div>
      )}

      {ready && tab === 'forecast' && (
        <div className="panel">
          <h2 className="panel-title">Traffic forecast method</h2>
          <Notes notes={payload.notes} />
          <div className="metric-row" style={{ margin: '0.75rem 0' }}>
            <div className="metric">
              <div className="label">Model</div>
              <div className="value" style={{ fontSize: '1rem' }}>
                {payload.model_order}
              </div>
            </div>
            <div className="metric">
              <div className="label">Holdout MAPE</div>
              <div className="value">{payload.holdout_mape_pct}%</div>
            </div>
            <div className="metric">
              <div className="label">RMSE</div>
              <div className="value">{payload.holdout_rmse}</div>
            </div>
          </div>
          <Table rows={payload.stationarity} />
        </div>
      )}

      {ready && tab === 'pipeline' && (
        <div className="panel">
          <h2 className="panel-title">Medallion pipeline</h2>
          <Notes notes={payload.notes} />
          <div className="grid-3" style={{ margin: '1rem 0' }}>
            {(payload.medallion || []).map((layer) => (
              <div className="metric" key={layer.layer}>
                <div className="label">{layer.layer}</div>
                <div className="sub" style={{ marginTop: '0.35rem' }}>
                  {layer.role}
                </div>
              </div>
            ))}
          </div>
          <h3 style={{ fontSize: '0.9rem' }}>Quality report (excerpt)</h3>
          <Table rows={payload.quality} />
        </div>
      )}
    </div>
  )
}
