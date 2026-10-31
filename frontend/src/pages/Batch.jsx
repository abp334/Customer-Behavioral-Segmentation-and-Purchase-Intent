import { useState } from 'react'
import { api, downloadBlob, segmentChipClass } from '../api'

export default function Batch() {
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function loadDemo() {
    setError('')
    setLoading(true)
    try {
      setResult(await api.batchDemo())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function onFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    setLoading(true)
    try {
      setResult(await api.batchUpload(file))
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
      e.target.value = ''
    }
  }

  function exportCsv() {
    if (!result?.rows?.length) return
    const headers = [
      'customer_id',
      'purchase_likelihood',
      'segment',
      'intent_rank',
      'total_spend',
      'cart_count',
      'recency_days',
    ]
    const lines = [headers.join(',')]
    result.rows.forEach((r) => {
      lines.push(headers.map((h) => r[h]).join(','))
    })
    downloadBlob(new Blob([lines.join('\n')], { type: 'text/csv' }), 'nexus_batch_scored.csv')
  }

  return (
    <div className="page-enter">
      <p className="eyebrow">Batch score</p>
      <h1 className="page-title">Score a customer list</h1>
      <p className="page-lead">
        Upload a simple CSV (customer ID, recency, spend, cart, sessions, events) or load the demo
        batch. Download a ranked list ready for targeting.
      </p>

      <div className="panel">
        <h2 className="panel-title">Upload</h2>
        <div className="cta-row">
          <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
            {loading ? 'Scoring…' : 'Upload CSV'}
            <input type="file" accept=".csv" hidden disabled={loading} onChange={onFile} />
          </label>
          <button className="btn btn-secondary" type="button" disabled={loading} onClick={loadDemo}>
            Load demo batch
          </button>
          <a className="btn btn-ghost" href={api.batchTemplateUrl}>
            Download template
          </a>
        </div>
        <p className="disclaimer">
          Simple template columns are mapped on the server. An advanced 17-feature template is also
          accepted if your team already has engineered features.
        </p>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result-block" style={{ marginTop: '1rem' }}>
          <div className="metric-row">
            <div className="metric">
              <div className="label">Customers scored</div>
              <div className="value">{result.row_count}</div>
            </div>
            <div className="metric">
              <div className="label">Predicted buyers</div>
              <div className="value">{result.expected_buyers}</div>
            </div>
            <div className="metric">
              <div className="label">Avg likelihood</div>
              <div className="value">{result.mean_likelihood}%</div>
            </div>
            <div className="metric">
              <div className="label">≥10% intent</div>
              <div className="value">{result.high_intent_count}</div>
            </div>
          </div>
          <div className="cta-row" style={{ margin: '1rem 0' }}>
            <button className="btn btn-secondary" type="button" onClick={exportCsv}>
              Download scored CSV
            </button>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Customer</th>
                  <th>Likelihood</th>
                  <th>Segment</th>
                  <th>Spend</th>
                  <th>Cart</th>
                  <th>Recency</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map((r) => (
                  <tr key={`${r.customer_id}-${r.intent_rank}`}>
                    <td>{r.intent_rank}</td>
                    <td>{r.customer_id}</td>
                    <td>{r.purchase_likelihood}%</td>
                    <td>
                      <span className={segmentChipClass(r.segment)}>{r.segment}</span>
                    </td>
                    <td>${r.total_spend}</td>
                    <td>{r.cart_count}</td>
                    <td>{r.recency_days}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
