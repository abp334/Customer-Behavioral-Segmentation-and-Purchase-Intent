import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, downloadBlob } from '../api'

const GOALS = [
  { value: 'maximize_profit', label: 'Maximize overall return' },
  { value: 'win_back_lapsed', label: 'Win back lapsed customers' },
  { value: 'convert_cart_abandoners', label: 'Convert cart abandoners' },
  { value: 'reward_loyal_champions', label: 'Reward loyal champions' },
]

export default function Campaign() {
  const [goal, setGoal] = useState('maximize_profit')
  const [budget, setBudget] = useState(5000)
  const [cpa, setCpa] = useState(1)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const body = {
    goal,
    total_budget: Number(budget),
    cost_per_contact: Number(cpa),
  }

  async function run() {
    setError('')
    setLoading(true)
    try {
      const data = await api.campaign(body)
      setResult(data)
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  async function exportPdf() {
    const blob = await api.campaignPdf(body)
    downloadBlob(blob, 'nexus_campaign_brief.pdf')
  }

  async function exportCsv() {
    const blob = await api.campaignCsv(body)
    downloadBlob(blob, 'nexus_campaign_targets.csv')
  }

  return (
    <div className="page-enter">
      <p className="eyebrow">Campaign plan</p>
      <h1 className="page-title">Who should we contact?</h1>
      <p className="page-lead">
        Set a goal, budget, and cost per contact. NEXUS ranks eligible customers by predicted
        expected value and fills your contact capacity.
      </p>

      <div className="grid-2">
        <div className="panel">
          <h2 className="panel-title">Inputs</h2>
          <div className="field">
            <label htmlFor="goal">Campaign goal</label>
            <select id="goal" value={goal} onChange={(e) => setGoal(e.target.value)}>
              {GOALS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="budget">Budget ($)</label>
            <input id="budget" type="number" min="1" value={budget} onChange={(e) => setBudget(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="cpa">Cost per contact ($)</label>
            <input id="cpa" type="number" min="0.01" step="0.01" value={cpa} onChange={(e) => setCpa(e.target.value)} />
          </div>
          <button className="btn btn-primary" type="button" disabled={loading} onClick={run}>
            {loading ? 'Planning…' : 'Build target list'}
          </button>
        </div>

        <div className="panel">
          <h2 className="panel-title">Predicted outcomes</h2>
          {error && <div className="error">{error}</div>}
          {!result && !error && <p className="muted">Run a plan to see capacity and expected return.</p>}
          {result && (
            <div className="result-block">
              <p className="muted" style={{ marginBottom: '0.85rem' }}>
                {result.story}
              </p>
              <div className="metric-row">
                <div className="metric">
                  <div className="label">Contacts funded</div>
                  <div className="value">{result.target_count.toLocaleString()}</div>
                  <div className="sub">${result.actual_spent.toLocaleString()} spent</div>
                </div>
                <div className="metric">
                  <div className="label">Predicted buyers</div>
                  <div className="value">{result.predicted_buyers}</div>
                  <div className="sub">{result.predicted_conversion_pct}% of targets</div>
                </div>
                <div className="metric">
                  <div className="label">Predicted revenue</div>
                  <div className="value">${result.predicted_revenue.toLocaleString()}</div>
                  <div className="sub">{result.predicted_roi_pct}% predicted ROI</div>
                </div>
              </div>
              <div className="cta-row" style={{ marginTop: '1rem' }}>
                <button className="btn btn-secondary" type="button" onClick={exportCsv}>
                  Download CSV
                </button>
                <button className="btn btn-secondary" type="button" onClick={exportPdf}>
                  Download PDF brief
                </button>
              </div>
              <p className="disclaimer">{result.disclaimer}</p>
              <p className="disclaimer">
                Segment budget study (Lagrange) is under{' '}
                <Link to="/methodology">Methodology → Optimization</Link>.
              </p>
            </div>
          )}
        </div>
      </div>

      {result?.targets?.length > 0 && (
        <div className="panel" style={{ marginTop: '1rem' }}>
          <h2 className="panel-title">Top targets (preview)</h2>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Customer</th>
                  <th>Segment</th>
                  <th>Likelihood</th>
                  <th>Expected value</th>
                  <th>Spend</th>
                  <th>Recency</th>
                </tr>
              </thead>
              <tbody>
                {result.targets.slice(0, 15).map((row) => (
                  <tr key={row['Customer ID']}>
                    <td>{row['Customer ID']}</td>
                    <td>{row['Segment Persona']}</td>
                    <td>{(Number(row['Purchase Probability']) * 100).toFixed(1)}%</td>
                    <td>${Number(row['Expected Value ($)']).toFixed(2)}</td>
                    <td>${Number(row['Historical Spend ($)']).toFixed(2)}</td>
                    <td>{row['Days Inactive']}</td>
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
