import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, segmentChipClass } from '../api'

const defaultProfile = {
  recency_days: 3,
  total_spend: 120,
  cart_count: 2,
  total_sessions: 6,
  total_events: 24,
}

const SEGMENTS = [
  { value: 'all', label: 'All segments' },
  { value: 'champions', label: 'Champions' },
  { value: 'explorers', label: 'Active Explorers' },
  { value: 'window', label: 'Window Shoppers' },
  { value: 'dormant', label: 'Dormant' },
]

export default function Customer() {
  const [mode, setMode] = useState('browse')
  const [filters, setFilters] = useState({
    segment: 'all',
    min_likelihood: 0,
    max_recency_days: '',
    min_spend: 0,
    has_cart: '',
    sort: 'likelihood',
  })
  const [list, setList] = useState([])
  const [total, setTotal] = useState(0)
  const [selectedId, setSelectedId] = useState(null)
  const [profile, setProfile] = useState(defaultProfile)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loadingList, setLoadingList] = useState(false)
  const [loadingScore, setLoadingScore] = useState(false)

  const loadBrowse = useCallback(async () => {
    setLoadingList(true)
    setError('')
    try {
      const params = {
        segment: filters.segment,
        min_likelihood: Number(filters.min_likelihood) || 0,
        min_spend: Number(filters.min_spend) || 0,
        sort: filters.sort,
        limit: 40,
        offset: 0,
      }
      if (filters.max_recency_days !== '') {
        params.max_recency_days = Number(filters.max_recency_days)
      }
      if (filters.has_cart === 'yes') params.has_cart = true
      if (filters.has_cart === 'no') params.has_cart = false
      const data = await api.browseCustomers(params)
      setList(data.customers || [])
      setTotal(data.total || 0)
    } catch (e) {
      setError(e.message)
      setList([])
    } finally {
      setLoadingList(false)
    }
  }, [filters])

  useEffect(() => {
    if (mode === 'browse') loadBrowse()
  }, [mode, loadBrowse])

  async function selectCustomer(id) {
    setSelectedId(id)
    setError('')
    setLoadingScore(true)
    setResult(null)
    try {
      setResult(await api.getCustomer(id))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingScore(false)
    }
  }

  async function scoreProfile() {
    setError('')
    setLoadingScore(true)
    setResult(null)
    setSelectedId(null)
    try {
      const body = Object.fromEntries(Object.entries(profile).map(([k, v]) => [k, Number(v)]))
      setResult(await api.scoreProfile(body))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingScore(false)
    }
  }

  return (
    <div className="page-enter">
      <p className="eyebrow">Customer lookup</p>
      <h1 className="page-title">Score one shopper</h1>
      <p className="page-lead">
        Filter the customer list and click a row—or describe behavior with a few knobs. No IDs to
        memorize.
      </p>

      <div className="tabs">
        <button type="button" className={mode === 'browse' ? 'active' : ''} onClick={() => setMode('browse')}>
          Browse customers
        </button>
        <button
          type="button"
          className={mode === 'profile' ? 'active' : ''}
          onClick={() => setMode('profile')}
        >
          Build a profile
        </button>
      </div>

      {mode === 'browse' && (
        <div className="panel" style={{ marginBottom: '1rem' }}>
          <h2 className="panel-title">Filters</h2>
          <div className="grid-4">
            <div className="field">
              <label htmlFor="seg">Segment</label>
              <select
                id="seg"
                value={filters.segment}
                onChange={(e) => setFilters({ ...filters, segment: e.target.value })}
              >
                {SEGMENTS.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="minL">Min. purchase likelihood (%)</label>
              <input
                id="minL"
                type="number"
                min="0"
                max="100"
                value={filters.min_likelihood}
                onChange={(e) => setFilters({ ...filters, min_likelihood: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor="rec">Active within (days)</label>
              <input
                id="rec"
                type="number"
                min="0"
                placeholder="Any"
                value={filters.max_recency_days}
                onChange={(e) => setFilters({ ...filters, max_recency_days: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor="spend">Min. spend ($)</label>
              <input
                id="spend"
                type="number"
                min="0"
                value={filters.min_spend}
                onChange={(e) => setFilters({ ...filters, min_spend: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor="cart">Cart activity</label>
              <select
                id="cart"
                value={filters.has_cart}
                onChange={(e) => setFilters({ ...filters, has_cart: e.target.value })}
              >
                <option value="">Any</option>
                <option value="yes">Has items in cart</option>
                <option value="no">Empty cart</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="sort">Sort by</label>
              <select
                id="sort"
                value={filters.sort}
                onChange={(e) => setFilters({ ...filters, sort: e.target.value })}
              >
                <option value="likelihood">Highest likelihood</option>
                <option value="spend">Highest spend</option>
                <option value="recency">Most recent visit</option>
              </select>
            </div>
          </div>
          <p className="muted">{loadingList ? 'Loading…' : `${total.toLocaleString()} customers match`}</p>
        </div>
      )}

      <div className="grid-2">
        <div className="panel">
          {mode === 'browse' ? (
            <>
              <h2 className="panel-title">Customers</h2>
              <p className="muted" style={{ marginBottom: '0.75rem' }}>
                Click a row to see likelihood, segment, and next action.
              </p>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Shopper</th>
                      <th>Segment</th>
                      <th>Likelihood</th>
                      <th>Spend</th>
                      <th>Recency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((c) => (
                      <tr
                        key={c.customer_id}
                        className={selectedId === c.customer_id ? 'row-selected' : 'row-clickable'}
                        onClick={() => selectCustomer(c.customer_id)}
                      >
                        <td>{c.display_name}</td>
                        <td>
                          <span className={segmentChipClass(c.segment)}>{c.segment}</span>
                        </td>
                        <td>{c.purchase_likelihood}%</td>
                        <td>${c.total_spend.toLocaleString()}</td>
                        <td>{c.recency_days}d</td>
                      </tr>
                    ))}
                    {!loadingList && list.length === 0 && (
                      <tr>
                        <td colSpan={5} className="muted">
                          No customers match these filters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <>
              <h2 className="panel-title">Behavior profile</h2>
              {[
                ['recency_days', 'Days since last visit'],
                ['total_spend', 'Historical spend ($)'],
                ['cart_count', 'Items added to cart'],
                ['total_sessions', 'Sessions'],
                ['total_events', 'Total events'],
              ].map(([key, label]) => (
                <div className="field" key={key}>
                  <label htmlFor={key}>{label}</label>
                  <input
                    id={key}
                    type="number"
                    min="0"
                    value={profile[key]}
                    onChange={(e) => setProfile({ ...profile, [key]: e.target.value })}
                  />
                </div>
              ))}
              <button className="btn btn-primary" type="button" disabled={loadingScore} onClick={scoreProfile}>
                {loadingScore ? 'Scoring…' : 'Score profile'}
              </button>
            </>
          )}
        </div>

        <div className="panel">
          <h2 className="panel-title">Result</h2>
          {error && <div className="error">{error}</div>}
          {loadingScore && <p className="muted">Scoring…</p>}
          {!result && !error && !loadingScore && (
            <p className="muted">
              {mode === 'browse' ? 'Select a customer from the list.' : 'Score a profile to see results.'}
            </p>
          )}
          {result && (
            <div className="result-block">
              <div className="metric-row">
                <div className="metric">
                  <div className="label">Purchase likelihood</div>
                  <div className="value">{result.purchase_likelihood}%</div>
                  <div className="sub">{result.intent_band}</div>
                </div>
                <div className="metric">
                  <div className="label">Segment</div>
                  <div className="value" style={{ fontSize: '1.15rem' }}>
                    <span className={segmentChipClass(result.segment)}>{result.segment}</span>
                  </div>
                  <div className="sub">{result.display_name || 'Custom profile'}</div>
                </div>
              </div>
              {result.why?.length > 0 && (
                <ul className="why-list">
                  {result.why.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              )}
              <div className="action-box">
                <strong>Recommended action</strong>
                {result.recommended_action}
              </div>
              <p className="disclaimer">
                How the model works lives in <Link to="/methodology">Methodology</Link>.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
