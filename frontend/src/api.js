const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      detail = body.detail || detail
      if (Array.isArray(detail)) detail = detail.map((d) => d.msg || d).join('; ')
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  const type = res.headers.get('content-type') || ''
  if (type.includes('application/json')) return res.json()
  return res.blob()
}

function toQuery(params = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null || v === '') return
    q.set(k, String(v))
  })
  const s = q.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => request('/api/health'),
  browseCustomers: (params) => request(`/api/customers/browse${toQuery(params)}`),
  getCustomer: (id) => request(`/api/customers/${id}`),
  scoreProfile: (body) =>
    request('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  batchDemo: () => request('/api/batch/demo'),
  batchUpload: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return request('/api/batch', { method: 'POST', body: fd })
  },
  batchTemplateUrl: '/api/batch/template',
  campaign: (body) =>
    request('/api/campaign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  campaignPdf: async (body) => {
    const res = await fetch('/api/campaign/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error('PDF generation failed')
    return res.blob()
  },
  campaignCsv: async (body) => {
    const res = await fetch('/api/campaign/csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error('CSV export failed')
    return res.blob()
  },
  personas: () => request('/api/personas'),
  segmentMap: () => request('/api/segments/map?limit=1500'),
  forecast: () => request('/api/forecast'),
  methodology: {
    overview: () => request('/api/methodology/overview'),
    models: () => request('/api/methodology/models'),
    calibration: () => request('/api/methodology/calibration'),
    segments: () => request('/api/methodology/segments'),
    optimization: () => request('/api/methodology/optimization'),
    forecast: () => request('/api/methodology/forecast'),
    pipeline: () => request('/api/methodology/pipeline'),
  },
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function segmentChipClass(name = '') {
  const n = name.toLowerCase()
  if (n.includes('champion')) return 'chip chip-champions'
  if (n.includes('explor')) return 'chip chip-explorers'
  if (n.includes('window') || n.includes('shopper')) return 'chip chip-shoppers'
  if (n.includes('dormant')) return 'chip chip-dormant'
  return 'chip'
}
