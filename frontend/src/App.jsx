import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API = 'http://localhost:8000'

export default function App() {
  const [modelTrained, setModelTrained] = useState(false)
  const [indexed, setIndexed]           = useState(false)
  const [stats, setStats]               = useState(null)
  const [trainMetrics, setTrainMetrics] = useState(null)
  const [sources, setSources]           = useState([])

  const [training, setTraining]         = useState(false)
  const [trainMsg, setTrainMsg]         = useState('')
  const [trainErr, setTrainErr]         = useState('')

  const [predicting, setPredicting]     = useState(false)
  const [predictResult, setPredictResult] = useState(null)
  const [predictErr, setPredictErr]     = useState('')

  const DISTRICTS      = ['Colombo', 'Kandy', 'Galle', 'Negombo', 'Jaffna', 'Matara', 'Kurunegala', 'Anuradhapura', 'Ratnapura', 'Badulla']
  const PROPERTY_TYPES = ['House', 'Apartment', 'Villa', 'Land & House']

  const [form, setForm] = useState({
    district: 'Colombo', property_type: 'House',
    bedrooms: 3, bathrooms: 2, floors: 1,
    land_perches: 10, floor_area_sqft: 1200,
    age_years: 5, has_garage: 0, has_pool: 0, furnished: 0,
  })

  const [messages, setMessages]         = useState([])
  const [question, setQuestion]         = useState('')
  const [thinking, setThinking]         = useState(false)
  const [resetting, setResetting]       = useState(false)
  const [sidebarOpen, setSidebarOpen]   = useState(true)
  const bottomRef = useRef(null)

  const DEMO_MESSAGES = [
    {
      role: 'user',
      content: 'Why is Colombo so expensive compared to other districts?',
    },
    {
      role: 'assistant',
      content: `Colombo is by far the most expensive district in Sri Lanka for property, averaging LKR 154,388,229 — which is 6.1x more expensive than the most affordable district (Badulla at LKR 25,306,288).\n\nKey reasons why Colombo commands such a premium:\n\n1. High land scarcity — Colombo is a dense urban district with very limited available land, driving land prices up to LKR 800,000 per perch vs LKR 130,000 in Badulla.\n2. CBD and business district proximity — properties near Fort, Kollupitiya, and Bambalapitiya attract premium buyers and investors.\n3. International schools and hospitals — areas near top schools command a strong location premium.\n4. Expat and corporate demand — multinational companies and expatriates drive rental and purchase demand.\n\nFull district price ranking:\nColombo (LKR 154,388,229) > Negombo (LKR 74,667,044) > Galle (LKR 62,672,159) > Kandy (LKR 56,458,331) > Jaffna (LKR 35,773,952) > Matara (LKR 32,385,069) > Ratnapura (LKR 27,996,739) > Kurunegala (LKR 27,317,814) > Anuradhapura (LKR 26,301,668) > Badulla (LKR 25,306,288)`,
      model_used: 'meta-llama/llama-3.3-70b-instruct:free',
      contexts: [
        'District price analysis [raw_houses.csv]: Colombo is the most expensive district at an average of LKR 154,388,229 (1,428 properties), which is 2.0x the overall average of LKR 77,717,913. This is driven by high land scarcity, proximity to the CBD, business districts, and international schools. Badulla is the most affordable at LKR 25,306,288 (138 properties) — 6.1x cheaper than Colombo.',
        'Key price drivers based on ML model [raw_houses.csv]: 1. district/location (combined) — 61.7% of model predictive power. 2. land size (perches) — 12.7%. 3. property type tier — 10.6%. Location alone explains 61.7% of price variation — confirming that location, location, location holds true in the Sri Lankan market.',
      ],
    },
    {
      role: 'user',
      content: 'How does land size affect house price?',
    },
    {
      role: 'assistant',
      content: `Land size (measured in perches) has a significant and progressive impact on house prices in Sri Lanka:\n\n• < 8 perches: LKR 63,924,101 average\n• 8–15 perches: LKR 70,837,183 average (↑11% vs previous tier)\n• 15–30 perches: LKR 78,526,736 average (↑11% vs previous tier)\n• 30+ perches: LKR 109,421,168 average (↑39% vs previous tier)\n\nLand size is the 2nd most important price driver after district, explaining 12.7% of price variation according to the ML model.\n\nWhy land matters so much in Sri Lanka: land is bought and sold separately from construction, and in high-demand districts like Colombo, each additional perch can add LKR 800,000+ to total value. Larger plots also allow future development or extensions, which buyers pay a premium for.`,
      model_used: 'meta-llama/llama-3.3-70b-instruct:free',
      contexts: [
        'How land size affects price [raw_houses.csv]: < 8 perches: LKR 63,924,101 average. 8–15 perches: LKR 70,837,183 average (↑11%). 15–30 perches: LKR 78,526,736 average (↑11%). 30+ perches: LKR 109,421,168 average (↑39%). Land size is one of the most direct price drivers in Sri Lanka.',
        'Dataset Overview [raw_houses.csv]: 5,000 properties. Price range: LKR 6,703,300 to LKR 420,205,591. Average estimated price: LKR 77,717,913. Average bedrooms: 3.',
      ],
    },
  ]

  function loadDemo() {
    setMessages(DEMO_MESSAGES)
    setStats({
      total_properties:    5000,
      avg_estimated_price: 77717913,
      min_estimated_price: 6703300,
      max_estimated_price: 420205591,
      avg_bedrooms:        3,
    })
    setTrainMetrics({ r2: 0.9535, accuracy: 88.6 })
    setSources(['raw_houses.csv'])
    setIndexed(true)
    setModelTrained(true)
  }

  useEffect(() => { fetchStatus() }, [])
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  async function fetchStatus() {
    try {
      const { data } = await axios.get(`${API}/status`)
      setModelTrained(data.model_trained)
      setIndexed(data.indexed)
      setStats(data.stats)
      setTrainMetrics(data.metrics && Object.keys(data.metrics).length > 0 ? data.metrics : null)
      setSources(data.sources)
    } catch {
      // backend not ready
    }
  }

  // ── Step 1: Train ─────────────────────────────────────────────────────────
  async function handleTrain(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setTraining(true)
    setTrainMsg('')
    setTrainErr('')
    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post(`${API}/train`, form)
      setModelTrained(true)
      setIndexed(true)
      setTrainMetrics(data.metrics)
      setSources([data.filename])
      setTrainMsg(`Model trained on ${data.train_rows} properties. Accuracy: ${data.metrics.accuracy?.toFixed(1)}%. ${data.indexed_docs} knowledge docs indexed — RAG is ready.`)
    } catch (err) {
      setTrainErr(err.response?.data?.detail || 'Training failed.')
    } finally {
      setTraining(false)
      e.target.value = ''
    }
  }

  // ── Step 2: Predict single event ─────────────────────────────────────────
  async function handlePredict(e) {
    e.preventDefault()
    if (!modelTrained) return
    setPredicting(true)
    setPredictResult(null)
    setPredictErr('')
    try {
      const { data } = await axios.post(`${API}/predict-single`, form)
      setPredictResult(data)
      setIndexed(true)
      setStats(data.stats)
      setSources(s => s.includes('user_inputs') ? s : [...s, 'user_inputs'])
    } catch (err) {
      setPredictErr(err.response?.data?.detail || 'Prediction failed.')
    } finally {
      setPredicting(false)
    }
  }

  // ── Chat ──────────────────────────────────────────────────────────────────
  async function handleSend(e) {
    e.preventDefault()
    if (!question.trim() || thinking) return

    const q = question.trim()
    setQuestion('')
    const history = messages.map(m => ({ role: m.role, content: m.content }))
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setThinking(true)

    try {
      const { data } = await axios.post(`${API}/chat`, { question: q, history })
      setMessages(prev => [...prev, {
        role:       'assistant',
        content:    data.answer,
        model_used: data.model_used,
        contexts:   data.contexts,
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role:    'assistant',
        content: `Error: ${err.response?.data?.detail || 'Something went wrong.'}`,
      }])
    } finally {
      setThinking(false)
    }
  }

  // ── Reset ─────────────────────────────────────────────────────────────────
  async function handleReset() {
    if (!confirm('Clear all indexed data and chat history?')) return
    setResetting(true)
    try {
      await axios.post(`${API}/reset`)
      setIndexed(false)
      setStats(null)
      setSources([])
      setMessages([])
      setUploadMsg('')
      setUploadErr('')
    } finally {
      setResetting(false)
    }
  }

  return (
    <div className="layout">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <div className="sidebar-logo">
          🏠 HousePrice RAG
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(false)} title="Hide sidebar">✕</button>
        </div>

        {/* Step 1: Train */}
        <div className="sidebar-section">
          <div className="step-badge">Step 1 — Train Model</div>
          <label className="sidebar-label">Upload CSV with actual costs</label>
          <label className={`upload-btn ${training ? 'disabled' : ''}`}>
            {training ? 'Training...' : modelTrained ? '↺ Retrain Model' : '+ Upload Training CSV'}
            <input type="file" accept=".csv" onChange={handleTrain} hidden disabled={training} />
          </label>
          {trainMsg && <p className="msg-success">{trainMsg}</p>}
          {trainErr && <p className="msg-error">{trainErr}</p>}
          <p className="sidebar-hint">Must include: district, property_type, bedrooms, bathrooms, floors, land_perches, floor_area_sqft, age_years, has_garage, has_pool, furnished, <strong>price_lkr</strong></p>
          {modelTrained && <div className="status-badge trained">Model ready</div>}
        </div>

        {/* Step 2: Predict via form */}
        <div className="sidebar-section">
          <div className="step-badge">Step 2 — Predict House Price</div>
          {!modelTrained
            ? <p className="sidebar-hint" style={{color:'#f59e0b'}}>Train a model first.</p>
            : (
            <form className="predict-form" onSubmit={handlePredict}>
              <div className="form-row">
                <label>District</label>
                <select value={form.district} onChange={e => setForm(f => ({...f, district: e.target.value}))}>
                  {DISTRICTS.map(v => <option key={v}>{v}</option>)}
                </select>
              </div>
              <div className="form-row">
                <label>Property Type</label>
                <select value={form.property_type} onChange={e => setForm(f => ({...f, property_type: e.target.value}))}>
                  {PROPERTY_TYPES.map(v => <option key={v}>{v}</option>)}
                </select>
              </div>
              <div className="form-row">
                <label>Bedrooms</label>
                <input type="number" min={1} max={10} value={form.bedrooms}
                  onChange={e => setForm(f => ({...f, bedrooms: +e.target.value}))} />
              </div>
              <div className="form-row">
                <label>Bathrooms</label>
                <input type="number" min={1} max={10} value={form.bathrooms}
                  onChange={e => setForm(f => ({...f, bathrooms: +e.target.value}))} />
              </div>
              <div className="form-row">
                <label>Floors</label>
                <input type="number" min={1} max={5} value={form.floors}
                  onChange={e => setForm(f => ({...f, floors: +e.target.value}))} />
              </div>
              <div className="form-row">
                <label>Land (perches)</label>
                <input type="number" min={1} max={500} step={0.5} value={form.land_perches}
                  onChange={e => setForm(f => ({...f, land_perches: +e.target.value}))} />
              </div>
              <div className="form-row">
                <label>Floor Area (sqft)</label>
                <input type="number" min={200} max={20000} step={50} value={form.floor_area_sqft}
                  onChange={e => setForm(f => ({...f, floor_area_sqft: +e.target.value}))} />
              </div>
              <div className="form-row">
                <label>Age (years)</label>
                <input type="number" min={0} max={100} value={form.age_years}
                  onChange={e => setForm(f => ({...f, age_years: +e.target.value}))} />
              </div>
              <div className="form-row">
                <label>Garage</label>
                <select value={form.has_garage} onChange={e => setForm(f => ({...f, has_garage: +e.target.value}))}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>
              <div className="form-row">
                <label>Pool</label>
                <select value={form.has_pool} onChange={e => setForm(f => ({...f, has_pool: +e.target.value}))}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>
              <div className="form-row">
                <label>Furnished</label>
                <select value={form.furnished} onChange={e => setForm(f => ({...f, furnished: +e.target.value}))}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>
              {predictErr && <p className="msg-error">{predictErr}</p>}
              <button className="predict-btn" type="submit" disabled={predicting}>
                {predicting ? 'Predicting...' : '⚡ Predict Price'}
              </button>
            </form>
          )}
          {predictResult && (
            <div className="predict-result">
              <div className="predict-label">Estimated Price</div>
              <div className="predict-cost">LKR {predictResult.estimated_price.toLocaleString()}</div>
              <div className="predict-hint">Added to RAG — you can now ask questions about it below.</div>
            </div>
          )}
        </div>

        {sources.length > 0 && (
          <div className="sidebar-section">
            <label className="sidebar-label">Indexed Files</label>
            {sources.map(s => (
              <div key={s} className="source-tag">{s}</div>
            ))}
          </div>
        )}

        {indexed && (
          <div className="sidebar-section" style={{ marginTop: 'auto' }}>
            <button className="reset-btn" onClick={handleReset} disabled={resetting}>
              {resetting ? 'Clearing...' : 'Clear Indexed Data'}
            </button>
          </div>
        )}
      </aside>

      {/* ── Main ── */}
      <main className="main">
        {!sidebarOpen && (
          <button className="sidebar-show-btn" onClick={() => setSidebarOpen(true)} title="Show sidebar">☰</button>
        )}
        {/* Stats bar */}
        {stats && stats.total_properties > 0 && (
          <div className="stats-bar">
            <StatCard label="Properties"    value={stats.total_properties?.toLocaleString()} />
            <StatCard label="Avg Price"     value={`LKR ${stats.avg_estimated_price?.toLocaleString()}`} />
            <StatCard label="Min Price"     value={`LKR ${stats.min_estimated_price?.toLocaleString()}`} />
            <StatCard label="Max Price"     value={`LKR ${stats.max_estimated_price?.toLocaleString()}`} />
            <StatCard label="Avg Bedrooms"  value={stats.avg_bedrooms?.toLocaleString()} />
            {trainMetrics?.r2      && <StatCard label="Model R²"   value={trainMetrics.r2?.toFixed(3)} />}
            {trainMetrics?.accuracy && <StatCard label="Accuracy"  value={`${trainMetrics.accuracy?.toFixed(1)}%`} />}
          </div>
        )}

        {/* Chat area */}
        <div className="chat-area">
          {messages.length === 0 && (
            <div className="chat-empty">
              <div className="chat-empty-icon">🏠</div>
              <h2>House Price Advisor</h2>
              <p>Train the model, predict your house price, then ask questions like:</p>
              <div className="example-questions">
                <ExampleQ text="Why is Colombo so expensive compared to other districts?" onClick={setQuestion} />
                <ExampleQ text="How does land size affect house price?" onClick={setQuestion} />
                <ExampleQ text="What is the price difference between a villa and an apartment?" onClick={setQuestion} />
                <ExampleQ text="How much does having a pool add to the price?" onClick={setQuestion} />
              </div>
              <button className="demo-btn" onClick={loadDemo}>▶ Load Demo Conversation</button>
            </div>
          )}

          {messages.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}

          {thinking && (
            <div className="message assistant">
              <div className="msg-bubble thinking">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <form className="chat-input-bar" onSubmit={handleSend}>
          <input
            className="chat-input"
            placeholder={indexed ? "Ask about house prices, districts, features..." : "Predict a house price first to start chatting"}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            disabled={!indexed || thinking}
          />
          <button className="send-btn" type="submit" disabled={!indexed || thinking || !question.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  )
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function ExampleQ({ text, onClick }) {
  return (
    <button className="example-q" onClick={() => onClick(text)}>
      {text}
    </button>
  )
}

function Message({ msg }) {
  const [showCtx, setShowCtx] = useState(false)
  return (
    <div className={`message ${msg.role}`}>
      <div className="msg-bubble">
        <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
        {msg.model_used && (
          <div className="msg-meta">Answered by: {msg.model_used}</div>
        )}
        {msg.contexts?.length > 0 && (
          <div className="ctx-section">
            <button className="ctx-toggle" onClick={() => setShowCtx(v => !v)}>
              {showCtx ? 'Hide' : 'Show'} sources ({msg.contexts.length})
            </button>
            {showCtx && (
              <div className="ctx-list">
                {msg.contexts.map((c, i) => (
                  <div key={i} className="ctx-item"><span className="ctx-num">[{i+1}]</span> {c}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
