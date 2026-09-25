const app = document.querySelector('#app')
const apiBase = new URLSearchParams(location.search).get('api') ?? ''
const defaultPrompt = 'Transcribe this recording in Swiss German. Return only the spoken words, without translating them.'
const highGermanPrompt = 'Transcribe this recording into High German. Return only the spoken words, without explanation.'
const dialectComparisonModelIds = ['openai-realtime-2', 'openai-realtime-2-1']
const dialectComparisonClipsPerDialect = 2
const utterancePageSize = 8
const resultPageSize = 50
let sampleSizeTimer = null
let pointerIsDown = false
let renderPending = false
let pollGeneration = 0
let runRequestGeneration = 0

const state = {
  view: 'benchmark',
  models: [],
  items: [],
  runs: [],
  historySummary: null,
  dialectAtlas: null,
  health: null,
  instructionPresets: [],
  instructionName: '',
  savingInstruction: false,
  selectedModelIds: [],
  selectedItemIds: [],
  selectedDialect: 'all',
  selectedDataset: '',
  uploadFile: null,
  uploadName: '',
  uploadingDataset: false,
  referenceMode: 'dialect',
  strategy: 'guided',
  search: '',
  selectedMetrics: ['validation', 'match', 'chrf'],
  chartMetric: 'match',
  summaryMetric: 'match',
  sampleSize: 16,
  sampleRound: 0,
  itemLimit: utterancePageSize,
  parameterOverrides: {},
  prompt: defaultPrompt,
  activeRun: null,
  resultDialect: 'all',
  resultModel: 'all',
  resultSort: 'order',
  resultSearch: '',
  resultStatus: 'all',
  resultPage: 1,
  historySearch: '',
  historyStatus: 'all',
  historyLimit: 10,
  message: 'Loading benchmark data...',
  error: '',
  pollError: '',
  loading: true,
  starting: false,
  runControlAction: null,
  metricInfoOpen: null,
  timer: null,
}

const views = {
  benchmark: 'Benchmark',
  results: 'Results',
  history: 'History',
  dialects: 'Dialects',
}

const taskModes = {
  dialect: {
    label: 'Transcribe dialect',
    description: 'Write what was said in Swiss German and score against the dialect transcript.',
  },
  'standard-german': {
    label: 'Translate to High German',
    description: 'Produce Swiss Standard German and score against the High German parallel text.',
  },
}

const strategyDefinitions = {
  baseline: {
    label: 'Baseline',
    description: 'Your prompt plus the dialect name. Comparable with earlier runs.',
  },
  guided: {
    label: 'Dialect-guided',
    description: 'Adds dialect features and four same-dialect example sentences (never the tested sentence).',
  },
  'two-pass': {
    label: 'Two-pass',
    description: 'Transcribes in dialect first, then translates in the same session. High German only.',
    highGermanOnly: true,
  },
  ensemble: {
    label: 'Ensemble',
    description: 'Three independent Realtime passes, reconciled by a text model. Most accurate; about 3× the Realtime calls and slower.',
    requiresRefiner: true,
  },
}

const passLabels = {
  'guided-dialect': 'Swiss German pass',
  'guided-standard': 'High German pass',
  baseline: 'Baseline pass',
}

const metricDefinitions = {
  validation: {
    label: 'Transcript',
    description: 'Show the model transcript with word differences against the reference highlighted.',
  },
  match: {
    label: 'Match',
    description: 'Normalized word match (1 - WER). 100% means the normalized word sequence matches the reference exactly.',
  },
  chrf: {
    label: 'chrF',
    description: 'Character n-gram F-score. Higher is better. More tolerant of inflection, spelling, and paraphrase than WER, which makes it useful for translations.',
  },
  wer: {
    label: 'WER',
    description: 'Word Error Rate. Lower is better; insertions can make it exceed 100%.',
  },
  cer: {
    label: 'CER',
    description: 'Character Error Rate after normalization. Lower is better.',
  },
  ttft: {
    label: 'First token',
    description: 'Realtime streaming time from the response request to the first non-empty text delta of the final answer.',
  },
  latency: {
    label: 'Completion',
    description: 'End-to-end time including decoding, authentication, connection setup, audio upload, and the completed transcript.',
  },
}

const metricProperties = {
  match: 'word_match_rate',
  chrf: 'chrf',
  wer: 'word_error_rate',
  cer: 'character_error_rate',
  ttft: 'time_to_first_token_ms',
  latency: 'latency_ms',
}

const chartMetricDefinitions = {
  match: { title: 'Mean word match by dialect', kind: 'rate', higherIsBetter: true },
  chrf: { title: 'Mean chrF by dialect', kind: 'rate', higherIsBetter: true },
  wer: { title: 'Mean WER by dialect', kind: 'rate', higherIsBetter: false },
  cer: { title: 'Mean CER by dialect', kind: 'rate', higherIsBetter: false },
  ttft: { title: 'Mean time to first token by dialect', kind: 'latency', higherIsBetter: false },
  latency: { title: 'Mean completion time by dialect', kind: 'latency', higherIsBetter: false },
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

function rate(value) {
  return value === null || value === undefined ? '–' : `${(value * 100).toFixed(1)}%`
}

function latency(value) {
  return value === null || value === undefined ? '–' : `${Math.round(value)} ms`
}

function formatMetric(metric, value) {
  return ['ttft', 'latency'].includes(metric) ? latency(value) : rate(value)
}

function compactNumber(value) {
  if (value === null || value === undefined) return '–'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${Math.round(value / 1_000)}k`
  return String(Math.round(value))
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value))
  return finite.length ? finite.reduce((total, value) => total + value, 0) / finite.length : null
}

function scoreTone(value) {
  if (value === null || value === undefined) return 'neutral'
  if (value >= 0.85) return 'positive'
  if (value >= 0.6) return 'warning'
  return 'negative'
}

function isRunInProgress(run = state.activeRun) {
  return Boolean(run && ['queued', 'running', 'paused', 'stopping'].includes(run.status))
}

function pendingRuns() {
  return [state.activeRun, ...state.runs.filter((run) => run.id !== state.activeRun?.id)].filter((run) => isRunInProgress(run))
}

function reportError(error, fallback) {
  state.error = error instanceof Error ? error.message : fallback
  state.message = state.error
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem('swissdial-theme', theme)
  } catch {}
  render()
}

function runProgress(run) {
  const total = run.model_ids.length * run.item_ids.length
  const completed = Number(run.result_count ?? run.results?.length ?? 0)
  return { total, completed, percent: total ? Math.min(100, Math.round((completed / total) * 100)) : 0 }
}

async function api(path, init) {
  const response = await fetch(`${apiBase}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const detail = Array.isArray(body.detail)
      ? body.detail.map((entry) => `${(entry.loc ?? []).filter((part) => part !== 'body').join('.')}: ${entry.msg}`).join('; ')
      : body.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed with ${response.status}`)
  }
  return response.json()
}

function selectedModels() {
  return state.models.filter((model) => state.selectedModelIds.includes(model.id))
}

function modelLabel(modelId) {
  return state.models.find((model) => model.id === modelId)?.label ?? modelId
}

function itemDialect(item) {
  return String(item?.metadata?.dialect ?? item?.metadata?.canton ?? '').toUpperCase()
}

function itemDialectName(item) {
  return item?.metadata?.dialect_name ?? (itemDialect(item) || 'Swiss German')
}

function itemById(itemId) {
  return state.items.find((item) => item.id === itemId)
}

function dialectProfile(code) {
  return state.dialectAtlas?.dialects?.find((dialect) => dialect.code === code) ?? null
}

function dialectShare(code) {
  return dialectProfile(code)?.share_of_german_speakers ?? null
}

function referenceModeLabel(mode = state.referenceMode) {
  return mode === 'standard-german' ? 'High German' : 'Swiss German'
}

function strategyLabel(strategy) {
  return strategyDefinitions[strategy ?? 'baseline']?.label ?? strategy
}

function itemReference(item, mode = state.referenceMode) {
  if (mode === 'standard-german') return item?.metadata?.standard_german_transcript ?? null
  return item?.reference_transcript ?? null
}

function itemCanRun(item) {
  return Boolean(item?.audio_available && (state.referenceMode !== 'standard-german' || itemReference(item)))
}

function syncSampleSizeToSelection() {
  if (state.selectedItemIds.length) state.sampleSize = state.selectedItemIds.length
}

function datasetLabel(item) {
  return item.metadata?.dataset || item.source || 'Other'
}

function datasetOptions() {
  const counts = new Map()
  for (const item of state.items) {
    const label = datasetLabel(item)
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  return [...counts].sort(([left], [right]) => left.localeCompare(right))
}

function sourceFilteredItems() {
  return state.selectedDataset === ''
    ? state.items
    : state.items.filter((item) => datasetLabel(item) === state.selectedDataset)
}

function availableDialects() {
  const dialects = new Map()
  for (const item of sourceFilteredItems()) {
    const code = itemDialect(item)
    if (!code) continue
    const entry = dialects.get(code) ?? { code, name: itemDialectName(item), count: 0 }
    entry.count += 1
    dialects.set(code, entry)
  }
  return [...dialects.values()].sort((left, right) => (dialectShare(right.code) ?? 0) - (dialectShare(left.code) ?? 0) || left.code.localeCompare(right.code))
}

function dialectFilteredItems() {
  return state.selectedDialect === 'all'
    ? sourceFilteredItems()
    : sourceFilteredItems().filter((item) => itemDialect(item) === state.selectedDialect)
}

function filteredItems() {
  const query = state.search.trim().toLowerCase()
  const items = dialectFilteredItems()
  if (!query) return items
  return items.filter((item) => [item.id, item.reference_transcript, item.metadata?.standard_german_transcript, item.metadata?.topic, item.source]
    .some((text) => String(text ?? '').toLowerCase().includes(query)))
}

function sampleCapacity() {
  return dialectFilteredItems().filter(itemCanRun).length
}

function updateSampleSize(value) {
  const requested = Math.floor(Number(value))
  const capacity = sampleCapacity()
  if (!Number.isFinite(requested) || requested < 1 || !capacity) return
  state.sampleSize = Math.min(requested, capacity)
  selectSample()
  if (requested > capacity) {
    state.message = `Sample limited to ${capacity} available utterance${capacity === 1 ? '' : 's'}`
  }
  render()
}

function selectSample(advance = false) {
  if (advance) state.sampleRound += 1
  const groups = new Map()
  for (const item of dialectFilteredItems().filter(itemCanRun)) {
    const dialect = itemDialect(item) || 'other'
    if (!groups.has(dialect)) groups.set(dialect, [])
    groups.get(dialect).push(item)
  }

  const queues = [...groups]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, items]) => {
      const offset = items.length ? state.sampleRound % items.length : 0
      return [...items.slice(offset), ...items.slice(0, offset)]
    })

  const selected = []
  while (selected.length < state.sampleSize && queues.some((queue) => queue.length)) {
    for (const queue of queues) {
      if (queue.length) selected.push(queue.shift().id)
      if (selected.length === state.sampleSize) break
    }
  }
  state.selectedItemIds = selected
}

function dialectComparisonModels() {
  return dialectComparisonModelIds
    .map((modelId) => state.models.find((model) => model.id === modelId))
    .filter(Boolean)
}

function dialectComparisonItems() {
  const groups = new Map()
  for (const item of state.items.filter((candidate) => !candidate.metadata?.dataset && itemCanRun(candidate) && itemDialect(candidate))) {
    const dialect = itemDialect(item)
    if (!groups.has(dialect)) groups.set(dialect, [])
    groups.get(dialect).push(item)
  }
  return [...groups]
    .sort(([left], [right]) => left.localeCompare(right))
    .flatMap(([, items]) => items.slice(0, dialectComparisonClipsPerDialect))
}

// ---------- Word diff ----------

function normalizeToken(token) {
  return token.normalize('NFKC').toLowerCase().replaceAll('ß', 'ss').replace(/[^\p{L}\p{N}]/gu, '')
}

function diffWords(reference, hypothesis) {
  const referenceWords = String(reference ?? '').split(/\s+/).filter(Boolean)
  const hypothesisWords = String(hypothesis ?? '').split(/\s+/).filter(Boolean)
  const left = referenceWords.map(normalizeToken)
  const right = hypothesisWords.map(normalizeToken)
  const table = Array.from({ length: left.length + 1 }, () => new Array(right.length + 1).fill(0))
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      table[i][j] = left[i] === right[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1])
    }
  }
  const operations = []
  let i = 0
  let j = 0
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      operations.push({ type: 'same', text: hypothesisWords[j] })
      i += 1
      j += 1
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      operations.push({ type: 'missing', text: referenceWords[i] })
      i += 1
    } else {
      operations.push({ type: 'extra', text: hypothesisWords[j] })
      j += 1
    }
  }
  while (i < left.length) operations.push({ type: 'missing', text: referenceWords[i++] })
  while (j < right.length) operations.push({ type: 'extra', text: hypothesisWords[j++] })
  return operations
}

function diffMarkup(reference, hypothesis) {
  if (!reference) return escapeHtml(hypothesis)
  return diffWords(reference, hypothesis).map(({ type, text }) => {
    if (type === 'same') return escapeHtml(text)
    if (type === 'extra') return `<mark class="diff-extra" title="Not in reference">${escapeHtml(text)}</mark>`
    return `<del class="diff-missing" title="Missing from transcript">${escapeHtml(text)}</del>`
  }).join(' ')
}

// ---------- Header and navigation ----------

function header() {
  const useLightMode = document.documentElement.dataset.theme === 'dark'
  const running = isRunInProgress()
  const tabs = Object.entries(views).map(([key, label]) => {
    const badge = key === 'results' && running
      ? '<span class="tab-live" aria-label="Run in progress"></span>'
      : key === 'history' && state.runs.length
        ? `<span class="tab-count">${state.runs.length}</span>`
        : ''
    return `<button type="button" class="tab ${state.view === key ? 'active' : ''}" ${state.view === key ? 'aria-current="page"' : ''} data-view="${key}" data-focus-key="nav-${key}">${escapeHtml(label)}${badge}</button>`
  }).join('')
  return `<header class="app-header">
    <div class="brand"><span class="brand-mark" aria-hidden="true"></span><div><span class="product-label">SwissDial benchmark</span><h1>Swiss German speech lab</h1></div></div>
    <nav class="tabs" aria-label="Sections">${tabs}</nav>
    <div class="header-status"><span class="status-dot ${running ? 'busy' : ''}"></span><span class="status-text" title="${escapeHtml(state.message)}">${escapeHtml(state.message)}</span>
      <button type="button" class="icon-button" data-action="toggle-theme" aria-label="${useLightMode ? 'Use light mode' : 'Use dark mode'}" title="${useLightMode ? 'Use light mode' : 'Use dark mode'}">${useLightMode ? '&#9728;' : '&#9790;'}</button>
      <button type="button" class="icon-button" data-action="refresh" aria-label="Refresh benchmark data" title="Refresh benchmark data">&#8635;</button>
    </div>
  </header>`
}

// ---------- Benchmark view ----------

function gettingStarted() {
  const available = state.items.filter(itemCanRun).length
  return `<details class="card getting-started" data-detail-key="getting-started" ${!state.runs.length || !available || !state.models.length ? 'open' : ''}>
    <summary>Getting started <span class="muted">${available} eligible clips · ${state.models.length} model definitions</span></summary>
    <ol class="setup-steps">
      <li><strong>1. Bring your audio</strong><p>Download SwissDial separately or upload a manifest-and-audio ZIP below. Local audio is required; reference text enables scoring. The High German task also requires parallel High German text.</p><a href="https://mtc.ethz.ch/publications/open-source/swiss-dial.html" target="_blank" rel="noopener">Get SwissDial (CC BY-NC 4.0)</a></li>
      <li><strong>2. Connect your deployments</strong><p>Set endpoints and authentication in <code>.env</code>, match deployment names in <code>config/models.json</code>, then restart the API. Listed models are definitions, not verified connections.</p><a href="https://github.com/flo7up/Swiss-german-transcription-test-bench#authentication-and-custom-deployments" target="_blank" rel="noopener">Configuration guide</a></li>
      <li><strong>3. Start small, then compare</strong><p>Choose a task, a few clips, and one deployed model. Inspect errors and transcripts before scaling up. Match and chrF are text similarity scores, not human comprehension ratings.</p><button type="button" class="secondary-button" data-action="small-sample" ${available && !state.loading ? '' : 'disabled'}>Select up to 4 clips</button></li>
    </ol>
    <p class="muted">Browsing, listening, and filtering stay local. Starting a run sends selected audio to your configured cloud deployments and may incur charges. Ensemble uses multiple inference calls per test.</p>
  </details>`
}

function taskPanel() {
  const modes = Object.entries(taskModes).map(([key, mode]) => (
    `<button type="button" class="segment ${state.referenceMode === key ? 'active' : ''}" aria-pressed="${state.referenceMode === key}" data-task-mode="${key}"><strong>${escapeHtml(mode.label)}</strong><small>${escapeHtml(mode.description)}</small></button>`
  )).join('')
  const strategies = Object.entries(strategyDefinitions).map(([key, strategy]) => {
    const needsRefiner = strategy.requiresRefiner && !state.health?.refiner_deployment
    const needsConversation = key === 'two-pass' && selectedTranscriptionModels().length > 0
    const disabled = (strategy.highGermanOnly && state.referenceMode !== 'standard-german') || needsRefiner || needsConversation
    const disabledReason = needsRefiner
      ? 'Set BENCHMARK_REFINER_DEPLOYMENT to a text deployment to enable this strategy'
      : needsConversation
        ? `Speech-to-text models cannot hold a follow-up turn: ${selectedTranscriptionModels().map((model) => model.label).join(', ')}`
        : disabled ? 'Available when translating to High German' : ''
    const badge = key === 'guided'
      ? '<span class="pill accent">Recommended</span>'
      : key === 'ensemble' && !needsRefiner
        ? `<span class="pill accent">Best accuracy · ${escapeHtml(state.health.refiner_deployment)}</span>`
        : ''
    return `<label class="strategy-option ${state.strategy === key ? 'selected' : ''} ${disabled ? 'disabled' : ''}" title="${escapeHtml(disabledReason)}">
      <input type="radio" name="strategy" data-strategy="${key}" ${state.strategy === key ? 'checked' : ''} ${disabled ? 'disabled' : ''}>
      <span><strong>${escapeHtml(strategy.label)} ${badge}</strong><small>${escapeHtml(strategy.description)}</small></span>
    </label>`
  }).join('')
  return `<section class="card task-card">
    <div class="card-heading"><div><span class="eyebrow">Step 1</span><h2>Task and prompting</h2></div></div>
    <div class="segmented" role="group" aria-label="Task">${modes}</div>
    <div class="strategy-options" role="radiogroup" aria-label="Prompt strategy">${strategies}</div>
  </section>`
}

function dialectChips() {
  const dialects = availableDialects()
  const chip = (code, label, count, share) => {
    const active = state.selectedDialect === code
    const shareLabel = share !== null && share !== undefined ? `<span class="chip-share" title="Share of Swiss German speakers in the dialect region">${(share * 100).toFixed(0)}%</span>` : ''
    return `<button type="button" class="chip ${active ? 'active' : ''}" aria-pressed="${active}" data-dialect-chip="${escapeHtml(code)}" title="${escapeHtml(label)}"><strong>${escapeHtml(code === 'all' ? 'All' : code)}</strong>${code === 'all' ? '' : `<span class="chip-name">${escapeHtml(label.replace(' German', ''))}</span>`}${shareLabel}<span class="chip-count">${count}</span></button>`
  }
  return `<div class="chip-row" role="group" aria-label="Dialect filter">${chip('all', 'All dialects', sourceFilteredItems().length, null)}${dialects.map((dialect) => chip(dialect.code, dialect.name, dialect.count, dialectShare(dialect.code))).join('')}</div>`
}

function datasetControls() {
  const sources = datasetOptions()
  const options = sources.map(([name, count]) =>
    `<option value="${escapeHtml(name)}" ${state.selectedDataset === name ? 'selected' : ''}>${escapeHtml(name)} (${count})</option>`
  ).join('')
  return `<div class="dataset-controls">
    <label class="inline-field"><span>Data source</span><select data-dataset-filter aria-label="Data source"><option value="" ${state.selectedDataset === '' ? 'selected' : ''}>All data sources (${state.items.length})</option>${options}</select></label>
    <details class="dataset-upload" ${state.uploadFile || state.uploadingDataset ? 'open' : ''}><summary>Upload your own audio dataset</summary>
      <p class="muted">Choose a ZIP with <code>manifest.jsonl</code> at its root and the referenced audio under <code>clips/</code>. Each non-empty line is a JSON object with a unique <code>id</code>, relative <code>audio_path</code>, and optional <code>reference_transcript</code> for scoring. Add <code>standard_german_transcript</code> to evaluate High German.</p>
      <pre>manifest.jsonl
clips/clip-01.wav
{"id":"clip-01","audio_path":"clips/clip-01.wav","reference_transcript":"grüezi","dialect":"ZH"}</pre>
      <p class="muted">Uploads are stored alongside existing clips and are not sent to a model until you start a run. See the <a href="https://github.com/flo7up/Swiss-german-transcription-test-bench#dataset-manifest" target="_blank" rel="noopener">full format and limits</a>.</p>
      <div class="dataset-upload-fields">
        <label>Source name <input type="text" data-upload-name maxlength="80" placeholder="My recordings" value="${escapeHtml(state.uploadName)}"></label>
        <label>ZIP archive <input type="file" data-dataset-file accept=".zip,application/zip"></label>
        <button type="button" class="secondary-button" data-action="upload-dataset" ${!state.uploadFile || state.uploadingDataset ? 'disabled' : ''}>${state.uploadingDataset ? 'Importing…' : 'Import dataset'}</button>
      </div>
      ${state.uploadFile ? `<small class="muted">Selected: ${escapeHtml(state.uploadFile.name)}</small>` : ''}
    </details>
  </div>`
}

function utteranceToolbar() {
  const capacity = sampleCapacity()
  const visibleRunnable = filteredItems().filter(itemCanRun)
  const presetSizes = [16, 40, 160, 200].filter((size) => size <= capacity)
  if (capacity && !presetSizes.includes(capacity)) presetSizes.push(capacity)
  const presets = presetSizes.map((size) =>
    `<button type="button" class="sample-preset ${state.sampleSize === size ? 'active' : ''}" data-sample-preset="${size}" aria-pressed="${state.sampleSize === size}">${size === capacity ? `All ${size}` : size}</button>`
  ).join('')
  return `<div class="toolbar">
    <label class="search-field"><span class="sr-only">Search utterances</span><input type="search" data-search data-focus-key="search" placeholder="Search text, topic, or ID" value="${escapeHtml(state.search)}"></label>
    <label class="inline-field"><span>Sample size</span><input data-sample-size data-focus-key="sample" type="number" min="1" max="${Math.max(1, capacity)}" step="1" value="${Math.min(state.sampleSize, Math.max(1, capacity))}" ${capacity ? '' : 'disabled'}></label>
    <div class="sample-presets" role="group" aria-label="Quick sample sizes">${presets}</div>
    <button type="button" class="secondary-button" data-action="resample-items" title="Pick a new balanced sample">Resample</button>
    <button type="button" class="ghost-button" data-action="select-visible" ${visibleRunnable.length ? '' : 'disabled'}>Select ${visibleRunnable.length} shown</button>
    <button type="button" class="ghost-button" data-action="clear-items" ${state.selectedItemIds.length ? '' : 'disabled'}>Clear</button>
  </div><p class="sample-hint">${availableDialects().length ? 'Balanced across dialects' : 'Available sample'} · ${capacity} clips in this selection. Each selected model runs on every selected clip; ensembles make multiple model calls per clip.</p>`
}

function utteranceRows() {
  if (!state.items.length) {
    return '<div class="empty-state"><strong>No utterances imported</strong><p>Upload your own audio dataset above, or import SwissDial with <code>scripts/import_swissdial_archive.py</code>.</p></div>'
  }
  const visibleItems = filteredItems()
  if (!visibleItems.length) {
    return '<div class="empty-state"><strong>No matching utterances</strong><p>Try another data source, dialect, or search term.</p></div>'
  }

  const rows = visibleItems.slice(0, state.itemLimit).map((item) => {
    const selected = state.selectedItemIds.includes(item.id)
    const selectable = itemCanRun(item)
    const swissGerman = item.reference_transcript
    const highGerman = item.metadata?.standard_german_transcript
    const primary = state.referenceMode === 'standard-german' ? 'de' : 'ch'
    const audioUrl = `${apiBase}/api/dataset/items/${encodeURIComponent(item.id)}/audio`
    const topic = item.metadata?.topic ? `<span>${escapeHtml(item.metadata.topic)}</span>` : ''
    return `<article class="utterance-row ${selected ? 'selected' : ''} ${selectable ? '' : 'unavailable'}">
      <label class="utterance-select">
        <input type="checkbox" data-item-id="${escapeHtml(item.id)}" ${selected ? 'checked' : ''} ${selectable ? '' : 'disabled'}>
        <span class="dialect-badge" title="${escapeHtml(itemDialectName(item))}">${escapeHtml(itemDialect(item) || '–')}</span>
        <span class="utterance-copy">
          <span class="line ${primary === 'ch' ? 'primary' : ''}"><em>CH</em>${escapeHtml(swissGerman || 'Swiss German reference unavailable')}</span>
          <span class="line ${primary === 'de' ? 'primary' : ''}"><em>DE</em>${escapeHtml(highGerman || 'High German reference unavailable')}</span>
          <small><span>${escapeHtml(itemDialectName(item))}</span><span>${escapeHtml(datasetLabel(item))}</span>${topic}<span>${escapeHtml(item.id)}</span></small>
        </span>
      </label>
      ${item.audio_available ? `<audio controls preload="none" src="${escapeHtml(audioUrl)}">Audio playback is not supported by this browser.</audio>` : '<span class="missing-audio">Audio unavailable</span>'}
    </article>`
  }).join('')

  const remaining = visibleItems.length - state.itemLimit
  const disclosure = remaining > 0
    ? `<div class="list-disclosure"><span>Showing ${state.itemLimit} of ${visibleItems.length}</span><button type="button" data-action="show-more">Show ${Math.min(utterancePageSize, remaining)} more</button><button type="button" data-action="show-all">Show all</button></div>`
    : ''
  return `${rows}${disclosure}`
}

const transportGroups = {
  'azure-openai-realtime': 'Realtime',
  'azure-openai-audio-chat': 'Audio chat',
  'azure-openai-transcription': 'Speech-to-text',
  ensemble: 'Multi-model (Ensemble strategy)',
}

function modelPicker() {
  if (!state.models.length) {
    return '<p class="muted">No runnable transcription model is configured in <code>config/models.json</code>.</p>'
  }
  const groups = new Map()
  for (const model of state.models) {
    const group = transportGroups[model.transport] ?? 'Foundry audio'
    if (!groups.has(group)) groups.set(group, [])
    groups.get(group).push(model)
  }
  return [...groups].map(([group, models]) => `<div class="model-group"><span class="field-label">${escapeHtml(group)}</span><div class="model-options">${models.map((model) => {
    const selected = state.selectedModelIds.includes(model.id)
    return `<label class="model-option ${selected ? 'selected' : ''}" title="${escapeHtml(model.description)}"><input type="checkbox" data-model-id="${escapeHtml(model.id)}" ${selected ? 'checked' : ''}><span><strong>${escapeHtml(model.label)}</strong><small>${escapeHtml(model.deployment)}</small></span></label>`
  }).join('')}</div></div>`).join('')
}

function selectedTranscriptionModels() {
  return selectedModels().filter((model) => model.transport === 'azure-openai-transcription')
}

function isVirtualEnsemble(modelId) {
  return state.models.find((model) => model.id === modelId)?.transport === 'ensemble'
}

function dialectComparisonControl() {
  const models = dialectComparisonModels()
  const items = dialectComparisonItems()
  const dialectCount = new Set(items.map(itemDialect)).size
  const taskCount = models.length * items.length
  const disabled = models.length !== dialectComparisonModelIds.length || dialectCount < 2 || pendingRuns().length || state.starting || state.loading
  return `<button type="button" class="quick-action" data-action="compare-dialects" ${disabled ? 'disabled' : ''}>
    <strong>Compare all dialects</strong><span>${models.length} models · ${dialectCount} dialects · ${taskCount} tests with the current task and strategy</span>
  </button>`
}

function metricPicker() {
  return `<div class="metric-options">${Object.entries(metricDefinitions).map(([key, definition]) => {
    const selected = state.selectedMetrics.includes(key)
    return `<label class="metric-option ${selected ? 'selected' : ''}" title="${escapeHtml(definition.description)}"><input type="checkbox" data-metric-id="${key}" ${selected ? 'checked' : ''}><span>${escapeHtml(definition.label)}</span></label>`
  }).join('')}</div>`
}

function parameterGroups() {
  const groups = selectedModels().map((model) => {
    if (!model.parameters.length) return ''
    const fields = model.parameters.map((parameter) => {
      const value = state.parameterOverrides[model.id]?.[parameter.name] ?? parameter.default
      if (parameter.kind === 'boolean') {
        return `<label class="parameter-field boolean-field"><span>${escapeHtml(parameter.label)}</span><input type="checkbox" data-parameter-model="${escapeHtml(model.id)}" data-parameter-name="${escapeHtml(parameter.name)}" ${value ? 'checked' : ''}></label>`
      }
      const type = parameter.kind === 'number' ? 'number' : 'text'
      return `<label class="parameter-field"><span>${escapeHtml(parameter.label)}</span><input type="${type}" data-parameter-model="${escapeHtml(model.id)}" data-parameter-name="${escapeHtml(parameter.name)}" value="${escapeHtml(value)}" ${parameter.minimum !== undefined && parameter.minimum !== null ? `min="${parameter.minimum}"` : ''} ${parameter.maximum !== undefined && parameter.maximum !== null ? `max="${parameter.maximum}"` : ''} ${parameter.step !== undefined && parameter.step !== null ? `step="${parameter.step}"` : ''}></label>`
    }).join('')
    return `<div class="parameter-group"><strong>${escapeHtml(model.label)}</strong><div class="parameter-fields">${fields}</div></div>`
  }).filter(Boolean)
  return groups.length ? `<div class="field-group"><span class="field-label">Model parameters</span>${groups.join('')}</div>` : ''
}

function instructionControls() {
  const options = state.instructionPresets.map((preset) => (
    `<option value="${escapeHtml(preset.name)}" ${state.instructionName === preset.name ? 'selected' : ''}>${escapeHtml(preset.name)}</option>`
  )).join('')
  return `<div class="instruction-controls">
    <label><span class="field-label">Saved prompt</span><select data-instruction-select><option value="">Default</option>${options}</select></label>
    <div class="save-row"><label><span class="field-label">Save as</span><input data-instruction-name type="text" maxlength="80" value="${escapeHtml(state.instructionName)}" placeholder="Prompt name"></label>
    <button type="button" class="secondary-button" data-action="save-instruction" ${state.savingInstruction ? 'disabled' : ''}>${state.savingInstruction ? 'Saving...' : 'Save'}</button></div>
  </div>`
}

function advancedSettings() {
  const strategyNote = state.strategy === 'baseline'
    ? 'The dialect name is appended to this prompt for each clip.'
    : state.strategy === 'ensemble'
      ? 'The ensemble uses this prompt for its baseline pass and adds two dialect-guided passes plus a fusion step.'
      : 'The selected strategy appends dialect features, example sentences, and output rules to this prompt for each clip.'
  return `<details class="card advanced-settings" data-detail-key="advanced-settings">
    <summary>Prompt and parameters</summary>
    <div class="advanced-content">
      <label class="prompt-field"><span class="field-label">Base prompt</span><textarea data-prompt>${escapeHtml(state.prompt)}</textarea><small class="muted">${escapeHtml(strategyNote)}</small></label>
      ${instructionControls()}
      ${parameterGroups()}
    </div>
  </details>`
}

function benchmarkView() {
  const selectedCount = state.selectedItemIds.length
  return `${gettingStarted()}<div class="benchmark-layout">
    <div class="benchmark-main">
      ${taskPanel()}
      <section class="card" aria-labelledby="utterances-title">
        <div class="card-heading"><div><span class="eyebrow">Step 2</span><h2 id="utterances-title">Utterances</h2></div><div class="selection-count"><strong>${selectedCount}</strong> selected</div></div>
        ${datasetControls()}
        ${dialectChips()}
        ${utteranceToolbar()}
        <div class="utterance-list">${utteranceRows()}</div>
      </section>
    </div>
    <aside class="benchmark-side">
      <section class="card"><div class="card-heading"><div><span class="eyebrow">Step 3</span><h2>Models</h2></div><span class="muted">${state.selectedModelIds.length} selected</span></div>${modelPicker()}${dialectComparisonControl()}</section>
      <section class="card"><div class="card-heading"><div><span class="eyebrow">Display</span><h2>Result fields</h2></div></div>${metricPicker()}</section>
      ${advancedSettings()}
    </aside>
  </div>
  ${runBar()}`
}

function runBar() {
  const taskCount = state.selectedModelIds.length * state.selectedItemIds.length
  const pending = pendingRuns()[0]
  const running = state.starting || Boolean(pending)
  const runDisabled = state.loading || running || !state.selectedModelIds.length || !state.selectedItemIds.length
  const runLabel = state.starting ? 'Starting benchmark…' : pending ? `Run ${pending.status}` : `Run ${taskCount} ${taskCount === 1 ? 'test' : 'tests'}`
  const dialectCount = new Set(state.selectedItemIds.map((itemId) => itemDialect(itemById(itemId))).filter(Boolean)).size
  return `<div class="run-bar">
    <div class="run-bar-summary">
      <span><strong>${state.selectedItemIds.length}</strong> utterances</span>
      ${dialectCount ? `<span><strong>${dialectCount}</strong> dialects</span>` : ''}
      <span><strong>${state.selectedModelIds.length}</strong> models</span>
      <span class="pill">${escapeHtml(taskModes[state.referenceMode].label)}</span>
      <span class="pill">${escapeHtml(strategyLabel(state.strategy))}</span>
    </div>
    ${pending ? `<button type="button" class="secondary-button" data-run-id="${escapeHtml(pending.id)}">View progress</button>` : ''}
    <button type="button" class="primary-button" data-action="start-run" ${runDisabled ? 'disabled' : ''}>${escapeHtml(runLabel)}</button>
  </div>`
}

// ---------- Results view ----------

function selectedResultMetrics() {
  return state.selectedMetrics.filter((metric) => metric !== 'validation')
}

function metricHeader(metric) {
  const definition = metricDefinitions[metric]
  const open = state.metricInfoOpen === metric
  return `<span class="metric-header">${escapeHtml(definition.label)}<button type="button" data-metric-info="${metric}" aria-label="Explain ${escapeHtml(definition.label)}" aria-expanded="${open}">i</button>${open ? `<span role="tooltip">${escapeHtml(definition.description)}</span>` : ''}</span>`
}

function modelScore(run, modelId, property) {
  return mean((run.results ?? []).filter((result) => result.model_id === modelId).map((result) => result[property]))
}

function weightedModelScore(run, modelId, property) {
  const byDialect = new Map()
  for (const result of run.results ?? []) {
    if (result.model_id !== modelId || !Number.isFinite(result[property])) continue
    const dialect = itemDialect(itemById(result.item_id))
    if (!byDialect.has(dialect)) byDialect.set(dialect, [])
    byDialect.get(dialect).push(result[property])
  }
  let total = 0
  let weight = 0
  for (const [dialect, values] of byDialect) {
    const share = dialectShare(dialect)
    if (!share) continue
    total += share * mean(values)
    weight += share
  }
  return weight ? total / weight : null
}

function modelScorecards(run) {
  const metric = state.summaryMetric === 'chrf' ? 'chrf' : 'match'
  const otherMetric = metric === 'match' ? 'chrf' : 'match'
  const title = metric === 'match' ? 'Mean word match' : 'Mean chrF'
  const spokenMetric = metric === 'match' ? 'mean word match' : 'mean chrF'
  const cards = run.model_ids.map((modelId, index) => {
    const results = (run.results ?? []).filter((result) => result.model_id === modelId)
    const failed = results.filter((result) => result.status === 'failed').length
    const score = modelScore(run, modelId, metricProperties[metric])
    const weighted = weightedModelScore(run, modelId, metricProperties[metric])
    const otherScore = modelScore(run, modelId, metricProperties[otherMetric])
    const completion = modelScore(run, modelId, 'latency_ms')
    const width = score === null ? 0 : Math.max(0, Math.min(100, score * 100))
    return `<article class="scorecard">
      <div class="scorecard-main">
        <strong class="scorecard-name"><i class="score-swatch series-${index % 4}"></i>${escapeHtml(modelLabel(modelId))}</strong>
        <span class="score-track ${score === null ? 'pending' : ''}" role="img" aria-label="${escapeHtml(`${modelLabel(modelId)}: ${score === null ? `no ${spokenMetric} score yet` : `${rate(score)} ${spokenMetric}`}`)}"><i class="score-fill series-${index % 4}" style="width:${width}%"></i></span>
        <strong class="score-value tone-${scoreTone(score)}">${rate(score)}</strong>
      </div>
      <dl>
        <div><dt>Results</dt><dd>${results.length}${failed ? ` · <span class="text-danger">${failed} failed</span>` : ''}</dd></div>
        <div><dt title="Mean of per-dialect ${escapeHtml(metricDefinitions[metric].label)} weighted by each dialect region's share of Swiss German speakers">Speaker-weighted ${escapeHtml(metricDefinitions[metric].label)}</dt><dd>${rate(weighted)}</dd></div>
        <div><dt>${escapeHtml(metricDefinitions[otherMetric].label)}</dt><dd>${rate(otherScore)}</dd></div>
        <div><dt>Completion</dt><dd>${latency(completion)}</dd></div>
      </dl>
    </article>`
  }).join('')
  return `<section class="scorecards" aria-label="Model comparison">
    <div class="scorecards-heading"><div><span class="eyebrow">${run.model_ids.length > 1 ? 'Model comparison' : 'Model results'}</span><h3>${title}</h3><p>Scores share a 0–100% scale for direct comparison.</p></div>
      <div class="chart-metric-switch" role="group" aria-label="Model comparison metric">
        ${['match', 'chrf'].map((key) => `<button type="button" data-summary-metric="${key}" aria-pressed="${metric === key}" class="${metric === key ? 'active' : ''}">${escapeHtml(metricDefinitions[key].label)}</button>`).join('')}
      </div>
    </div>
    <div class="score-axis" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
    ${cards}
  </section>`
}

function niceChartMaximum(value) {
  if (value <= 0) return 1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  const normalized = value / magnitude
  const rounded = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10
  return rounded * magnitude
}

function chartValue(value, kind, compact = false) {
  if (kind === 'rate') return `${(value * 100).toFixed(compact ? 0 : 1)}%`
  if (compact && value >= 1000) return `${(value / 1000).toFixed(value % 1000 ? 1 : 0)}s`
  return `${Math.round(value)} ms`
}

function dialectComparisonChart(run) {
  const models = run.model_ids.map((modelId) => ({ id: modelId, label: modelLabel(modelId) }))
  const dialects = [...new Map(run.item_ids.map((itemId) => itemById(itemId)).filter(Boolean).map((item) => [itemDialect(item), itemDialectName(item)]))]
    .filter(([dialect]) => dialect)
    .sort(([left], [right]) => (dialectShare(right) ?? 0) - (dialectShare(left) ?? 0) || left.localeCompare(right))
  const availableMetrics = selectedResultMetrics().filter((metric) => chartMetricDefinitions[metric])
  if (!models.length || !dialects.length || !availableMetrics.length) return ''

  const metric = availableMetrics.includes(state.chartMetric) ? state.chartMetric : availableMetrics[0]
  const definition = chartMetricDefinitions[metric]
  const property = metricProperties[metric]
  const referenceLabel = referenceModeLabel(run.reference_mode)
  const direction = definition.higherIsBetter ? 'Higher is better' : 'Lower is better'
  const chartDescription = definition.kind === 'rate'
    ? `${direction}; scored against the ${referenceLabel} reference. Dialects are ordered by number of speakers.`
    : `${direction}. Dialects are ordered by number of speakers.`
  const series = dialects.map(([dialect, dialectName]) => ({
    dialect,
    dialectName,
    models: models.map((model) => {
      const values = (run.results ?? [])
        .filter((result) => result.model_id === model.id && itemDialect(itemById(result.item_id)) === dialect)
        .map((result) => result[property])
        .filter((value) => Number.isFinite(value))
      return { model, values, average: mean(values) }
    }),
  }))
  const observedValues = series.flatMap((group) => group.models.map((entry) => entry.average).filter((value) => value !== null))
  const scaleMaximum = definition.kind === 'rate'
    ? niceChartMaximum(Math.max(1, ...observedValues))
    : niceChartMaximum(Math.max(1000, ...observedValues))
  const scaleLabels = [0, 0.5, 1]
    .map((position) => `<span>${chartValue(scaleMaximum * position, definition.kind, true)}</span>`)
    .join('')

  const legend = models.map((model, index) => `<span><i class="series-${index % 4}"></i>${escapeHtml(model.label)}</span>`).join('')
  const metricControls = availableMetrics.length > 1
    ? `<div class="chart-metric-switch" role="group" aria-label="Chart metric">${availableMetrics.map((key) => `<button type="button" data-chart-metric="${key}" aria-pressed="${metric === key}" class="${metric === key ? 'active' : ''}">${escapeHtml(metricDefinitions[key].label)}</button>`).join('')}</div>`
    : ''
  const groups = series.map(({ dialect, dialectName, models: modelSeries }) => {
    const bars = modelSeries.map(({ model, values, average }, index) => {
      const width = average === null ? 0 : Math.max(0, Math.min(100, average / scaleMaximum * 100))
      const valueLabel = average === null ? (isRunInProgress(run) ? 'Pending' : 'No score') : chartValue(average, definition.kind)
      const utteranceCount = `${values.length} ${values.length === 1 ? 'utterance' : 'utterances'}`
      const accessibleLabel = `${model.label}, ${dialectName}, ${metricDefinitions[metric].label}: ${valueLabel}${values.length ? ` across ${utteranceCount}` : ''}`
      return `<span class="chart-bar series-${index % 4} ${average === null ? 'pending' : ''}" tabindex="0" role="img" aria-label="${escapeHtml(accessibleLabel)}"><span class="chart-track"><i class="chart-bar-fill" style="--bar-width:${width}%"></i></span><strong class="chart-value">${escapeHtml(valueLabel)}</strong><span class="chart-tooltip">${escapeHtml(model.label)}<strong>${escapeHtml(valueLabel)}</strong><small>${values.length ? utteranceCount : 'Waiting for results'}</small></span></span>`
    }).join('')
    const share = dialectShare(dialect)
    return `<div class="chart-group"><span class="chart-label" title="${escapeHtml(dialectName)}"><strong>${escapeHtml(dialect)}</strong>${share ? `<small>${(share * 100).toFixed(0)}% of speakers</small>` : ''}</span><div class="chart-bars">${bars}</div></div>`
  }).join('')

  return `<section class="dialect-chart" aria-labelledby="dialect-chart-title">
    <div class="dialect-chart-heading"><div><span class="eyebrow">By dialect</span><h3 id="dialect-chart-title">${escapeHtml(definition.title)}</h3><p>${escapeHtml(chartDescription)}</p></div><div class="chart-heading-actions">${metricControls}<div class="chart-legend">${legend}</div></div></div>
    <div class="chart-scroll"><div class="chart-canvas">
      <div class="chart-scale" aria-hidden="true">${scaleLabels}</div>
      <div class="chart-groups">${groups}</div>
    </div></div>
  </section>`
}

function runSetupDetails(run) {
  const parameterGroups = Object.entries(run.parameters ?? {}).flatMap(([modelId, parameters]) => {
    const values = Object.entries(parameters ?? {})
    if (!values.length) return []
    return [`${modelLabel(modelId)}: ${values.map(([name, value]) => `${name}=${value}`).join(', ')}`]
  })
  return `<details class="run-setup" data-detail-key="run-setup-${escapeHtml(run.id)}">
    <summary>Run setup</summary>
    <div><span>Models</span><strong>${escapeHtml(run.model_ids.map(modelLabel).join(', '))}</strong></div>
    <div><span>Strategy</span><strong>${escapeHtml(strategyLabel(run.strategy))}</strong></div>
    <div><span>Base prompt</span><p>${escapeHtml(run.prompt)}</p></div>
    ${parameterGroups.length ? `<div><span>Parameters</span><p>${escapeHtml(parameterGroups.join(' / '))}</p></div>` : ''}
    <div><span>Run ID</span><p>${escapeHtml(run.id)}</p></div>
  </details>`
}

function resultFilters(run) {
  const dialects = [...new Set(run.item_ids.map((itemId) => itemDialect(itemById(itemId))).filter(Boolean))].sort()
  const dialectOptions = dialects.map((code) => `<option value="${escapeHtml(code)}" ${state.resultDialect === code ? 'selected' : ''}>${escapeHtml(code)}</option>`).join('')
  const modelOptions = run.model_ids.map((modelId) => `<option value="${escapeHtml(modelId)}" ${state.resultModel === modelId ? 'selected' : ''}>${escapeHtml(modelLabel(modelId))}</option>`).join('')
  const sorts = { order: 'Run order', worst: 'Lowest match first', best: 'Highest match first', slowest: 'Slowest first' }
  return `<div class="toolbar result-toolbar">
    <label class="search-field"><span class="sr-only">Search results</span><input type="search" data-result-search data-focus-key="result-search" placeholder="Search reference, output, error, or clip ID" value="${escapeHtml(state.resultSearch)}"></label>
    <label class="inline-field"><span>Dialect</span><select data-result-dialect data-focus-key="result-dialect"><option value="all">All</option>${dialectOptions}</select></label>
    ${run.model_ids.length > 1 ? `<label class="inline-field"><span>Model</span><select data-result-model data-focus-key="result-model"><option value="all">All</option>${modelOptions}</select></label>` : ''}
    <label class="inline-field"><span>Sort</span><select data-result-sort data-focus-key="result-sort">${Object.entries(sorts).map(([key, label]) => `<option value="${key}" ${state.resultSort === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
    <label class="inline-field"><span>Outcome</span><select data-result-status data-focus-key="result-status">${Object.entries({ all: 'All results', completed: 'Successful', failed: 'Failed' }).map(([key, label]) => `<option value="${key}" ${state.resultStatus === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
    <button type="button" class="secondary-button" data-action="clear-result-filters">Reset filters</button>
    <span class="legend-inline"><mark class="diff-extra">extra</mark><del class="diff-missing">missing</del> word vs. reference</span>
  </div>`
}

function visibleResults(run) {
  let results = [...(run.results ?? [])]
  if (state.resultDialect !== 'all') results = results.filter((result) => itemDialect(itemById(result.item_id)) === state.resultDialect)
  if (state.resultModel !== 'all') results = results.filter((result) => result.model_id === state.resultModel)
  if (state.resultStatus !== 'all') results = results.filter((result) => result.status === state.resultStatus)
  const query = state.resultSearch.trim().toLocaleLowerCase()
  if (query) results = results.filter((result) => [result.item_id, result.model_label, result.reference_transcript, result.transcript, result.error].some((value) => String(value ?? '').toLocaleLowerCase().includes(query)))
  if (state.resultSort !== 'order') {
    const property = state.resultSort === 'slowest' ? 'latency_ms' : 'word_match_rate'
    const direction = state.resultSort === 'worst' ? 1 : -1
    results.sort((left, right) => {
      const a = left[property]
      const b = right[property]
      if (!Number.isFinite(a)) return Number.isFinite(b) ? 1 : 0
      if (!Number.isFinite(b)) return -1
      return direction * (a - b)
    })
  }
  return results
}

function resetResultFilters() {
  state.resultDialect = 'all'
  state.resultModel = 'all'
  state.resultStatus = 'all'
  state.resultSearch = ''
  state.resultSort = 'order'
  state.resultPage = 1
}

function resultPagination(run) {
  const count = visibleResults(run).length
  const pages = Math.max(1, Math.ceil(count / resultPageSize))
  state.resultPage = Math.min(state.resultPage, pages)
  const start = count ? (state.resultPage - 1) * resultPageSize + 1 : 0
  const end = Math.min(count, state.resultPage * resultPageSize)
  return `<div class="result-pagination">
    <span role="status">${start}–${end} of ${count} matching results · ${(run.results ?? []).length} total</span>
    <div class="button-row"><button type="button" class="secondary-button" data-action="results-previous" ${state.resultPage === 1 ? 'disabled' : ''}>Previous</button><span>Page ${state.resultPage} / ${pages}</span><button type="button" class="secondary-button" data-action="results-next" ${state.resultPage === pages ? 'disabled' : ''}>Next</button></div>
  </div>`
}

function errorMarkup(error) {
  const text = String(error ?? 'Unknown error')
  if (text.length <= 160) return `<span class="result-error">${escapeHtml(text)}</span>`
  return `<details class="result-error"><summary>${escapeHtml(text.slice(0, 140))}…</summary><span>${escapeHtml(text)}</span></details>`
}

function resultRows(run) {
  const results = visibleResults(run)
  const metrics = selectedResultMetrics()
  const showValidation = state.selectedMetrics.includes('validation')
  const columnCount = 2 + metrics.length + (showValidation ? 1 : 0)
  if (!results.length) {
    return `<tr><td colspan="${columnCount}" class="table-empty">${run.results?.length ? 'No results match the current filters.' : 'Results will appear as each utterance is transcribed.'}</td></tr>`
  }

  return results.slice((state.resultPage - 1) * resultPageSize, state.resultPage * resultPageSize).map((result) => {
    const item = itemById(result.item_id)
    const reference = result.reference_transcript ?? item?.reference_transcript ?? 'Reference unavailable'
    const audioUrl = `${apiBase}/api/dataset/items/${encodeURIComponent(result.item_id)}/audio`
    const playerId = `result-audio-${result.id}`
    const dialectPass = (result.conversation ?? []).find((entry) => entry.pass === 'dialect')?.contents?.[0]
    const hypotheses = (result.conversation ?? []).filter((entry) => passLabels[String(entry.pass ?? '').split('#')[0]])
    const fusionModel = (result.conversation ?? []).find((entry) => entry.pass === 'fusion')?.model
    const hypothesisBlock = hypotheses.length
      ? `<details class="hypotheses" data-detail-key="hypotheses-${escapeHtml(result.id)}"><summary>${hypotheses.length} hypotheses${fusionModel ? ` · fused by ${escapeHtml(fusionModel)}` : ''}</summary>${hypotheses.map((entry) => `<span><em>${escapeHtml(passLabels[String(entry.pass).split('#')[0]])}${entry.model && entry.model !== result.model_label ? ` · ${escapeHtml(entry.model)}` : ''}</em>${escapeHtml(entry.contents?.[0] ?? '')}</span>`).join('')}</details>`
      : ''
    const validation = showValidation
      ? `<td class="transcript-cell">${result.status === 'failed'
        ? errorMarkup(result.error)
        : `<span class="transcript">${diffMarkup(result.reference_transcript, result.transcript ?? '')}</span>${dialectPass ? `<span class="dialect-pass"><em>Dialect pass</em>${escapeHtml(dialectPass)}</span>` : ''}${hypothesisBlock}`}</td>`
      : ''
    const metricCells = metrics.map((metric) => {
      const value = result[metricProperties[metric]]
      const tone = ['match', 'chrf'].includes(metric) ? `tone-${scoreTone(value)}` : ''
      return `<td class="metric-cell ${tone}">${formatMetric(metric, value)}</td>`
    }).join('')
    return `<tr class="${result.status === 'failed' ? 'failed-result' : ''}">
      <td class="result-utterance"><div class="utterance-cell"><button type="button" data-action="toggle-result-audio" data-audio-id="${playerId}" aria-label="Play utterance" title="${item?.audio_available ? 'Play utterance' : 'Audio no longer available'}" ${item?.audio_available ? '' : 'disabled'}>&#9654;</button><span><strong>${escapeHtml(reference)}</strong><small><span class="dialect-badge small">${escapeHtml(itemDialect(item) || '–')}</span>${escapeHtml(itemDialectName(item))} · ${escapeHtml(result.item_id)}</small></span><audio id="${playerId}" preload="none" src="${escapeHtml(audioUrl)}"></audio></div></td>
      <td class="model-cell">${escapeHtml(result.model_label)}${result.status === 'failed' && !showValidation ? errorMarkup(result.error) : ''}</td>${validation}${metricCells}
    </tr>`
  }).join('')
}

function resultsView() {
  if (!state.activeRun) {
    return `<section class="card empty-results"><strong>No run selected</strong><p>Start a benchmark or open a previous run from History.</p><div class="button-row"><button type="button" class="primary-button" data-view="benchmark">Set up a run</button>${state.runs.length ? '<button type="button" class="secondary-button" data-view="history">Open history</button>' : ''}</div></section>`
  }
  const run = state.activeRun
  const progress = runProgress(run)
  const metrics = selectedResultMetrics()
  const showValidation = state.selectedMetrics.includes('validation')
  const metricHeaders = metrics.map((metric) => `<th>${metricHeader(metric)}</th>`).join('')
  const validationHeader = showValidation ? '<th>Model output</th>' : ''
  const runReferenceMode = run.reference_mode ?? 'dialect'
  const runDialects = [...new Set(run.item_ids.map((itemId) => itemDialect(itemById(itemId))).filter(Boolean))].sort()
  const dialectSummary = runDialects.length === 1 ? itemDialectName(itemById(run.item_ids[0])) : `${runDialects.length} dialects`
  const controlsDisabled = Boolean(state.runControlAction)
  const controls = run.status === 'paused'
    ? `<button type="button" class="secondary-button" data-action="resume-run" ${controlsDisabled ? 'disabled' : ''}>Resume</button><button type="button" class="danger-button" data-action="stop-run" ${controlsDisabled ? 'disabled' : ''}>Stop</button>`
    : run.status === 'stopping'
      ? '<span class="control-note">Stopping after the current utterance</span>'
      : isRunInProgress(run)
        ? `<button type="button" class="secondary-button" data-action="pause-run" ${controlsDisabled ? 'disabled' : ''}>Pause</button><button type="button" class="danger-button" data-action="stop-run" ${controlsDisabled ? 'disabled' : ''}>Stop</button>`
        : ''
  const progressPanel = isRunInProgress(run)
    ? `<div class="run-progress"><div><span>${progress.completed} of ${progress.total} complete</span><strong>${progress.percent}%</strong></div><div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="${progress.total}" aria-valuenow="${progress.completed}"><span style="width:${progress.percent}%"></span></div><div class="run-controls">${controls}</div></div>`
    : ''
  const resultActions = `<div class="button-row">${!isRunInProgress(run) ? '<button type="button" class="secondary-button" data-action="reuse-run">Use this setup</button>' : ''}<a class="secondary-button" href="${escapeHtml(apiBase)}/api/runs/${encodeURIComponent(run.id)}/export.csv" download title="Exports all saved results, not just filtered rows">Download full CSV</a></div>`
  const partialNote = run.status === 'stopped'
    ? `<p class="muted">This run was stopped after ${progress.completed} of ${progress.total} tests. Scores below cover only completed clips.</p>`
    : ''
  return `<section class="card results-card">
    <div class="card-heading"><div><span class="eyebrow">${escapeHtml(new Date(run.started_at).toLocaleString())}</span><h2>${escapeHtml(taskModes[runReferenceMode].label)} · ${escapeHtml(strategyLabel(run.strategy))}</h2></div>${resultActions}</div>
    <div class="result-summary">
      <div><span>Status</span><strong class="status-value ${escapeHtml(run.status)}">${escapeHtml(run.status)}</strong></div>
      <div><span>Completed</span><strong>${progress.completed} / ${progress.total}</strong></div>
      <div><span>Audio</span><strong>${escapeHtml(dialectSummary || 'Unknown dialect')}</strong></div>
      <div><span>Reference</span><strong>${escapeHtml(referenceModeLabel(runReferenceMode))}</strong></div>
    </div>
    ${partialNote}
    ${progressPanel}
    ${modelScorecards(run)}
    <p class="results-note">Charts summarize the whole run; filters below apply only to the table. Scores exclude unscored results. Compare models on the same clips and check failures before drawing conclusions.</p>
    ${runSetupDetails(run)}
    ${dialectComparisonChart(run)}
    ${resultFilters(run)}
    ${resultPagination(run)}
    <div class="table-wrap" tabindex="0" role="region" aria-label="Benchmark results; scroll horizontally for all columns"><table><thead><tr><th>${escapeHtml(referenceModeLabel(runReferenceMode))} reference</th><th>Model</th>${validationHeader}${metricHeaders}</tr></thead><tbody>${resultRows(run)}</tbody></table></div>
  </section>`
}

// ---------- History view ----------

function historySummaryStrip() {
  const summary = state.historySummary
  if (!summary) return ''
  const indicator = summary.indicator ?? { label: '', tone: 'neutral' }
  return `<div class="summary-strip">
    <div><span>Total runs</span><strong>${summary.total_run_count}</strong></div>
    <div><span>Successful results</span><strong>${summary.successful_result_count} / ${summary.result_count}</strong></div>
    <div><span>Mean match</span><strong>${rate(summary.average_word_match_rate)}</strong></div>
    <div><span>Mean completion</span><strong>${latency(summary.average_latency_ms)}</strong></div>
    <div><span>Overall</span><strong><span class="status-chip ${escapeHtml(indicator.tone)}">${escapeHtml(indicator.label)}</span></strong></div>
  </div>`
}

function filteredRuns() {
  const query = state.historySearch.trim().toLocaleLowerCase()
  return state.runs.filter((run) => {
    const matchesStatus = state.historyStatus === 'all' || (state.historyStatus === 'attention'
      ? isRunInProgress(run)
      : state.historyStatus === 'errors' ? run.failed_result_count > 0 || run.status === 'failed' : run.status === state.historyStatus)
    return matchesStatus && (!query || [run.id, ...run.model_ids.map(modelLabel), referenceModeLabel(run.reference_mode), strategyLabel(run.strategy), new Date(run.started_at).toLocaleString()].some((value) => value.toLocaleLowerCase().includes(query)))
  })
}

function historyView() {
  if (!state.runs.length) {
    return '<section class="card empty-results"><strong>No runs yet</strong><p>Completed and active runs will appear here.</p></section>'
  }
  const runs = filteredRuns()
  const rows = runs.slice(0, state.historyLimit).map((run) => {
    const total = run.total_task_count ?? run.model_ids.length * run.item_ids.length
    const indicator = run.indicator ?? { label: run.status, tone: 'neutral' }
    const active = state.activeRun?.id === run.id
    return `<button class="history-row ${active ? 'active' : ''}" type="button" data-run-id="${escapeHtml(run.id)}">
      <span class="history-primary"><strong>${escapeHtml(run.model_ids.map(modelLabel).join(' vs '))}</strong><small>${new Date(run.started_at).toLocaleString()} · ${run.item_ids.length} utterances · ${escapeHtml(run.status)}${run.failed_result_count ? ` · ${run.failed_result_count} failed` : ''}</small><small>${escapeHtml(referenceModeLabel(run.reference_mode))} · ${escapeHtml(strategyLabel(run.strategy))}</small></span>
      <span class="history-tags"><span class="pill">${escapeHtml(taskModes[run.reference_mode ?? 'dialect'].label)}</span><span class="pill">${escapeHtml(strategyLabel(run.strategy))}</span></span>
      <span class="status-chip ${escapeHtml(indicator.tone)}">${escapeHtml(indicator.label)}</span>
      <span class="history-metric"><small>Match</small><strong class="tone-${scoreTone(run.average_word_match_rate)}">${rate(run.average_word_match_rate)}</strong></span>
      <span class="history-count">${run.result_count ?? 0} / ${total}</span>
      <span aria-hidden="true" class="chevron">&#8250;</span>
    </button>`
  }).join('')
  const more = runs.length > state.historyLimit
    ? `<div class="list-disclosure centered"><button type="button" data-action="history-more">Show more (${runs.length - state.historyLimit} remaining)</button></div>`
    : ''
  return `<section class="card">
    <div class="card-heading"><div><span class="eyebrow">All runs</span><h2>History</h2></div></div>
    ${historySummaryStrip()}
    <div class="toolbar">
      <label class="search-field"><span class="sr-only">Search history</span><input type="search" data-history-search data-focus-key="history-search" placeholder="Search model, task, strategy, date, or run ID" value="${escapeHtml(state.historySearch)}"></label>
      <label class="inline-field"><span>Status</span><select data-history-status data-focus-key="history-status">${Object.entries({ all: 'All runs', attention: 'Active or paused', completed: 'Completed', errors: 'With failures', stopped: 'Stopped' }).map(([key, label]) => `<option value="${key}" ${state.historyStatus === key ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
      <button type="button" class="secondary-button" data-action="clear-history-filters">Reset filters</button>
      <span class="muted" role="status">${runs.length} of ${state.runs.length} runs</span>
    </div>
    ${rows ? `<div class="history-list">${rows}</div>${more}` : '<p class="empty-state">No runs match these filters. Try another search or reset filters.</p>'}
  </section>`
}

// ---------- Dialect atlas view ----------

function latestDialectScores(code) {
  const run = state.activeRun
  if (!run?.results?.length) return []
  return run.model_ids.map((modelId) => {
    const values = run.results
      .filter((result) => result.model_id === modelId && itemDialect(itemById(result.item_id)) === code)
      .map((result) => result.word_match_rate)
    return { modelId, value: mean(values), count: values.filter(Number.isFinite).length }
  }).filter((entry) => entry.count)
}

function uncoveredCantons(atlas) {
  const covered = new Set(atlas.dialects.flatMap((dialect) => dialect.region_cantons))
  return Object.entries(atlas.cantons ?? {})
    .filter(([code]) => !covered.has(code))
    .map(([code, canton]) => ({ code, name: canton.name, speakers: canton.population * canton.german_share }))
    .sort((left, right) => right.speakers - left.speakers)
}

function dialectsView() {
  const atlas = state.dialectAtlas
  if (!atlas?.dialects?.length) {
    return '<section class="card empty-results"><strong>Dialect data unavailable</strong><p>Add <code>config/dialects.json</code> to show dialect demographics.</p></section>'
  }
  const maxShare = Math.max(...atlas.dialects.map((dialect) => dialect.share_of_german_speakers ?? 0))
  const uncovered = uncoveredCantons(atlas)
  const uncoveredSpeakers = atlas.total_german_speakers - atlas.covered_german_speakers
  const bars = atlas.dialects.map((dialect) => {
    const share = dialect.share_of_german_speakers ?? 0
    return `<div class="share-row">
      <span class="dialect-badge">${escapeHtml(dialect.code)}</span>
      <span class="share-label"><strong>${escapeHtml(dialect.name)}</strong><small>${escapeHtml(dialect.region_label)}</small></span>
      <span class="share-track"><span style="width:${(share / maxShare) * 100}%"></span></span>
      <span class="share-value"><strong>${(share * 100).toFixed(1)}%</strong><small>${compactNumber(dialect.region_speakers)}</small></span>
    </div>`
  }).join('')
  const uncoveredRow = `<div class="share-row muted-row">
    <span class="dialect-badge ghost">–</span>
    <span class="share-label"><strong>Not in SwissDial</strong><small title="${escapeHtml(uncovered.filter((canton) => atlas.cantons[canton.code].german_share >= 0.5).map((canton) => canton.name).join(', '))} and German speakers in bilingual and Latin cantons">${escapeHtml(uncovered.filter((canton) => atlas.cantons[canton.code].german_share >= 0.5).map((canton) => canton.code).join(', '))} and German speakers in bilingual and Latin cantons</small></span>
    <span class="share-track"><span style="width:${((1 - atlas.coverage_share) / maxShare) * 100}%"></span></span>
    <span class="share-value"><strong>${((1 - atlas.coverage_share) * 100).toFixed(1)}%</strong><small>${compactNumber(uncoveredSpeakers)}</small></span>
  </div>`

  const cards = atlas.dialects.map((dialect) => {
    const scores = latestDialectScores(dialect.code)
    const itemCount = state.items.filter((item) => itemDialect(item) === dialect.code).length
    const scoreBlock = scores.length
      ? `<div class="dialect-scores"><span class="field-label">Open run · word match</span>${scores.map((entry) => `<span><span>${escapeHtml(modelLabel(entry.modelId))}</span><strong class="tone-${scoreTone(entry.value)}">${rate(entry.value)}</strong></span>`).join('')}</div>`
      : ''
    return `<article class="dialect-card">
      <header><span class="dialect-badge large">${escapeHtml(dialect.code)}</span><div><h3>${escapeHtml(dialect.name)}</h3><small>${escapeHtml(dialect.native_name)} · ${escapeHtml(dialect.group)}</small></div></header>
      <div class="dialect-stats">
        <div><span>Share of speakers</span><strong>${rate(dialect.share_of_german_speakers)}</strong></div>
        <div><span>Region speakers</span><strong>${compactNumber(dialect.region_speakers)}</strong></div>
        <div><span>Of Swiss population</span><strong>${rate(dialect.share_of_population)}</strong></div>
      </div>
      <p>${escapeHtml(dialect.summary)}</p>
      <details><summary>Typical features</summary><ul>${dialect.features.map((feature) => `<li>${escapeHtml(feature)}</li>`).join('')}</ul></details>
      ${scoreBlock}
      <footer><small class="muted">${escapeHtml(dialect.region_label)} · ${itemCount} clips imported</small><button type="button" class="secondary-button" data-benchmark-dialect="${escapeHtml(dialect.code)}" ${itemCount ? '' : 'disabled'}>Benchmark ${escapeHtml(dialect.code)}</button></footer>
    </article>`
  }).join('')

  return `<section class="card atlas-hero">
    <div class="card-heading"><div><span class="eyebrow">Dialect importance</span><h2>Who speaks which Swiss German dialect?</h2><p class="muted">Estimated German-speaking residents per SwissDial dialect region. Use this to prioritise which dialects matter most for real users; results views weight scores by these shares.</p></div></div>
    <div class="summary-strip">
      <div><span>Residents in Switzerland</span><strong>${compactNumber(atlas.total_population)}</strong></div>
      <div><span>German/Swiss German main language</span><strong>${compactNumber(atlas.total_german_speakers)}</strong></div>
      <div><span>Covered by SwissDial dialects</span><strong>${rate(atlas.coverage_share)}</strong></div>
      <div><span>Data years</span><strong>${atlas.population_year} / ${atlas.language_survey_year}</strong></div>
    </div>
    <div class="share-chart">${bars}${uncoveredRow}</div>
  </section>
  <div class="dialect-grid">${cards}</div>
  <section class="card methodology"><h3>Methodology and sources</h3><p>${escapeHtml(atlas.methodology)}</p><ul>${(atlas.sources ?? []).map((source) => `<li><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.label)}</a></li>`).join('')}</ul></section>`
}

// ---------- Rendering ----------

function render() {
  renderPending = false
  const details = new Map([...app.querySelectorAll('details[data-detail-key]')].map((element) => [element.dataset.detailKey, element.open]))
  const tableScroll = app.querySelector('.table-wrap')?.scrollLeft ?? 0
  const focused = document.activeElement?.dataset?.focusKey
  const selection = focused ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : null
  const content = state.view === 'results'
    ? resultsView()
    : state.view === 'history'
      ? historyView()
      : state.view === 'dialects'
        ? dialectsView()
        : benchmarkView()
  const pending = pendingRuns()
  const notice = state.error || state.pollError
  app.innerHTML = `<div class="app-shell">${header()}<main id="main-content" class="view view-${state.view}" tabindex="-1">
    ${notice ? `<div class="notice error-notice" role="alert"><span>${escapeHtml(notice)}</span><button type="button" class="secondary-button" data-action="refresh">Refresh data</button><button type="button" class="icon-button" data-action="dismiss-error" aria-label="Dismiss error">&times;</button></div>` : ''}
    ${pending.length && !isRunInProgress() ? `<div class="notice"><span>${pending.length} active or paused run${pending.length === 1 ? '' : 's'}. Reopen to monitor, resume, or stop.</span><button type="button" class="secondary-button" data-run-id="${escapeHtml(pending[0].id)}">Open run</button></div>` : ''}
    ${content}</main></div>`
  for (const element of app.querySelectorAll('details[data-detail-key]')) {
    if (details.has(element.dataset.detailKey)) element.open = details.get(element.dataset.detailKey)
  }
  const table = app.querySelector('.table-wrap')
  if (table) table.scrollLeft = tableScroll
  if (focused) {
    const element = app.querySelector(`[data-focus-key="${focused}"]`)
    if (element) {
      element.focus()
      try {
        element.setSelectionRange(...selection)
      } catch {}
    }
  }
}

// Background poll updates must not replace the DOM between pointerdown and click, or the click is lost.
function renderFromPoll() {
  const interacting = app.contains(document.activeElement) && document.activeElement?.matches('input, textarea, select, summary')
  const playing = [...app.querySelectorAll('audio')].some((audio) => !audio.paused && !audio.ended)
  if (pointerIsDown || interacting || playing) {
    renderPending = true
    return
  }
  render()
}

function flushPendingRender() {
  pointerIsDown = false
  if (renderPending) setTimeout(() => { if (renderPending) renderFromPoll() }, 0)
}

function writeRoute() {
  const hash = `#${state.view}${state.view === 'results' && state.activeRun ? `/${encodeURIComponent(state.activeRun.id)}` : ''}`
  if (location.hash !== hash) history.pushState(null, '', hash)
}

async function restoreRoute() {
  const [view, runId] = location.hash.slice(1).split('/')
  if (view === 'results' && runId) {
    try {
      await openRun(decodeURIComponent(runId), false)
    } catch (error) {
      reportError(error, 'Invalid run link.')
      render()
    }
    return
  }
  ++runRequestGeneration
  state.view = Object.hasOwn(views, view) ? view : 'benchmark'
  render()
}

function setView(view) {
  ++runRequestGeneration
  state.view = view
  state.metricInfoOpen = null
  writeRoute()
  render()
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

async function refreshWorkspace(fromPoll = false) {
  try {
    const [models, items, runs, historySummary, instructionPresets, dialectAtlas, health] = await Promise.all([
      api('/api/models'),
      api('/api/dataset/items'),
      api('/api/runs'),
      api('/api/runs/summary'),
      api('/api/instructions'),
      api('/api/dialects'),
      api('/api/health'),
    ])
    state.models = models
    state.items = items
    state.runs = runs
    state.historySummary = historySummary
    state.instructionPresets = instructionPresets
    state.dialectAtlas = dialectAtlas
    state.health = health
    if (state.strategy === 'ensemble' && !health?.refiner_deployment) state.strategy = 'guided'
    if (!state.selectedModelIds.length) state.selectedModelIds = models.slice(0, 1).map((model) => model.id)
    if (state.selectedDataset && !datasetOptions().some(([name]) => name === state.selectedDataset)) state.selectedDataset = ''
    if (state.selectedDialect !== 'all' && !availableDialects().some(({ code }) => code === state.selectedDialect)) state.selectedDialect = 'all'
    const availableItemIds = new Set(items.filter((item) => item.audio_available).map((item) => item.id))
    state.selectedItemIds = state.selectedItemIds.filter((itemId) => availableItemIds.has(itemId))
    if (!state.selectedItemIds.length) selectSample()
    for (const model of models) {
      state.parameterOverrides[model.id] ??= Object.fromEntries(model.parameters.map((parameter) => [parameter.name, parameter.default]))
    }
    if (!isRunInProgress() && !state.starting) {
      state.message = !models.length
        ? 'Configure a runnable model to begin.'
        : items.length
          ? `${items.length} utterances ready`
          : 'Import SwissDial utterances to begin.'
    }
  } catch (error) {
    reportError(error, 'Unable to load the benchmark.')
  }
  state.loading = false
  if (fromPoll) renderFromPoll()
  else render()
}

async function uploadDataset() {
  if (!state.uploadFile || state.uploadingDataset) return
  const file = state.uploadFile
  const name = state.uploadName.trim() || file.name.replace(/\.zip$/i, '').trim()
  if (!name || name.length > 80) {
    state.message = 'Enter a source name of up to 80 characters.'
    render()
    return
  }
  state.uploadingDataset = true
  state.message = `Importing ${file.name}…`
  render()
  try {
    const imported = await api(`/api/datasets?name=${encodeURIComponent(name)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file,
    })
    state.uploadFile = null
    state.uploadName = ''
    state.selectedDataset = imported.name
    state.selectedDialect = 'all'
    state.selectedItemIds = []
    state.sampleSize = Math.min(16, imported.item_count)
    state.sampleRound = 0
    state.itemLimit = utterancePageSize
    state.search = ''
    await refreshWorkspace()
    state.message = state.selectedItemIds.length
      ? `Imported ${imported.item_count} clips from ${imported.name}. Select models and run when ready.`
      : `Imported ${imported.item_count} clips from ${imported.name}. Select a task with available references to run them.`
  } catch (error) {
    reportError(error, 'Unable to import the dataset.')
  } finally {
    state.uploadingDataset = false
    render()
  }
}

async function openRun(runId, updateRoute = true) {
  const request = ++runRequestGeneration
  try {
    const run = await api(`/api/runs/${encodeURIComponent(runId)}`)
    if (request !== runRequestGeneration) return
    state.activeRun = run
    state.pollError = ''
    resetResultFilters()
    state.view = 'results'
    state.message = `Run ${run.status}`
    if (updateRoute) writeRoute()
    render()
    schedulePoll()
  } catch (error) {
    if (request !== runRequestGeneration) return
    reportError(error, 'Unable to load run details.')
    render()
  }
}

function schedulePoll() {
  clearTimeout(state.timer)
  const generation = ++pollGeneration
  if (!isRunInProgress()) return
  const runId = state.activeRun.id
  const poll = async () => {
    try {
      const previous = state.activeRun
      const run = await api(`/api/runs/${encodeURIComponent(runId)}`)
      if (generation !== pollGeneration || state.activeRun?.id !== runId) return
      state.activeRun = run
      const recovered = Boolean(state.pollError)
      state.pollError = ''
      const progress = runProgress(state.activeRun)
      state.message = `${run.status} · ${progress.completed} / ${progress.total}`
      if (!isRunInProgress()) {
        state.runControlAction = null
        await refreshWorkspace(true)
        if (generation !== pollGeneration) return
        state.message = `Run ${state.activeRun.status}`
        renderFromPoll()
        return
      }
      if (!recovered && previous?.status === run.status && previous?.results?.length === run.results?.length) return
      if (state.view !== 'results' && !recovered) {
        const statusText = app.querySelector('.status-text')
        if (statusText) statusText.textContent = state.message
        return
      }
      renderFromPoll()
    } catch (error) {
      if (generation !== pollGeneration || state.activeRun?.id !== runId) return
      state.pollError = `Live updates interrupted. Retrying automatically; do not start another run. ${error instanceof Error ? error.message : 'Connection unavailable.'}`
      renderFromPoll()
    } finally {
      if (generation === pollGeneration && state.activeRun?.id === runId && isRunInProgress()) {
        state.timer = setTimeout(poll, 1200)
      }
    }
  }
  state.timer = setTimeout(poll, 1200)
}

async function startRun() {
  if (state.starting || state.loading || pendingRuns().length) return
  if (!state.selectedModelIds.length || !state.selectedItemIds.length) {
    state.message = 'Select at least one model and utterance.'
    render()
    return
  }
  state.starting = true
  state.error = ''
  render()
  try {
    const parameterOverrides = Object.fromEntries(state.selectedModelIds.map((id) => [id, state.parameterOverrides[id] ?? {}]))
    const queued = await api('/api/runs', {
      method: 'POST',
      body: JSON.stringify({
        model_ids: state.selectedModelIds,
        item_ids: state.selectedItemIds,
        parameter_overrides: parameterOverrides,
        prompt: state.prompt,
        reference_mode: state.referenceMode,
        strategy: state.strategy,
      }),
    })
    ++runRequestGeneration
    // Track an accepted run even if the next read fails; never offer an accidental second submission.
    state.activeRun = {
      ...queued, model_ids: [...state.selectedModelIds], item_ids: [...state.selectedItemIds],
      reference_mode: state.referenceMode, strategy: state.strategy, prompt: state.prompt,
      parameters: parameterOverrides, results: [], started_at: new Date().toISOString(),
    }
    resetResultFilters()
    state.message = 'Benchmark running'
    state.view = 'results'
    writeRoute()
    schedulePoll()
    await refreshWorkspace()
  } catch (error) {
    reportError(error, 'Unable to start the benchmark.')
  } finally {
    state.starting = false
    render()
  }
}

async function startDialectComparison() {
  const models = dialectComparisonModels()
  const items = dialectComparisonItems()
  const dialectCount = new Set(items.map(itemDialect)).size
  if (models.length !== dialectComparisonModelIds.length || dialectCount < 2) {
    state.message = 'Both Realtime models and at least two dialects are required.'
    render()
    return
  }

  state.selectedModelIds = models.map((model) => model.id)
  state.selectedItemIds = items.map((item) => item.id)
  state.selectedDialect = 'all'
  state.sampleSize = items.length
  const chartMetrics = selectedResultMetrics().filter((metric) => chartMetricDefinitions[metric])
  if (!chartMetrics.length) {
    state.selectedMetrics = [...state.selectedMetrics, 'match']
    state.chartMetric = 'match'
  } else if (!chartMetrics.includes(state.chartMetric)) {
    state.chartMetric = chartMetrics[0]
  }
  state.message = `Preparing ${models.length * items.length} dialect comparison tests`
  render()
  await startRun()
}

async function controlRun(action) {
  if (!state.activeRun || !isRunInProgress() || state.runControlAction) return
  const runId = state.activeRun.id
  clearTimeout(state.timer)
  ++pollGeneration
  state.runControlAction = action
  render()
  try {
    const run = await api(`/api/runs/${encodeURIComponent(runId)}/${action}`, { method: 'POST' })
    if (state.activeRun?.id !== runId) return
    state.activeRun = run
    state.message = action === 'pause' ? 'Benchmark paused' : action === 'resume' ? 'Benchmark resumed' : 'Stopping benchmark'
    const [runs, historySummary] = await Promise.all([api('/api/runs'), api('/api/runs/summary')])
    state.runs = runs
    state.historySummary = historySummary
  } catch (error) {
    reportError(error, `Unable to ${action} the benchmark.`)
  } finally {
    state.runControlAction = null
    schedulePoll()
    render()
  }
}

async function saveInstructionPreset() {
  const name = state.instructionName.trim()
  if (!name) {
    state.message = 'Enter a prompt name before saving.'
    render()
    return
  }
  state.savingInstruction = true
  render()
  try {
    const saved = await api('/api/instructions', {
      method: 'POST',
      body: JSON.stringify({ name, prompt: state.prompt }),
    })
    state.instructionName = saved.name
    state.instructionPresets = [saved, ...state.instructionPresets.filter((preset) => preset.name !== saved.name)]
    state.message = `Saved prompt "${saved.name}"`
  } catch (error) {
    reportError(error, 'Unable to save the prompt.')
  } finally {
    state.savingInstruction = false
    render()
  }
}

function setReferenceMode(mode) {
  if (mode === state.referenceMode) return
  const previousSelectionCount = state.selectedItemIds.length
  state.referenceMode = mode
  if (state.prompt === defaultPrompt || state.prompt === highGermanPrompt) {
    state.prompt = mode === 'standard-german' ? highGermanPrompt : defaultPrompt
  }
  if (strategyDefinitions[state.strategy]?.highGermanOnly && mode !== 'standard-german') state.strategy = 'guided'
  state.sampleRound = 0
  const eligibleItemIds = new Set(dialectFilteredItems().filter(itemCanRun).map((item) => item.id))
  state.selectedItemIds = state.selectedItemIds.filter((itemId) => eligibleItemIds.has(itemId))
  if (previousSelectionCount && !state.selectedItemIds.length) selectSample()
  syncSampleSizeToSelection()
  const removedCount = previousSelectionCount - state.selectedItemIds.length
  state.message = removedCount > 0
    ? `${referenceModeLabel()} selected; ${removedCount} utterance${removedCount === 1 ? '' : 's'} lacked this reference`
    : `${taskModes[mode].label} selected`
  render()
}

function reuseActiveRun() {
  if (!state.activeRun) return
  const availableModelIds = new Set(state.models.map((model) => model.id))
  state.selectedModelIds = state.activeRun.model_ids.filter((modelId) => availableModelIds.has(modelId))
  state.referenceMode = state.activeRun.reference_mode ?? 'dialect'
  state.strategy = state.activeRun.strategy ?? 'baseline'
  const availableItemIds = new Set(state.items.filter(itemCanRun).map((item) => item.id))
  state.selectedItemIds = state.activeRun.item_ids.filter((itemId) => availableItemIds.has(itemId))
  state.selectedDataset = ''
  state.search = ''
  state.itemLimit = utterancePageSize
  const selectedDialects = [...new Set(state.selectedItemIds.map((itemId) => itemDialect(itemById(itemId))).filter(Boolean))]
  state.selectedDialect = selectedDialects.length === 1 ? selectedDialects[0] : 'all'
  syncSampleSizeToSelection()
  state.parameterOverrides = { ...state.parameterOverrides, ...structuredClone(state.activeRun.parameters ?? {}) }
  state.prompt = state.activeRun.prompt
  state.message = `Restored ${state.selectedItemIds.length} utterances and ${state.selectedModelIds.length} models`
  setView('benchmark')
}

function benchmarkDialect(code) {
  state.selectedDialect = code
  state.search = ''
  state.sampleRound = 0
  state.itemLimit = utterancePageSize
  state.sampleSize = Math.min(Math.max(state.sampleSize, 4), sampleCapacity())
  selectSample()
  state.message = `${dialectProfile(code)?.name ?? code} selected`
  setView('benchmark')
}

async function toggleResultAudio(button) {
  const player = document.getElementById(button.dataset.audioId)
  if (!(player instanceof HTMLAudioElement)) return
  if (player.paused) {
    document.querySelectorAll('.result-utterance audio').forEach((audio) => {
      if (audio !== player) {
        audio.pause()
        const other = app.querySelector(`[data-audio-id="${audio.id}"]`)
        if (other) { other.innerHTML = '&#9654;'; other.setAttribute('aria-label', 'Play utterance') }
      }
    })
    try {
      await player.play()
      button.textContent = 'II'
      button.setAttribute('aria-label', 'Pause utterance')
      player.addEventListener('ended', () => { button.innerHTML = '&#9654;'; button.setAttribute('aria-label', 'Play utterance') }, { once: true })
    } catch (error) {
      reportError(error, 'Unable to play this utterance.')
      render()
    }
  } else {
    player.pause()
    button.innerHTML = '&#9654;'
    button.setAttribute('aria-label', 'Play utterance')
  }
}

app.addEventListener('click', (event) => {
  const button = event.target.closest('[data-action], [data-run-id], [data-metric-info], [data-chart-metric], [data-summary-metric], [data-view], [data-task-mode], [data-dialect-chip], [data-benchmark-dialect], [data-sample-preset]')
  if (!button || button.disabled) return
  if (button.dataset.view) setView(button.dataset.view)
  if (button.dataset.runId) void openRun(button.dataset.runId)
  if (button.dataset.taskMode) setReferenceMode(button.dataset.taskMode)
  if (button.dataset.benchmarkDialect) benchmarkDialect(button.dataset.benchmarkDialect)
  if (button.dataset.samplePreset) updateSampleSize(button.dataset.samplePreset)
  if (button.dataset.dialectChip) {
    state.selectedDialect = button.dataset.dialectChip
    state.sampleRound = 0
    state.itemLimit = utterancePageSize
    selectSample()
    render()
  }
  if (button.dataset.chartMetric) {
    state.chartMetric = button.dataset.chartMetric
    render()
  }
  if (button.dataset.summaryMetric) {
    state.summaryMetric = button.dataset.summaryMetric
    render()
  }
  if (button.dataset.metricInfo) {
    state.metricInfoOpen = state.metricInfoOpen === button.dataset.metricInfo ? null : button.dataset.metricInfo
    render()
  }
  const action = button.dataset.action
  if (action === 'toggle-theme') setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark')
  if (action === 'refresh') { state.error = ''; void refreshWorkspace() }
  if (action === 'dismiss-error') { state.error = ''; state.pollError = ''; render() }
  if (action === 'small-sample') updateSampleSize(Math.min(4, sampleCapacity()))
  if (action === 'clear-result-filters') { resetResultFilters(); render() }
  if (action === 'results-previous') { state.resultPage = Math.max(1, state.resultPage - 1); render() }
  if (action === 'results-next') { state.resultPage += 1; render() }
  if (action === 'clear-history-filters') { state.historySearch = ''; state.historyStatus = 'all'; state.historyLimit = 10; render() }
  if (action === 'upload-dataset') void uploadDataset()
  if (action === 'resample-items') { selectSample(true); render() }
  if (action === 'clear-items') { state.selectedItemIds = []; render() }
  if (action === 'select-visible') {
    const visible = filteredItems().filter(itemCanRun).map((item) => item.id)
    state.selectedItemIds = [...new Set([...state.selectedItemIds, ...visible])]
    syncSampleSizeToSelection()
    render()
  }
  if (action === 'show-more') { state.itemLimit = Math.min(state.itemLimit + utterancePageSize, filteredItems().length); render() }
  if (action === 'show-all') { state.itemLimit = filteredItems().length; render() }
  if (action === 'history-more') { state.historyLimit += 10; render() }
  if (action === 'compare-dialects') void startDialectComparison()
  if (action === 'start-run') void startRun()
  if (action === 'pause-run') void controlRun('pause')
  if (action === 'resume-run') void controlRun('resume')
  if (action === 'stop-run') void controlRun('stop')
  if (action === 'reuse-run') reuseActiveRun()
  if (action === 'toggle-result-audio') void toggleResultAudio(button)
  if (action === 'save-instruction') void saveInstructionPreset()
})

app.addEventListener('input', (event) => {
  const target = event.target
  if (target.matches('[data-result-search]')) { state.resultSearch = target.value; state.resultPage = 1; render(); return }
  if (target.matches('[data-history-search]')) { state.historySearch = target.value; state.historyLimit = 10; render(); return }
  if (target.matches('[data-search]')) {
    state.search = target.value
    state.itemLimit = utterancePageSize
    render()
    return
  }
  if (target.matches('[data-upload-name]')) {
    state.uploadName = target.value
    return
  }
  if (!target.matches('[data-sample-size]') || target.value === '') return
  window.clearTimeout(sampleSizeTimer)
  sampleSizeTimer = window.setTimeout(() => updateSampleSize(target.value), 250)
})

app.addEventListener('change', (event) => {
  const target = event.target
  if (target.matches('[data-dataset-file]')) {
    state.uploadFile = target.files?.[0] ?? null
    render()
    return
  }
  if (target.matches('[data-dataset-filter]')) {
    state.selectedDataset = target.value
    state.selectedDialect = 'all'
    state.itemLimit = utterancePageSize
    state.sampleRound = 0
    selectSample()
    render()
    return
  }
  if (target.matches('[data-model-id]')) {
    state.selectedModelIds = target.checked
      ? [...state.selectedModelIds, target.dataset.modelId]
      : state.selectedModelIds.filter((id) => id !== target.dataset.modelId)
    if (target.checked && isVirtualEnsemble(target.dataset.modelId) && state.strategy !== 'ensemble' && state.health?.refiner_deployment) {
      state.strategy = 'ensemble'
      state.message = 'Multi-model ensembles run with the Ensemble strategy'
    }
    if (state.strategy === 'two-pass' && selectedTranscriptionModels().length) {
      state.strategy = 'guided'
      state.message = 'Two-pass needs a conversational model; switched to Dialect-guided'
    }
    render()
  }
  if (target.matches('[data-item-id]')) {
    state.selectedItemIds = target.checked
      ? [...state.selectedItemIds, target.dataset.itemId]
      : state.selectedItemIds.filter((id) => id !== target.dataset.itemId)
    syncSampleSizeToSelection()
    render()
  }
  if (target.matches('[data-strategy]')) {
    state.strategy = target.dataset.strategy
    if (state.strategy !== 'ensemble' && state.selectedModelIds.some(isVirtualEnsemble)) {
      state.selectedModelIds = state.selectedModelIds.filter((id) => !isVirtualEnsemble(id))
      state.message = 'Multi-model ensembles need the Ensemble strategy and were deselected'
    }
    render()
  }
  if (target.matches('[data-metric-id]')) {
    state.selectedMetrics = target.checked
      ? [...state.selectedMetrics, target.dataset.metricId]
      : state.selectedMetrics.filter((metric) => metric !== target.dataset.metricId)
    if (target.checked && chartMetricDefinitions[target.dataset.metricId]) state.chartMetric = target.dataset.metricId
    if (!state.selectedMetrics.includes(state.chartMetric)) state.chartMetric = selectedResultMetrics()[0] ?? 'match'
    render()
  }
  if (target.matches('[data-result-dialect]')) { state.resultDialect = target.value; state.resultPage = 1; render() }
  if (target.matches('[data-result-model]')) { state.resultModel = target.value; state.resultPage = 1; render() }
  if (target.matches('[data-result-sort]')) { state.resultSort = target.value; state.resultPage = 1; render() }
  if (target.matches('[data-result-status]')) { state.resultStatus = target.value; state.resultPage = 1; render() }
  if (target.matches('[data-history-status]')) { state.historyStatus = target.value; state.historyLimit = 10; render() }
  if (target.matches('[data-sample-size]')) {
    window.clearTimeout(sampleSizeTimer)
    updateSampleSize(target.value || 1)
  }
  if (target.matches('[data-parameter-model]')) {
    const model = target.dataset.parameterModel
    const name = target.dataset.parameterName
    const definition = state.models.find((candidate) => candidate.id === model)?.parameters.find((parameter) => parameter.name === name)
    const value = target.type === 'checkbox' ? target.checked : definition?.kind === 'number' ? Number(target.value) : target.value
    state.parameterOverrides[model] = { ...state.parameterOverrides[model], [name]: value }
  }
  if (target.matches('[data-prompt]')) state.prompt = target.value
  if (target.matches('[data-instruction-name]')) state.instructionName = target.value
  if (target.matches('[data-instruction-select]')) {
    const preset = state.instructionPresets.find((candidate) => candidate.name === target.value)
    if (preset) {
      state.instructionName = preset.name
      state.prompt = preset.prompt
      state.message = `Loaded prompt "${preset.name}"`
      render()
    }
  }
})

document.addEventListener('pointerdown', () => { pointerIsDown = true }, true)
document.addEventListener('pointerup', flushPendingRender, true)
document.addEventListener('pointercancel', flushPendingRender, true)
window.addEventListener('blur', flushPendingRender)
app.addEventListener('focusout', flushPendingRender)
app.addEventListener('pause', flushPendingRender, true)
app.addEventListener('ended', flushPendingRender, true)
window.addEventListener('popstate', () => { void restoreRoute() })
document.querySelector('.skip-link')?.addEventListener('click', (event) => {
  event.preventDefault()
  document.getElementById('main-content')?.focus()
})

async function initializeWorkspace() {
  await refreshWorkspace()
  await restoreRoute()
}

void initializeWorkspace()
