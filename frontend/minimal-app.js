const app = document.querySelector('#app')
const apiBase = new URLSearchParams(location.search).get('api') ?? ''
const defaultPrompt = 'Transcribe this recording in Swiss German. Return only the spoken words, without translating them.'
const highGermanPrompt = 'Transcribe this recording into High German. Return only the spoken words, without explanation.'
const dialectComparisonModelIds = ['openai-realtime-2', 'openai-realtime-2-1']
const dialectComparisonClipsPerDialect = 2
let sampleSizeTimer = null

const state = {
  models: [],
  items: [],
  runs: [],
  historySummary: null,
  instructionPresets: [],
  instructionName: '',
  savingInstruction: false,
  selectedModelIds: [],
  selectedItemIds: [],
  selectedDialect: 'all',
  referenceMode: 'dialect',
  selectedMetrics: ['validation', 'match'],
  chartMetric: 'match',
  sampleSize: 16,
  sampleRound: 0,
  itemLimit: 5,
  parameterOverrides: {},
  prompt: defaultPrompt,
  activeRun: null,
  message: 'Loading benchmark data...',
  starting: false,
  runControlAction: null,
  metricInfoOpen: null,
  timer: null,
}

const metricDefinitions = {
  validation: {
    label: 'Transcript',
    description: 'Show the model transcript beside the reference utterance for direct validation.',
  },
  match: {
    label: 'Match',
    description: 'Normalized word match. 100% means the normalized word sequence matches the reference exactly.',
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
    description: 'Realtime streaming time from the response request to the first non-empty text delta.',
  },
  latency: {
    label: 'Completion',
    description: 'End-to-end time including decoding, authentication, connection setup, audio upload, and the completed transcript.',
  },
}

const chartMetricDefinitions = {
  match: {
    property: 'word_match_rate',
    title: 'Mean word match by dialect',
    description: 'Higher bars indicate closer transcription to the matching Swiss German reference.',
    kind: 'rate',
  },
  wer: {
    property: 'word_error_rate',
    title: 'Mean WER by dialect',
    description: 'Lower bars indicate fewer word-level transcription errors.',
    kind: 'rate',
  },
  cer: {
    property: 'character_error_rate',
    title: 'Mean CER by dialect',
    description: 'Lower bars indicate fewer character-level transcription errors.',
    kind: 'rate',
  },
  ttft: {
    property: 'time_to_first_token_ms',
    title: 'Mean time to first token by dialect',
    description: 'Lower bars indicate a faster first streamed text delta after the response request.',
    kind: 'latency',
  },
  latency: {
    property: 'latency_ms',
    title: 'Mean completion time by dialect',
    description: 'Lower bars indicate faster completed transcriptions.',
    kind: 'latency',
  },
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
  return value === null || value === undefined ? 'Unavailable' : `${(value * 100).toFixed(1)}%`
}

function latency(value) {
  return value === null || value === undefined ? 'Unavailable' : `${Math.round(value)} ms`
}

function isRunInProgress(run = state.activeRun) {
  return Boolean(run && ['queued', 'running', 'paused', 'stopping'].includes(run.status))
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem('swissdial-theme', theme)
  } catch {}
  const button = document.querySelector('[data-action="toggle-theme"]')
  if (!button) return
  const useLightMode = theme === 'dark'
  button.innerHTML = useLightMode ? '&#9728;' : '&#9790;'
  button.setAttribute('aria-label', useLightMode ? 'Use light mode' : 'Use dark mode')
  button.setAttribute('title', useLightMode ? 'Use light mode' : 'Use dark mode')
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
    throw new Error(body.detail ?? `Request failed with ${response.status}`)
  }
  return response.json()
}

function selectedModels() {
  return state.models.filter((model) => state.selectedModelIds.includes(model.id))
}

function itemDialect(item) {
  return String(item?.metadata?.dialect ?? item?.metadata?.canton ?? '').toUpperCase()
}

function itemDialectName(item) {
  return item?.metadata?.dialect_name ?? itemDialect(item) ?? 'Swiss German'
}

function itemById(itemId) {
  return state.items.find((item) => item.id === itemId)
}

function referenceModeLabel(mode = state.referenceMode) {
  return mode === 'standard-german' ? 'High German' : 'Matching Swiss German'
}

function itemReference(item, mode = state.referenceMode) {
  if (mode === 'standard-german') return item?.metadata?.standard_german_transcript ?? null
  return item?.reference_transcript ?? null
}

function itemCanRun(item) {
  return Boolean(item?.audio_available && (state.referenceMode !== 'standard-german' || itemReference(item)))
}

function syncSampleSizeToSelection() {
  if (state.selectedItemIds.length) {
    state.sampleSize = state.selectedItemIds.length
  }
}

function availableDialects() {
  const dialects = new Map()
  for (const item of state.items) {
    const code = itemDialect(item)
    if (code) dialects.set(code, itemDialectName(item))
  }
  return [...dialects].sort(([left], [right]) => left.localeCompare(right))
}

function filteredItems() {
  return state.selectedDialect === 'all'
    ? state.items
    : state.items.filter((item) => itemDialect(item) === state.selectedDialect)
}

function sampleCapacity() {
  return filteredItems().filter(itemCanRun).length
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
  for (const item of filteredItems().filter(itemCanRun)) {
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
  for (const item of state.items.filter((candidate) => itemCanRun(candidate) && itemDialect(candidate))) {
    const dialect = itemDialect(item)
    if (!groups.has(dialect)) groups.set(dialect, [])
    groups.get(dialect).push(item)
  }
  return [...groups]
    .sort(([left], [right]) => left.localeCompare(right))
    .flatMap(([, items]) => items.slice(0, dialectComparisonClipsPerDialect))
}

function modelPicker() {
  if (!state.models.length) {
    return '<p class="empty-state">No runnable transcription model is configured.</p>'
  }
  return `<div class="model-options">${state.models.map((model) => {
    const selected = state.selectedModelIds.includes(model.id)
    const capability = model.capabilities.includes('realtime') ? 'Realtime transcription' : 'Audio transcription'
    return `<label class="model-option ${selected ? 'selected' : ''}"><input type="checkbox" data-model-id="${escapeHtml(model.id)}" ${selected ? 'checked' : ''}><span><strong>${escapeHtml(model.label)}</strong><small>${escapeHtml(capability)}</small></span></label>`
  }).join('')}</div>`
}

function dialectComparisonControl() {
  const models = dialectComparisonModels()
  const items = dialectComparisonItems()
  const dialectCount = new Set(items.map(itemDialect)).size
  const taskCount = models.length * items.length
  const disabled = models.length !== dialectComparisonModelIds.length || dialectCount < 2 || isRunInProgress() || state.starting
  return `<div class="dialect-compare-control">
    <button type="button" data-action="compare-dialects" ${disabled ? 'disabled' : ''}><strong>Compare all dialects</strong><span>${models.length} models / ${dialectCount} dialects / ${taskCount} tests</span></button>
    <small>Uses up to two utterances per dialect, the selected reference, and the selected metrics.</small>
  </div>`
}

function metricPicker() {
  return `<div class="metric-options">${Object.entries(metricDefinitions).map(([key, definition]) => {
    const selected = state.selectedMetrics.includes(key)
    return `<label class="metric-option ${selected ? 'selected' : ''}" title="${escapeHtml(definition.description)}"><input type="checkbox" data-metric-id="${key}" ${selected ? 'checked' : ''}><span>${escapeHtml(definition.label)}</span></label>`
  }).join('')}</div>`
}

function sampleControls() {
  const dialectOptions = availableDialects().map(([code, name]) => (
    `<option value="${escapeHtml(code)}" ${state.selectedDialect === code ? 'selected' : ''}>${escapeHtml(name)}</option>`
  )).join('')
  const capacity = sampleCapacity()
  return `<div class="sample-controls">
    <label><span>Audio dialect</span><select data-dialect-filter><option value="all" ${state.selectedDialect === 'all' ? 'selected' : ''}>All dialects</option>${dialectOptions}</select></label>
    <label><span>Evaluation reference</span><select data-reference-mode><option value="dialect" ${state.referenceMode === 'dialect' ? 'selected' : ''}>Matching Swiss German</option><option value="standard-german" ${state.referenceMode === 'standard-german' ? 'selected' : ''}>High German</option></select></label>
    <label class="sample-size"><span>Sample size</span><input data-sample-size type="number" min="1" max="${Math.max(1, capacity)}" step="1" value="${Math.min(state.sampleSize, Math.max(1, capacity))}" ${capacity ? '' : 'disabled'}></label>
    <button type="button" class="secondary-button" data-action="resample-items">Resample</button>
  </div>`
}

function utteranceRows() {
  if (!state.items.length) {
    return '<p class="empty-state">Import SwissDial to choose benchmark utterances.</p>'
  }
  const visibleItems = filteredItems()
  if (!visibleItems.length) {
    return '<p class="empty-state">No utterances are available for this dialect.</p>'
  }

  const rows = visibleItems.slice(0, state.itemLimit).map((item) => {
    const selected = state.selectedItemIds.includes(item.id)
    const reference = itemReference(item)
    const selectable = itemCanRun(item)
    const utterance = reference || `${referenceModeLabel()} reference unavailable`
    const audioUrl = `${apiBase}/api/dataset/items/${encodeURIComponent(item.id)}/audio`
    const topic = item.metadata?.topic ? `<span>${escapeHtml(item.metadata.topic)}</span>` : ''
    return `<article class="utterance-row ${selected ? 'selected' : ''} ${selectable ? '' : 'unavailable'}">
      <label class="utterance-select">
        <input type="checkbox" data-item-id="${escapeHtml(item.id)}" ${selected ? 'checked' : ''} ${selectable ? '' : 'disabled'}>
        <span class="utterance-copy"><strong>${escapeHtml(utterance)}</strong><small><span>${escapeHtml(itemDialectName(item))}</span>${topic}</small></span>
      </label>
      ${item.audio_available ? `<audio controls preload="none" src="${escapeHtml(audioUrl)}">Audio playback is not supported by this browser.</audio>` : '<span class="missing-audio">Audio unavailable</span>'}
    </article>`
  }).join('')

  const remaining = visibleItems.length - state.itemLimit
  const disclosure = remaining > 0
    ? `<div class="list-disclosure"><button type="button" data-action="show-more">Show 5 more</button><button type="button" data-action="show-all">Show all ${visibleItems.length}</button></div>`
    : ''
  return `${rows}${disclosure}`
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
      return `<label class="parameter-field"><span>${escapeHtml(parameter.label)}</span><input type="${type}" data-parameter-model="${escapeHtml(model.id)}" data-parameter-name="${escapeHtml(parameter.name)}" value="${escapeHtml(value)}" ${parameter.minimum !== undefined ? `min="${parameter.minimum}"` : ''} ${parameter.maximum !== undefined ? `max="${parameter.maximum}"` : ''} ${parameter.step !== undefined ? `step="${parameter.step}"` : ''}></label>`
    }).join('')
    return `<div class="parameter-group"><strong>${escapeHtml(model.label)}</strong><div class="parameter-fields">${fields}</div></div>`
  }).filter(Boolean)
  return groups.length ? `<div class="advanced-subsection"><span>Model parameters</span>${groups.join('')}</div>` : ''
}

function instructionControls() {
  const options = state.instructionPresets.map((preset) => (
    `<option value="${escapeHtml(preset.name)}" ${state.instructionName === preset.name ? 'selected' : ''}>${escapeHtml(preset.name)}</option>`
  )).join('')
  return `<div class="instruction-controls">
    <label><span>Saved prompt</span><select data-instruction-select><option value="">Default</option>${options}</select></label>
    <label><span>Save as</span><input data-instruction-name type="text" maxlength="80" value="${escapeHtml(state.instructionName)}" placeholder="Prompt name"></label>
    <button type="button" class="secondary-button" data-action="save-instruction" ${state.savingInstruction ? 'disabled' : ''}>${state.savingInstruction ? 'Saving...' : 'Save'}</button>
  </div>`
}

function advancedSettings() {
  return `<details class="advanced-settings">
    <summary>Advanced settings</summary>
    <div class="advanced-content">
      ${parameterGroups()}
      <label class="prompt-field"><span>Transcription prompt</span><textarea data-prompt>${escapeHtml(state.prompt)}</textarea></label>
      ${instructionControls()}
    </div>
  </details>`
}

function metricHeader(metric) {
  const definition = metricDefinitions[metric]
  const open = state.metricInfoOpen === metric
  return `<span class="metric-header">${escapeHtml(definition.label)}<button type="button" data-metric-info="${metric}" aria-label="Explain ${escapeHtml(definition.label)}" aria-expanded="${open}">i</button>${open ? `<span role="tooltip">${escapeHtml(definition.description)}</span>` : ''}</span>`
}

function resultMetricValue(result, metric) {
  if (metric === 'match') return rate(result.word_match_rate)
  if (metric === 'wer') return rate(result.word_error_rate)
  if (metric === 'cer') return rate(result.character_error_rate)
  if (metric === 'ttft') return latency(result.time_to_first_token_ms)
  return latency(result.latency_ms)
}

function selectedResultMetrics() {
  return state.selectedMetrics.filter((metric) => metric !== 'validation')
}

function resultRows() {
  const results = state.activeRun?.results ?? []
  const metrics = selectedResultMetrics()
  const showValidation = state.selectedMetrics.includes('validation')
  const columnCount = 2 + metrics.length + (showValidation ? 1 : 0)
  if (!results.length) {
    return `<tr><td colspan="${columnCount}" class="table-empty">Results will appear as each utterance is transcribed.</td></tr>`
  }

  return results.map((result) => {
    const item = itemById(result.item_id)
    const reference = result.reference_transcript ?? item?.reference_transcript ?? 'Reference unavailable'
    const audioUrl = `${apiBase}/api/dataset/items/${encodeURIComponent(result.item_id)}/audio`
    const playerId = `result-audio-${result.id}`
    const validation = showValidation
      ? `<td class="transcript-cell">${result.status === 'failed' ? `<span class="result-error">${escapeHtml(result.error)}</span>` : escapeHtml(result.transcript ?? 'No transcript returned')}</td>`
      : ''
    const metricCells = metrics.map((metric) => `<td class="metric-cell">${resultMetricValue(result, metric)}</td>`).join('')
    return `<tr class="${result.status === 'failed' ? 'failed-result' : ''}">
      <td class="result-utterance"><strong>${escapeHtml(reference)}</strong><span>${escapeHtml(itemDialectName(item))}</span><button type="button" data-action="toggle-result-audio" data-audio-id="${playerId}" aria-label="Play utterance" title="Play utterance">&#9654;</button><audio id="${playerId}" preload="none" src="${escapeHtml(audioUrl)}"></audio></td>
      <td>${escapeHtml(result.model_label)}</td>${validation}${metricCells}
    </tr>`
  }).join('')
}

function averageResultMetric(run, metric) {
  const property = {
    match: 'word_match_rate',
    wer: 'word_error_rate',
    cer: 'character_error_rate',
    ttft: 'time_to_first_token_ms',
    latency: 'latency_ms',
  }[metric]
  const values = (run.results ?? []).map((result) => result[property]).filter((value) => value !== null && value !== undefined)
  if (!values.length) return null
  return values.reduce((total, value) => total + value, 0) / values.length
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
  const models = run.model_ids
    .map((modelId) => state.models.find((model) => model.id === modelId))
    .filter(Boolean)
  const dialects = [...new Map(run.item_ids.map((itemId) => itemById(itemId)).filter(Boolean).map((item) => [itemDialect(item), itemDialectName(item)]))]
    .filter(([dialect]) => dialect)
    .sort(([left], [right]) => left.localeCompare(right))
  const availableMetrics = selectedResultMetrics().filter((metric) => chartMetricDefinitions[metric])
  if (!models.length || !dialects.length || !availableMetrics.length) return ''

  const metric = availableMetrics.includes(state.chartMetric) ? state.chartMetric : availableMetrics[0]
  const definition = chartMetricDefinitions[metric]
  const referenceLabel = referenceModeLabel(run.reference_mode)
  const chartDescription = metric === 'match'
    ? `Higher bars indicate closer transcription to the ${referenceLabel} reference.`
    : metric === 'wer'
      ? `Lower bars indicate fewer word-level errors against the ${referenceLabel} reference.`
      : metric === 'cer'
        ? `Lower bars indicate fewer character-level errors against the ${referenceLabel} reference.`
        : definition.description
  const series = dialects.map(([dialect, dialectName]) => ({
    dialect,
    dialectName,
    models: models.map((model) => {
      const values = (run.results ?? [])
        .filter((result) => result.model_id === model.id && itemDialect(itemById(result.item_id)) === dialect)
        .map((result) => result[definition.property])
        .filter((value) => Number.isFinite(value))
      return {
        model,
        values,
        average: values.length ? values.reduce((total, value) => total + value, 0) / values.length : null,
      }
    }),
  }))
  const observedValues = series.flatMap((group) => group.models.map((entry) => entry.average).filter((value) => value !== null))
  const scaleMaximum = definition.kind === 'rate'
    ? niceChartMaximum(Math.max(1, ...observedValues))
    : niceChartMaximum(Math.max(1000, ...observedValues))
  const scaleLabels = [1, 0.75, 0.5, 0.25, 0]
    .map((position) => `<span>${chartValue(scaleMaximum * position, definition.kind, true)}</span>`)
    .join('')

  const legend = models.map((model, index) => `<span><i class="series-${index % 2}"></i>${escapeHtml(model.label)}</span>`).join('')
  const metricControls = availableMetrics.length > 1
    ? `<div class="chart-metric-switch" role="group" aria-label="Chart metric">${availableMetrics.map((key) => `<button type="button" data-chart-metric="${key}" aria-pressed="${metric === key}" class="${metric === key ? 'active' : ''}">${escapeHtml(metricDefinitions[key].label)}</button>`).join('')}</div>`
    : ''
  const groups = series.map(({ dialect, dialectName, models: modelSeries }) => {
    const bars = modelSeries.map(({ model, values, average }, index) => {
      const height = average === null ? 0 : Math.max(0, Math.min(100, average / scaleMaximum * 100))
      const valueLabel = average === null ? (isRunInProgress(run) ? 'Pending' : 'No score') : chartValue(average, definition.kind)
      const utteranceCount = `${values.length} ${values.length === 1 ? 'utterance' : 'utterances'}`
      const accessibleLabel = `${model.label}, ${dialectName}, ${metricDefinitions[metric].label}: ${valueLabel}${values.length ? ` across ${utteranceCount}` : ''}`
      return `<span class="chart-bar series-${index % 2} ${average === null ? 'pending' : ''}" tabindex="0" role="img" aria-label="${escapeHtml(accessibleLabel)}"><i class="chart-bar-fill" style="--bar-height:${height}%"></i><span class="chart-tooltip">${escapeHtml(model.label)}<strong>${escapeHtml(valueLabel)}</strong><small>${values.length ? utteranceCount : 'Waiting for results'}</small></span></span>`
    }).join('')
    return `<div class="chart-group"><div class="chart-bars">${bars}</div><span title="${escapeHtml(dialectName)}">${escapeHtml(dialect)}</span></div>`
  }).join('')

  return `<section class="dialect-chart" aria-labelledby="dialect-chart-title">
    <div class="dialect-chart-heading"><div><span>${models.length > 1 ? 'Model comparison' : 'Model results'}</span><h3 id="dialect-chart-title">${escapeHtml(definition.title)}</h3><p>${escapeHtml(chartDescription)}</p></div><div class="chart-heading-actions">${metricControls}<div class="chart-legend">${legend}</div></div></div>
    <div class="chart-scroll"><div class="chart-canvas">
      <div class="chart-scale" aria-hidden="true">${scaleLabels}</div>
      <div class="chart-grid" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
      <div class="chart-groups" style="--dialect-count:${dialects.length}">${groups}</div>
    </div></div>
  </section>`
}

function runSetupDetails(run) {
  const modelLabels = run.model_ids.map((modelId) => (
    state.models.find((model) => model.id === modelId)?.label ?? modelId
  ))
  const parameterGroups = Object.entries(run.parameters ?? {}).flatMap(([modelId, parameters]) => {
    const values = Object.entries(parameters ?? {})
    if (!values.length) return []
    const modelLabel = state.models.find((model) => model.id === modelId)?.label ?? modelId
    return [`${modelLabel}: ${values.map(([name, value]) => `${name}=${value}`).join(', ')}`]
  })
  return `<details class="run-setup">
    <summary>Run setup</summary>
    <div><span>Models</span><strong>${escapeHtml(modelLabels.join(', '))}</strong></div>
    <div><span>Prompt</span><p>${escapeHtml(run.prompt)}</p></div>
    ${parameterGroups.length ? `<div><span>Parameters</span><p>${escapeHtml(parameterGroups.join(' / '))}</p></div>` : ''}
  </details>`
}

function activeRunPanel() {
  if (!state.activeRun) {
    return '<div class="empty-results"><strong>No results yet</strong><p>Select utterances, models, and result fields, then run the benchmark.</p></div>'
  }
  const run = state.activeRun
  const progress = runProgress(run)
  const metrics = selectedResultMetrics()
  const showValidation = state.selectedMetrics.includes('validation')
  const metricHeaders = metrics.map((metric) => `<th>${metricHeader(metric)}</th>`).join('')
  const validationHeader = showValidation ? '<th>Model transcript</th>' : ''
  const runReferenceMode = run.reference_mode ?? 'dialect'
  const referenceHeader = `${referenceModeLabel(runReferenceMode)} reference`
  const runDialects = [...new Set(run.item_ids.map((itemId) => itemDialect(itemById(itemId))).filter(Boolean))].sort()
  const dialectSummary = runDialects.length === 1
    ? itemDialectName(run.item_ids.map((itemId) => itemById(itemId)).find((item) => itemDialect(item) === runDialects[0]))
    : `${runDialects.length} dialects (${runDialects.join(', ')})`
  const controlsDisabled = Boolean(state.runControlAction)
  const controls = run.status === 'paused'
    ? `<button type="button" class="secondary-button" data-action="resume-run" ${controlsDisabled ? 'disabled' : ''}>Resume</button><button type="button" class="danger-button" data-action="stop-run" ${controlsDisabled ? 'disabled' : ''}>Stop</button>`
    : run.status === 'stopping'
      ? '<span class="control-note">Stopping after the current utterance</span>'
      : isRunInProgress(run)
        ? `<button type="button" class="secondary-button" data-action="pause-run" ${controlsDisabled ? 'disabled' : ''}>Pause</button><button type="button" class="danger-button" data-action="stop-run" ${controlsDisabled ? 'disabled' : ''}>Stop</button>`
        : ''
  const primaryMetric = metrics[0]
  const primaryAverage = primaryMetric ? averageResultMetric(run, primaryMetric) : null
  const primaryValue = ['ttft', 'latency'].includes(primaryMetric) ? latency(primaryAverage) : rate(primaryAverage)
  const metricSummary = primaryMetric ? `<div><span>Mean ${escapeHtml(metricDefinitions[primaryMetric].label)}</span><strong>${primaryValue}</strong></div>` : ''
  const progressPanel = isRunInProgress(run)
    ? `<div class="run-progress"><div><span>${progress.completed} of ${progress.total} complete</span><strong>${progress.percent}%</strong></div><div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="${progress.total}" aria-valuenow="${progress.completed}"><span style="width:${progress.percent}%"></span></div><div class="run-controls">${controls}</div></div>`
    : ''
  const resultActions = !isRunInProgress(run)
    ? `<div class="result-actions"><button type="button" class="secondary-button" data-action="reuse-run">Use this setup</button><a class="secondary-button" href="${apiBase}/api/runs/${encodeURIComponent(run.id)}/export.csv" download>Download CSV</a></div>`
    : ''
  const comparisonChart = dialectComparisonChart(run)
  const setupDetails = runSetupDetails(run)
  return `<div class="result-summary">
      <div><span>Status</span><strong class="status-value ${escapeHtml(run.status)}">${escapeHtml(run.status)}</strong></div>
      <div><span>Completed</span><strong>${progress.completed} / ${progress.total}</strong></div>
      <div><span>Audio</span><strong>${escapeHtml(dialectSummary || 'Unknown dialect')}</strong></div>
      <div><span>Reference</span><strong>${escapeHtml(referenceModeLabel(runReferenceMode))}</strong></div>
      ${metricSummary}
    </div>
    ${setupDetails}${progressPanel}${comparisonChart}${resultActions}
    <div class="table-wrap"><table><thead><tr><th>${escapeHtml(referenceHeader)}</th><th>Model</th>${validationHeader}${metricHeaders}</tr></thead><tbody>${resultRows()}</tbody></table></div>`
}

function historyRows() {
  if (!state.runs.length) {
    return '<p class="empty-state">Completed runs will appear here.</p>'
  }
  const preferredMetric = selectedResultMetrics()[0] ?? 'match'
  return state.runs.slice(0, 6).map((run) => {
    const total = run.total_task_count ?? run.model_ids.length * run.item_ids.length
    const indicator = run.indicator ?? { label: run.status, tone: 'neutral' }
    const value = ['ttft', 'latency'].includes(preferredMetric)
      ? latency(preferredMetric === 'ttft' ? run.average_time_to_first_token_ms : run.average_latency_ms)
      : preferredMetric === 'wer'
        ? rate(run.average_word_error_rate)
        : preferredMetric === 'match'
          ? rate(run.average_word_match_rate)
          : 'Open run'
    return `<button class="history-row" type="button" data-run-id="${escapeHtml(run.id)}">
      <span class="history-primary"><strong>${run.model_ids.length} ${run.model_ids.length === 1 ? 'model' : 'models'} / ${run.item_ids.length} utterances</strong><small>${new Date(run.started_at).toLocaleString()} / ${escapeHtml(referenceModeLabel(run.reference_mode))}</small></span>
      <span class="history-status ${escapeHtml(indicator.tone)}">${escapeHtml(indicator.label)}</span>
      <span class="history-metric"><small>${escapeHtml(metricDefinitions[preferredMetric]?.label ?? 'Result')}</small><strong>${value}</strong></span>
      <span class="history-count">${run.result_count ?? 0} / ${total}</span>
      <span aria-hidden="true">&#8250;</span>
    </button>`
  }).join('')
}

function render() {
  const taskCount = state.selectedModelIds.length * state.selectedItemIds.length
  const runDisabled = state.starting || isRunInProgress() || !state.selectedModelIds.length || !state.selectedItemIds.length
  const runLabel = state.starting || isRunInProgress() ? 'Benchmark running' : `Run ${taskCount} ${taskCount === 1 ? 'test' : 'tests'}`
  const useLightMode = document.documentElement.dataset.theme === 'dark'
  app.innerHTML = `<div class="app-shell">
    <header class="app-header">
      <div><span class="product-label">SwissDial benchmark</span><h1>Swiss German transcription</h1></div>
      <div class="header-status"><span class="status-dot"></span><span>${escapeHtml(state.message)}</span><button type="button" class="icon-button" data-action="toggle-theme" aria-label="${useLightMode ? 'Use light mode' : 'Use dark mode'}" title="${useLightMode ? 'Use light mode' : 'Use dark mode'}">${useLightMode ? '&#9728;' : '&#9790;'}</button><button type="button" class="icon-button" data-action="refresh" aria-label="Refresh benchmark data" title="Refresh benchmark data">&#8635;</button></div>
    </header>

    <main>
      <section class="builder" aria-labelledby="builder-title">
        <div class="utterance-panel">
          <div class="section-title"><div><span>Benchmark set</span><h2 id="builder-title">Utterances</h2></div><div class="selection-count"><strong>${state.selectedItemIds.length}</strong> selected <button type="button" data-action="clear-items">Clear</button></div></div>
          ${sampleControls()}
          <div class="utterance-list">${utteranceRows()}</div>
        </div>

        <aside class="setup-panel">
          <section><div class="section-title compact"><div><span>Compare</span><h2>Models</h2></div><strong>${state.selectedModelIds.length}</strong></div>${modelPicker()}${dialectComparisonControl()}</section>
          <section><div class="section-title compact"><div><span>Display</span><h2>Result fields</h2></div></div>${metricPicker()}</section>
          ${advancedSettings()}
        </aside>
      </section>

      <div class="run-bar"><div><strong>${state.selectedItemIds.length} utterances</strong><span>${state.selectedModelIds.length} models / ${taskCount} total tests</span></div><button type="button" class="primary-button" data-action="start-run" ${runDisabled ? 'disabled' : ''}>${escapeHtml(runLabel)}</button></div>

      <section class="results-section"><div class="section-heading"><div><span>Current run</span><h2>Validation and metrics</h2></div></div>${activeRunPanel()}</section>
      <section class="history-section"><div class="section-heading"><div><span>Previous runs</span><h2>History</h2></div></div><div class="history-list">${historyRows()}</div></section>
    </main>
  </div>`
}

async function refreshWorkspace() {
  try {
    const [models, items, runs, historySummary, instructionPresets] = await Promise.all([
      api('/api/models'),
      api('/api/dataset/items'),
      api('/api/runs'),
      api('/api/runs/summary'),
      api('/api/instructions'),
    ])
    state.models = models
    state.items = items
    state.runs = runs
    state.historySummary = historySummary
    state.instructionPresets = instructionPresets
    if (!state.selectedModelIds.length) state.selectedModelIds = models.slice(0, 1).map((model) => model.id)
    if (state.selectedDialect !== 'all' && !availableDialects().some(([code]) => code === state.selectedDialect)) state.selectedDialect = 'all'
    const availableItemIds = new Set(items.filter((item) => item.audio_available).map((item) => item.id))
    state.selectedItemIds = state.selectedItemIds.filter((itemId) => availableItemIds.has(itemId))
    if (!state.selectedItemIds.length) selectSample()
    for (const model of models) {
      state.parameterOverrides[model.id] ??= Object.fromEntries(model.parameters.map((parameter) => [parameter.name, parameter.default]))
    }
    state.message = !models.length
      ? 'Configure a runnable model to begin.'
      : items.length
        ? `${items.length} utterances ready`
        : 'Import SwissDial utterances to begin.'
  } catch (error) {
    state.message = error instanceof Error ? error.message : 'Unable to load the benchmark.'
  }
  render()
}

async function openRun(runId) {
  try {
    state.activeRun = await api(`/api/runs/${encodeURIComponent(runId)}`)
    state.starting = isRunInProgress()
    render()
    schedulePoll()
  } catch (error) {
    state.message = error instanceof Error ? error.message : 'Unable to load run details.'
    render()
  }
}

function schedulePoll() {
  clearInterval(state.timer)
  if (!isRunInProgress()) return
  state.timer = setInterval(async () => {
    try {
      state.activeRun = await api(`/api/runs/${encodeURIComponent(state.activeRun.id)}`)
      if (!isRunInProgress()) {
        state.starting = false
        state.runControlAction = null
        clearInterval(state.timer)
        await refreshWorkspace()
        return
      }
      render()
    } catch (error) {
      state.message = error instanceof Error ? error.message : 'Unable to refresh benchmark progress.'
      render()
    }
  }, 1200)
}

async function startRun() {
  if (!state.selectedModelIds.length || !state.selectedItemIds.length) {
    state.message = 'Select at least one model and utterance.'
    render()
    return
  }
  state.starting = true
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
      }),
    })
    state.activeRun = await api(`/api/runs/${queued.id}`)
    state.message = 'Benchmark running'
    await refreshWorkspace()
    schedulePoll()
  } catch (error) {
    state.message = error instanceof Error ? error.message : 'Unable to start the benchmark.'
  } finally {
    if (!isRunInProgress()) state.starting = false
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
  if (!state.activeRun || !isRunInProgress()) return
  state.runControlAction = action
  render()
  try {
    state.activeRun = await api(`/api/runs/${encodeURIComponent(state.activeRun.id)}/${action}`, { method: 'POST' })
    state.starting = isRunInProgress()
    state.message = action === 'pause' ? 'Benchmark paused' : action === 'resume' ? 'Benchmark resumed' : 'Stopping benchmark'
    const [runs, historySummary] = await Promise.all([api('/api/runs'), api('/api/runs/summary')])
    state.runs = runs
    state.historySummary = historySummary
    schedulePoll()
  } catch (error) {
    state.message = error instanceof Error ? error.message : `Unable to ${action} the benchmark.`
  } finally {
    state.runControlAction = null
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
    state.message = error instanceof Error ? error.message : 'Unable to save the prompt.'
  } finally {
    state.savingInstruction = false
    render()
  }
}

function reuseActiveRun() {
  if (!state.activeRun) return
  const availableModelIds = new Set(state.models.map((model) => model.id))
  state.selectedModelIds = state.activeRun.model_ids.filter((modelId) => availableModelIds.has(modelId))
  state.referenceMode = state.activeRun.reference_mode ?? 'dialect'
  const availableItemIds = new Set(state.items.filter(itemCanRun).map((item) => item.id))
  state.selectedItemIds = state.activeRun.item_ids.filter((itemId) => availableItemIds.has(itemId))
  const selectedDialects = [...new Set(state.selectedItemIds.map((itemId) => itemDialect(itemById(itemId))).filter(Boolean))]
  state.selectedDialect = selectedDialects.length === 1 ? selectedDialects[0] : 'all'
  syncSampleSizeToSelection()
  state.parameterOverrides = { ...state.parameterOverrides, ...state.activeRun.parameters }
  state.prompt = state.activeRun.prompt
  state.message = `Restored ${state.selectedItemIds.length} utterances and ${state.selectedModelIds.length} models`
  render()
  document.querySelector('.builder')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

async function toggleResultAudio(button) {
  const player = document.getElementById(button.dataset.audioId)
  if (!(player instanceof HTMLAudioElement)) return
  if (player.paused) {
    document.querySelectorAll('.result-utterance audio').forEach((audio) => {
      if (audio !== player) audio.pause()
    })
    try {
      await player.play()
      button.textContent = 'II'
      player.addEventListener('ended', () => { button.innerHTML = '&#9654;' }, { once: true })
    } catch (error) {
      state.message = error instanceof Error ? error.message : 'Unable to play this utterance.'
      render()
    }
  } else {
    player.pause()
    button.innerHTML = '&#9654;'
  }
}

app.addEventListener('click', (event) => {
  const button = event.target.closest('[data-action], [data-run-id], [data-metric-info], [data-chart-metric]')
  if (!button) return
  if (button.dataset.runId) void openRun(button.dataset.runId)
  if (button.dataset.chartMetric) {
    state.chartMetric = button.dataset.chartMetric
    render()
  }
  if (button.dataset.metricInfo) {
    state.metricInfoOpen = state.metricInfoOpen === button.dataset.metricInfo ? null : button.dataset.metricInfo
    render()
  }
  if (button.dataset.action === 'toggle-theme') setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark')
  if (button.dataset.action === 'refresh') void refreshWorkspace()
  if (button.dataset.action === 'resample-items') { selectSample(true); render() }
  if (button.dataset.action === 'clear-items') { state.selectedItemIds = []; render() }
  if (button.dataset.action === 'show-more') { state.itemLimit = Math.min(state.itemLimit + 5, filteredItems().length); render() }
  if (button.dataset.action === 'show-all') { state.itemLimit = filteredItems().length; render() }
  if (button.dataset.action === 'compare-dialects') void startDialectComparison()
  if (button.dataset.action === 'start-run') void startRun()
  if (button.dataset.action === 'pause-run') void controlRun('pause')
  if (button.dataset.action === 'resume-run') void controlRun('resume')
  if (button.dataset.action === 'stop-run') void controlRun('stop')
  if (button.dataset.action === 'reuse-run') reuseActiveRun()
  if (button.dataset.action === 'toggle-result-audio') void toggleResultAudio(button)
  if (button.dataset.action === 'save-instruction') void saveInstructionPreset()
})

app.addEventListener('input', (event) => {
  const target = event.target
  if (!target.matches('[data-sample-size]') || target.value === '') return
  window.clearTimeout(sampleSizeTimer)
  sampleSizeTimer = window.setTimeout(() => updateSampleSize(target.value), 250)
})

app.addEventListener('change', (event) => {
  const target = event.target
  if (target.matches('[data-model-id]')) {
    state.selectedModelIds = target.checked
      ? [...state.selectedModelIds, target.dataset.modelId]
      : state.selectedModelIds.filter((id) => id !== target.dataset.modelId)
    render()
  }
  if (target.matches('[data-item-id]')) {
    state.selectedItemIds = target.checked
      ? [...state.selectedItemIds, target.dataset.itemId]
      : state.selectedItemIds.filter((id) => id !== target.dataset.itemId)
    syncSampleSizeToSelection()
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
  if (target.matches('[data-dialect-filter]')) {
    state.selectedDialect = target.value
    state.sampleRound = 0
    state.itemLimit = 5
    selectSample()
    render()
  }
  if (target.matches('[data-reference-mode]')) {
    const previousMode = state.referenceMode
    const previousSelectionCount = state.selectedItemIds.length
    state.referenceMode = target.value
    if (state.prompt === defaultPrompt || state.prompt === highGermanPrompt) {
      state.prompt = state.referenceMode === 'standard-german' ? highGermanPrompt : defaultPrompt
    }
    state.sampleRound = 0
    const eligibleItemIds = new Set(filteredItems().filter(itemCanRun).map((item) => item.id))
    state.selectedItemIds = state.selectedItemIds.filter((itemId) => eligibleItemIds.has(itemId))
    if (previousSelectionCount && !state.selectedItemIds.length) selectSample()
    syncSampleSizeToSelection()
    const removedCount = previousSelectionCount - state.selectedItemIds.length
    state.message = removedCount > 0
      ? `${referenceModeLabel()} selected; ${removedCount} utterance${removedCount === 1 ? '' : 's'} lacked this reference`
      : `${referenceModeLabel()} selected for display and scoring`
    if (previousMode !== state.referenceMode && removedCount > 0) state.itemLimit = 5
    render()
  }
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

void refreshWorkspace()
