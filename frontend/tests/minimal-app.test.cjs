const assert = require('node:assert/strict')
const { readFileSync } = require('node:fs')
const path = require('node:path')
const { test } = require('node:test')
const vm = require('node:vm')

const source = readFileSync(path.join(__dirname, '..', 'minimal-app.js'), 'utf8')

function workbench() {
  const timers = new Map()
  let timerId = 0
  const app = {
    addEventListener() {},
    querySelectorAll: () => [],
    querySelector: () => null,
    contains: () => false,
  }
  const context = vm.createContext({
    URLSearchParams, structuredClone, Error,
    location: { search: '', hash: '' },
    history: { pushState(_state, _title, hash) { context.location.hash = hash } },
    document: {
      querySelector: () => app,
      addEventListener() {},
      documentElement: { dataset: { theme: 'light' } },
      activeElement: null,
    },
    window: { addEventListener() {}, scrollTo() {} },
    fetch: () => new Promise(() => {}),
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId },
    clearTimeout: (id) => timers.delete(id),
  })
  vm.runInContext(source, context)
  const state = vm.runInContext('state', context)
  context.render = () => {}
  state.loading = false
  return {
    context, state, timers, app,
    async tick() {
      const [id, callback] = timers.entries().next().value
      timers.delete(id)
      await callback()
    },
  }
}

function run(id = 'run-a', status = 'completed') {
  return {
    id, status, model_ids: ['m'], item_ids: ['clip'], results: [],
    reference_mode: 'dialect', strategy: 'guided', parameters: {},
    started_at: '2026-01-01T00:00:00Z',
  }
}

function deferred() {
  let resolve
  let reject
  const promise = new Promise((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

test('result search and status compose; null scores sort last in both directions', () => {
  const { context, state } = workbench()
  const fixture = run()
  fixture.results = [
    { id: 1, status: 'failed', error: 'Quota exceeded', word_match_rate: null },
    { id: 2, status: 'completed', transcript: 'GRÜEZI', word_match_rate: 0.8, latency_ms: 100 },
    { id: 3, status: 'completed', transcript: 'grüezi mitenand', word_match_rate: 0.2, latency_ms: 300 },
  ]
  state.resultSort = 'worst'
  assert.equal(context.visibleResults(fixture).map((r) => r.id).join(), '3,2,1')
  state.resultSort = 'best'
  assert.equal(context.visibleResults(fixture).map((r) => r.id).join(), '2,3,1')
  state.resultSort = 'slowest'
  assert.equal(context.visibleResults(fixture).map((r) => r.id).join(), '3,2,1')
  state.resultSearch = ' GRÜEZI '
  state.resultStatus = 'completed'
  assert.equal(context.visibleResults(fixture).length, 2)
  state.resultSearch = 'quota'
  state.resultStatus = 'failed'
  assert.equal(context.visibleResults(fixture)[0].id, 1)
})

test('result pages contain at most 50 rows and clamp when the filter shrinks', () => {
  const { context, state } = workbench()
  const fixture = run()
  fixture.results = Array.from({ length: 123 }, (_, id) => ({ id, model_id: 'm', item_id: 'clip', status: 'completed' }))
  assert.match(context.resultPagination(fixture), /1–50 of 123/)
  assert.equal((context.resultRows(fixture).match(/<tr /g) ?? []).length, 50)
  state.resultPage = 3
  assert.match(context.resultPagination(fixture), /101–123 of 123/)
  assert.equal((context.resultRows(fixture).match(/<tr /g) ?? []).length, 23)
  state.resultSearch = 'not present'
  assert.match(context.resultPagination(fixture), /0–0 of 0/)
  assert.equal(state.resultPage, 1)
  assert.match(context.resultRows(fixture), /No results match/)
})

test('filters reset fully and result error text is escaped even with transcripts hidden', () => {
  const { context, state } = workbench()
  state.resultSearch = 'quota'
  state.resultPage = 4
  state.resultStatus = 'failed'
  context.resetResultFilters()
  assert.equal(state.resultSearch, '')
  assert.equal(state.resultStatus, 'all')
  assert.equal(state.resultPage, 1)
  state.selectedMetrics = ['match']
  const fixture = run()
  fixture.results = [{ id: 1, item_id: 'missing', status: 'failed', error: '<script>bad</script>' }]
  const html = context.resultRows(fixture)
  assert.match(html, /&lt;script&gt;bad&lt;\/script&gt;/)
  assert.match(html, /Audio no longer available" disabled/)
})

test('history searches task, strategy, model, and run ID while filtering failures', () => {
  const { context, state } = workbench()
  state.models = [{ id: 'm', label: 'My model' }]
  state.runs = [run('healthy'), { ...run('errors'), failed_result_count: 2 }, run('paused', 'paused')]
  for (const query of ['my model', 'dialect', 'guided']) {
    state.historySearch = query
    assert.equal(context.filteredRuns().length, 3)
  }
  state.historySearch = ''
  state.historyStatus = 'errors'
  assert.equal(context.filteredRuns()[0].id, 'errors')
  state.historyStatus = 'attention'
  assert.equal(context.filteredRuns()[0].id, 'paused')
  state.historySearch = 'healthy'
  assert.equal(context.filteredRuns().length, 0)
})

test('active run tracking does not confuse a terminal detail with a stale summary', () => {
  const { context, state } = workbench()
  state.runs = [run('a', 'running'), run('b', 'paused')]
  state.activeRun = run('a', 'completed')
  assert.equal(context.pendingRuns().length, 1)
  assert.equal(context.pendingRuns()[0].id, 'b')
})

test('latest requested run wins when detail requests finish out of order', async () => {
  const { context, state } = workbench()
  const first = deferred()
  context.api = (url) => url.endsWith('/a') ? first.promise : Promise.resolve(run('b'))
  const old = context.openRun('a')
  await context.openRun('b')
  first.resolve(run('a'))
  await old
  assert.equal(state.activeRun.id, 'b')
  assert.equal(context.location.hash, '#results/b')
})

test('navigating away invalidates an in-flight run-opening request', async () => {
  const { context, state } = workbench()
  const response = deferred()
  context.api = () => response.promise
  const opening = context.openRun('a')
  context.setView('history')
  response.resolve(run('a'))
  await opening
  assert.equal(state.view, 'history')
  assert.equal(state.activeRun, null)
})

test('run links restore details without submitting another benchmark', async () => {
  const { context, state } = workbench()
  context.location.hash = '#results/saved%20run'
  context.api = async (url, init) => {
    assert.equal(url, '/api/runs/saved%20run')
    assert.equal(init, undefined)
    return run('saved run')
  }
  await context.restoreRoute()
  assert.equal(state.activeRun.id, 'saved run')
  assert.equal(state.view, 'results')
})

test('polling waits for each response and ignores stale results after switching runs', async () => {
  const bench = workbench()
  const { context, state, timers } = bench
  const response = deferred()
  state.activeRun = run('a', 'running')
  context.api = () => response.promise
  context.schedulePoll()
  const pending = bench.tick()
  assert.equal(timers.size, 0)
  state.activeRun = run('b')
  context.schedulePoll()
  response.resolve(run('a', 'running'))
  await pending
  assert.equal(state.activeRun.id, 'b')
  assert.equal(timers.size, 0)
})

test('poll failures are visible, retry automatically, and clear on recovery', async () => {
  const bench = workbench()
  const { context, state, timers } = bench
  state.activeRun = run('a', 'paused')
  context.api = async () => { throw new Error('Offline') }
  context.schedulePoll()
  await bench.tick()
  assert.match(state.pollError, /Retrying automatically.*Offline/)
  assert.equal(timers.size, 1)
  context.api = async () => run('a', 'paused')
  await bench.tick()
  assert.equal(state.pollError, '')
  assert.match(state.message, /paused/)
  assert.equal(timers.size, 1)
})

test('an accepted submission stays tracked if workspace refresh fails', async () => {
  const { context, state, timers } = workbench()
  state.selectedModelIds = ['m']
  state.selectedItemIds = ['clip']
  let submissions = 0
  context.api = async (url, init) => {
    if (init?.method === 'POST') { submissions++; return { id: 'accepted', status: 'queued' } }
    throw new Error('Connection interrupted')
  }
  await context.startRun()
  assert.equal(state.activeRun.id, 'accepted')
  assert.equal(context.location.hash, '#results/accepted')
  assert.equal(timers.size, 1)
  assert.match(state.error, /Connection interrupted/)
  await context.startRun()
  assert.equal(submissions, 1)
})

test('structured API validation errors are readable', async () => {
  const { context } = workbench()
  context.fetch = async () => ({
    ok: false, status: 422,
    json: async () => ({ detail: [{ loc: ['body', 'prompt'], msg: 'Field required' }] }),
  })
  await assert.rejects(context.api('/api/runs'), /prompt: Field required/)
})

test('setup guidance distinguishes model definitions from verified connections', () => {
  const { context } = workbench()
  const html = context.gettingStarted()
  assert.match(html, /not verified connections/)
  assert.match(html, /may incur charges/)
  assert.match(html, /data-detail-key="getting-started" open/)
})

test('poll rendering waits for audio and focused form controls', () => {
  const { context, app } = workbench()
  let renders = 0
  context.render = () => { renders++ }
  app.querySelectorAll = () => [{ paused: false, ended: false }]
  context.renderFromPoll()
  assert.equal(renders, 0)
  app.querySelectorAll = () => []
  app.contains = () => true
  context.document.activeElement = { matches: () => true }
  context.renderFromPoll()
  assert.equal(renders, 0)
  context.document.activeElement = null
  context.renderFromPoll()
  assert.equal(renders, 1)
})

test('control responses cannot replace a different run opened while waiting', async () => {
  const { context, state } = workbench()
  state.activeRun = run('a', 'running')
  const response = deferred()
  context.api = () => response.promise
  const controlling = context.controlRun('pause')
  state.activeRun = run('b')
  response.resolve(run('a', 'paused'))
  await controlling
  assert.equal(state.activeRun.id, 'b')
  assert.equal(state.runControlAction, null)
})

test('reusing a run clears stale source filters without mutating saved parameters', () => {
  const { context, state } = workbench()
  state.activeRun = { ...run(), parameters: { m: { temperature: 0.5 } } }
  state.models = [{ id: 'm' }]
  state.items = [{ id: 'clip', audio_available: true, reference_transcript: 'test', metadata: { dialect: 'ZH' } }]
  state.selectedDataset = 'different source'
  state.search = 'old search'
  context.reuseActiveRun()
  assert.equal(state.selectedDataset, '')
  assert.equal(state.search, '')
  assert.equal(state.selectedItemIds.join(), 'clip')
  state.parameterOverrides.m.temperature = 1
  assert.equal(state.activeRun.parameters.m.temperature, 0.5)
})
