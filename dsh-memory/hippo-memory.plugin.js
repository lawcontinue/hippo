/**
 * hippo-memory — cross-session memory tools for DeepSeek Harness (DSH) agents,
 * backed by the hippo VectorStore (BM25 + bge-small-zh hybrid retrieval).
 *
 * Static Cordis plugin intended to travel with an agent preset. Spawns a
 * persistent Python bridge (the embedding model stays warm in memory) and
 * registers five model-facing tools into the host `tools` registry:
 *
 *   memory_store    write a durable fact (source-graded, confidence-weighted)
 *   memory_recall   hybrid retrieval: BM25 + dense vectors, RRF fusion,
 *                   confidence-weighted ranking
 *   memory_forget   delete one memory by id
 *   memory_list     browse recently written memories
 *   memory_rebuild  re-embed memories that predate hybrid mode
 *
 * Consumes host services only (`tools`, `subprocess`) and publishes nothing,
 * so it needs no realm. Configuration via environment variables:
 *
 *   HIPPO_DSH_DIR     workspace holding hippo_bridge.py and the Python venv
 *                     (default: this file's directory)
 *   HIPPO_DSH_PYTHON  python executable for the bridge
 *                     (default: <HIPPO_DSH_DIR>/.venv/Scripts/python.exe on
 *                      Windows, <HIPPO_DSH_DIR>/.venv/bin/python elsewhere)
 *
 * The bridge itself is configured through HIPPO_MEMORY_DB, HIPPO_MEMORY_MODE
 * and HIPPO_EMBED_MODEL_PATH — see dsh-memory/hippo_bridge.py.
 */
const name = 'hippo-memory'
const inject = ['tools']

function bridgeDir() {
  return process.env.HIPPO_DSH_DIR || new URL('.', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')
}

function bridgePython() {
  if (process.env.HIPPO_DSH_PYTHON) return process.env.HIPPO_DSH_PYTHON
  const dir = bridgeDir()
  return process.platform === 'win32'
    ? dir + '\\.venv\\Scripts\\python.exe'
    : dir + '/.venv/bin/python'
}

function bridgeScript() {
  const dir = bridgeDir()
  return process.platform === 'win32'
    ? dir + '\\hippo_bridge.py'
    : dir + '/hippo_bridge.py'
}

function textBlock(text) {
  return [{ type: 'text', text: text }]
}

function asString(v, fallback) {
  return typeof v === 'string' ? v : fallback
}

function asNumber(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

const apply = (ctx) => {
  const subprocess = ctx.get('subprocess')
  if (subprocess === undefined) {
    ctx.logger.error('hippo-memory: subprocess service unavailable, plugin disabled')
    return
  }

  // ---- persistent bridge server (model stays warm in memory) ----
  // Protocol: one JSON line per request/response, correlated by `rid`
  // (never `id` — that key belongs to business ops like delete/update).

  let handle = null
  let nextRid = 1
  const pending = new Map()
  let buffer = ''
  const decoder = new TextDecoder('utf-8')
  let readyResolve = null
  let readyReject = null

  function onLine(line) {
    let msg = null
    try { msg = JSON.parse(line) } catch (e) {
      ctx.logger.error('hippo-memory: unparsable bridge line: ' + line.slice(0, 200))
      return
    }
    if (msg.ready) {
      if (readyResolve) { readyResolve(msg); readyResolve = null; readyReject = null }
      return
    }
    if (msg.rid == null) {
      if (readyReject) { readyReject(new Error(msg.error || 'bridge startup failed')); readyReject = null; readyResolve = null }
      return
    }
    const p = pending.get(msg.rid)
    if (p) {
      pending.delete(msg.rid)
      if (msg.ok) p.resolve(msg)
      else p.reject(new Error(msg.error || 'hippo operation failed'))
    }
  }

  function spawnServer() {
    const h = subprocess.spawn({
      argv: [bridgePython(), '-X', 'utf8', bridgeScript(), '--serve'],
      cwd: bridgeDir(),
      stdio: {
        stdin: 'pipe',
        stdout: 'pipe',
        stderr: { maxBytes: 65536 },
      },
      graceMs: 8000,
      env: {
        TRANSFORMERS_OFFLINE: '1',
        HF_HUB_OFFLINE: '1',
        TQDM_DISABLE: '1',
        TOKENIZERS_PARALLELISM: 'false',
      },
    })
    handle = h
    buffer = ''
    const readyPromise = new Promise(function (resolve, reject) {
      readyResolve = resolve
      readyReject = reject
    })
    h.stdout.on('data', function (chunk) {
      buffer += decoder.decode(chunk, { stream: true })
      let idx = buffer.indexOf('\n')
      while (idx >= 0) {
        const line = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 1)
        if (line) onLine(line)
        idx = buffer.indexOf('\n')
      }
    })
    function onExit(outcome) {
      const code = outcome && outcome.exitCode
      const err = new Error('hippo bridge process exited (exit=' + code + ')')
      if (readyReject) { readyReject(err); readyReject = null; readyResolve = null }
      pending.forEach(function (p) { p.reject(err) })
      pending.clear()
      handle = null
      ctx.logger.error('hippo-memory: bridge exited, code=' + code)
    }
    h.done.then(onExit, onExit)
    return readyPromise
  }

  const serverReady = spawnServer()
  serverReady.then(function (info) {
    ctx.logger.info('hippo-memory: bridge ready, mode=' + info.mode + ', db=' + info.db)
  }, function (e) {
    ctx.logger.error('hippo-memory: bridge failed to start: ' + e.message)
  })

  function bridge(cmd) {
    return serverReady.then(function () {
      const current = handle
      if (!current || !current.stdin) throw new Error('hippo bridge not running')
      const rid = nextRid++
      cmd.rid = rid
      return new Promise(function (resolve, reject) {
        pending.set(rid, { resolve: resolve, reject: reject })
        current.stdin.write(JSON.stringify(cmd) + '\n', 'utf8', function (err) {
          if (err) { pending.delete(rid); reject(err) }
        })
      })
    })
  }

  ctx.effect(function () {
    return function () {
      if (handle) { handle.terminate(); handle = null }
    }
  }, 'hippo-memory: bridge-server')

  // ---- model-facing tools (raw ToolDefinition objects) ----

  ctx.tools.register({
    name: 'memory_store',
    description: '将值得跨会话记住的事实写入全局长期记忆库（hippo 本地检索，所有会话共享）。适合记录：用户偏好与习惯、项目结构与关键决策、常用路径/命令/配置、用户明确要求记住的内容。不要记录：一次性闲聊、可直接从代码读取的事实、敏感凭据。用户亲口陈述的事实用 source=user 且置信度 0.9；模型推断用 source=inference 且较低置信度。',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: '要记住的事实，一句完整、自包含的陈述。' },
        source: { type: 'string', enum: ['user', 'model', 'inference'], description: '来源：user=用户亲口所述，model=模型总结，inference=模型推断。默认 model。' },
        confidence: { type: 'number', description: '置信度 0-1，可选；缺省按来源取默认值。' },
        tags: { type: 'array', items: { type: 'string' }, description: '可选标签，如 ["偏好", "项目"]。' },
      },
      required: ['text'],
    },
    output: {
      schema: {},
      render: function (args, value) {
        return textBlock('已记住 #' + value.id + '（当前共 ' + value.count + ' 条）：' + args.text)
      },
    },
    timeoutMs: 90000,
    async execute(args, exec) {
      const a = args || {}
      const sessionId = exec && exec.agent && exec.agent.id ? String(exec.agent.id) : undefined
      return bridge({
        op: 'store',
        text: asString(a.text, '').trim(),
        source: asString(a.source, 'model'),
        confidence: asNumber(a.confidence),
        tags: Array.isArray(a.tags) ? a.tags.filter(function (t) { return typeof t === 'string' }) : undefined,
        session: sessionId,
      })
    },
  })

  ctx.tools.register({
    name: 'memory_recall',
    description: '在全局长期记忆库中检索与查询相关的记忆（BM25 + bge-small-zh 语义向量 RRF 融合，置信度加权）。当用户的问题可能依赖之前会话的上下文（偏好、项目决定、历史事实、环境配置）时调用；回答涉及"我之前/上次/以前"时必先调用。',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '检索查询，用关键词短语而非整句。' },
        top_k: { type: 'integer', description: '返回条数，默认 5。' },
        min_confidence: { type: 'number', description: '最低置信度过滤 0-1，默认 0。' },
      },
      required: ['query'],
    },
    output: {
      schema: {},
      render: function (args, value) {
        const rs = value.results || []
        if (!rs.length) return textBlock('记忆库中没有找到与「' + args.query + '」相关的记忆。')
        const lines = rs.map(function (r) {
          return '#' + r.id + ' [' + r.source + ', conf=' + r.confidence + '] ' + r.text
        })
        return textBlock('找到 ' + rs.length + ' 条相关记忆：\n' + lines.join('\n'))
      },
    },
    timeoutMs: 90000,
    isConcurrencySafe: function () { return true },
    async execute(args) {
      const a = args || {}
      const topK = asNumber(a.top_k)
      const minConf = asNumber(a.min_confidence)
      return bridge({
        op: 'recall',
        query: asString(a.query, '').trim(),
        top_k: topK == null ? 5 : Math.max(1, Math.floor(topK)),
        min_confidence: minConf == null ? 0 : minConf,
      })
    },
  })

  ctx.tools.register({
    name: 'memory_forget',
    description: '按 id 删除一条长期记忆。仅在记忆错误、过期或用户明确要求忘记时使用。',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'integer', description: '记忆条目的 id（memory_recall / memory_list 返回的 #id）。' },
      },
      required: ['id'],
    },
    output: {
      schema: {},
      render: function (args, value) {
        return textBlock('已删除记忆 #' + args.id + '（剩余 ' + value.count + ' 条）。')
      },
    },
    timeoutMs: 90000,
    async execute(args) {
      const a = args || {}
      const id = asNumber(a.id)
      if (id == null) throw new Error('memory_forget 需要整数 id')
      return bridge({ op: 'delete', id: Math.floor(id) })
    },
  })

  ctx.tools.register({
    name: 'memory_list',
    description: '列出长期记忆库中最近写入的记忆条目（按 id 倒序），用于浏览或定位要删除/更新的记忆。',
    parameters: {
      type: 'object',
      properties: {
        limit: { type: 'integer', description: '最多返回条数，默认 20。' },
        offset: { type: 'integer', description: '跳过条数，默认 0。' },
      },
    },
    output: {
      schema: {},
      render: function (args, value) {
        const items = value.items || []
        if (!items.length) return textBlock('记忆库为空。')
        const lines = items.map(function (m) {
          return '#' + m.id + ' [' + m.source + ', conf=' + m.confidence + '] ' + m.text
        })
        return textBlock('记忆库共 ' + value.total + ' 条，最近 ' + items.length + ' 条：\n' + lines.join('\n'))
      },
    },
    timeoutMs: 90000,
    isConcurrencySafe: function () { return true },
    async execute(args) {
      const a = args || {}
      const limit = asNumber(a.limit)
      const offset = asNumber(a.offset)
      return bridge({
        op: 'list',
        limit: limit == null ? 20 : Math.max(1, Math.floor(limit)),
        offset: offset == null ? 0 : Math.max(0, Math.floor(offset)),
      })
    },
  })

  ctx.tools.register({
    name: 'memory_rebuild',
    description: '为缺少语义向量的记忆条目重建向量（sparse 时代写入的旧记忆需要一次重建才能参与语义检索）。',
    parameters: {
      type: 'object',
      properties: {},
    },
    output: {
      schema: {},
      render: function (args, value) {
        return textBlock('已为 ' + value.rebuilt + ' 条记忆重建语义向量。')
      },
    },
    timeoutMs: 120000,
    async execute() {
      return bridge({ op: 'rebuild' })
    },
  })

  ctx.logger.info('hippo-memory: 5 memory tools registered')
}

export { name, inject, apply }
