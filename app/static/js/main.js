/* SciFact Verifier — Streaming Chat */

document.addEventListener('DOMContentLoaded', () => {
  const form       = document.getElementById('chatForm');
  const input      = document.getElementById('claimInput');
  const sendBtn    = document.getElementById('sendBtn');
  const messages   = document.getElementById('messages');
  const welcome    = document.getElementById('welcome');
  const modeSelect  = document.getElementById('modeSelect');
  const modelSelect = document.getElementById('modelSelect');

  // ── Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  // ── Enter to submit
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event('submit'));
    }
  });

  // ── Example buttons
  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      input.value = btn.dataset.claim;
      input.dispatchEvent(new Event('input'));
      input.focus();
    });
  });

  // ── Submit — streaming
  form.addEventListener('submit', async e => {
    e.preventDefault();

    const claim = input.value.trim();
    if (!claim) return;

    const mode  = modeSelect.value;
    const model = modelSelect.value;

    if (welcome) welcome.style.display = 'none';

    appendUser(claim);

    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    // Create bot message with loading state
    const botMsg = createBotMessage();
    const contentEl  = botMsg.querySelector('.bot-content');
    const docsEl     = botMsg.querySelector('.docs-container');
    const verdictEl  = botMsg.querySelector('.verdict-container');
    const textEl     = botMsg.querySelector('.generation-text');
    const cursorEl   = botMsg.querySelector('.cursor');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim, mode, model }),
      });

      if (!res.ok) {
        const err = await res.json();
        contentEl.innerHTML = `<div class="error-msg">${esc(err.error || 'Error')}</div>`;
        sendBtn.disabled = false;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE lines
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'retrieval') {
              renderDocs(docsEl, data.documents, data.mode);
            } else if (data.type === 'token') {
              textEl.textContent += data.content;
              scrollBottom();
            } else if (data.type === 'verdict') {
              renderVerdict(verdictEl, data.verdict);
              scrollBottom();
            } else if (data.type === 'done') {
              cursorEl.style.display = 'none';
            }
          } catch (parseErr) {
            // skip malformed JSON
          }
        }
      }

      cursorEl.style.display = 'none';

    } catch (err) {
      contentEl.innerHTML = `<div class="error-msg">Connection error. Is the server running?</div>`;
    }

    sendBtn.disabled = false;
    input.focus();
  });

  // ── Render helpers

  function appendUser(text) {
    const div = document.createElement('div');
    div.className = 'msg msg-user';
    div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    messages.appendChild(div);
    scrollBottom();
  }

  function createBotMessage() {
    const div = document.createElement('div');
    div.className = 'msg msg-bot';
    div.innerHTML = `
      <div class="bubble">
        <div class="bot-content">
          <div class="docs-container"></div>
          <div class="verdict-container"></div>
          <div class="generation-section">
            <div class="section-label">Raw output</div>
            <div class="generation-text"></div><span class="cursor"></span>
          </div>
        </div>
      </div>`;
    messages.appendChild(div);
    scrollBottom();
    return div;
  }

  function renderDocs(container, docs, mode) {
    const count = docs.length;
    const docsId = 'docs-' + Date.now();

    const items = docs.map(doc => `
      <div class="doc-item">
        <div class="doc-header">
          <span class="doc-rank">${doc.rank}</span>
          <span class="doc-title">${esc(doc.title)}</span>
          <span class="doc-score">${doc.score}</span>
        </div>
        <p class="doc-abstract">${esc(doc.abstract.substring(0, 200))}${doc.abstract.length > 200 ? '...' : ''}</p>
      </div>`).join('');

    container.innerHTML = `
      <div class="docs-header">
        <button class="docs-toggle" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          ${count} documents retrieved
          <span class="toggle-icon">&#9660;</span>
        </button>
        <div class="docs-list" id="${docsId}">
          ${items}
        </div>
      </div>
      <div class="docs-meta">
        <span class="meta-tag">${esc(mode)}</span>
      </div>`;

    scrollBottom();
  }

  function renderVerdict(container, v) {
    const verdictClass = {
      'SUPPORTED': 'verdict-supported',
      'REFUTED': 'verdict-refuted',
      'NOT ENOUGH INFO': 'verdict-nei',
    }[v.verdict] || 'verdict-nei';

    const conf = v.confidence != null ? Math.round(v.confidence * 100) : '?';
    const cited = (v.cited_docs || []).map(d => `Doc ${d}`).join(', ') || 'none';
    const evidence = (v.evidence || []).map(e => `<li>${esc(e)}</li>`).join('');

    container.innerHTML = `
      <div class="verdict-card ${verdictClass}">
        <div class="verdict-header">
          <span class="verdict-label">${esc(v.verdict)}</span>
          <span class="verdict-conf">${conf}% confidence</span>
        </div>
        <p class="verdict-explanation">${esc(v.explanation || '')}</p>
        ${evidence ? `<div class="verdict-evidence"><div class="section-label">Key evidence</div><ul>${evidence}</ul></div>` : ''}
        <div class="verdict-cited">Cited: ${esc(cited)}</div>
      </div>`;
  }

  function scrollBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function esc(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  // ── Sidebar metrics ───────────────────────────────────────
  const sidebarMetrics = document.getElementById('sidebarMetrics');

  async function loadMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (!res.ok) return;
      const data = await res.json();
      if (!data || Object.keys(data).length === 0) return;

      const rows = Object.entries(data).map(([name, m]) => {
        const recall = m['Recall@5'] != null ? m['Recall@5'].toFixed(3) : '—';
        const mrr    = m['MRR']      != null ? m['MRR'].toFixed(3)      : '—';
        const ms     = m.latency_ms_per_query != null ? m.latency_ms_per_query.toFixed(0) + 'ms' : '—';
        return `
          <div class="metric-row">
            <div class="metric-name">${esc(name)}</div>
            <div class="metric-stats">
              <span title="Recall@5">R@5 ${recall}</span>
              <span title="Mean Reciprocal Rank">MRR ${mrr}</span>
              <span title="Latency per query">${ms}</span>
            </div>
          </div>`;
      }).join('');

      sidebarMetrics.innerHTML = rows;
    } catch (e) {
      // keep default empty state
    }
  }

  loadMetrics();
});
