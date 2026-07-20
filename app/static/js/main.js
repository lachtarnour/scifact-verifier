/* SciFact Verifier - streaming chat */

document.addEventListener('DOMContentLoaded', () => {
  const app = document.querySelector('.app');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('claimInput');
  const sendBtn = document.getElementById('sendBtn');
  const messages = document.getElementById('messages');
  const welcome = document.getElementById('welcome');
  const modeSelect = document.getElementById('modeSelect');
  const modelSelect = document.getElementById('modelSelect');
  const inputMeta = document.getElementById('inputMeta');
  const configBtn = document.getElementById('configBtn');
  const configPopover = document.getElementById('configPopover');
  const randomClaimBtn = document.getElementById('randomClaimBtn');

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 132)}px`;
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event('submit'));
    }
  });

  randomClaimBtn.addEventListener('click', async () => {
    closeConfigPopover();
    randomClaimBtn.disabled = true;

    try {
      const res = await fetch('/api/random-claim');
      if (!res.ok) throw new Error('Random claim failed');

      const data = await res.json();
      input.value = data.claim || '';
      input.dispatchEvent(new Event('input'));
      input.focus();
      inputMeta.textContent = 'Random SciFact claim';
    } catch (err) {
      inputMeta.textContent = 'Unable to generate a claim';
    } finally {
      randomClaimBtn.disabled = false;
    }
  });

  configBtn.addEventListener('click', e => {
    e.stopPropagation();
    const isOpen = !configPopover.hidden;
    configPopover.hidden = isOpen;
    const nextOpen = !isOpen;
    configBtn.setAttribute('aria-expanded', String(nextOpen));
    app.classList.toggle('config-open', nextOpen);
  });

  document.addEventListener('click', e => {
    if (configPopover.hidden) return;
    if (configPopover.contains(e.target) || configBtn.contains(e.target)) return;
    closeConfigPopover();
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !configPopover.hidden) {
      closeConfigPopover();
      configBtn.focus();
    }
  });

  document.querySelectorAll('.option-list').forEach(list => {
    list.addEventListener('click', e => {
      const button = e.target.closest('.option-item');
      if (!button) return;

      const select = document.getElementById(list.dataset.select);
      select.value = button.dataset.value;

      list.querySelectorAll('.option-item').forEach(item => {
        item.classList.toggle('is-selected', item === button);
      });
    });
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();

    const claim = input.value.trim();
    if (!claim) return;

    const mode = modeSelect.value;
    const model = modelSelect.value;

    app.classList.add('has-conversation');
    if (welcome) welcome.hidden = true;

    appendUser(claim);
    closeConfigPopover();

    input.value = '';
    input.style.height = 'auto';
    inputMeta.textContent = 'Developed by Nour Lachtar · © 2026 SciFact Verifier';
    sendBtn.disabled = true;

    const botMsg = createBotMessage();
    const contentEl = botMsg.querySelector('.bot-content');
    const verdictEl = botMsg.querySelector('.verdict-container');
    const stepEl = botMsg.querySelector('.process-step');

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
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'retrieval') {
              scrollBottom();
            } else if (data.type === 'token') {
              scrollBottom();
            } else if (data.type === 'verdict') {
              renderVerdict(verdictEl, data.verdict);
              stepEl.classList.add('is-hidden');
              scrollBottom();
            } else if (data.type === 'done') {
              stepEl.classList.add('is-hidden');
            }
          } catch (parseErr) {
            // Ignore malformed SSE chunks.
          }
        }
      }
    } catch (err) {
      contentEl.innerHTML = '<div class="error-msg">Connection error. Is the server running?</div>';
    }

    sendBtn.disabled = false;
    input.focus();
  });

  function closeConfigPopover() {
    configPopover.hidden = true;
    configBtn.setAttribute('aria-expanded', 'false');
    app.classList.remove('config-open');
  }

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
      <div class="bubble response-card">
        <div class="bot-content response-body">
          <div class="process-step">
            <span class="process-spinner" aria-hidden="true"></span>
            <span class="process-text">Retrieving documents...</span>
          </div>
          <div class="verdict-container"></div>
        </div>
      </div>`;
    messages.appendChild(div);
    scrollBottom();
    return div;
  }

  function renderVerdict(container, v) {
    const verdictClass = {
      'SUPPORTED': 'verdict-supported',
      'REFUTED': 'verdict-refuted',
      'NOT ENOUGH INFO': 'verdict-nei',
    }[v.verdict] || 'verdict-nei';

    const conf = v.confidence != null ? Math.round(v.confidence * 100) : '?';
    const evidence = (v.evidence || []).map(e => `<li>${esc(e)}</li>`).join('');

    container.innerHTML = `
      <div class="answer-section">
        <div class="verdict-card ${verdictClass}">
          <div class="verdict-header">
            <span class="verdict-label">${esc(v.verdict)}</span>
            <span class="verdict-conf">${conf}% confidence</span>
          </div>
          <p class="verdict-explanation">${esc(v.explanation || '')}</p>
          ${evidence ? `<div class="verdict-evidence"><div class="section-label">Key evidence</div><ul>${evidence}</ul></div>` : ''}
        </div>
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

  const metricsOpenBtn = document.getElementById('metricsOpenBtn');
  const metricsModal = document.getElementById('metricsModal');
  const metricsCloseBtn = metricsModal.querySelector('.metrics-close-btn');
  const metricsTableBody = document.getElementById('metricsTableBody');
  let previousFocus = null;
  const metricNames = {
    Dense: 'Dense SPECTER2',
    'Hybrid+Reranker': 'Hybrid + Reranker'
  };

  function metricValue(metrics, key) {
    return metrics[key] != null ? metrics[key].toFixed(3) : '—';
  }

  function openMetricsModal() {
    previousFocus = document.activeElement;
    metricsModal.hidden = false;
    document.body.classList.add('modal-open');
    metricsCloseBtn.focus();
  }

  function closeMetricsModal() {
    metricsModal.hidden = true;
    document.body.classList.remove('modal-open');
    if (previousFocus) previousFocus.focus();
  }

  metricsOpenBtn.addEventListener('click', openMetricsModal);

  metricsModal.querySelectorAll('[data-close-metrics]').forEach(el => {
    el.addEventListener('click', closeMetricsModal);
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !metricsModal.hidden) {
      closeMetricsModal();
    }
  });

  async function loadMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (!res.ok) return;
      const data = await res.json();
      if (!data || Object.keys(data).length === 0) return;
      const bestNdcg = Math.max(
        ...Object.values(data).map(metrics => metrics['nDCG@10'] ?? -Infinity)
      );

      const rows = Object.entries(data).map(([name, m]) => {
        const displayName = metricNames[name] || name;
        const latency = m.latency_ms_per_query != null
          ? `${m.latency_ms_per_query.toFixed(0)}ms`
          : '—';
        const isBest = m['nDCG@10'] === bestNdcg;

        return `
          <tr class="${isBest ? 'is-best' : ''}">
            <td>
              <span>${esc(displayName)}</span>
              ${isBest ? '<span class="metric-best">Best</span>' : ''}
            </td>
            <td>${metricValue(m, 'Recall@1')}</td>
            <td>${metricValue(m, 'Recall@5')}</td>
            <td>${metricValue(m, 'Recall@10')}</td>
            <td>${metricValue(m, 'MRR')}</td>
            <td>${metricValue(m, 'nDCG@10')}</td>
            <td>${latency}</td>
          </tr>`;
      }).join('');

      metricsTableBody.innerHTML = rows;
    } catch (e) {
      // Keep default empty state.
    }
  }

  loadMetrics();
});
