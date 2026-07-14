/* Tool cards, activity spinner, slash-command results, input requests,
 * interactive menus. Everything that renders agent activity. */

Object.assign(ChatApp.prototype, {

  _addToolCard(name, inputs, status, result, toolId) {
    const id = 'tool-' + (this._toolCounter++);
    const card = document.createElement('details');
    card.className = 'tool-card'
      + (status === 'done' ? ' done' : '')
      + (status === 'denied' ? ' denied' : '');
    card.id = id;
    const inputStr = typeof inputs === 'string'
      ? inputs : JSON.stringify(inputs || {}, null, 2);
    card.innerHTML = `
      <summary>
        ${status === 'running' ? '<div class="spinner"></div>' : ''}
        <span class="tool-name">${this._esc(name)}</span>
        <span class="tool-badge ${status}">${status}</span>
      </summary>
      <div class="tool-body">
        <div class="label">Input</div>
        <pre>${this._esc(inputStr)}</pre>
        ${result ? `<div class="label">Output</div><pre>${this._esc(result)}</pre>` : ''}
      </div>`;
    document.getElementById('messages').appendChild(card);
    // Key by unique per-call id when available so multiple calls of the same
    // tool (e.g. parallel Read) each get their own completed card.  Fall back
    // to a per-(name,counter) key plus a name pointer for calls without an id.
    const key = toolId ? ('__id__' + toolId)
                       : ('__n__' + name + ':' + (this._toolCounter - 1));
    this._toolCards[key] = card;
    if (!toolId) this._toolCards['__last__' + name] = card;
    this._scrollBottom();
  },

  _completeToolCard(name, result, permitted, toolId) {
    const card = toolId ? this._toolCards['__id__' + toolId]
                        : this._toolCards['__last__' + name];
    if (!card) return;
    const status = permitted ? 'done' : 'denied';
    card.className = 'tool-card ' + status;
    const summary = card.querySelector('summary');
    const spinner = summary.querySelector('.spinner');
    if (spinner) spinner.remove();
    const badge = summary.querySelector('.tool-badge');
    badge.className = 'tool-badge ' + status;
    badge.textContent = status;
    if (result) {
      const body = card.querySelector('.tool-body');
      const existing = body.querySelectorAll('.label');
      if (existing.length < 2) {
        body.innerHTML += `<div class="label">Output</div><pre>${this._esc(result)}</pre>`;
      }
    }
  },

  _showActivity(type, label, detail) {
    if (!this._activityEl) {
      this._activityEl = document.createElement('div');
      this._activityEl.className = 'activity-indicator';
      this._activityEl.innerHTML = `
        <div class="ai-spinner"></div>
        <div class="ai-text">
          <span class="ai-label"></span>
          <span class="ai-dots"></span>
          <span class="ai-detail"></span>
        </div>
        <div class="ai-progress"><div class="ai-fill"></div></div>`;
      document.getElementById('messages').appendChild(this._activityEl);
      this._scrollBottom();
    }
    this._activityEl.className = 'activity-indicator' + (type ? ' ' + type : '');
    this._activityEl.querySelector('.ai-label').textContent = label || 'Working';
    const detailEl = this._activityEl.querySelector('.ai-detail');
    if (detail) { detailEl.textContent = detail; detailEl.style.display = ''; }
    else { detailEl.style.display = 'none'; }
    this._scrollBottom();
  },

  _removeActivity() {
    if (this._activityEl) { this._activityEl.remove(); this._activityEl = null; }
    if (this._thinkEl) { this._thinkEl.remove(); this._thinkEl = null; }
  },

  _addInputRequest(data) {
    const el = document.createElement('div');
    el.className = 'msg assistant';
    const uid = 'ir-' + Date.now();
    el.innerHTML = `
      <div class="role-tag" style="color:var(--accent)">Input Required</div>
      <div style="background:var(--surface);border:1px solid var(--accent);border-radius:var(--radius-sm);
        padding:12px 14px;max-width:min(500px,90%);">
        <div style="font-size:13px;color:var(--text);margin-bottom:8px;">${this._esc(data.prompt)}</div>
        <div style="display:flex;gap:6px;">
          <input id="${uid}" type="text" placeholder="${this._esc(data.placeholder || '')}"
            style="flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
            border-radius:var(--radius-sm);padding:6px 10px;font-size:13px;font-family:var(--font);outline:none;"
            onkeydown="if(event.key==='Enter'){document.getElementById('${uid}-go').click()}">
          <button id="${uid}-go" style="background:var(--accent);color:#000;border:none;padding:6px 14px;
            border-radius:var(--radius-sm);font-weight:600;font-size:12px;cursor:pointer;"
            onclick="(function(){
              var v=document.getElementById('${uid}').value.trim();
              var cmd='${data.command}' + (v ? ' ' + v : ' general project improvement');
              document.getElementById('prompt-input').value=cmd;
              app.send();
              this.parentElement.parentElement.style.opacity='0.5';
              this.parentElement.parentElement.style.pointerEvents='none';
            }).call(this)">Go</button>
        </div>
      </div>`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
    setTimeout(() => { const inp = document.getElementById(uid); if (inp) inp.focus(); }, 100);
  },

  _addAskRequest(data) {
    const el = document.createElement('div');
    el.className = 'ask-card';
    const uid = 'ask-' + Date.now();
    const options = data.options || [];
    const allowFree = data.allow_freetext !== false;
    const optHtml = options.map((o, i) =>
      `<button class="ask-opt" data-value="${this._esc(o.value)}"
        style="display:block;width:100%;text-align:left;margin:4px 0;background:var(--surface);
          border:1px solid var(--border);color:var(--text);border-radius:var(--radius-sm);
          padding:8px 12px;font-size:13px;cursor:pointer;font-family:var(--font);"
        onmouseenter="this.style.background='var(--panel)'"
        onmouseleave="this.style.background='var(--surface)'"
        onclick="app._answerAsk(this.getAttribute('data-value'))">
        <span style="color:var(--accent);font-weight:600;margin-right:8px;">${i+1}.</span>
        ${this._esc(o.label)}</button>`).join('');
    el.innerHTML = `
      <div class="ask-hdr">&#10067; Question</div>
      <div class="ask-q">${this._esc(data.prompt)}</div>
      ${optHtml}
      ${allowFree ? `
      <div style="display:flex;gap:6px;margin-top:8px;">
        <input id="${uid}" type="text" placeholder="Type your answer…"
          style="flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
            border-radius:var(--radius-sm);padding:6px 10px;font-size:13px;font-family:var(--font);outline:none;"
          onkeydown="if(event.key==='Enter'){document.getElementById('${uid}-go').click()}">
        <button id="${uid}-go" style="background:var(--accent);color:#000;border:none;padding:6px 14px;
          border-radius:var(--radius-sm);font-weight:600;font-size:12px;cursor:pointer;"
          onclick="app._answerAsk(document.getElementById('${uid}').value)">Send</button>
      </div>` : ''}`;
    document.getElementById('messages').appendChild(el);
    this._askEl = el;
    this._pendingAsk = true;
    this._scrollBottom();
    if (allowFree) {
      setTimeout(() => { const inp = document.getElementById(uid); if (inp) inp.focus(); }, 100);
    }
  },

  // Read-only variant used when replaying a persisted ask block from history
  // (e.g. after a page refresh). The agent already received the answer, so
  // this must NOT be interactive — no buttons/input, no _pendingAsk state.
  // `answer` (taken from the matching AskUserQuestion tool block's result)
  // is shown so the card mirrors the live session exactly.
  _addAskHistory(data) {
    const options = data.options || [];
    const optHtml = options.map((o, i) =>
      `<div style="display:block;width:100%;text-align:left;margin:4px 0;background:var(--surface);
        border:1px solid var(--border);color:var(--text);border-radius:var(--radius-sm);
        padding:8px 12px;font-size:13px;font-family:var(--font);opacity:.8;">
        <span style="color:var(--accent);font-weight:600;margin-right:8px;">${i+1}.</span>
        ${this._esc(o.label)}</div>`).join('');
    const answer = (data.answer || '').toString().trim();
    const answerHtml = answer
      ? `<div style="margin-top:8px;font-size:13px;color:var(--text);">
           <span style="color:var(--green);font-weight:600;margin-right:6px;">✓ Answer:</span>
           <span>${this._esc(answer)}</span></div>`
      : '';
    const el = document.createElement('div');
    el.className = 'ask-card resolved';
    el.innerHTML = `
      <div class="ask-hdr">&#10067; Question</div>
      <div class="ask-q">${this._esc(data.prompt || '')}</div>
      ${optHtml}
      ${answerHtml}`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  },

  _answerAsk(value) {
    if (!this._pendingAsk) return;
    const v = (value || '').trim();
    if (this.ws && this.ws.readyState === 1) {
      this.ws.send(JSON.stringify({type: 'ask_response', value: v}));
    } else {
      fetch('/api/ask-response', {
        method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({session_id: this.sessionId, value: v}),
      }).catch(e => console.error('ask-response:', e));
    }
    // Reflect the chosen answer in the card so it reads as a normal turn.
    if (this._askEl) {
      const qEl = this._askEl.querySelector('.ask-q');
      if (qEl && v) qEl.innerHTML += ` <span style="color:var(--accent);font-weight:600;">→ ${this._esc(v)}</span>`;
      this._askEl.style.opacity = '0.6';
      this._askEl.style.pointerEvents = 'none';
    }
    this._scrollBottom();
  },

  _resolveAsk(data) {
    if (this._askEl) {
      this._askEl.classList.add('resolved');
      this._askEl = null;
    }
    this._pendingAsk = false;
  },

  _addInteractiveMenu(data) {
    const icons = {bulb:'&#128161;',clipboard:'&#128203;',worker:'&#128119;',
      brain:'&#129504;',sparkle:'&#10024;',search:'&#128270;',book:'&#128214;',
      chat:'&#128172;',test:'&#129514;',note:'&#128221;',monitor:'&#128225;',
      robot:'&#129302;'};
    const el = document.createElement('div');
    el.className = 'msg assistant';
    const items = (data.items || []).map(it =>
      `<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;cursor:pointer;
        border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);
        transition:background .15s;"
        onmouseenter="this.style.background='var(--panel)'"
        onmouseleave="this.style.background='var(--surface)'"
        onclick="document.getElementById('prompt-input').value='${it.cmd}';app.send()">
        <span style="font-size:16px;">${icons[it.icon]||'&#9654;'}</span>
        <div>
          <div style="font-size:12px;font-weight:600;color:var(--text);">${this._esc(it.label)}</div>
          <div style="font-size:10px;font-family:var(--mono);color:var(--text-muted);">${this._esc(it.cmd)}</div>
        </div>
      </div>`).join('');
    el.innerHTML = `<div class="role-tag" style="color:var(--accent)">SSJ Developer Mode</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:min(640px,95%);
        margin-top:6px;">${items}</div>`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  },

  _addCommandResult(command, output) {
    const el = document.createElement('div');
    el.className = 'msg assistant';
    el.innerHTML = `<div class="role-tag" style="color:var(--accent)">System</div>
      <div class="bubble" style="background:var(--surface);border:1px solid var(--border);
        border-left:3px solid var(--accent);border-radius:var(--radius-sm);padding:12px 14px;">
        <div style="font-family:var(--mono);font-size:11px;color:var(--accent);margin-bottom:6px;">${this._esc(command)}</div>
        <pre style="white-space:pre-wrap;font-family:var(--mono);font-size:12px;color:var(--text-dim);
          margin:0;background:none;border:none;padding:0;line-height:1.5;">${this._esc(output)}</pre>
      </div>`;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  },
});
