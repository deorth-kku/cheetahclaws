/* ChatApp — core class, constructor, send/WS/streaming/event dispatch.
 *
 * Other /static/js modules extend ChatApp.prototype via Object.assign, so
 * this file must load FIRST (after marked.min.js). The global `app` instance
 * is created in init.js once all mixins have registered their methods.
 */

class ChatApp {
  constructor() {
    this.sessionId = null;
    this.ws = null;
    this.streaming = false;
    this._textBuf = '';
    this._curMsgEl = null;
    this._thinkEl = null;
    this._toolCards = {};
    this._toolCounter = 0;
    this._approvalEl = null;
    this._activityEl = null;
    this._pendingApproval = false;
    this._askEl = null;
    this._pendingAsk = false;
    this._authed = false;
    this._authMode = 'login';   // or 'register'
    this._sessions = [];        // last fetched list (for search filter)
    this._user = null;
    this._pendingImage = null;  // {name, dataUrl, b64} awaiting next prompt
  }

  // ── Image attachment ───────────────────────────────────────────

  pickImage() {
    const input = document.getElementById('image-input');
    if (input) input.click();
  }

  // Render a persisted message (user or assistant) in order, preserving
  // interleaving of text / tool / ask blocks. Falls back to the legacy
  // content + tool_calls shape for rows written before blocks existed.
  _renderMessage(m) {
    if (!m) return;
    if (m.role === 'user') {
      this._addUserBubble(m.content || '');
      return;
    }
    // assistant
    const blocks = m.blocks;
    if (Array.isArray(blocks) && blocks.length) {
      // Answers to AskUserQuestion live in the matching tool block's result,
      // which is rendered separately. Collect them so the read-only ask card
      // (history replay) can show the already-given answer, matching the
      // live session exactly.
      const askAnswers = blocks
        .filter(b => b.type === 'tool' && b.name === 'AskUserQuestion')
        .map(b => b.result || '');
      let _ai = 0;
      for (const b of blocks) {
        if (b.type === 'text') {
          if (b.text && b.text.trim()) this._addAssistantBubble(b.text);
        } else if (b.type === 'tool') {
          this._addToolCard(b.name, b.inputs, b.status || 'done',
                            b.result || '', b.tool_id);
        } else if (b.type === 'ask') {
          // History replay (post-refresh): render read-only, never interactive.
          this._addAskHistory({
            prompt: b.prompt || '',
            options: b.options || null,
            allow_freetext: b.allow_freetext !== false,
            answer: askAnswers[_ai++],
          });
        }
      }
      return;
    }
    // Legacy fallback
    if (m.content) this._addAssistantBubble(m.content);
    if (m.tool_calls) m.tool_calls.forEach(tc => {
      this._addToolCard(tc.name, tc.inputs, tc.status, tc.result, tc.tool_id);
    });
  }

  _onImagePicked(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const b64 = dataUrl.split(',')[1] || '';
      this._pendingImage = {name: file.name, dataUrl, b64};
      this._renderAttachPreview();
    };
    reader.readAsDataURL(file);
  }

  _renderAttachPreview() {
    const el = document.getElementById('attach-preview');
    if (!el) return;
    el.innerHTML = '';
    if (!this._pendingImage) return;
    const chip = document.createElement('div');
    chip.className = 'thumb';
    chip.innerHTML =
      `<img src="${this._pendingImage.dataUrl}">` +
      `<span class="name"></span>` +
      `<button class="rm" title="Remove">&times;</button>`;
    chip.querySelector('.name').textContent = this._pendingImage.name;
    chip.querySelector('.rm').onclick = () => {
      this._pendingImage = null;
      this._renderAttachPreview();
    };
    el.appendChild(chip);
  }

  async _uploadPendingImage() {
    if (!this._pendingImage) return;
    const body = JSON.stringify({
      session_id: this.sessionId || '',
      image: this._pendingImage.b64,
    });
    const r = await this._fetchAuth('/api/upload-image', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body,
    });
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      this._addError(data.error || 'Failed to upload image');
      throw new Error('image upload failed');
    }
    const img = this._pendingImage;
    this._pendingImage = null;
    this._renderAttachPreview();
    return img;
  }

  // ── Send prompt ─────────────────────────────────────────────────

  async send() {
    const input = document.getElementById('prompt-input');
    const text = input.value.trim();
    const hasImg = !!this._pendingImage;
    if (!text && !hasImg) return;
    input.value = '';
    input.style.height = 'auto';

    try {
      // Create the session first (if needed) so an attached image binds to
      // the real session id — not the empty "" placeholder.
      if (!this.sessionId) {
        const r = await this._fetchAuth('/api/prompt', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({prompt: '', session_id: ''})
        });
        const data = await r.json();
        if (!r.ok) {
          input.value = text;
          this._addError(data.error || `Server error (${r.status})`);
          return;
        }
        this.sessionId = data.session_id;
        // If user is "in" a folder, drop the auto-created session there.
        const fid = this._getActiveFolderId && this._getActiveFolderId();
        if (fid) {
          try {
            await this._fetchAuth(
              `/api/sessions/${data.session_id}/folder`, {
                method: 'PATCH',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({folder_id: fid}),
              });
          } catch(e) { /* non-fatal */ }
        }
        this._connectWS(this.sessionId);
        this.loadSessions();
      }

      // Upload any attached image before the prompt so it binds to this turn.
      let sentImg = null;
      if (hasImg) {
        try {
          sentImg = await this._uploadPendingImage();
        } catch(e) {
          input.value = text;
          return;
        }
      }

      this._addUserBubble(text, sentImg);
      this._showActivity('', 'Processing', 'connecting...');
      this._scrollBottom();

      // Slash commands
      if (text.startsWith('/')) {
        const longRunning = ['/brainstorm','/worker','/plan','/agent'];
        const isLong = longRunning.some(c => text === c || text.startsWith(c + ' '));
        if (isLong) {
          this._showActivity('', 'Running', text.split(' ')[0] + '...');
          this._runSlashSSE(text);
        } else {
          this._showActivity('', 'Running', text.split(' ')[0] + '...');
          const r = await this._fetchAuth('/api/prompt', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({prompt: text, session_id: this.sessionId})
          });
          const data = await r.json();
          if (!r.ok) {
            this._removeActivity();
            this._addError(data.error || `Server error (${r.status})`);
            return;
          }
          this._removeActivity();
          (data.events || []).forEach(evt => this._handleEvent(evt));
          if (!this.sessionId) this.sessionId = data.session_id;
        }
        return;
      }

      // Regular prompts — prefer WS
      await this._ensureWS();
      const wsOK = this.ws && this.ws.readyState === 1;
      if (wsOK) {
        this._showActivity('', 'Processing', 'sending to agent...');
        this.ws.send(JSON.stringify({type: 'prompt', prompt: text}));
      } else {
        this._showActivity('', 'Processing', 'sending (http)...');
        const r = await this._fetchAuth('/api/prompt', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({prompt: text, session_id: this.sessionId})
        });
        if (!r.ok) {
          const data = await r.json();
          this._addError(data.error || `Server error (${r.status})`);
          return;
        }
        this._pollForResult();
      }
    } catch(e) {
      input.value = text;
      this._addError('Failed to send: ' + e.message);
    }
  }

  _pollForResult() {
    if (this._polling) return;
    this._polling = true;
    this._pollCount = 0;
    this.setStatus('running');
    this._showActivity('', 'Working', 'waiting for response...');
    const poll = async () => {
      this._pollCount++;
      try {
        const r = await fetch(`/api/sessions/${this.sessionId}`, {credentials:'same-origin'});
        if (!r.ok) { this._polling = false; this._removeActivity(); return; }
        const data = await r.json();
        const secs = this._pollCount * 2;
        this._showActivity('', 'Working',
          data.busy ? `running... (${secs}s)` : 'finishing...');
        if (!data.busy) {
          this._polling = false;
          this._removeActivity();
          this.setStatus('idle');
          const msgs = data.messages || [];
          const last = msgs[msgs.length - 1];
          if (last && last.role === 'assistant') {
            this._renderMessage(last);
          }
          this.loadSessions();
          if (this.sessionId && (!this.ws || this.ws.readyState !== 1)) {
            this._connectWS(this.sessionId);
          }
          return;
        }
      } catch(e) { /* ignore */ }
      if (this._polling) setTimeout(poll, 2000);
    };
    setTimeout(poll, 2000);
  }

  // ── WebSocket ────────────────────────────────────────────────────

  _connectWS(sid) {
    this._disconnectWS();
    this._wsRetries = (this._wsRetries || 0);
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/api/events`;
    try {
      this.ws = new WebSocket(url);
    } catch(e) {
      console.warn('[chat] WebSocket constructor failed:', e);
      this.setStatus('no-ws');
      return;
    }
    this._wsSessionId = sid;

    this.ws.onopen = () => {
      this._wsRetries = 0;
      this.ws.send(JSON.stringify({session_id: sid}));
      this.setStatus('connected');
    };
    this.ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.error) {
          console.warn('[chat] WS server error:', data.error);
          return;
        }
        this._handleEvent(data);
      } catch(err) { console.error('[chat] ws parse:', err); }
    };
    this.ws.onclose = (ev) => {
      if (ev.code === 1000) {
        this.setStatus('idle');
        return;
      }
      if (this._wsSessionId && this.sessionId === this._wsSessionId) {
        const delay = Math.min(1000 * Math.pow(2, this._wsRetries), 10000);
        this._wsRetries++;
        this.setStatus(this._wsRetries <= 2 ? 'connecting...' : 'reconnecting...');
        setTimeout(() => {
          if (this.sessionId === this._wsSessionId) {
            this._connectWS(this._wsSessionId);
          }
        }, delay);
      } else {
        this.setStatus('idle');
      }
    };
    this.ws.onerror = () => {};
  }

  _disconnectWS() {
    if (this.ws) { try { this.ws.close(); } catch(e){} this.ws = null; }
  }

  _runSlashSSE(cmd) {
    const body = JSON.stringify({prompt: cmd, session_id: this.sessionId || ''});
    fetch('/api/prompt', {
      method: 'POST',
      credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'Accept': 'text/event-stream'},
      body,
    }).then(response => {
      if (!response.ok) {
        this._removeActivity();
        this._addError(`Server error (${response.status})`);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const processChunk = ({done, value}) => {
        if (done) {
          this._removeActivity();
          this.loadSessions();
          return;
        }
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const evt = JSON.parse(line.slice(6));
              if (evt.type === 'session') {
                if (!this.sessionId) {
                  this.sessionId = evt.data.session_id;
                  this.loadSessions();
                }
              } else if (evt.type === 'done') {
                this._removeActivity();
                this.loadSessions();
              } else {
                this._handleEvent(evt);
              }
            } catch(e) { /* skip bad JSON */ }
          }
        }
        reader.read().then(processChunk);
      };
      reader.read().then(processChunk);
    }).catch(err => {
      this._removeActivity();
      this._addError('Connection error: ' + err.message);
    });
  }

  _ensureWS() {
    return new Promise(resolve => {
      if (this.ws && this.ws.readyState === 1) { resolve(); return; }
      if (!this.ws || this.ws.readyState >= 2) {
        if (this.sessionId) this._connectWS(this.sessionId);
      }
      let elapsed = 0;
      const iv = setInterval(() => {
        elapsed += 50;
        if (this.ws && this.ws.readyState === 1) {
          clearInterval(iv); resolve(); return;
        }
        if (elapsed >= 3000) { clearInterval(iv); resolve(); }
      }, 50);
    });
  }

  // ── Event dispatch ──────────────────────────────────────────────

  _handleEvent(evt) {
    switch (evt.type) {
      case 'text_chunk':
        this._removeActivity();
        if (!this._curMsgEl) this._startAssistantStream();
        this._textBuf += evt.data.text;
        this._renderStream();
        break;
      case 'thinking_chunk':
        this._showActivity('thinking', 'Thinking',
          evt.data.text ? evt.data.text.slice(0, 60) : '');
        break;
      case 'tool_start':
        this._removeActivity();
        this._addToolCard(evt.data.name, evt.data.inputs, 'running', '', evt.data.tool_id);
        this._showActivity('tool-running', `Running ${evt.data.name}`, '');
        break;
      case 'tool_end':
        this._removeActivity();
        this._completeToolCard(evt.data.name, evt.data.result, evt.data.permitted, evt.data.tool_id);
        if (evt.data.name === 'AskUserQuestion') {
          // Show a Processing spinner while the agent processes the answer,
          // mirroring the same UX as a regular user prompt. The spinner is
          // cleared automatically when the agent emits its first event
          // (text_chunk → _startAssistantStream → _removeActivity).
          this._resolveAsk(evt.data);
          this._showActivity('', 'Processing', 'waiting for agent…');
        }
        break;
      case 'permission_request':
        this._removeActivity();
        this._showApproval(evt.data.description);
        break;
      case 'permission_response':
        this._resolveApproval(evt.data.granted);
        break;
      case 'turn_done':
        this._removeActivity();
        this._finishTurn(evt.data.input_tokens, evt.data.output_tokens);
        break;
      case 'status':
        if (evt.data.state === 'running') {
          this.setStatus('running');
          this._showActivity('', 'Processing', '');
        } else if (evt.data.state === 'idle') {
          this._removeActivity();
          this.setStatus('connected');
          this.loadSessions();
        }
        break;
      case 'command_result':
        this._removeActivity();
        this._addCommandResult(evt.data.command, evt.data.output);
        break;
      case 'interactive_menu':
        this._removeActivity();
        this._addInteractiveMenu(evt.data);
        break;
      case 'input_request':
        this._removeActivity();
        this._addInputRequest(evt.data);
        break;
      case 'ask_request':
        this._removeActivity();
        this._addAskRequest(evt.data);
        break;
      case 'ask_response':
        this._resolveAsk(evt.data);
        break;
      case 'error':
        this._removeActivity();
        this._addError(evt.data.message);
        break;
    }
  }

  // ── Message rendering (bubbles + streaming) ────────────────────

  _clearChat() {
    const el = document.getElementById('messages');
    el.innerHTML = '<div style="flex:1"></div>';
    this._curMsgEl = null; this._thinkEl = null; this._activityEl = null;
    this._textBuf = ''; this._toolCards = {};
    this._toolCounter = 0; this._approvalEl = null;
    this._pendingApproval = false;
    this._askEl = null; this._pendingAsk = false;
  }

  _addUserBubble(text, img) {
    const el = document.createElement('div');
    el.className = 'msg user';
    el.innerHTML = `<div class="role-tag">You</div><div class="bubble"></div>`;
    const bubble = el.querySelector('.bubble');
    if (img && img.dataUrl) {
      const imgEl = document.createElement('img');
      imgEl.src = img.dataUrl;
      imgEl.style.cssText =
        'display:block;max-width:240px;max-height:240px;border-radius:8px;' +
        'margin-bottom:6px;';
      bubble.appendChild(imgEl);
    }
    if (text) bubble.appendChild(document.createTextNode(text));
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  }

  _addAssistantBubble(content) {
    const el = document.createElement('div');
    el.className = 'msg assistant';
    el.innerHTML = `<div class="role-tag">Assistant</div><div class="bubble"></div>`;
    el.querySelector('.bubble').innerHTML = this._renderMd(content);
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  }

  _startAssistantStream() {
    this._removeActivity();
    this._textBuf = '';
    const el = document.createElement('div');
    el.className = 'msg assistant';
    el.innerHTML = `<div class="role-tag">Assistant</div><div class="bubble"></div>`;
    document.getElementById('messages').appendChild(el);
    this._curMsgEl = el.querySelector('.bubble');
    this.streaming = true;
  }

  _renderStream() {
    if (!this._curMsgEl) return;
    if (!this._rafPending) {
      this._rafPending = true;
      requestAnimationFrame(() => {
        this._rafPending = false;
        if (this._curMsgEl) {
          this._curMsgEl.innerHTML = this._renderMd(this._textBuf);
          this._scrollBottom();
        }
      });
    }
  }

  _finishTurn(tokIn, tokOut) {
    this._removeActivity();
    this.streaming = false;
    this._curMsgEl = null;
    if (tokIn || tokOut) {
      const meta = document.createElement('div');
      meta.className = 'turn-meta';
      meta.textContent = `${(tokIn||0).toLocaleString()} tokens in / ${(tokOut||0).toLocaleString()} tokens out`;
      document.getElementById('messages').appendChild(meta);
    }
    this._scrollBottom();
  }

  _addError(msg) {
    const el = document.createElement('div');
    el.style.cssText = 'color:var(--red);font-size:13px;padding:8px 12px;background:var(--red-dim);border-radius:var(--radius-sm);margin:8px 0;max-width:min(640px,90%)';
    el.textContent = msg;
    document.getElementById('messages').appendChild(el);
    this._scrollBottom();
  }

  setStatus(state) {
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    dot.className = 'dot' + (state==='disconnected'?' off':'') + (state==='running'?' busy':'');
    txt.textContent = state;
    this.setRunning(state === 'running');
  }

  // ── Send / Stop toggle ───────────────────────────────────────────

  setRunning(running) {
    const btn = document.getElementById('send-btn');
    if (!btn) return;
    if (running) {
      btn.textContent = 'Stop';
      btn.classList.add('stop');
      btn.dataset.mode = 'stop';
    } else {
      btn.textContent = 'Send';
      btn.classList.remove('stop');
      btn.dataset.mode = 'send';
    }
  }

  onSendOrStop() {
    const btn = document.getElementById('send-btn');
    if (btn && btn.dataset.mode === 'stop') {
      this.stop();
    } else {
      this.send();
    }
  }

  stop() {
    this.setRunning(false);
    this._showActivity('', 'Stopping', 'requesting stop...');
    const doStop = () => {
      if (this.ws && this.ws.readyState === 1) {
        this.ws.send(JSON.stringify({type: 'stop'}));
      } else {
        this._fetchAuth(`/api/stop?session_id=${encodeURIComponent(this.sessionId || '')}`, {
          method: 'POST',
        }).catch(() => {});
      }
    };
    if (this._authed) {
      if (this.ws && this.ws.readyState === 1) {
        doStop();
      } else {
        this._ensureWS().then(doStop);
      }
    } else {
      doStop();
    }
  }
}
