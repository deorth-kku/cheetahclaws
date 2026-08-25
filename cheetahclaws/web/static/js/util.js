/* Small helpers shared across ChatApp modules. */

Object.assign(ChatApp.prototype, {
  _escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  },

  _esc(s) {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
  },

  _fmtRelTime(epochSec) {
    if (!epochSec) return '';
    const diff = Date.now() / 1000 - epochSec;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
    if (diff < 86400 * 7) return `${Math.floor(diff/86400)}d ago`;
    return new Date(epochSec * 1000).toLocaleDateString();
  },

  _scrollBottom() {
    // While steers are queued (dimmed, not-yet-effective) they must always
    // sit at the very bottom — below any assistant output streaming in right
    // now. Re-pinning here (the chokepoint every render calls) keeps them
    // anchored to the bottom until `steer_applied` finalizes + locks them.
    if (typeof this._keepSteersAtBottom === 'function') this._keepSteersAtBottom();
    const el = document.getElementById('messages');
    requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  },

  _renderMd(text) {
    try {
      // Strip raw HTML tags before passing to marked, so that model output
      // can't inject <script>/<img onerror> through markdown.
      const clean = (text || '').replace(/<\/?[a-zA-Z][^>]*>/g, (tag) =>
        tag.replace(/</g, '&lt;').replace(/>/g, '&gt;')
      );
      return marked.parse(clean, {breaks: true});
    } catch(e) { return this._esc(text); }
  },
});
