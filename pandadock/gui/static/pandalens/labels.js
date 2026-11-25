// Residue names drawn over the 3D view as HTML, so they can be dragged clear of the sticks and
// exported at any resolution. Each label follows its 3D anchor; a dragged label keeps its offset
// and grows a leader line back to the residue.

const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif';
const LEADER_FROM = 16; // px of offset before a leader line is drawn

export class LabelLayer {
  constructor(el, v3d, { onChange } = {}) {
    this.el = el;
    this.v = v3d;
    this.items = [];
    this.offsets = new Map();
    this.onChange = onChange;
    el.innerHTML = '<svg class="leaders" aria-hidden="true"></svg>';
    this.svg = el.firstChild;
    this.nodes = new Map();
    this.lastView = '';
    const tick = () => {
      if (this.items.length) {
        const view = this.v.getView().join(',') + `|${innerWidth}x${innerHeight}`;
        if (view !== this.lastView) { this.lastView = view; this.place(); }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    this.bindDrag();
  }

  /** items: { key, text, pos: {x,y,z}, color, strong } */
  set(items) {
    this.items = items;
    const keep = new Set(items.map((i) => i.key));
    for (const [k, n] of this.nodes) if (!keep.has(k)) { n.remove(); this.nodes.delete(k); }
    for (const it of items) {
      let n = this.nodes.get(it.key);
      if (!n) {
        n = document.createElement('div');
        n.className = 'rlabel';
        n.dataset.key = it.key;
        n.title = 'Drag to move. Double-click to put it back.';
        this.el.appendChild(n);
        this.nodes.set(it.key, n);
      }
      n.textContent = it.text;
      n.classList.toggle('strong', !!it.strong);
      n.style.color = it.color || '';
    }
    this.place();
  }

  offsetOf(key) { return this.offsets.get(key) || { dx: 0, dy: 0 }; }

  /** Screen geometry of every label, in CSS pixels. */
  layout() {
    return this.items.map((it) => {
      const s = this.v.modelToScreen(it.pos);
      const o = this.offsetOf(it.key);
      return { it, ax: s.x, ay: s.y, x: s.x + o.dx, y: s.y + o.dy, leader: Math.hypot(o.dx, o.dy) > LEADER_FROM };
    });
  }

  place() {
    let lines = '';
    for (const L of this.layout()) {
      const n = this.nodes.get(L.it.key);
      if (!n) continue;
      n.style.transform = `translate(${L.x}px, ${L.y}px) translate(-50%, -50%)`;
      if (L.leader) lines += `<line x1="${L.ax}" y1="${L.ay}" x2="${L.x}" y2="${L.y}"/><circle cx="${L.ax}" cy="${L.ay}" r="2.2"/>`;
    }
    this.svg.innerHTML = lines;
  }

  bindDrag() {
    let drag = null;
    this.el.addEventListener('pointerdown', (e) => {
      const n = e.target.closest('.rlabel');
      if (!n) return;
      e.preventDefault();
      const o = this.offsetOf(n.dataset.key);
      drag = { key: n.dataset.key, x: e.clientX, y: e.clientY, dx: o.dx, dy: o.dy, node: n };
      n.setPointerCapture(e.pointerId);
      n.classList.add('dragging');
    });
    this.el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      this.offsets.set(drag.key, { dx: drag.dx + e.clientX - drag.x, dy: drag.dy + e.clientY - drag.y });
      this.place();
    });
    const end = () => {
      if (!drag) return;
      drag.node.classList.remove('dragging');
      drag = null;
      this.onChange?.();
    };
    this.el.addEventListener('pointerup', end);
    this.el.addEventListener('pointercancel', end);
    this.el.addEventListener('dblclick', (e) => {
      const n = e.target.closest('.rlabel');
      if (!n) return;
      this.offsets.delete(n.dataset.key);
      this.place();
      this.onChange?.();
    });
  }

  /** Paint the labels onto an export canvas. `k` is output pixels per CSS pixel. */
  paint(g, k, { dark }) {
    const ink = dark ? '#F5F5F7' : '#1D1D1F';
    const bg = dark ? 'rgba(28,28,30,0.82)' : 'rgba(255,255,255,0.86)';
    const lay = this.layout();
    g.save();
    g.lineWidth = 1.1 * k;
    g.strokeStyle = dark ? 'rgba(245,245,247,0.7)' : 'rgba(29,29,31,0.6)';
    g.fillStyle = g.strokeStyle;
    for (const L of lay) {
      if (!L.leader) continue;
      g.beginPath(); g.moveTo(L.ax * k, L.ay * k); g.lineTo(L.x * k, L.y * k); g.stroke();
      g.beginPath(); g.arc(L.ax * k, L.ay * k, 2.2 * k, 0, Math.PI * 2); g.fill();
    }
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    for (const L of lay) {
      const size = (L.it.strong ? 15 : 13) * k;
      g.font = `${L.it.strong ? 700 : 600} ${size}px ${FONT}`;
      const w = g.measureText(L.it.text).width + 10 * k, h = size + 7 * k;
      const x = L.x * k - w / 2, y = L.y * k - h / 2, r = 6 * k;
      g.fillStyle = bg;
      g.beginPath();
      g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
      g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); g.fill();
      g.fillStyle = L.it.color && !(dark === false && L.it.color === '#F5F5F7') ? L.it.color : ink;
      g.fillText(L.it.text, L.x * k, L.y * k + 0.5 * k);
    }
    g.restore();
  }
}
