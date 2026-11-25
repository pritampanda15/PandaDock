// PandaMap-style 2D interaction diagram.
// The focus molecule is laid out from its own 3D geometry (PCA projection, then stress relaxation
// against true 3D distances), partner residues are placed in the direction they actually sit,
// and the whole map is rotated to match the current 3D camera.
import { TYPES } from './interactions.js';
import { resLabel } from './structure.js';
import { dist, sub, dot, centroid } from './vec.js';

const SCALE = 30;          // px per Å in the ligand drawing
const RES_R = 25;          // residue disc radius, px
const RANK = ['covalent', 'disulfide', 'saltbridge', 'metal', 'hbond', 'halogen', 'cationpi', 'pipi', 'tshape', 'amidepi', 'donorpi', 'water', 'attractive', 'sulfurpi', 'carbonpi', 'hydrophobic', 'repulsion', 'clash'];

/**
 * @param focusAtoms atoms of the focus molecule/residue
 * @param interactions visible interactions (from computeInteractions, already filtered)
 * @param project optional fn(atom) → {x, y} screen coords, used to orient the map like the 3D view
 * @param manual Map(resIndex → {x, y}) of residues the user dragged
 * @param rings aromatic rings of the structure (only those wholly inside the focus are drawn)
 */
export function layoutMap(focusAtoms, interactions, { project, manual, rings = [] } = {}) {
  const atoms = focusAtoms.filter((a) => a.elem !== 'H');
  const n = atoms.length;
  const idx = new Map(atoms.map((a, i) => [a.index, i]));
  const bonds = [];
  for (const a of atoms) for (const j of a.bonds) {
    const k = idx.get(j);
    if (k != null && k > idx.get(a.index)) bonds.push({ i: idx.get(a.index), j: k });
  }

  // --- Ligand coordinates ---
  const inFocus = new Set(atoms);
  const S_rings = rings.filter((r) => r.atoms.every((x) => inFocus.has(x)));
  const pos = pcaProject(atoms);
  const topo = topoDistances(n, bonds);
  relax(atoms, pos, topo, 400);
  // Projection can fold a substituent into its ring; point every ring substituent outward and settle again.
  const ringIdx = S_rings.map((r) => r.atoms.map((a) => idx.get(a.index)));
  const inAnyRing = new Set(ringIdx.flat());
  for (const ring of ringIdx) {
    const c = { x: 0, y: 0 };
    for (const k of ring) { c.x += pos[k].x / ring.length; c.y += pos[k].y / ring.length; }
    for (const k of ring) for (const b of bonds) {
      const other = b.i === k ? b.j : b.j === k ? b.i : null;
      if (other == null || inAnyRing.has(other)) continue;
      const dx = pos[k].x - c.x, dy = pos[k].y - c.y, d = Math.hypot(dx, dy) || 1;
      const L = dist(atoms[k], atoms[other]) * SCALE;
      const nx = pos[k].x + (dx / d) * L, ny = pos[k].y + (dy / d) * L;
      const sx = nx - pos[other].x, sy = ny - pos[other].y;
      // Drag the whole branch beyond `other` along with it.
      for (const m of branch(other, k, bonds, n)) { pos[m].x += sx; pos[m].y += sy; }
    }
  }
  relax(atoms, pos, topo, 250);

  // Rotate (and mirror if needed) so the map matches what the camera shows.
  if (project) {
    const screen = atoms.map(project);
    alignTo(pos, screen);
  }

  const at2 = (a) => pos[idx.get(a.index)];
  const groupCenter2 = (g) => {
    const pts = g.atoms.filter((x) => idx.has(x.index)).map(at2);
    return pts.length ? centroid(pts.map((p) => ({ ...p, z: 0 }))) : null;
  };

  // --- Partner residues ---
  const byRes = new Map();
  for (const [k, it] of interactions.entries()) {
    if (!it.res) continue;
    // Ligand-side endpoint: whichever end belongs to the focus.
    const fp = it.from;
    const anchor2 = fp.atom && idx.has(fp.atom.index) ? at2(fp.atom) : fp.group ? groupCenter2(fp.group) : null;
    if (!anchor2) continue;
    const anchor3 = fp;
    const partner3 = it.to;
    let e = byRes.get(it.res.index);
    if (!e) byRes.set(it.res.index, (e = { res: it.res, links: [], dirs: [], anchors: [] }));
    e.links.push({ k, it, anchor: anchor2, isGroup: !!fp.group });
    e.anchors.push(anchor2);
    e.dirs.push({ a3: anchor3, p3: partner3 });
  }

  const lig = centroid(pos.map((p) => ({ ...p, z: 0 })));
  const residues = [...byRes.values()].map((e) => {
    const anchor = centroid(e.anchors.map((p) => ({ ...p, z: 0 })));
    // Direction: projection of anchor→partner onto the screen when we can, else away from the centre.
    let dx = 0, dy = 0;
    if (project) {
      for (const d of e.dirs) {
        const a = project(d.a3), b = project(d.p3);
        dx += b.x - a.x; dy += b.y - a.y;
      }
    }
    const len = Math.hypot(dx, dy);
    if (len < 1e-3) { dx = anchor.x - lig.x; dy = anchor.y - lig.y; }
    let L = Math.hypot(dx, dy) || 1;
    dx /= L; dy /= L;
    const reach = 2.6 * SCALE + RES_R;
    const want = { x: anchor.x + dx * reach, y: anchor.y + dy * reach };
    const types = [...new Set(e.links.map((l) => l.it.type))].sort((a, b) => RANK.indexOf(a) - RANK.indexOf(b));
    const m = manual?.get(e.res.index);
    return { res: e.res, links: e.links, anchor, want, x: m ? m.x : want.x, y: m ? m.y : want.y, fixed: !!m, types, main: types[0] };
  });
  const obstacles = [
    ...pos.map((p) => ({ x: p.x, y: p.y, r: RES_R + 14 })),
    ...bonds.map((b) => ({ x: (pos[b.i].x + pos[b.j].x) / 2, y: (pos[b.i].y + pos[b.j].y) / 2, r: RES_R + 10 })),
  ];
  for (const r of S_rings) {
    const pts = r.atoms.map(at2).filter(Boolean);
    const c = centroid(pts.map((p) => ({ ...p, z: 0 })));
    obstacles.push({ x: c.x, y: c.y, r: 1.4 * SCALE + RES_R + 8 });
  }
  spreadResidues(residues, obstacles);

  // Bond orders from geometry.
  const aromatic = new Set();
  for (const r of S_rings) for (const a of r.atoms) aromatic.add(a.index);
  const drawnBonds = bonds.map((b) => {
    const A = atoms[b.i], B = atoms[b.j];
    const d = dist(A, B);
    const inArom = aromatic.has(A.index) && aromatic.has(B.index) && S_rings.some((r) => r.atoms.includes(A) && r.atoms.includes(B));
    let order = 1;
    if (!inArom) order = bondOrder(A, B, d);
    return { ...b, order, arom: inArom };
  });
  const ringDraw = S_rings.map((r) => {
    const pts = r.atoms.map(at2).filter(Boolean);
    const c = centroid(pts.map((p) => ({ ...p, z: 0 })));
    const rad = pts.reduce((s, p) => s + Math.hypot(p.x - c.x, p.y - c.y), 0) / pts.length;
    return { c, r: rad, atoms: r.atoms };
  });

  return { atoms, pos, bonds: drawnBonds, rings: ringDraw, residues, interactions };
}

/** Atoms reachable from `start` without crossing back through `from` (stops if it loops into a ring). */
function branch(start, from, bonds, n) {
  const adj = Array.from({ length: n }, () => []);
  for (const b of bonds) { adj[b.i].push(b.j); adj[b.j].push(b.i); }
  const seen = new Set([start]);
  const q = [start];
  for (let h = 0; h < q.length; h++) for (const w of adj[q[h]]) {
    if (w === from && q[h] === start) continue;
    if (w === from) return [start]; // branch closes a ring back to `from`: move only the first atom
    if (!seen.has(w)) { seen.add(w); q.push(w); }
  }
  return [...seen];
}

function bondOrder(A, B, d) {
  const pair = [A.elem, B.elem].sort().join('');
  if ((pair === 'CC' && d < 1.26) || (pair === 'CN' && d < 1.19)) return 3;
  return isDouble(A, B, d) ? 2 : 1;
}

function isDouble(A, B, d) {
  const pair = [A.elem, B.elem].sort().join('');
  if (pair === 'CO') return d < 1.29;
  if (pair === 'CN') return d < 1.31;
  if (pair === 'CC') return d < 1.37;
  if (pair === 'NO') return d < 1.25;
  if (pair === 'OS' || pair === 'OP') return d < 1.52;
  if (pair === 'NN') return d < 1.28;
  return false;
}

function pcaProject(atoms) {
  const c = centroid(atoms);
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const a of atoms) {
    const v = [a.x - c.x, a.y - c.y, a.z - c.z];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += v[i] * v[j];
  }
  const { vecs } = jacobi3(C);
  const e1 = vecs[0], e2 = vecs[1];
  return atoms.map((a) => {
    const v = { x: a.x - c.x, y: a.y - c.y, z: a.z - c.z };
    return { x: dot(v, e1) * SCALE, y: dot(v, e2) * SCALE };
  });
}

function jacobi3(A) {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let s = 0; s < 40; s++) {
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-12) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      for (let k = 0; k < 3; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - sn * y; a[k][q] = sn * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - sn * y; a[q][k] = sn * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = v[k][p], y = v[k][q]; v[k][p] = c * x - sn * y; v[k][q] = sn * x + c * y; }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]);
  return { vecs: order.map((k) => ({ x: v[0][k], y: v[1][k], z: v[2][k] })) };
}

function topoDistances(n, bonds) {
  const adj = Array.from({ length: n }, () => []);
  for (const b of bonds) { adj[b.i].push(b.j); adj[b.j].push(b.i); }
  const D = Array.from({ length: n }, () => new Int16Array(n).fill(99));
  for (let s = 0; s < n; s++) {
    D[s][s] = 0;
    const q = [s];
    for (let h = 0; h < q.length; h++) {
      const u = q[h];
      if (D[s][u] >= 4) continue;
      for (const w of adj[u]) if (D[s][w] === 99) { D[s][w] = D[s][u] + 1; q.push(w); }
    }
  }
  return D;
}

/**
 * Stress relaxation: pairs within three bonds keep their true 3D separation (these are realisable
 * in a plane for planar fragments, so rings come out regular); everything else only repels.
 */
function relax(atoms, pos, D, iters) {
  const n = atoms.length;
  const tgt = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (D[i][j] <= 3) tgt.push([i, j, dist(atoms[i], atoms[j]) * SCALE, D[i][j] === 1 ? 1 : 0.5]);
  }
  const minSep = 2.3 * SCALE;
  for (let it = 0; it < iters; it++) {
    const lr = 0.25 * (1 - it / (iters * 1.12));
    for (const [i, j, t, w] of tgt) {
      const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
      const d = Math.hypot(dx, dy) || 0.01;
      const f = (lr * w * (d - t)) / d / 2;
      pos[i].x += dx * f; pos[i].y += dy * f;
      pos[j].x -= dx * f; pos[j].y -= dy * f;
    }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (D[i][j] <= 2) continue;
      const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
      const d = Math.hypot(dx, dy) || 0.01;
      if (d >= minSep) continue;
      const f = (lr * 0.5 * (d - minSep)) / d / 2;
      pos[i].x += dx * f; pos[i].y += dy * f;
      pos[j].x -= dx * f; pos[j].y -= dy * f;
    }
  }
}

/** Best rotation (allowing a mirror) of `pos` onto screen points, about the centroid. */
function alignTo(pos, screen) {
  const n = pos.length;
  if (n < 2) return;
  const cp = pos.reduce((s, p) => ({ x: s.x + p.x / n, y: s.y + p.y / n }), { x: 0, y: 0 });
  const cs = screen.reduce((s, p) => ({ x: s.x + p.x / n, y: s.y + p.y / n }), { x: 0, y: 0 });
  const fit = (mirror) => {
    let sxx = 0, sxy = 0;
    for (let k = 0; k < n; k++) {
      const px = pos[k].x - cp.x, py = (pos[k].y - cp.y) * (mirror ? -1 : 1);
      const qx = screen[k].x - cs.x, qy = screen[k].y - cs.y;
      sxx += px * qx + py * qy;
      sxy += px * qy - py * qx;
    }
    return { th: Math.atan2(sxy, sxx), score: Math.hypot(sxx, sxy), mirror };
  };
  const a = fit(false), b = fit(true);
  const best = b.score > a.score ? b : a;
  const c = Math.cos(best.th), s = Math.sin(best.th);
  for (const p of pos) {
    const x = p.x - cp.x, y = (p.y - cp.y) * (best.mirror ? -1 : 1);
    p.x = c * x - s * y; p.y = s * x + c * y;
  }
}

/** Push residue discs off the ligand and off each other, while each stays near where it wants to be. */
function spreadResidues(res, obstacles) {
  const blocked = (p) => obstacles.some((o) => Math.hypot(p.x - o.x, p.y - o.y) < o.r);
  // March each target outward along its direction until it sits in free space.
  for (const r of res) {
    const dx = r.want.x - r.anchor.x, dy = r.want.y - r.anchor.y, d = Math.hypot(dx, dy) || 1;
    let k = 0;
    while (blocked(r.want) && k++ < 80) { r.want.x += (dx / d) * 6; r.want.y += (dy / d) * 6; }
    if (!r.fixed) { r.x = r.want.x; r.y = r.want.y; }
  }
  const pushOut = (r, o, strength) => {
    const dx = r.x - o.x, dy = r.y - o.y, d = Math.hypot(dx, dy) || 0.01;
    if (d < o.r) { r.x += (dx / d) * (o.r - d) * strength; r.y += (dy / d) * (o.r - d) * strength; }
  };
  for (let it = 0; it < 300; it++) {
    for (const r of res) {
      if (r.fixed) continue;
      r.x += (r.want.x - r.x) * 0.03;
      r.y += (r.want.y - r.y) * 0.03;
      for (const o of obstacles) pushOut(r, o, 0.6);
    }
    for (let i = 0; i < res.length; i++) for (let j = i + 1; j < res.length; j++) {
      const A = res[i], B = res[j];
      const dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 0.01;
      const min = RES_R * 2 + 10;
      if (d >= min) continue;
      const push = (min - d) / 2;
      const wa = A.fixed ? 0 : B.fixed ? 2 : 1, wb = B.fixed ? 0 : A.fixed ? 2 : 1;
      A.x -= (dx / d) * push * wa; A.y -= (dy / d) * push * wa;
      B.x += (dx / d) * push * wb; B.y += (dy / d) * push * wb;
    }
  }
  // Last word goes to the ligand: nothing may sit on top of it.
  for (let it = 0; it < 30; it++) for (const r of res) if (!r.fixed) for (const o of obstacles) pushOut(r, o, 1);
}

// ---------- Rendering ----------

const ELEM = { N: '#2F62D6', O: '#E5352B', S: '#C99A00', P: '#E07A10', F: '#2E9E48', CL: '#2E9E48', BR: '#9A4A26', I: '#7B3FA0', SE: '#C98A00' };

export function linkGeometry(link, r) {
  const a = link.anchor;
  const dx = r.x - a.x, dy = r.y - a.y, d = Math.hypot(dx, dy) || 1;
  const end = { x: r.x - (dx / d) * RES_R, y: r.y - (dy / d) * RES_R };
  const start = link.isGroup ? a : { x: a.x + (dx / d) * 9, y: a.y + (dy / d) * 9 };
  const mid = { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 };
  return { start, end, mid };
}

/**
 * `compact` drops the legend and background, for the hover bubble.
 * `crop` (a set of focus atoms) frames the view on those atoms and the residues, like a loupe.
 * `fit` ({w, h} in screen px) is the box the SVG will be shown in: text and discs are enlarged
 * by however much the drawing gets shrunk, so labels keep a readable on-screen size.
 */
export function renderMap(model, { dark, labels, title, compact = false, crop = null, fit = null }) {
  const ink = dark ? '#F5F5F7' : '#1D1D1F', ink2 = dark ? '#A1A1A6' : '#6E6E73';
  const bg = dark ? '#1C1C1E' : '#FFFFFF';
  const framed = crop ? model.pos.filter((_, i) => crop.has(model.atoms[i])) : model.pos;
  const xs = [...framed.map((p) => p.x), ...model.residues.map((r) => r.x - RES_R), ...model.residues.map((r) => r.x + RES_R)];
  const ys = [...framed.map((p) => p.y), ...model.residues.map((r) => r.y - RES_R), ...model.residues.map((r) => r.y + RES_R)];
  const pad = compact ? 8 : 26;
  let x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const legendTypes = compact ? [] : [...new Set(model.residues.flatMap((r) => r.types))].sort((a, b) => RANK.indexOf(a) - RANK.indexOf(b));
  const legendH = legendTypes.length ? Math.ceil(legendTypes.length / 3) * 20 + 18 : 0;
  let W = x1 - x0, H = y1 - y0 + legendH;
  // Compact drawings start 30% larger: the loupe is about reading labels at a glance.
  const ts = fit ? Math.min(2.4, Math.max(compact ? 1.3 : 1, W / fit.w, H / fit.h)) : 1;
  const rs = Math.min(ts, 1.7);
  if (rs > 1) {
    // Bigger discs need a little more room at the edges.
    const grow = RES_R * (rs - 1);
    x0 -= grow; x1 += grow; y0 -= grow; y1 += grow;
    W = x1 - x0; H = y1 - y0 + legendH;
  }

  const out = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${x0} ${y0} ${W} ${H}" width="${Math.round(W)}" height="${Math.round(H)}" font-family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif" class="map2d-svg">`);
  if (title) out.push(`<title>${esc(title)}</title>`);
  if (!compact) out.push(`<rect class="map-bg" x="${x0}" y="${y0}" width="${W}" height="${H}" fill="${bg}"/>`);

  // Links first, under everything.
  out.push('<g class="links">');
  for (const [ri, r] of model.residues.entries()) {
    for (const l of r.links) {
      const t = TYPES[l.it.type];
      const { start, end, mid } = linkGeometry(l, r);
      const dash = l.it.type === 'hydrophobic' ? '1.5 4' : l.it.type === 'covalent' || l.it.type === 'disulfide' ? '' : '5 4';
      out.push(`<g class="link" data-i="${l.k}" data-res="${ri}">`);
      out.push(`<line class="hit" x1="${f(start.x)}" y1="${f(start.y)}" x2="${f(end.x)}" y2="${f(end.y)}" stroke="transparent" stroke-width="12"/>`);
      out.push(`<line class="vis" x1="${f(start.x)}" y1="${f(start.y)}" x2="${f(end.x)}" y2="${f(end.y)}" stroke="${t.color}" stroke-width="${f((l.it.type === 'hydrophobic' ? 1.8 : 1.7) * Math.min(ts, 1.6))}" stroke-linecap="round" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`);
      if (l.it.type === 'water') out.push(`<circle class="water" cx="${f(mid.x)}" cy="${f(mid.y)}" r="6.5" fill="#5AC8FA" stroke="${bg}" stroke-width="1.5"/>`);
      if (l.isGroup) out.push(`<circle class="centroid" cx="${f(start.x)}" cy="${f(start.y)}" r="3.2" fill="${t.color}"/>`);
      if (labels && l.it.type !== 'water') {
        const txt = l.it.d.toFixed(1);
        out.push(`<g class="dlabel" transform="translate(${f(mid.x)} ${f(mid.y)})${ts > 1 ? ` scale(${f(ts)})` : ''}"><rect x="-11" y="-7.5" width="22" height="15" rx="7.5" fill="${bg}" opacity="0.92"/><text text-anchor="middle" y="3.6" font-size="10" font-weight="600" fill="${ink2}">${txt}</text></g>`);
      }
      out.push('</g>');
    }
  }
  out.push('</g>');

  // Ligand.
  out.push('<g class="ligand">');
  const P = model.pos;
  const ringOf = (b) => model.rings.find((r) => r.atoms.includes(model.atoms[b.i]) && r.atoms.includes(model.atoms[b.j]));
  for (const b of model.bonds) {
    const p = P[b.i], q = P[b.j];
    out.push(`<line x1="${f(p.x)}" y1="${f(p.y)}" x2="${f(q.x)}" y2="${f(q.y)}" stroke="${ink}" stroke-width="${f(2 * Math.min(ts, 1.5))}" stroke-linecap="round"/>`);
    if (b.order === 3) {
      const dx = q.x - p.x, dy = q.y - p.y, d = Math.hypot(dx, dy) || 1, nx = -dy / d * 4.5, ny = dx / d * 4.5;
      for (const sgn of [1, -1]) out.push(`<line x1="${f(p.x + nx * sgn)}" y1="${f(p.y + ny * sgn)}" x2="${f(q.x + nx * sgn)}" y2="${f(q.y + ny * sgn)}" stroke="${ink}" stroke-width="1.5" stroke-linecap="round"/>`);
    }
    if (b.order === 2) {
      const dx = q.x - p.x, dy = q.y - p.y, d = Math.hypot(dx, dy) || 1;
      let nx = -dy / d, ny = dx / d;
      const ring = ringOf(b);
      const m = { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 };
      if (ring && (ring.c.x - m.x) * nx + (ring.c.y - m.y) * ny < 0) { nx = -nx; ny = -ny; }
      const o = 5, sh = 0.15;
      if (ring) {
        out.push(`<line x1="${f(p.x + nx * o + dx * sh)}" y1="${f(p.y + ny * o + dy * sh)}" x2="${f(q.x + nx * o - dx * sh)}" y2="${f(q.y + ny * o - dy * sh)}" stroke="${ink}" stroke-width="1.6" stroke-linecap="round"/>`);
      } else {
        out.push(`<line x1="${f(p.x + nx * o)}" y1="${f(p.y + ny * o)}" x2="${f(q.x + nx * o)}" y2="${f(q.y + ny * o)}" stroke="${ink}" stroke-width="1.6" stroke-linecap="round"/>`);
      }
    }
  }
  for (const r of model.rings) {
    out.push(`<circle cx="${f(r.c.x)}" cy="${f(r.c.y)}" r="${f(r.r * 0.6)}" fill="none" stroke="${ink}" stroke-width="1.4" opacity="0.8"/>`);
  }
  for (const [i, a] of model.atoms.entries()) {
    const p = P[i];
    const col = ELEM[a.elem];
    const hit = `<circle class="atom-hit" data-atom="${i}" cx="${f(p.x)}" cy="${f(p.y)}" r="9" fill="transparent"/>`;
    if (a.elem === 'C') { out.push(hit); continue; }
    const label = a.elem.charAt(0) + a.elem.slice(1).toLowerCase();
    out.push(`<g class="atom" data-atom="${i}"><circle cx="${f(p.x)}" cy="${f(p.y)}" r="${f((label.length > 1 ? 10 : 8.5) * ts)}" fill="${bg}"/><text x="${f(p.x)}" y="${f(p.y + 4.6 * ts)}" text-anchor="middle" font-size="${f(13.5 * ts)}" font-weight="600" fill="${col || ink}">${label}</text></g>`);
    out.push(hit);
  }
  out.push('</g>');

  // Residues.
  out.push('<g class="residues">');
  for (const [ri, r] of model.residues.entries()) {
    const t = TYPES[r.main];
    const otherChain = model.atoms[0] && r.res.chain !== model.atoms[0].res.chain;
    const name = r.res.kind === 'protein' ? resLabel(r.res) : r.res.resn;
    out.push(`<g class="res" data-res="${ri}" transform="translate(${f(r.x)} ${f(r.y)})${rs > 1 ? ` scale(${f(rs)})` : ''}">
      <circle r="${RES_R}" fill="${mix(t.color, bg, dark ? 0.78 : 0.86)}" stroke="${t.color}" stroke-width="1.8"/>
      <text text-anchor="middle" y="${otherChain ? 1 : 4.2}" font-size="11.5" font-weight="650" fill="${ink}">${esc(name)}</text>
      ${otherChain ? `<text text-anchor="middle" y="13" font-size="9.5" fill="${ink2}">chain ${esc(r.res.chain)}</text>` : ''}
    </g>`);
  }
  out.push('</g>');

  // Legend.
  if (legendTypes.length) {
    const ly = y1 + 6;
    const colW = W / 3;
    out.push(`<g class="legend" font-size="11" fill="${ink2}">`);
    legendTypes.forEach((tp, k) => {
      const x = x0 + 14 + (k % 3) * colW, y = ly + Math.floor(k / 3) * 20 + 8;
      const t = TYPES[tp];
      out.push(`<line x1="${f(x)}" y1="${f(y)}" x2="${f(x + 18)}" y2="${f(y)}" stroke="${t.color}" stroke-width="2" ${tp === 'hydrophobic' ? 'stroke-dasharray="1.5 3.5"' : tp === 'covalent' ? '' : 'stroke-dasharray="4 3"'} stroke-linecap="round"/>`);
      out.push(`<text x="${f(x + 24)}" y="${f(y + 3.8)}">${esc(t.label)}</text>`);
    });
    out.push('</g>');
  }
  out.push('</svg>');
  return out.join('');
}

const f = (v) => (Math.round(v * 10) / 10).toString();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const c = [16, 8, 0].map((s) => Math.round(((pa >> s) & 255) * (1 - t) + ((pb >> s) & 255) * t));
  return '#' + c.map((x) => x.toString(16).padStart(2, '0')).join('');
}

export { RES_R };
