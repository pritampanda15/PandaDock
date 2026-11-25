// A Structure is our own view of a 3Dmol model: residues, chains, a spatial grid and chemistry.
import { residueKind, isAdditive, perceive, AA3 } from './chem.js';
import { dist } from './vec.js';

const CELL = 4.5;

export class Structure {
  constructor(name, model) {
    this.name = name;
    this.model = model;
    this.atoms = [];
    this.residues = [];
    this.build(model.selectedAtoms({}));
  }

  build(raw) {
    const resMap = new Map();
    raw.forEach((r, i) => {
      const a = {
        index: i, src: r, x: r.x, y: r.y, z: r.z,
        elem: (r.elem || '').toUpperCase(), name: (r.atom || '').trim(),
        resn: (r.resn || '').trim().toUpperCase(), resi: r.resi, icode: r.icode || '',
        chain: r.chain || '', het: !!r.hetflag, b: r.b ?? 0, ss: r.ss,
        bonds: (r.bonds || []).slice(),
      };
      // 3Dmol bonds index into the model's atom array; keep that contract.
      if (r.index !== i) this.reindex = true;
      const key = `${a.chain}|${a.resi}|${a.icode}|${a.resn}`;
      let res = resMap.get(key);
      if (!res) {
        res = { key, chain: a.chain, resi: a.resi, icode: a.icode, resn: a.resn, atoms: [], byName: {}, het: a.het };
        resMap.set(key, res);
        this.residues.push(res);
      }
      a.res = res;
      res.atoms.push(a);
      if (!res.byName[a.name]) res.byName[a.name] = a;
      this.atoms.push(a);
    });
    if (this.reindex) {
      const map = new Map(raw.map((r, i) => [r.index, i]));
      for (const a of this.atoms) a.bonds = a.bonds.map((j) => map.get(j)).filter((j) => j != null);
    }

    this.chains = new Map();
    this.residues.forEach((r, i) => {
      r.index = i;
      r.kind = residueKind(r.resn, r.het, r.atoms.length, r.atoms[0].elem);
      r.additive = isAdditive(r.resn);
      r.ca = r.byName.CA || null;
      if (r.kind === 'protein' || r.kind === 'nucleic') {
        if (!this.chains.has(r.chain)) this.chains.set(r.chain, []);
        this.chains.get(r.chain).push(r);
      }
    });
    // Sequential neighbours, only where the backbone is actually connected.
    for (const list of this.chains.values()) {
      for (let k = 0; k < list.length; k++) {
        const r = list[k], p = list[k - 1];
        r.prev = null; r.next = null;
        if (p && p.byName.C && r.byName.N && dist(p.byName.C, r.byName.N) < 2.0) {
          r.prev = p.index; p.next = r.index;
        } else if (p) r.chainBreakBefore = true;
      }
    }
    this.buildGrid();
    const chem = perceive(this);
    this.rings = chem.rings;
    this.charges = chem.charges;
    this.amides = chem.amides;
  }

  buildGrid() {
    this.grid = new Map();
    for (const a of this.atoms) {
      const k = this.cellKey(a.x, a.y, a.z);
      let c = this.grid.get(k);
      if (!c) this.grid.set(k, (c = []));
      c.push(a.index);
    }
  }

  cellKey(x, y, z) {
    return `${Math.floor(x / CELL)},${Math.floor(y / CELL)},${Math.floor(z / CELL)}`;
  }

  /** Atoms within r of point p. */
  near(p, r) {
    const out = [], r2 = r * r, span = Math.ceil(r / CELL);
    const cx = Math.floor(p.x / CELL), cy = Math.floor(p.y / CELL), cz = Math.floor(p.z / CELL);
    for (let i = -span; i <= span; i++)
      for (let j = -span; j <= span; j++)
        for (let k = -span; k <= span; k++) {
          const cell = this.grid.get(`${cx + i},${cy + j},${cz + k}`);
          if (!cell) continue;
          for (const idx of cell) {
            const a = this.atoms[idx];
            const dx = a.x - p.x, dy = a.y - p.y, dz = a.z - p.z;
            if (dx * dx + dy * dy + dz * dz <= r2) out.push(a);
          }
        }
    return out;
  }

  /**
   * Attach hydrogens read from the file to the heavy atom they sit on (nearest within 1.3 Å).
   * A residue that received any is marked hasH, which switches its hydrogen bonds to D–H···A geometry.
   */
  attachHydrogens(points) {
    let n = 0;
    for (const h of points) {
      let best = null, bd = 1.3;
      for (const a of this.near(h, 1.3)) {
        if (a.elem === 'H') continue;
        const d = dist(a, h);
        if (d < bd) { bd = d; best = a; }
      }
      if (!best) continue;
      (best.hs ||= []).push({ x: h.x, y: h.y, z: h.z });
      best.res.hasH = true;
      n++;
    }
    this.hydrogens = (this.hydrogens || 0) + n;
    return n;
  }

  /** Re-read coordinates after a transform was applied to the 3Dmol atoms. */
  syncCoords() {
    for (const a of this.atoms) { a.x = a.src.x; a.y = a.src.y; a.z = a.src.z; }
    this.refreshGeometry();
  }

  /** Rebuild everything derived from coordinates: the grid and the centres of rings and groups. */
  refreshGeometry() {
    this.buildGrid();
    for (const g of [...this.rings, ...this.charges, ...this.amides]) {
      const pts = g.atoms;
      g.center = pts.length === 2
        ? { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2, z: (pts[0].z + pts[1].z) / 2 }
        : avg(pts);
      if (g.normal) g.normal = newell(pts);
    }
  }

  /**
   * Ligands sitting in the same space as `res` are alternatives to it (a docked pose over the
   * co-crystal ligand), not partners, so contacts between them mean nothing. Returns those ligands.
   */
  markAlternatives(res) {
    const found = new Set();
    const hits = new Map();
    for (const a of res.atoms) {
      for (const b of this.near(a, 1.2)) {
        if (b.res === res || b.res.kind !== 'ligand') continue;
        hits.set(b.res, (hits.get(b.res) || 0) + 1);
      }
    }
    for (const [other, n] of hits) {
      if (n < 3) continue;
      found.add(other);
      (other.alternatives ||= new Set()).add(res);
    }
    res.alternatives = found;
    return [...found];
  }

  ligands() {
    return this.residues
      .filter((r) => r.kind === 'ligand')
      .sort((a, b) => (a.additive - b.additive) || (b.atoms.length - a.atoms.length));
  }

  findResidue(chain, resi) {
    return this.residues.find((r) => r.chain === chain && String(r.resi) === String(resi));
  }

  sequence(chain) {
    return (this.chains.get(chain) || []).map((r) => AA3[r.resn] || 'X').join('');
  }

  summary() {
    const protein = this.residues.filter((r) => r.kind === 'protein').length;
    const chains = [...this.chains.keys()];
    return { atoms: this.atoms.length, protein, chains, ligands: this.ligands() };
  }
}

function avg(pts) {
  let x = 0, y = 0, z = 0;
  for (const p of pts) { x += p.x; y += p.y; z += p.z; }
  return { x: x / pts.length, y: y / pts.length, z: z / pts.length };
}
function newell(pts) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    nx += (p.y - q.y) * (p.z + q.z); ny += (p.z - q.z) * (p.x + q.x); nz += (p.x - q.x) * (p.y + q.y);
  }
  const n = Math.hypot(nx, ny, nz) || 1;
  return { x: nx / n, y: ny / n, z: nz / n };
}

export const resLabel = (r) => (r.kind === 'protein' ? `${cap(r.resn)}${r.resi}` : `${r.resn} ${r.resi}`);
export const cap = (s) => s.charAt(0) + s.slice(1).toLowerCase();
