// Non-covalent interaction detection, covering the interaction classes PandaMap reports.
// Geometry thresholds follow PLIP / Discovery Studio conventions where they exist.
import { dist, angle, planeAngle, offsetFromAxis, sub, dot, norm } from './vec.js';
import { vdw } from './chem.js';

// `rule` is the detection criterion as shown in the in-app legend; keep it in step with the code below.
export const TYPES = {
  hbond:      { label: 'Hydrogen bond',        color: '#0A84FF', short: 'H-bond',      rule: 'Donor–acceptor N/O/S 2.4–3.6 Å; every heavy-atom substituent at donor and acceptor makes ≥ 85° with the D···A line. When the file has hydrogens on the donor’s residue, the donor is tested on them instead: H···A ≤ 2.7 Å and D–H···A ≥ 120°.' },
  saltbridge: { label: 'Salt bridge',          color: '#FF2D55', short: 'Salt bridge', rule: 'Closest atoms of two oppositely charged groups ≤ 4.0 Å.' },
  attractive: { label: 'Attractive charge',    color: '#FF7AA2', short: 'Charge',      rule: 'Opposite charges whose group centres are ≤ 5.5 Å apart but too far for a salt bridge.' },
  repulsion:  { label: 'Charge repulsion',     color: '#B3261E', short: 'Repulsion',   rule: 'Like charges with centres ≤ 5.0 Å (His excluded, its charge is uncertain).' },
  hydrophobic:{ label: 'Hydrophobic',          color: '#E8A33D', short: 'Hydrophobic', rule: 'Carbon bonded only to C/S, Met Sδ or halogen pairs ≤ 4.0 Å; closest pair per residue pair.' },
  pipi:       { label: 'π–π stacking',         color: '#20A4B8', short: 'π–π',         rule: 'Aromatic ring centroids ≤ 5.5 Å, planes within 30°, lateral offset ≤ 2.0 Å.' },
  tshape:     { label: 'π–π T-shaped',         color: '#3FC1C9', short: 'π–π T',       rule: 'Ring centroids ≤ 6.0 Å, planes at ≥ 60°, offset ≤ 2.0 Å.' },
  cationpi:   { label: 'Cation–π',             color: '#AF52DE', short: 'Cation–π',    rule: 'Cationic group centre ≤ 6.0 Å from a ring centroid and ≤ 2.0 Å off its axis.' },
  amidepi:    { label: 'Amide–π',              color: '#5BB98C', short: 'Amide–π',     rule: 'Backbone or Asn/Gln amide plane ≤ 4.5 Å from a ring, planes within 30°, offset ≤ 2.0 Å.' },
  donorpi:    { label: 'Donor–π',              color: '#4A7FD6', short: 'Donor–π',     rule: 'N/O donor ≤ 4.0 Å from a ring centroid, within 35° of the ring normal.' },
  carbonpi:   { label: 'Alkyl–π (C–H···π)',    color: '#C9A06B', short: 'Alkyl–π',     rule: 'Hydrophobic carbon ≤ 4.5 Å from a ring centroid, within 40° of the normal.' },
  sulfurpi:   { label: 'Sulfur–π',             color: '#D4B106', short: 'Sulfur–π',    rule: 'Sulfur ≤ 5.5 Å from a ring centroid, within 50° of the normal.' },
  halogen:    { label: 'Halogen bond',         color: '#34C759', short: 'Halogen',     rule: 'C–X···A with X = Cl, Br, I; X···A ≤ 3.5, 3.6, 3.8 Å respectively; C–X···A ≥ 140°.' },
  metal:      { label: 'Metal coordination',   color: '#9C7B5B', short: 'Metal',       rule: 'Metal ion to O, N or S ≤ 2.8 Å.' },
  water:      { label: 'Water bridge',         color: '#5AC8FA', short: 'Water',       rule: 'Polar atom – water oxygen – polar atom, both legs 2.4–3.5 Å.' },
  covalent:   { label: 'Covalent bond',        color: '#8E8E93', short: 'Covalent',    rule: 'A bond leaving the focus to another residue (peptide bonds excluded), e.g. a covalent warhead.' },
  disulfide:  { label: 'Disulfide',            color: '#C7A600', short: 'S–S',         rule: 'S–S bond between cysteines.' },
  clash:      { label: 'Steric clash',         color: '#FF3B30', short: 'Clash',       rule: 'Non-bonded heavy atoms closer than the sum of van der Waals radii − 0.6 Å (− 1.0 Å for polar pairs).' },
};

const ENV_RADIUS = 7.5;

/**
 * Interactions between a focus (a set of atoms) and everything around it.
 * Returns every interaction found within generous cutoffs; callers filter by the probe radius.
 */
export function computeInteractions(S, focusAtoms) {
  const focus = new Set(focusAtoms);
  const focusRes = new Set(focusAtoms.map((a) => a.res));
  const focusIsWater = [...focusRes].every((r) => r.kind === 'water');
  const excluded = new Set(focusRes);
  for (const r of focusRes) {
    if (r.prev != null) excluded.add(S.residues[r.prev]);
    if (r.next != null) excluded.add(S.residues[r.next]);
    for (const alt of r.alternatives || []) excluded.add(alt);
  }

  const env = new Set();
  for (const f of focusAtoms) for (const a of S.near(f, ENV_RADIUS)) if (!excluded.has(a.res)) env.add(a);
  const envAtoms = [...env].filter((a) => a.elem !== 'H');
  const heavyFocus = focusAtoms.filter((a) => a.elem !== 'H');
  const out = [];
  const bonded = (a, b) => a.bonds.includes(b.index);
  const nbrs = (a) => a.bonds.map((j) => S.atoms[j]).filter((x) => x && x.elem !== 'H');
  const add = (type, from, to, d, extra = {}) => out.push({ type, from, to, d, ...extra });

  // Covalent links that leave the focus (ligand–Cys warheads, disulfides, glycans).
  for (const f of heavyFocus) {
    for (const j of f.bonds) {
      const b = S.atoms[j];
      if (!b || focusRes.has(b.res)) continue;
      const peptide = (f.name === 'C' && b.name === 'N') || (f.name === 'N' && b.name === 'C');
      if (peptide && f.res.kind === 'protein' && b.res.kind === 'protein') continue;
      const type = f.elem === 'S' && b.elem === 'S' ? 'disulfide' : 'covalent';
      add(type, pt(f), pt(b), dist(f, b), { res: b.res });
    }
  }

  // Hydrogen bonds (heavy-atom geometry; hydrogens are rarely modelled in crystal structures).
  const polar = (a) => a.elem === 'N' || a.elem === 'O' || (a.elem === 'S' && (a.don || a.acc));
  for (const f of heavyFocus) {
    if (!polar(f) || !(f.don || f.acc)) continue;
    for (const a of envAtoms) {
      if (!polar(a) || !(a.don || a.acc)) continue;
      if (a.res.kind === 'water' && !focusIsWater) continue;
      const d = dist(f, a);
      if (d < 2.4 || d > 3.6 || bonded(f, a)) continue;
      let role = null, viaH = null;
      // With hydrogens in the file the donor is tested on its real hydrogen; otherwise on heavy-atom geometry.
      const donates = (D, A) => {
        if (!D.res.hasH) return geomOK(D, A, nbrs);
        const h = hydrogenTo(D, A);
        if (h && (!viaH || h.angle > viaH.angle)) viaH = h;
        return !!h;
      };
      if (f.don && a.acc && donates(f, a)) role = 'donor';
      if (f.acc && a.don && donates(a, f)) role = role ? 'either' : 'acceptor';
      if (!role) continue;
      const D = role === 'acceptor' ? a : f, A = role === 'acceptor' ? f : a;
      const dn = nbrs(D);
      const ang = dn.length ? Math.min(...dn.map((x) => angle(x, D, A))) : null;
      add('hbond', pt(f), pt(a), d, {
        res: a.res, role, angle: ang, strength: d < 3.0 ? 'strong' : d < 3.3 ? 'moderate' : 'weak',
        ...(viaH ? { hAngle: viaH.angle, dHA: viaH.d } : {}),
      });
    }
  }

  // Water bridges: focus polar atom – water – partner polar atom.
  if (!focusIsWater) {
    const waters = envAtoms.filter((a) => a.res.kind === 'water' && a.elem === 'O');
    const seen = new Set();
    for (const f of heavyFocus) {
      if (!polar(f)) continue;
      for (const w of waters) {
        const d1 = dist(f, w);
        if (d1 > 3.5 || d1 < 2.4) continue;
        for (const p of S.near(w, 3.5)) {
          if (p === w || focus.has(p) || excluded.has(p.res) || p.res.kind === 'water' || !polar(p)) continue;
          if (!(p.don || p.acc)) continue;
          const d2 = dist(w, p);
          if (d2 < 2.4) continue;
          const key = `${f.index}-${p.index}`;
          if (seen.has(key)) continue;
          seen.add(key);
          add('water', pt(f), pt(p), Math.max(d1, d2), { res: p.res, via: pt(w), water: w, d1, d2 });
        }
      }
    }
  }

  // Hydrophobic contacts: closest carbon pair per focus/partner residue pair.
  const best = new Map();
  for (const f of heavyFocus) {
    if (!f.hyd) continue;
    for (const a of envAtoms) {
      if (!a.hyd) continue;
      const d = dist(f, a);
      if (d > 4.0) continue;
      const key = `${f.res.index}-${a.res.index}`;
      if (!best.has(key) || best.get(key).d > d) best.set(key, { f, a, d });
    }
  }
  for (const { f, a, d } of best.values()) add('hydrophobic', pt(f), pt(a), d, { res: a.res });

  // Charges.
  const inFocus = (g) => g.atoms.some((x) => focus.has(x));
  const fCharges = S.charges.filter(inFocus);
  const pCharges = S.charges.filter((g) => !excluded.has(g.res) && !inFocus(g));
  for (const g of fCharges) {
    for (const h of pCharges) {
      const dc = dist(g.center, h.center);
      if (dc > 8) continue;
      if (g.sign !== h.sign) {
        let dmin = Infinity, pa = null, pb = null;
        for (const x of g.atoms) for (const y of h.atoms) {
          const d = dist(x, y);
          if (d < dmin) { dmin = d; pa = x; pb = y; }
        }
        if (dmin <= 4.0) add('saltbridge', pt(pa), pt(pb), dmin, { res: h.res, note: noteFor(g, h) });
        else if (dc <= 5.5) add('attractive', gpt(g), gpt(h), dc, { res: h.res, note: noteFor(g, h) });
      } else if (dc <= 5.0 && g.res.resn !== 'HIS' && h.res.resn !== 'HIS') {
        add('repulsion', gpt(g), gpt(h), dc, { res: h.res });
      }
    }
  }

  // Aromatic interactions.
  const fRings = S.rings.filter(inFocus);
  const pRings = S.rings.filter((r) => !excluded.has(r.res) && !inFocus(r));
  for (const r1 of fRings) {
    for (const r2 of pRings) {
      const d = dist(r1.center, r2.center);
      if (d > 6.0) continue;
      const ang = planeAngle(r1.normal, r2.normal);
      const off = Math.min(offsetFromAxis(r2.center, r1.center, r1.normal), offsetFromAxis(r1.center, r2.center, r2.normal));
      if (ang <= 30 && d <= 5.5 && off <= 2.0) add('pipi', gpt(r1), gpt(r2), d, { res: r2.res, angle: ang, offset: off });
      else if (ang >= 60 && d <= 6.0 && off <= 2.0) add('tshape', gpt(r1), gpt(r2), d, { res: r2.res, angle: ang, offset: off });
    }
  }
  const cationPi = (g, r, gIsFocus) => {
    const d = dist(g.center, r.center);
    const off = offsetFromAxis(g.center, r.center, r.normal);
    if (d <= 6.0 && off <= 2.0) {
      add('cationpi', gIsFocus ? gpt(g) : gpt(r), gIsFocus ? gpt(r) : gpt(g), d, { res: gIsFocus ? r.res : g.res, offset: off });
    }
  };
  for (const g of fCharges) if (g.sign > 0) for (const r of pRings) if (r.res !== g.res) cationPi(g, r, true);
  for (const r of fRings) for (const g of pCharges) if (g.sign > 0 && g.res !== r.res) cationPi(g, r, false);

  const fAmides = S.amides.filter(inFocus);
  const pAmides = S.amides.filter((m) => !excluded.has(m.res) && !inFocus(m));
  const amidePi = (m, r, mIsFocus) => {
    const d = dist(m.center, r.center);
    if (d > 4.5) return;
    const ang = planeAngle(m.normal, r.normal);
    if (ang > 30 || offsetFromAxis(m.center, r.center, r.normal) > 2.0) return;
    add('amidepi', mIsFocus ? gpt(m) : gpt(r), mIsFocus ? gpt(r) : gpt(m), d, { res: mIsFocus ? r.res : m.res, angle: ang, note: m.label });
  };
  for (const m of fAmides) for (const r of pRings) if (r.res !== m.res) amidePi(m, r, true);
  for (const r of fRings) for (const m of pAmides) if (r.res !== m.res) amidePi(m, r, false);

  // Single atoms over a ring face: donor–π, sulfur–π, alkyl–π.
  const overFace = (a, r) => {
    const v = sub(a, r.center);
    const d = norm(v);
    const tilt = (Math.acos(Math.min(1, Math.abs(dot(v, r.normal)) / (d || 1))) * 180) / Math.PI;
    return { d, tilt };
  };
  const atomPi = (a, r, aIsFocus) => {
    if (r.atoms.includes(a) || a.arom) return;
    const { d, tilt } = overFace(a, r);
    let type = null;
    if ((a.elem === 'N' || a.elem === 'O') && a.don && d <= 4.0 && tilt <= 35 && !isCation(S, a)) type = 'donorpi';
    else if (a.elem === 'S' && d <= 5.5 && tilt <= 50) type = 'sulfurpi';
    else if (a.elem === 'C' && a.hyd && d <= 4.5 && tilt <= 40) type = 'carbonpi';
    if (!type) return;
    const other = aIsFocus ? r.res : a.res;
    add(type, aIsFocus ? pt(a) : gpt(r), aIsFocus ? gpt(r) : pt(a), d, { res: other, angle: tilt, atom: a });
  };
  for (const a of heavyFocus) for (const r of pRings) if (r.res !== a.res) atomPi(a, r, true);
  for (const r of fRings) for (const a of envAtoms) if (a.res !== r.res && a.res.kind !== 'water') atomPi(a, r, false);

  // Halogen bonds: C–X···A, near-linear.
  const halo = (x, acc, xIsFocus) => {
    const c = nbrs(x).find((n) => n.elem === 'C');
    if (!c) return;
    const d = dist(x, acc);
    const lim = x.elem === 'I' ? 3.8 : x.elem === 'BR' ? 3.6 : 3.5;
    const ang = angle(c, x, acc);
    if (d <= lim && ang >= 140) add('halogen', pt(xIsFocus ? x : acc), pt(xIsFocus ? acc : x), d, { res: xIsFocus ? acc.res : x.res, angle: ang });
  };
  const isXB = (a) => a.elem === 'CL' || a.elem === 'BR' || a.elem === 'I';
  const isAcc = (a) => (a.elem === 'O' || a.elem === 'N' || a.elem === 'S') && a.acc;
  for (const f of heavyFocus) for (const a of envAtoms) {
    if (isXB(f) && isAcc(a)) halo(f, a, true);
    else if (isXB(a) && isAcc(f)) halo(a, f, false);
  }

  // Metal coordination.
  for (const f of heavyFocus) for (const a of envAtoms) {
    const m = f.metal ? f : a.metal ? a : null;
    const l = m === f ? a : f;
    if (!m || !['O', 'N', 'S'].includes(l.elem)) continue;
    const d = dist(m, l);
    if (d <= 2.8) add('metal', pt(f), pt(a), d, { res: a.res, note: m.elem });
  }

  // Steric clashes between non-bonded heavy atoms.
  for (const f of heavyFocus) for (const a of envAtoms) {
    if (a.res.kind === 'water' || bonded(f, a) || f.metal || a.metal) continue;
    const d = dist(f, a);
    const bothPolar = polar(f) && polar(a);
    const limit = vdw(f.elem) + vdw(a.elem) - (bothPolar ? 1.0 : 0.6);
    if (d < limit && d > 0.5) add('clash', pt(f), pt(a), d, { res: a.res, overlap: vdw(f.elem) + vdw(a.elem) - d });
  }

  // One alkyl–π / sulfur–π / donor–π per ring and partner residue: the closest.
  const keep = new Map();
  const rest = [];
  for (const it of out) {
    if (it.type !== 'carbonpi' && it.type !== 'sulfurpi' && it.type !== 'donorpi') { rest.push(it); continue; }
    const ring = (it.from.group || it.to.group);
    const k = `${it.type}|${ring.atoms[0].index}|${it.atom.res.index}`;
    if (!keep.has(k) || keep.get(k).d > it.d) keep.set(k, it);
  }
  return rest.concat([...keep.values()]).sort((x, y) => x.d - y.d);
}

/** The donor's hydrogen that points at the acceptor: H···A ≤ 2.7 Å and D–H···A ≥ 120°. Best angle wins. */
export const HB_H = { maxHA: 2.7, minAngle: 120 };
function hydrogenTo(D, A) {
  let best = null;
  for (const h of D.hs || []) {
    const d = dist(h, A);
    if (d > HB_H.maxHA) continue;
    const ang = angle(D, h, A);
    if (ang >= HB_H.minAngle && (!best || ang > best.angle)) best = { angle: ang, d };
  }
  return best;
}
function geomOK(D, A, nbrs) {
  const dn = nbrs(D), an = nbrs(A);
  return dn.every((x) => angle(x, D, A) >= 85) && an.every((y) => angle(y, A, D) >= 85);
}
function isCation(S, a) {
  return S.charges.some((g) => g.sign > 0 && g.atoms.includes(a));
}
function noteFor(g, h) {
  return [g.label, h.label].filter(Boolean).join(' / ') || null;
}
const pt = (a) => ({ x: a.x, y: a.y, z: a.z, atom: a, label: `${a.name}` });
const gpt = (g) => ({ x: g.center.x, y: g.center.y, z: g.center.z, group: g, label: g.size ? `ring(${g.size})` : g.label || 'group' });
