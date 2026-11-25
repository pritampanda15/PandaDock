// Residue tables and chemical perception: who can donate, accept, stack, or carry charge.
import { centroid, ringNormal, dist, mid, sub, dot } from './vec.js';

export const AA3 = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  MSE: 'M', SEP: 'S', TPO: 'T', PTR: 'Y', HID: 'H', HIE: 'H', HIP: 'H', HSD: 'H', HSE: 'H', HSP: 'H',
  CYX: 'C', CSO: 'C', KCX: 'K', LLP: 'K', MLY: 'K', SEC: 'U', PYL: 'O',
};

export const AA_NAME = {
  ALA: 'Alanine', ARG: 'Arginine', ASN: 'Asparagine', ASP: 'Aspartate', CYS: 'Cysteine', GLN: 'Glutamine',
  GLU: 'Glutamate', GLY: 'Glycine', HIS: 'Histidine', ILE: 'Isoleucine', LEU: 'Leucine', LYS: 'Lysine',
  MET: 'Methionine', PHE: 'Phenylalanine', PRO: 'Proline', SER: 'Serine', THR: 'Threonine',
  TRP: 'Tryptophan', TYR: 'Tyrosine', VAL: 'Valine', MSE: 'Selenomethionine', SEP: 'Phosphoserine',
  TPO: 'Phosphothreonine', PTR: 'Phosphotyrosine',
};

export const NUCLEIC = new Set(['A', 'C', 'G', 'U', 'T', 'DA', 'DC', 'DG', 'DT', 'DU', 'I', 'DI']);
const WATER = new Set(['HOH', 'WAT', 'DOD', 'H2O', 'SOL', 'TIP', 'TIP3']);
export const METALS = new Set(['ZN', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'NA', 'K', 'CD', 'HG', 'LI', 'SR', 'BA', 'PT', 'AU', 'AG', 'MO', 'W', 'V', 'CR']);
// Crystallisation additives: kept, but ranked below the real ligand.
const ADDITIVES = new Set(['GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'SO4', 'PO4', 'ACT', 'ACY', 'DMS', 'CL', 'BR', 'IOD', 'FMT', 'TRS', 'MPD', 'EPE', 'MES', 'BME', 'NO3', 'SCN', 'IMD', 'CIT', 'TAR', 'MLI', 'NH4', 'UNX', 'UNL', 'P6G', 'PE4', 'BU3', 'EOH', 'IPA', 'NAG', 'MAN', 'BMA', 'FUC', 'GAL']);

export function residueKind(resn, het, atomCount, elem) {
  if (WATER.has(resn)) return 'water';
  if (AA3[resn] && !(het && atomCount < 4)) return 'protein';
  if (NUCLEIC.has(resn)) return 'nucleic';
  if (atomCount === 1 && METALS.has(elem)) return 'ion';
  if (atomCount === 1) return 'ion';
  return 'ligand';
}
export const isAdditive = (resn) => ADDITIVES.has(resn);

// Kyte–Doolittle hydropathy.
export const KD = {
  ALA: 1.8, ARG: -4.5, ASN: -3.5, ASP: -3.5, CYS: 2.5, GLN: -3.5, GLU: -3.5, GLY: -0.4, HIS: -3.2, ILE: 4.5,
  LEU: 3.8, LYS: -3.9, MET: 1.9, PHE: 2.8, PRO: -1.6, SER: -0.8, THR: -0.7, TRP: -0.9, TYR: -1.3, VAL: 4.2,
};

// Formal side-chain charge at pH 7.
export const CHARGE = { ARG: 1, LYS: 1, ASP: -1, GLU: -1, HIS: 0.1 };

export const VDW = { H: 1.1, C: 1.7, N: 1.55, O: 1.52, S: 1.8, P: 1.8, F: 1.47, CL: 1.75, BR: 1.85, I: 1.98, SE: 1.9 };
export const vdw = (e) => VDW[e] ?? 1.8;

const SC_DONORS = {
  ARG: ['NE', 'NH1', 'NH2'], ASN: ['ND2'], GLN: ['NE2'], HIS: ['ND1', 'NE2'], LYS: ['NZ'],
  SER: ['OG'], THR: ['OG1'], TYR: ['OH'], TRP: ['NE1'], CYS: ['SG'],
};
const SC_ACCEPTORS = {
  ASP: ['OD1', 'OD2'], GLU: ['OE1', 'OE2'], ASN: ['OD1'], GLN: ['OE1'], HIS: ['ND1', 'NE2'],
  SER: ['OG'], THR: ['OG1'], TYR: ['OH'], MET: ['SD'],
};
const RINGS = {
  PHE: [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
  TYR: [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
  PTR: [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
  HIS: [['CG', 'ND1', 'CE1', 'NE2', 'CD2']],
  TRP: [['CG', 'CD1', 'NE1', 'CE2', 'CD2'], ['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3']],
};
const SC_CHARGED = {
  ARG: [1, ['NE', 'NH1', 'NH2']], LYS: [1, ['NZ']], HIS: [1, ['ND1', 'NE2']],
  ASP: [-1, ['OD1', 'OD2']], GLU: [-1, ['OE1', 'OE2']],
};
const SC_AMIDES = { ASN: ['CG', 'OD1', 'ND2'], GLN: ['CD', 'OE1', 'NE2'] };

/**
 * Annotate atoms and residues of a Structure in place.
 * Adds atom.don / atom.acc / atom.hyd / atom.arom, and returns rings, charged groups and amide planes.
 */
export function perceive(S) {
  const rings = [], charges = [], amides = [];
  const A = S.atoms;
  const heavyNbrs = (a) => a.bonds.map((j) => A[j]).filter((b) => b && b.elem !== 'H');

  for (const a of A) {
    a.don = false; a.acc = false; a.hyd = false; a.arom = false;
    a.metal = METALS.has(a.elem) && a.res.kind === 'ion';
    a.halogen = a.elem === 'CL' || a.elem === 'BR' || a.elem === 'I' || a.elem === 'F';
  }

  for (const r of S.residues) {
    if (r.kind === 'water') {
      const o = r.atoms.find((a) => a.elem === 'O');
      if (o) { o.don = true; o.acc = true; }
      continue;
    }
    if (r.kind === 'protein') perceiveProtein(r, rings, charges, amides, S);
    else if (r.kind === 'ligand' || r.kind === 'nucleic') perceiveSmallMolecule(r, rings, charges, heavyNbrs);
  }

  // Hydrophobic carbons: carbon bonded only to carbon/hydrogen; plus thioether sulfur and halogens.
  for (const a of A) {
    if (a.res.kind === 'water' || a.res.kind === 'ion') continue;
    if (a.elem === 'C') {
      const nb = heavyNbrs(a);
      a.hyd = nb.length > 0 && nb.every((b) => b.elem === 'C' || b.elem === 'S');
    } else if (a.elem === 'S' && a.res.resn === 'MET') a.hyd = true;
    else if (a.halogen) a.hyd = true;
  }
  for (const ring of rings) for (const a of ring.atoms) a.arom = true;

  return { rings, charges, amides };
}

function perceiveProtein(r, rings, charges, amides, S) {
  const by = r.byName;
  if (by.N && r.resn !== 'PRO') by.N.don = true;
  if (by.O) by.O.acc = true;
  if (by.OXT) {
    by.OXT.acc = true;
    if (by.O) charges.push(group(-1, [by.O, by.OXT], r, 'C-terminus'));
  }
  for (const n of SC_DONORS[r.resn] || []) if (by[n]) by[n].don = true;
  for (const n of SC_ACCEPTORS[r.resn] || []) if (by[n]) by[n].acc = true;
  for (const names of RINGS[r.resn] || []) {
    const atoms = names.map((n) => by[n]).filter(Boolean);
    if (atoms.length === names.length) rings.push(ring(atoms, r));
  }
  const ch = SC_CHARGED[r.resn];
  if (ch) {
    const atoms = ch[1].map((n) => by[n]).filter(Boolean);
    if (atoms.length) charges.push(group(ch[0], atoms, r, r.resn === 'HIS' ? 'imidazolium (if protonated)' : null));
  }
  if (r.prev == null && by.N && !r.chainBreakBefore) {
    // First residue of a chain: free amine.
    charges.push(group(1, [by.N], r, 'N-terminus'));
  }
  // Backbone amide: C, O of this residue and N of the next.
  const next = r.next != null ? S.residues[r.next] : null;
  if (by.C && by.O && next && next.byName.N && dist(by.C, next.byName.N) < 1.6) {
    const atoms = [by.CA, by.C, by.O, next.byName.N].filter(Boolean);
    amides.push({ atoms, res: r, center: centroid(atoms), normal: ringNormal(atoms), label: 'backbone amide' });
  }
  const sc = SC_AMIDES[r.resn];
  if (sc && sc.every((n) => by[n])) {
    const atoms = sc.map((n) => by[n]);
    amides.push({ atoms, res: r, center: centroid(atoms), normal: ringNormal(atoms), label: 'side-chain amide' });
  }
}

function perceiveSmallMolecule(r, rings, charges, heavyNbrs) {
  const atoms = r.atoms.filter((a) => a.elem !== 'H');
  const inRes = new Set(atoms);
  const nbrs = new Map(atoms.map((a) => [a, heavyNbrs(a)]));
  const localNbrs = (a) => nbrs.get(a).filter((b) => inRes.has(b));

  // Rings of size 5 and 6 by bounded DFS, deduplicated by atom set.
  const seen = new Set();
  for (const start of atoms) {
    const stack = [[start, [start]]];
    while (stack.length) {
      const [cur, path] = stack.pop();
      if (path.length > 6) continue;
      for (const nb of localNbrs(cur)) {
        if (nb === start && path.length >= 5) {
          const key = path.map((a) => a.index).sort((x, y) => x - y).join(',');
          if (!seen.has(key)) {
            seen.add(key);
            const pts = path.slice();
            if (isPlanarAromatic(pts, localNbrs)) rings.push(ring(pts, r));
          }
        } else if (!path.includes(nb) && nb.index > start.index) {
          stack.push([nb, path.concat(nb)]);
        }
      }
    }
  }
  const aromatic = new Set(rings.filter((x) => x.res === r).flatMap((x) => x.atoms));

  for (const a of atoms) {
    const nb = nbrs.get(a);
    if (a.elem === 'O') {
      a.acc = true;
      // Bond length separates C=O (≈1.23 Å) from C–OH (≈1.36–1.43 Å).
      if (nb.length === 1 && dist(a, nb[0]) > 1.31 && nb[0].elem !== 'P' && nb[0].elem !== 'S') a.don = true;
      if (nb.length === 0) a.don = true;
    } else if (a.elem === 'N') {
      const arom = aromatic.has(a);
      if (nb.length < 4) a.acc = !(arom && nb.length === 3) && !isAmideN(a, nb, nbrs);
      if (nb.length <= 2 && !(arom && nb.length === 2 && ringSizeOf(a, rings) === 6)) a.don = true;
      if (nb.length === 3 && arom && ringSizeOf(a, rings) === 5) a.don = false;
    }
  }

  // Charged groups.
  for (const a of atoms) {
    const nb = nbrs.get(a);
    if (a.elem === 'C' || a.elem === 'P' || a.elem === 'S') {
      // A neighbour outside this residue (the next nucleotide, a covalent link) isn't terminal.
      const termO = nb.filter((b) => b.elem === 'O' && nbrs.get(b)?.length === 1);
      if (a.elem === 'C' && termO.length === 2 && nb.length === 3) charges.push(group(-1, termO, r, 'carboxylate'));
      if (a.elem === 'P' && termO.length >= 2) charges.push(group(-1, termO, r, 'phosphate'));
      if (a.elem === 'S' && termO.length >= 3) charges.push(group(-1, termO, r, 'sulfonate'));
      if (a.elem === 'C' && !aromatic.has(a)) {
        const ns = nb.filter((b) => b.elem === 'N' && !aromatic.has(b));
        if (ns.length >= 2 && nb.length === 3 && nb.every((b) => b.elem === 'N' || b.elem === 'C')) {
          charges.push(group(1, ns, r, ns.length === 3 ? 'guanidinium' : 'amidinium'));
        }
      }
    }
    if (a.elem === 'N' && !aromatic.has(a)) {
      if (nb.length === 4) charges.push(group(1, [a], r, 'quaternary ammonium'));
      else if (nb.length >= 1 && nb.every((b) => b.elem === 'C' && !aromatic.has(b) && isSp3Carbon(b, nbrs) && dist(a, b) > 1.42) && !isAmideN(a, nb, nbrs)) {
        charges.push(group(1, [a], r, 'aliphatic amine (protonated at pH 7)'));
      }
    }
  }
}

function isSp3Carbon(c, nbrs) {
  // An sp3 carbon has no short bond to O/N/C, which would indicate a double bond.
  return (nbrs.get(c) || []).every((b) => dist(c, b) > 1.42 || b.elem === 'H');
}
function isAmideN(n, nb, nbrs) {
  return nb.some((c) => c.elem === 'C' && (nbrs.get(c) || []).some((o) => o.elem === 'O' && dist(c, o) < 1.3));
}
function ringSizeOf(a, rings) {
  const r = rings.find((x) => x.atoms.includes(a));
  return r ? r.atoms.length : 0;
}
function isPlanarAromatic(pts, localNbrs) {
  if (!pts.every((a) => ['C', 'N', 'O', 'S'].includes(a.elem))) return false;
  if (!pts.every((a) => localNbrs(a).length <= 3)) return false;
  const c = centroid(pts), n = ringNormal(pts);
  return pts.every((p) => Math.abs(dot(sub(p, c), n)) < 0.15);
}
function ring(atoms, res) {
  return { atoms, res, center: centroid(atoms), normal: ringNormal(atoms), size: atoms.length };
}
function group(sign, atoms, res, label) {
  return { sign, atoms, res, center: atoms.length === 2 ? mid(atoms[0], atoms[1]) : centroid(atoms), label };
}
