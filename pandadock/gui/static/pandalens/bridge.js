// PandaLens Bridge for PandaDock Studio
// Integrates PandaLens's exact 18-class chemical perception & interaction engine
// (vec.js, chem.js, structure.js, interactions.js, labels.js) with PandaDock Studio's
// 3Dmol.js WebGL viewer and explicit-H RDKit binding-site pocket models.

import { Structure, resLabel } from './structure.js';
import { computeInteractions, TYPES } from './interactions.js';
import { LabelLayer } from './labels.js';
import { layoutMap, renderMap } from './map2d.js';
import { dist } from './vec.js';

export const PANDALENS_TYPES = TYPES;

/**
 * Default hidden noisy classes so key directional contacts (H-bonds, salt bridges,
 * metal coordination, covalent warheads, pi-stacking, halogen, water bridges, clashes)
 * read cleanly immediately; users can click any pill to toggle hydrophobic/alkyl-pi on.
 */
export const DEFAULT_HIDDEN_TYPES = new Set(['hydrophobic', 'carbonpi']);

const COV_MAX_DIST = 2.05;
const METAL_SET = new Set(['ZN', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'NA', 'K', 'CD', 'HG', 'PT', 'AU', 'AG', 'MO', 'W', 'V', 'CR']);

function cloneAtom(a, newIdx, override = null) {
  const resn = override ? override.resn : (a.resn || '').trim().toUpperCase();
  let elem = (a.elem || '').toUpperCase();
  if (!override && METAL_SET.has(resn)) {
    elem = resn;
  }
  return {
    index: newIdx,
    x: a.x,
    y: a.y,
    z: a.z,
    elem,
    atom: (a.atom || elem || '').trim(),
    resn,
    resi: override ? override.resi : a.resi,
    icode: override ? '' : (a.icode || ''),
    chain: override ? override.chain : (a.chain || 'A'),
    hetflag: override ? true : (METAL_SET.has(resn) ? true : !!a.hetflag),
    b: a.b ?? 0,
    ss: a.ss || '',
    bonds: [],
  };
}

/**
 * Build a PandaLens `Structure` from the live 3Dmol pocket & ligand models (both
 * containing explicit 3D hydrogens), attach all explicit H coordinates for true
 * D-H...A hydrogen-bond geometry, link any covalent warhead bond, and compute
 * all 18 PandaLens / PLIP / PandaMap interaction classes.
 */
export function analyzeModels(pocketModel, ligModel, fallbackRecModel = null) {
  const recSrcModel = pocketModel || fallbackRecModel;
  if (!recSrcModel) {
    return { structure: null, interactions: [], hCount: 0, pocketResidues: [] };
  }

  const recRaw = recSrcModel.selectedAtoms({});
  const ligRaw = ligModel ? ligModel.selectedAtoms({}) : [];

  const heavyAtoms = [];
  const hPoints = [];

  // Map original 3Dmol atom.index -> new heavyAtoms index (per model)
  const recIdxMap = new Map();
  const ligIdxMap = new Map();

  // 1. Receptor / pocket atoms (skip embedded LIG/COV if we have a separate ligModel)
  for (const a of recRaw) {
    const elem = (a.elem || '').toUpperCase();
    const resn = (a.resn || '').trim().toUpperCase();
    if (ligRaw.length && (resn === 'LIG' || resn === 'COV' || resn === 'UNL')) continue;
    if (elem === 'H') {
      hPoints.push({ x: a.x, y: a.y, z: a.z });
      continue;
    }
    const ni = heavyAtoms.length;
    recIdxMap.set(a.index, ni);
    heavyAtoms.push(cloneAtom(a, ni));
  }
  // Wire intra-receptor bonds
  for (const a of recRaw) {
    const u = recIdxMap.get(a.index);
    if (u == null) continue;
    for (const j of a.bonds || []) {
      const v = recIdxMap.get(j);
      if (v != null && v !== u && !heavyAtoms[u].bonds.includes(v)) {
        heavyAtoms[u].bonds.push(v);
      }
    }
  }

  // 2. Ligand atoms (normalized to LIG Z:1 so PandaLens cleanly isolates focus)
  const ligOverride = { resn: 'LIG', chain: 'Z', resi: 1 };
  const ligHeavyIndices = [];
  const elemCount = {};
  for (const a of ligRaw) {
    const elem = (a.elem || '').toUpperCase();
    if (elem === 'H') {
      hPoints.push({ x: a.x, y: a.y, z: a.z });
      continue;
    }
    const ni = heavyAtoms.length;
    ligIdxMap.set(a.index, ni);
    const cloned = cloneAtom(a, ni, ligOverride);
    if (!cloned.atom || cloned.atom === elem) {
      elemCount[elem] = (elemCount[elem] || 0) + 1;
      cloned.atom = `${elem}${elemCount[elem]}`;
    }
    heavyAtoms.push(cloned);
    ligHeavyIndices.push(ni);
  }
  // Wire intra-ligand bonds
  for (const a of ligRaw) {
    const u = ligIdxMap.get(a.index);
    if (u == null) continue;
    for (const j of a.bonds || []) {
      const v = ligIdxMap.get(j);
      if (v != null && v !== u && !heavyAtoms[u].bonds.includes(v)) {
        heavyAtoms[u].bonds.push(v);
      }
    }
  }

  // 3. Detect inter-molecular covalent warhead link (e.g. CYS SG - ligand C <= 2.05 A)
  const recCount = ligHeavyIndices.length ? ligHeavyIndices[0] : heavyAtoms.length;
  for (const li of ligHeavyIndices) {
    const la = heavyAtoms[li];
    for (let ri = 0; ri < recCount; ri++) {
      const ra = heavyAtoms[ri];
      if (ra.resn === 'HOH' || ra.resn === 'WAT') continue;
      // Exclude metal ions from covalent bonding (handled by metal coordination)
      if (['ZN', 'MG', 'FE', 'MN', 'CU', 'CA', 'CO', 'NI', 'NA', 'K'].includes(ra.elem)) continue;
      const d = dist(la, ra);
      if (d > 0.8 && d <= COV_MAX_DIST) {
        if (!la.bonds.includes(ri)) la.bonds.push(ri);
        if (!ra.bonds.includes(li)) ra.bonds.push(li);
      }
    }
  }

  // 4. Build PandaLens Structure + attach explicit hydrogens for true D-H...A geometry
  const fakeModel = { selectedAtoms: () => heavyAtoms };
  const S = new Structure('pandadock-pocket', fakeModel);
  const hCount = hPoints.length ? S.attachHydrogens(hPoints) : 0;

  // 5. Identify focus ligand residue
  let focusRes = S.residues.find((r) => r.chain === 'Z' && r.resn === 'LIG');
  if (!focusRes) {
    const ligs = S.ligands();
    focusRes = ligs.length ? ligs[0] : null;
  }
  if (!focusRes) {
    return { structure: S, interactions: [], hCount, pocketResidues: [] };
  }

  const rawInts = computeInteractions(S, focusRes.atoms);
  // Deduplicate & annotate each interaction cleanly for PandaDock Studio UI
  const interactions = rawInts.map((it, idx) => {
    const meta = TYPES[it.type] || { label: it.type, short: it.type, color: '#64748B', rule: '' };
    const r = it.res;
    const partnerLabel = r ? (r.kind === 'water' ? `HOH ${r.chain}:${r.resi}` : `${resLabel(r)} (${r.chain})`) : '';
    const anchor = r ? (r.ca || r.atoms[0]) : it.to;
    return {
      ...it,
      id: idx,
      color: meta.color,
      label: meta.label,
      short: meta.short,
      rule: meta.rule,
      partnerLabel,
      partnerKey: r ? r.key : `p-${idx}`,
      anchorPos: anchor ? { x: anchor.x, y: anchor.y, z: anchor.z } : { x: it.to.x, y: it.to.y, z: it.to.z },
      isSolid: it.type === 'covalent' || it.type === 'disulfide' || it.type === 'clash',
    };
  });

  const pocketResidues = S.residues.filter((r) => r !== focusRes && r.kind !== 'water');
  return { structure: S, focusRes, interactions, hCount, pocketResidues };
}

/**
 * Draw PandaLens 3D interaction cylinders, ring/group centroid markers, distance pills,
 * and draggable HTML residue labels with SVG leader lines.
 */
export class PandaLensOverlay {
  constructor(viewer, labelContainerEl, getDark) {
    this.v = viewer;
    this.getDark = getDark;
    this.shapes = [];
    this.distLabels = [];
    this.hlShapes = [];
    this.labelLayer = labelContainerEl ? new LabelLayer(labelContainerEl, viewer) : null;
  }

  clear() {
    for (const s of this.shapes) this.v.removeShape(s);
    for (const l of this.distLabels) this.v.removeLabel(l);
    this.clearHighlight();
    this.shapes = [];
    this.distLabels = [];
    if (this.labelLayer) this.labelLayer.set([]);
  }

  clearHighlight() {
    for (const s of this.hlShapes) this.v.removeShape(s);
    this.hlShapes = [];
    if (this.atomMark) {
      this.v.removeShape(this.atomMark);
      this.atomMark = null;
    }
  }

  markAtom(atom) {
    if (this.atomMark) {
      this.v.removeShape(this.atomMark);
      this.atomMark = null;
    }
    if (atom) {
      this.atomMark = this.v.addSphere({
        center: { x: atom.x, y: atom.y, z: atom.z },
        radius: 0.55,
        color: '#FF9F0A',
        opacity: 0.58,
      });
    }
    this.v.render();
  }

  drawOne(it, targetShapes, targetLabels, showDistLabels, emphasis = 1) {
    const segs = it.via ? [[it.from, it.via], [it.via, it.to]] : [[it.from, it.to]];
    const solid = !!it.isSolid;
    for (const [p, q] of segs) {
      targetShapes.push(this.v.addCylinder({
        start: { x: p.x, y: p.y, z: p.z },
        end: { x: q.x, y: q.y, z: q.z },
        radius: (solid ? 0.08 : 0.055) * emphasis,
        color: it.color,
        dashed: !solid,
        dashLength: it.type === 'hydrophobic' ? 0.14 : 0.22,
        gapLength: it.type === 'hydrophobic' ? 0.22 : 0.16,
        fromCap: 1,
        toCap: 1,
      }));
    }
    // Ring / charge-group centroid sphere marker (PandaLens signature visual)
    for (const p of [it.from, it.to]) {
      if (p && p.group) {
        targetShapes.push(this.v.addSphere({
          center: { x: p.x, y: p.y, z: p.z },
          radius: 0.17 * emphasis,
          color: it.color,
        }));
      }
    }
    if (showDistLabels && targetLabels && it.type !== 'hydrophobic' && it.type !== 'carbonpi') {
      const dark = this.getDark();
      const anchor = it.via
        ? it.via
        : { x: (it.from.x + it.to.x) / 2, y: (it.from.y + it.to.y) / 2, z: (it.from.z + it.to.z) / 2 };
      targetLabels.push(this.v.addLabel(`${it.d.toFixed(2)} Å`, {
        position: { x: anchor.x, y: anchor.y, z: anchor.z },
        fontSize: 10.5,
        fontColor: '#FFFFFF',
        backgroundColor: it.color,
        backgroundOpacity: dark ? 0.90 : 0.92,
        borderRadius: 5,
        padding: 1.5,
        inFront: true,
        alignment: 'center',
        font: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif',
      }));
    }
  }

  render(interactions, { hiddenTypes = DEFAULT_HIDDEN_TYPES, showDistLabels = true, showResLabels = true } = {}) {
    this.clear();
    const visible = interactions.filter((it) => !hiddenTypes.has(it.type));
    const resLabelItems = new Map();

    for (const it of visible) {
      this.drawOne(it, this.shapes, this.distLabels, showDistLabels, 1);
      if (showResLabels && it.partnerLabel && !resLabelItems.has(it.partnerKey)) {
        resLabelItems.set(it.partnerKey, {
          key: it.partnerKey,
          text: it.partnerLabel,
          pos: it.anchorPos,
          color: it.type === 'covalent' || it.type === 'metal' ? it.color : '',
          strong: it.type === 'covalent' || it.type === 'metal' || it.type === 'saltbridge',
        });
      }
    }

    if (this.labelLayer) {
      this.labelLayer.set(showResLabels ? [...resLabelItems.values()] : []);
    }
    this.v.render();
  }

  highlight(items) {
    this.clearHighlight();
    const list = !items ? [] : Array.isArray(items) ? items : [items];
    for (const it of list) {
      this.drawOne(it, this.hlShapes, null, false, 2.3);
    }
    this.v.render();
  }

  resetLabelPositions() {
    if (this.labelLayer) {
      this.labelLayer.offsets.clear();
      this.labelLayer.place();
    }
  }
}

export function render2DInteractionMap(analysis, viewer, { hiddenTypes = DEFAULT_HIDDEN_TYPES, dark = false, title = '2D Interaction Map', fit = { w: 420, h: 300 } } = {}) {
  if (!analysis || !analysis.focusRes || !analysis.structure) return null;
  const visible = (analysis.interactions || []).filter((it) => !hiddenTypes.has(it.type));
  const project = viewer && typeof viewer.modelToScreen === 'function'
    ? (p) => viewer.modelToScreen({ x: p.x, y: p.y, z: p.z })
    : null;
  const model = layoutMap(analysis.focusRes.atoms, visible, {
    project,
    rings: analysis.structure.rings || [],
  });
  const svg = renderMap(model, { dark, labels: true, title, compact: false, fit });
  return { svg, model, visible };
}

window.PandaLensBridge = {
  TYPES: PANDALENS_TYPES,
  DEFAULT_HIDDEN_TYPES,
  analyzeModels,
  PandaLensOverlay,
  render2DInteractionMap,
};
window.dispatchEvent(new CustomEvent('pandalens-ready'));
