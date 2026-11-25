// Small vector toolkit. Points are {x, y, z}; nothing here allocates more than it must.

export const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const scale = (a, s) => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a, b) => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const norm = (a) => Math.hypot(a.x, a.y, a.z);
export const unit = (a) => {
  const n = norm(a) || 1;
  return { x: a.x / n, y: a.y / n, z: a.z / n };
};
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
export const dist2 = (a, b) => {
  const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
};
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });

export function centroid(pts) {
  let x = 0, y = 0, z = 0;
  for (const p of pts) { x += p.x; y += p.y; z += p.z; }
  const n = pts.length || 1;
  return { x: x / n, y: y / n, z: z / n };
}

/** Angle at b, in degrees. */
export function angle(a, b, c) {
  const u = sub(a, b), v = sub(c, b);
  const cos = dot(u, v) / ((norm(u) * norm(v)) || 1);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

/** IUPAC dihedral a-b-c-d in degrees, range (-180, 180]. */
export function dihedral(a, b, c, d) {
  const b0 = sub(a, b), b1 = sub(c, b), b2 = sub(d, c);
  const b1u = unit(b1);
  const v = sub(b0, scale(b1u, dot(b0, b1u)));
  const w = sub(b2, scale(b1u, dot(b2, b1u)));
  const x = dot(v, w);
  const y = dot(cross(b1u, v), w);
  return (Math.atan2(y, x) * 180) / Math.PI;
}

/** Ring plane normal via Newell's method; robust for slightly puckered rings. */
export function ringNormal(pts) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    nx += (p.y - q.y) * (p.z + q.z);
    ny += (p.z - q.z) * (p.x + q.x);
    nz += (p.x - q.x) * (p.y + q.y);
  }
  return unit({ x: nx, y: ny, z: nz });
}

/** Acute angle between two plane normals, 0–90°. */
export function planeAngle(n1, n2) {
  const c = Math.abs(dot(n1, n2));
  return (Math.acos(Math.min(1, c)) * 180) / Math.PI;
}

/** Distance from p to the axis through c along unit normal n, i.e. lateral offset. */
export function offsetFromAxis(p, c, n) {
  const v = sub(p, c);
  const along = dot(v, n);
  return Math.sqrt(Math.max(0, dot(v, v) - along * along));
}

/** Smallest signed difference between two angles in degrees. */
export function angleDelta(a, b) {
  let d = b - a;
  while (d > 180) d -= 360;
  while (d <= -180) d += 360;
  return d;
}

export function applyTransform(p, T) {
  const R = T.R;
  return {
    x: R[0] * p.x + R[1] * p.y + R[2] * p.z + T.t.x,
    y: R[3] * p.x + R[4] * p.y + R[5] * p.z + T.t.y,
    z: R[6] * p.x + R[7] * p.y + R[8] * p.z + T.t.z,
  };
}
