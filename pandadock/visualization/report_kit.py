"""
Brochure layout kit for PandaDock publication PDF reports.

Shared visual language with the Panda family (PandaDock / PandaInsight):
- A4 pages (8.27 x 11.69 in) drawn in one unified matplotlib inch coordinate
  system (origin top-left, y pointing down) via `PdfPages` — zero external C
  dependencies (no WeasyPrint/Cairo required) and 100% vector typography & plots.
- Deep indigo/cobalt gradient accents (#4A3AA8 -> #3453B8), warm beige figure
  panels (#F4F0EF), dark ruled tables (#161A2E), NPG colorblind-safe plot
  palette, interlocking ribbon PandaDock logo, and curved midnight-navy footers.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from .. import __version__  # noqa: E402

logger = logging.getLogger("pandadock.visualization.report_kit")

# A4 dimensions in inches
W, H = 8.27, 11.69
M = 0.62                    # outer margin
CW = W - 2 * M              # content width

# Editorial colour tokens (Panda family)
INK = "#12142B"             # headlines
NIGHT = "#161A2E"           # footer / table header
INDIGO = "#4A3AA8"          # gradient start
COBALT = "#3453B8"          # gradient end
ACCENT = "#2F4FB5"          # highlight values / links
BEIGE = "#F4F0EF"           # figure card surface
TEXT = "#2C2E3E"            # body text
SOFT = "#6B6F83"            # secondary labels / footers
RULE = "#D9DCE8"            # table hairlines
SPARK = "#7DD3FC"           # cyan accent on dark background
BRAND_A = "#4F46E5"         # logo gradient start
BRAND_B = "#0A84FF"         # logo gradient mid
BRAND_C = "#38BDF8"         # logo gradient end

# NPG-inspired scientific plot palette
NPG_RED = "#E64B35"
NPG_BLUE = "#4DBBD5"
NPG_TEAL = "#00A087"
NPG_NAVY = "#3C5488"
NPG_ORANGE = "#F39B7F"
NPG_SLATE = "#8491B4"
NPG_AMBER = "#D98A32"
NPG_CATEGORICAL = [NPG_BLUE, NPG_RED, NPG_TEAL, NPG_NAVY, NPG_ORANGE, NPG_AMBER, NPG_SLATE]

HEAD = ["Avenir Next", "Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"]
BODY = ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"]

KAPPA = 0.5522847498

PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": BODY,
    "font.size": 7.2,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.6,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": SOFT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "axes.facecolor": "none",
    "mathtext.fontset": "dejavusans",
}


def seq_cmap():
    return LinearSegmentedColormap.from_list(
        "pandadock_seq", ["#F8F6F2", "#CBE7EC", "#4DBBD5", "#2B7A9E", "#193A6F"]
    )


def fp(size: float, bold: bool = False, head: bool = False, italic: bool = False) -> FontProperties:
    return FontProperties(
        family=HEAD if head else BODY,
        weight="bold" if bold else "regular",
        style="italic" if italic else "normal",
        size=size,
    )


_FONTS = {}


def _adv(s: str, size: float, bold: bool, head: bool) -> float:
    """Unhinted advance width (inches) of `s`: glyph advances + kerning."""
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import Kerning, LoadFlags

    key = (size, bold, head)
    if key not in _FONTS:
        f = fm.get_font(fm.findfont(fp(size, bold, head)))
        f.set_size(size, 72)
        _FONTS[key] = f
    f = _FONTS[key]
    f.set_size(size, 72)
    adv, prev = 0.0, None
    for ch in s:
        g = f.load_char(ord(ch), flags=LoadFlags.NO_HINTING)
        cur = f.get_char_index(ord(ch))
        if prev is not None:
            adv += f.get_kerning(prev, cur, Kerning.UNFITTED) / 64
        adv += g.linearHoriAdvance / 65536
        prev = cur
    return adv / 72


def rect_path(x: float, y: float, w: float, h: float, radii=(0, 0, 0, 0)) -> MPath:
    """Closed Bezier path of a rectangle with per-corner radii (tl, tr, br, bl); y points down."""
    tl, tr, br, bl = radii
    k = KAPPA
    v, c = [], []

    v.append((x + tl, y))
    c.append(MPath.MOVETO)
    v.append((x + w - tr, y))
    c.append(MPath.LINETO)
    if tr:
        v += [(x + w - tr + k * tr, y), (x + w, y + tr - k * tr), (x + w, y + tr)]
        c += [MPath.CURVE4] * 3
    v.append((x + w, y + h - br))
    c.append(MPath.LINETO)
    if br:
        v += [(x + w, y + h - br + k * br), (x + w - br + k * br, y + h), (x + w - br, y + h)]
        c += [MPath.CURVE4] * 3
    v.append((x + bl, y + h))
    c.append(MPath.LINETO)
    if bl:
        v += [(x + bl - k * bl, y + h), (x, y + h - bl + k * bl), (x, y + h - bl)]
        c += [MPath.CURVE4] * 3
    v.append((x, y + tl))
    c.append(MPath.LINETO)
    if tl:
        v += [(x, y + tl - k * tl), (x + tl - k * tl, y), (x + tl, y)]
        c += [MPath.CURVE4] * 3
    v.append((x + tl, y))
    c.append(MPath.CLOSEPOLY)
    return MPath(v, c)


def parse_runs(text: str):
    """'a **bold** b' -> [(word, bold)]."""
    out, bold = [], False
    for part in re.split(r"(\*\*)", text):
        if part == "**":
            bold = not bold
            continue
        for w in part.split():
            out.append((w, bold))
    return out


class Page:
    def __init__(self, doc: "Doc", number_visible: bool = True):
        self.doc = doc
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.ax.set_autoscale_on(False)
        self.number_visible = number_visible
        self.y = M
        self._widths = {}

    def _w(self, s: str, size: float, bold: bool, head: bool = False) -> float:
        key = (s, size, bold, head)
        if key not in self._widths:
            if "$" in s:
                r = self.fig.canvas.get_renderer()
                w, _, _ = r.get_text_width_height_descent(s, fp(size, bold, head), True)
                self._widths[key] = w / self.fig.dpi
            else:
                self._widths[key] = _adv(s, size, bold, head)
        return self._widths[key]

    def t(self, x, y, s, size=9, color=TEXT, bold=False, head=False, ha="left", va="top",
          italic=False, z=5, **kw):
        return self.ax.text(
            x, y, s, fontproperties=fp(size, bold, head, italic), color=color,
            ha=ha, va=va, zorder=z, **kw,
        )

    def _lay(self, runs, w, size, head=False):
        sp = self._w(" ", size, False, head)
        lines, cur, cw = [], [], 0.0
        for word, bold in runs:
            ww = self._w(word, size, bold, head)
            if cur and cw + sp + ww > w:
                lines.append(cur)
                cur, cw = [], 0.0
            cw += (sp if cur else 0) + ww
            cur.append((word, bold, ww))
        if cur:
            lines.append(cur)
        return lines, sp

    def para_height(self, text, w, size=9.2, lead=1.5, head=False):
        lines, _ = self._lay(parse_runs(text), w, size, head)
        return len(lines) * size * lead / 72

    def flow(self, x, y, w, text, size=9.2, color=TEXT, lead=1.5, head=False, bold_all=False,
             ha="left", draw=True, z=5):
        runs = parse_runs(text)
        if bold_all:
            runs = [(wd, True) for wd, _ in runs]
        lines, sp = self._lay(runs, w, size, head)
        lh = size * lead / 72
        for line in lines:
            if draw:
                total = sum(ww for _, _, ww in line) + sp * (len(line) - 1)
                cx = x if ha == "left" else x + (w - total) / 2 if ha == "center" else x + w - total
                for word, bold, ww in line:
                    self.ax.text(cx, y, word, fontproperties=fp(size, bold, head), color=color,
                                 ha="left", va="top", zorder=z)
                    cx += ww + sp
            y += lh
        return y

    def paras(self, x, y, w, paragraphs: Iterable[str], size=9.2, gap=0.55, **kw):
        for p in paragraphs:
            y = self.flow(x, y, w, p, size=size, **kw) + size * gap / 72 * 1.5
        return y

    def columns(self, x, y, w, paragraphs: Sequence[str], n=2, gutter=0.32, size=9.0, lead=1.48):
        cw = (w - gutter * (n - 1)) / n
        gap = size * 0.55 / 72 * 1.5
        hs = [self.para_height(p, cw, size, lead) + gap for p in paragraphs]
        best, split = None, None
        if n == 2 and len(paragraphs) > 1:
            for k in range(1, len(paragraphs)):
                m = max(sum(hs[:k]), sum(hs[k:]))
                if best is None or m < best:
                    best, split = m, k
            groups = [paragraphs[:split], paragraphs[split:]]
        else:
            groups = [paragraphs] + [[] for _ in range(n - 1)]
        bottoms = []
        for i, g in enumerate(groups):
            bottoms.append(self.paras(x + i * (cw + gutter), y, cw, g, size=size, lead=lead))
        return max(bottoms)

    def heading(self, text, y=None, size=20, x=M, w=CW, color=INK):
        y = self.y if y is None else y
        lines, _ = self._lay([(wd, True) for wd in text.split()], w, size, head=True)
        for line in lines:
            self.t(x, y, " ".join(wd for wd, _, _ in line), size=size, bold=True, head=True, color=color)
            y += size * 1.12 / 72
        self.y = y + 0.1
        return self.y

    def shape(self, path, fc=BEIGE, ec="none", lw=0, z=1, alpha=1.0):
        p = PathPatch(path, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
        self.ax.add_patch(p)
        return p

    def rrect(self, x, y, w, h, r=0.14, fc=BEIGE, ec="none", lw=0, z=1, alpha=1.0, radii=None):
        return self.shape(rect_path(x, y, w, h, radii or (r, r, r, r)), fc, ec, lw, z, alpha)

    def grad(self, x, y, w, h, r=0.1, c1=INDIGO, c2=COBALT, z=1, radii=None, angle=0.0, cmap=None):
        patch = self.shape(rect_path(x, y, w, h, radii or (r, r, r, r)), fc="none", z=z)
        n = 64
        gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        a = np.radians(angle)
        arr = gx * np.cos(a) + gy * np.sin(a)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-9)
        cm = cmap or LinearSegmentedColormap.from_list("g", [c1, c2])
        im = self.ax.imshow(
            arr, extent=(x, x + w, y + h, y), cmap=cm, aspect="auto", zorder=z,
            interpolation="bilinear", vmin=0, vmax=1,
        )
        im.set_clip_path(patch)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        return patch

    def line(self, x0, y0, x1, y1, color=RULE, lw=0.8, z=4, **kw):
        self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=z, solid_capstyle="butt", **kw)

    def axes(self, x, y, w, h, **kw):
        ax = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)
        ax.set_zorder(10)
        return ax

    def panel_label(self, x, y, s):
        self.t(x, y, s, size=10, bold=True, head=True, color=INK, z=12)

    def figure_panel(self, x, y, w, h, fc=BEIGE):
        return self.rrect(x, y, w, h, r=0.16, fc=fc, z=1)

    def caption(self, x, y, w, name, text, size=7.8):
        s = f"**{name}:** {text}"
        return self.flow(x, y, w, s, size=size, lead=1.38)

    def sidebar(self, title, paragraphs: Sequence[str], h=5.0, w=3.37, size=8.6):
        self.grad(-0.4, -0.4, w + 0.4, h + 0.4, radii=(0, 0, 0.45, 0), angle=40)
        self.t(M, 0.78, title.upper(), size=8.6, bold=True, color="white", head=True)
        self.paras(M, 1.08, w - M - 0.34, paragraphs, size=size, color="#F0F0FA", lead=1.48)

    def table(self, x, y, widths, header, rows, size=8.0, row_h=0.29, head_h=0.33, align=None,
              bold_first=False, highlight=None, zebra=False, header_fc=NIGHT):
        n = len(widths)
        align = align or (["left"] + ["center"] * (n - 1))
        total = sum(widths)
        self.shape(rect_path(x, y, total, head_h), fc=header_fc, z=2)
        cx = x
        for wd, h_, a in zip(widths, header, align):
            tx = cx + 0.11 if a == "left" else cx + wd / 2 if a == "center" else cx + wd - 0.11
            self.t(tx, y + head_h / 2, h_, size=size, bold=True, color="white", ha=a, va="center", z=6)
            cx += wd
        yy = y + head_h
        for ri, row in enumerate(rows):
            if zebra and ri % 2:
                self.shape(rect_path(x, yy, total, row_h), fc="#F7F8FC", z=1)
            cx = x
            for ci, (wd, v, a) in enumerate(zip(widths, row, align)):
                tx = cx + 0.11 if a == "left" else cx + wd / 2 if a == "center" else cx + wd - 0.11
                b = (bold_first and ci == 0) or (highlight is not None and ri == highlight)
                self.t(
                    tx, yy + row_h / 2, str(v), size=size, bold=b,
                    color=ACCENT if (ci == 0 and not bold_first) else TEXT, ha=a, va="center", z=6,
                )
                cx += wd
            yy += row_h
            self.line(x, yy, x + total, yy, color=RULE, lw=0.7)
        return yy + 0.08

    def _nlines(self, text, w, size, bold=False):
        runs = parse_runs(text)
        if bold:
            runs = [(a, True) for a, _ in runs]
        return len(self._lay(runs, w, size)[0])

    def fact_table(self, x, y, w, rows, label_w=1.12, row_h=0.58, size=8.0):
        h = row_h * len(rows)
        pad = 0.18
        self.rrect(x, y, w + 2 * pad, h + 2 * pad, r=0.2, fc=BEIGE)
        for i, (k, v) in enumerate(rows):
            yy = y + pad + i * row_h
            self.grad(x + pad, yy, label_w, row_h - 0.012, r=0, angle=60)
            self.flow(
                x + pad + 0.12,
                yy + row_h / 2 - self._nlines(k, label_w - 0.22, 8.6, True) * 8.6 * 1.25 / 144,
                label_w - 0.22, k, size=8.6, bold_all=True, lead=1.25, color="white",
            )
            self.shape(rect_path(x + pad + label_w, yy, w - label_w, row_h - 0.012), fc="white", z=1)
            ny = self._nlines(v, w - label_w - 0.28, size)
            self.flow(
                x + pad + label_w + 0.13,
                yy + row_h / 2 - ny * size * 1.3 / 144,
                w - label_w - 0.28, v, size=size, lead=1.3,
            )
        return y + h + 2 * pad

    def tile(self, x, y, w, h, title, body, size=8.1):
        self.grad(x, y, w, h, r=0.09, angle=70)
        self.t(x + 0.16, y + 0.2, title, size=11.8, bold=True, head=True, color="white")
        self.flow(x + 0.16, y + 0.52, w - 0.3, body, size=size, color="#F0F0FA", lead=1.45)

    def icon(self, kind, cx, cy, s=0.55, color="white", lw=1.4):
        draw_icon(self.ax, kind, cx, cy, s, color, lw)

    def navy_footer(self, y0=10.34, show_logo=True, right_text=None, brand_size=17):
        x1, x2 = 5.3, 6.55
        v = [
            (0, y0), (x1, y0), (x1 + 0.65, y0), (x2 - 0.2, y0 + 0.25), (x2 + 0.05, y0 + 0.62),
            (W, H + 1), (0, H + 1), (0, y0),
        ]
        c = [
            MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY,
        ]
        self.shape(MPath(v, c), fc=NIGHT, z=1)
        if show_logo:
            self.logo(M, y0 + 0.62, size=brand_size)
        if right_text:
            self.line(M + 2.55, y0 + 0.5, M + 2.55, y0 + 1.0, color="#FFFFFF", lw=0.9, z=6)
            self.t(M + 2.80, y0 + 0.75, right_text, size=11, bold=True, color="white", head=True, va="center")

    def logo(self, x, y, size=17, color="white", mark=True):
        """Official PandaDock logo mark (panda head with atom & Python eyes) + wordmark."""
        s = size / 72
        if mark:
            box = s * 2.15
            x0, y0 = x, y + s * 0.62 - box / 2
            logo_candidates = [
                Path(__file__).parent / "assets" / "pandadock_mark.png",
                Path(__file__).parent.parent.parent / "PandaDock.png",
            ]
            logo_path = next((p for p in logo_candidates if p.exists()), None)
            if logo_path is not None:
                import matplotlib.image as mpimg

                img = mpimg.imread(str(logo_path))
                badge = self.rrect(
                    x0, y0, box, box, r=box * 0.22,
                    fc="#FFFFFF", ec="#3B82F6", lw=0.6, z=6.8,
                )
                im = self.ax.imshow(
                    img, extent=(x0, x0 + box, y0 + box, y0),
                    aspect="equal", zorder=7.2, interpolation="bilinear",
                )
                im.set_clip_path(badge)
                self.ax.set_xlim(0, W)
                self.ax.set_ylim(H, 0)
            else:
                self.rrect(x0, y0, box, box, r=box * 0.22, fc=BRAND_B, ec="none", z=7.0)
                self.t(x0 + box / 2, y0 + box / 2, "P", size=size * 0.9, bold=True, head=True,
                       color="white", ha="center", va="center", z=8.0)

        tx = x + (s * 2.55 if mark else 0)
        self.t(tx, y + s * 0.62, "Panda", size=size, bold=True, head=True, color=color, va="center")
        wp = self._w("Panda", size, True, True)
        self.t(tx + wp, y + s * 0.62, "Dock", size=size, head=True,
               color="#48B9D0" if color == "white" else ACCENT, va="center", bold=True)

    def page_footer(self, label=None):
        if self.number_visible:
            self.t(W - M, H - 0.46, str(self.doc.pages.index(self) + 1),
                   size=9, color=SOFT, ha="right", va="center")
        self.t(M, H - 0.46, label or f"PandaDock v{__version__}  ·  {self.doc.title}",
               size=7, color=SOFT, va="center")


def draw_icon(ax, kind, cx, cy, s=0.55, color="white", lw=1.4):
    """Crisp vector line icons matching the Panda family brochure kit."""
    z = 8
    kw = dict(color=color, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")

    def L(pts):
        ax.plot([p[0] * s + cx for p in pts], [p[1] * s + cy for p in pts], **kw)

    def C(x, y, r, fill=False):
        ax.add_patch(Circle((cx + x * s, cy + y * s), r * s,
                            fc=color if fill else "none", ec=color, lw=lw, zorder=z))

    if kind == "target":
        C(0, 0, 0.8)
        C(0, 0, 0.5)
        C(0, 0, 0.2, True)
        L([(0.0, 0.0), (0.8, -0.8)])
        L([(0.5, -0.8), (0.8, -0.8), (0.8, -0.5)])
    elif kind == "chart":
        L([(-0.8, -0.8), (-0.8, 0.8), (0.8, 0.8)])
        L([(-0.6, 0.4), (-0.2, -0.1), (0.1, 0.2), (0.7, -0.6)])
        C(0.7, -0.6, 0.08, True)
    elif kind == "check":
        C(0, 0, 0.8)
        L([(-0.4, 0.0), (-0.1, 0.32), (0.45, -0.3)])
    elif kind == "molecule":
        hexa = [(0.52 * np.cos(a), 0.52 * np.sin(a)) for a in np.radians(np.arange(0, 361, 60))]
        L([(p[0] - 0.22, p[1]) for p in hexa])
        L([(0.30, 0.0), (0.72, -0.36)])
        L([(-0.48, 0.45), (-0.78, 0.72)])
    elif kind == "flame":
        L([(0, -0.85), (0.5, -0.1), (0.35, 0.6), (0, 0.85), (-0.35, 0.6), (-0.5, -0.1), (0, -0.85)])
    elif kind == "layers":
        for dy in (-0.45, 0.0, 0.45):
            L([(-0.8, dy), (0, dy - 0.35), (0.8, dy), (0, dy + 0.35), (-0.8, dy)])


class Doc:
    def __init__(self, title: str, kind: str = "Molecular Docking Report"):
        self.title = title
        self.kind = kind
        self.pages: List[Page] = []
        self.fig_n = 0

    def page(self, number_visible: bool = True) -> Page:
        p = Page(self, number_visible)
        self.pages.append(p)
        return p

    def next_fig(self) -> str:
        self.fig_n += 1
        return f"Figure {self.fig_n}"

    def save(self, path, footer: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
        with PdfPages(
            path,
            metadata={
                "Title": self.title,
                "Creator": f"PandaDock {__version__}",
                "CreationDate": datetime.now(),
            },
        ) as pdf:
            for i, p in enumerate(self.pages):
                if footer and i > 0 and not getattr(p, "no_footer", False):
                    p.page_footer()
                pdf.savefig(p.fig)
                plt.close(p.fig)
        return path


def cover(doc: Doc, title: str, subtitle: str, highlights: Sequence[tuple], quote: str,
          quote_by: str, facts: Sequence[tuple], art: Optional[Callable] = None, kind: Optional[str] = None):
    """Cover page: kicker, bold title, indigo-ruled subtitle, 3-KPI banner, pull-quote + fact table, navy footer."""
    p = doc.page(number_visible=False)
    p.no_footer = True
    p.grad(0, 0, W, 5.35, r=0, c1="#F6F5F5", c2="#ECEBEC", angle=90, z=0)
    p.rrect(0, 5.0, W, 0.5, r=0, fc="white", z=0)
    if art is not None:
        art(p)
    p.t(M, 0.62, (kind or doc.kind).upper(), size=9.5, color=INK, head=True)

    size = 34
    while size > 20:
        lines, _ = p._lay([(w, True) for w in title.split()], 3.9, size, head=True)
        if len(lines) <= 4:
            break
        size -= 2
    y = 0.95
    for line in lines:
        p.t(M, y, " ".join(w for w, _, _ in line), size=size, bold=True, head=True, color=INK)
        y += size * 1.06 / 72
    y += 0.25
    sub_h = p.flow(M + 0.16, y, 3.65, subtitle, size=9.3, color=TEXT, lead=1.45, draw=True) - y
    p.line(M, y + 0.02, M, y + sub_h - 0.05, color=INDIGO, lw=2.0)

    # 3-KPI gradient banner
    by = 4.55
    p.grad(M, by, CW, 1.28, r=0.12, angle=15)
    n = len(highlights)
    cw = CW / n
    for i, (icon, text) in enumerate(highlights):
        cx = M + cw * (i + 0.5)
        p.icon(icon, cx, by + 0.42, 0.25, lw=1.5)
        p.flow(M + cw * i + 0.24, by + 0.80, cw - 0.48, text, size=8.6, color="#EEF0FB", lead=1.38, ha="center")
        if i:
            p.line(M + cw * i, by + 0.2, M + cw * i, by + 1.08, color="#FFFFFF", lw=0.6, alpha=0.4)

    # Pull-quote + fact table
    qy = 6.32
    p.ax.text(M - 0.02, qy - 0.2, "\u201c", fontsize=70, fontweight="bold", family="DejaVu Serif",
              color=ACCENT, ha="left", va="top", zorder=5)
    qsize = 14.5
    while qsize > 9.5:
        if p.para_height(quote, 3.2, qsize, 1.3, head=True) < 2.9:
            break
        qsize -= 1
    ye = p.flow(M, qy + 0.85, 3.2, quote, size=qsize, bold_all=True, head=True, lead=1.28, color=INK)
    p.t(M + 3.2, ye + 0.12, "— " + quote_by.split("|")[0].strip(), size=7.5, bold=True, color=INK, ha="right")
    if "|" in quote_by:
        p.t(M, ye + 0.35, quote_by.split("|")[1].strip(), size=7.2, color=SOFT)

    p.fact_table(4.28, qy, 3.10, facts, label_w=1.08, row_h=0.58)
    p.navy_footer(y0=10.34, right_text=doc.kind)
    return p


def sidebar_page(doc: Doc, sidebar_title: str, sidebar_text: Sequence[str], heading: str,
                 body: Sequence[str], sidebar_h: Optional[float] = None):
    p = doc.page()
    x = 3.78
    w = W - M - x
    sw = 3.37
    side_end = p.paras(M, 1.08, sw - M - 0.34, sidebar_text, size=8.6, draw=False, lead=1.48)
    lines, _ = p._lay([(wd, True) for wd in heading.split()], w, 19.0, head=True)
    head_end = 0.78 + len(lines) * 19.0 * 1.12 / 72 + 0.2
    body_end = p.paras(x, head_end, w, body, size=9.0, draw=False)
    h = sidebar_h or max(3.4, side_end + 0.42, min(body_end, 5.4) - 0.1)
    p.sidebar(sidebar_title, sidebar_text, h=h, w=sw)
    p.heading(heading, y=0.78, size=19.0, x=x, w=w)
    y = p.paras(x, p.y + 0.1, w, body, size=9.0)
    p.sidebar_bottom = h
    return p, y


def figure_block(doc: Doc, p: Page, x, y, w, h, caption_text: str,
                 draw: Callable[[Page, float, float, float, float], None], panel: bool = True):
    if panel:
        p.figure_panel(x, y, w, h)
    draw(p, x, y, w, h)
    name = doc.next_fig()
    return p.caption(x + 0.05, y + h + 0.09, w - 0.1, name, caption_text)


def end_page(doc: Doc, tiles: Sequence[tuple], methods_rows: Sequence[tuple], references: Sequence[str],
             contact: str = "https://github.com/pritampanda15/PandaDock",
             heading: str = "Analysis modules in this report",
             methods_heading: str = "Methods and parameters"):
    p = doc.page()
    p.heading(heading, y=0.70, size=19)
    n = len(tiles)
    gap = 0.17
    tw = (CW - gap * (n - 1)) / n
    for i, (t_head, body) in enumerate(tiles):
        p.tile(M + i * (tw + gap), 1.32, tw, 1.58, t_head, body)

    y = 3.18
    p.heading(methods_heading, y=y, size=16.5)
    y = p.y + 0.02
    for k, v in methods_rows:
        p.t(M, y, k, size=8.2, bold=True, color=INK)
        y = max(p.flow(M + 1.42, y, CW - 1.42, v, size=8.2, lead=1.38), y + 0.19) + 0.06

    # Curved beige references band
    by = max(y + 0.14, 7.15)
    tt = np.linspace(0, 1, 30)
    c0, c1, c2, c3 = (
        np.array([W * 0.40, by + 0.42]),
        np.array([W * 0.58, by + 0.42]),
        np.array([W * 0.55, by]),
        np.array([W * 0.74, by]),
    )
    curve = [
        tuple((1 - t) ** 3 * c0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * c3)
        for t in tt
    ]
    pts = [(0, by + 0.42)] + curve + [(W - 0.9, by), (W - 0.2, by + 0.3), (W, by + 0.68), (W, 10.6), (0, 10.6)]
    p.shape(
        MPath(pts + [pts[0]], [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1) + [MPath.CLOSEPOLY]),
        fc=BEIGE, z=0,
    )
    p.t(M - 0.04, by + 0.52, "References", size=16.5, bold=True, head=True, color=INK)
    yy = by + 1.00
    for i, ref in enumerate(references, 1):
        p.t(M, yy, f"{i}.", size=7.8, bold=True, color=INK)
        yy = p.flow(M + 0.28, yy, CW - 0.28, ref, size=7.6, lead=1.35) + 0.05

    y0 = max(10.18, yy + 0.08)
    k = min(1.0, (H - y0) / 1.49)
    p.navy_footer(y0=y0, show_logo=False)
    for dy, key, val in (
        (0.24, "Contact us:", contact),
        (0.46, "Learn more:", "pip install pandadock   ·   pandadock dock"),
    ):
        p.t(M, y0 + dy * k, key, size=8.0, bold=True, color="white")
        p.t(M + 0.86, y0 + dy * k, val, size=8.0, color="white")
    p.logo(M, y0 + 0.84 * k, size=19 * max(k, 0.8))
    p.t(M + 2.55, y0 + 1.04 * k, f"v{__version__}  ·  generated {datetime.now():%Y-%m-%d %H:%M}",
        size=6.8, color="#9AA0B8", va="center")
    p.no_footer = True
    return p


def art_trace(p: Page, xyz: np.ndarray, extra: Optional[np.ndarray] = None,
              box=(4.55, 0.12, 3.65, 4.15)):
    """3-layer depth-shaded grey Cα ribbon trace + highlighted bound ligand on cover page."""
    from matplotlib.collections import LineCollection

    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 3:
        return
    center = xyz.mean(0)
    xyz_c = xyz - center
    _, _, vt = np.linalg.svd(xyz_c, full_matrices=False)
    th = np.radians(25)
    rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    pts = (xyz_c @ vt.T) @ rot.T
    x0, y0, w, h = box
    scale = 0.45 * min(w, h) / max(np.abs(pts[:, :2]).max(), 1e-6)
    sx = x0 + w / 2 + pts[:, 0] * scale
    sy = y0 + h / 2 + pts[:, 1] * scale
    d = (pts[:, 2] - pts[:, 2].min()) / max(np.ptp(pts[:, 2]), 1e-6)

    # Skip chain breaks (> 6 A in 3D) so multi-chain proteins don't draw stray chords
    step_dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    valid = [i for i in range(len(sx) - 1) if step_dists[i] < 6.0]
    if not valid:
        return
    segs = [[(sx[i], sy[i]), (sx[i + 1], sy[i + 1])] for i in valid]
    depth_mid = np.array([(d[i] + d[i + 1]) / 2.0 for i in valid])
    order = np.argsort(depth_mid)
    cols = [(0.78 - 0.50 * depth_mid[k], 0.78 - 0.50 * depth_mid[k], 0.795 - 0.50 * depth_mid[k]) for k in order]

    for lw, shade, a in ((9.0, 0.0, 1.0), (4.2, 0.12, 0.9), (1.7, 0.28, 0.8)):
        lc = LineCollection(
            [segs[k] for k in order],
            linewidths=lw * (0.6 + 0.8 * depth_mid[order]) * 0.82,
            colors=[tuple(min(1.0, c + shade) for c in col) for col in cols],
            alpha=a, capstyle="round", zorder=2,
        )
        p.ax.add_collection(lc)

    if extra is not None and len(extra):
        e = ((np.asarray(extra, dtype=float) - center) @ vt.T) @ rot.T
        ex, ey = x0 + w / 2 + e[:, 0] * scale, y0 + h / 2 + e[:, 1] * scale
        p.ax.scatter([float(np.mean(ex))], [float(np.mean(ey))], s=340, color=INDIGO, alpha=0.18, zorder=3)
        p.ax.scatter(ex, ey, s=30, color="#FFFFFF", edgecolors=INDIGO, linewidths=0.7, zorder=4)


def style_axes(ax, grid=False, spines=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
    ax.tick_params(length=2.5, width=0.6)
    if grid:
        ax.grid(color=RULE, lw=0.5)
