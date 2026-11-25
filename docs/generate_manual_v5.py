#!/usr/bin/env python3
"""
Generate the official publishable PandaDock v5.0 User Manual PDF.

Uses the exact Panda-family A4 brochure layout engine (`pandadock.visualization.report_kit`)
with deep indigo/cobalt gradients, warm beige cards, midnight-navy headers & curved footers,
official PandaDock logo, real 3D cover ribbon artwork, auto-paginating option tables,
terminal recipe boxes, clickable internal PDF Table of Contents links, clickable
"↑ Contents" back-links on every page, and full PDF sidebar bookmarks via PyMuPDF.

Usage:
    python docs/generate_manual_v5.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pymupdf as fitz  # PyMuPDF

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

plt.rcParams["figure.max_open_warning"] = 0

# Ensure repo root is importable when executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pandadock import __version__  # noqa: E402
from pandadock.visualization.report_kit import (  # noqa: E402
    ACCENT,
    BEIGE,
    COBALT,
    CW,
    H,
    INDIGO,
    INK,
    M,
    NIGHT,
    NPG_AMBER,
    NPG_BLUE,
    NPG_RED,
    NPG_TEAL,
    PLOT_RC,
    RULE,
    SOFT,
    SPARK,
    TEXT,
    W,
    Doc,
    Page,
    art_trace,
    cover,
    end_page,
    rect_path,
)

MONO = ["DejaVu Sans Mono", "Menlo", "Courier New"]
Y_TOP = 0.88
Y_MAX = 10.82


def _clean_md(text: str) -> str:
    """Convert inline markdown backticks into clean bold typography without literal ` characters."""
    # Avoid double-bolding **`code`** -> **code**, convert standalone `code` -> **code**
    text = re.sub(r"\*\*`([^`]+)`\*\*", r"**\1**", text)
    text = re.sub(r"`([^`]+)`", r"**\1**", text)
    return text


def _fp_mono(size: float, bold: bool = False) -> FontProperties:
    return FontProperties(family=MONO, weight="bold" if bold else "regular", size=size)


def _extract_cover_coords() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract real 3D C-alpha ribbon and bound ligand coordinates for cover artwork."""
    candidates = [
        REPO_ROOT / "test" / "v5_validation" / "03_waters_3hs4" / "complex1.pdb",
        REPO_ROOT / "test" / "v5_validation" / "04_covalent_6lu7_mpro" / "covalent_complex1.pdb",
    ]
    pdb_path = next((p for p in candidates if p.exists()), None)
    if pdb_path is None:
        return None, None
    ca_pts, lig_pts = [], []
    for ln in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.startswith(("ATOM  ", "HETATM")):
            continue
        aname = ln[12:16].strip()
        resn = ln[17:20].strip().upper()
        try:
            xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        except ValueError:
            continue
        if ln.startswith("ATOM  ") and aname == "CA":
            ca_pts.append(xyz)
        elif resn in ("LIG", "COV", "AZM", "UNL"):
            lig_pts.append(xyz)
    ca = np.asarray(ca_pts, dtype=float) if len(ca_pts) >= 10 else None
    lig = np.asarray(lig_pts, dtype=float) if len(lig_pts) >= 3 else None
    return ca, lig


class ManualBuilder:
    """Auto-paginating Panda-family A4 manual builder with exact link tracking."""

    def __init__(self, doc: Doc, toc_page_index: int = 1):
        self.doc = doc
        self.toc_page_index = toc_page_index  # 0-based index of TOC page (Page 2 = index 1)
        self.p: Optional[Page] = None
        self.y: float = Y_TOP
        self.chapter_label: str = ""
        self.entries: List[Dict[str, Any]] = []  # {level, number, title, subtitle, page_1based}
        self.back_links: List[Tuple[int, Tuple[float, float, float, float]]] = []  # (page_idx, (x,y,w,h))

    def _draw_running_header(self, p: Page) -> None:
        page_idx = self.doc.pages.index(p)
        # Subtle top rule
        p.line(M, 0.66, W - M, 0.66, color=RULE, lw=0.7, z=3)
        p.t(M, 0.45, f"PANDADOCK {__version__} OFFICIAL USER MANUAL", size=7.0, bold=True, head=True, color=SOFT)
        if self.chapter_label:
            p.t(W - M - 1.05, 0.45, self.chapter_label.upper(), size=7.0, bold=True, head=True, color=INDIGO, ha="right")
        # Clickable "↑ Contents" back-pill (top right)
        bx, by, bw, bh = W - M - 0.92, 0.38, 0.92, 0.21
        p.rrect(bx, by, bw, bh, r=0.06, fc="#EEF1FA", ec="#C8D2EE", lw=0.5, z=4)
        p.t(bx + bw / 2, by + bh / 2, "↑ Contents", size=6.8, bold=True, head=True, color=ACCENT, ha="center", va="center", z=6)
        self.back_links.append((page_idx, (bx, by, bw, bh)))

    def new_page(self) -> Page:
        p = self.doc.page(number_visible=True)
        self.p = p
        self.y = Y_TOP
        self._draw_running_header(p)
        return p

    def ensure_space(self, needed: float) -> None:
        if self.p is None or (self.y + needed) > Y_MAX:
            self.new_page()

    def current_page_1based(self) -> int:
        return self.doc.pages.index(self.p) + 1

    def chapter(
        self,
        number: str,
        title: str,
        subtitle: str,
        start_new_page: bool = True,
        toc_title: Optional[str] = None,
    ) -> None:
        self.chapter_label = f"Ch. {number} · {toc_title or title}"
        if start_new_page or self.p is None or (self.y + 2.4) > Y_MAX:
            self.new_page()
        else:
            self.y += 0.18
        pg_num = self.current_page_1based()
        self.entries.append({
            "level": 1,
            "number": number,
            "title": toc_title or title,
            "full_title": title,
            "subtitle": _clean_md(subtitle),
            "page": pg_num,
        })
        p = self.p
        # Gradient chapter pill badge
        p.grad(M, self.y, 1.18, 0.26, r=0.07, angle=20, z=2)
        p.t(M + 0.59, self.y + 0.13, f"CHAPTER {number}", size=7.4, bold=True, head=True, color="white", ha="center", va="center", z=6)
        self.y += 0.34
        self.y = p.heading(title, y=self.y, size=18.5, x=M, w=CW, color=INK)
        sub_end = p.flow(M + 0.14, self.y - 0.02, CW - 0.14, _clean_md(subtitle), size=9.0, color=SOFT, lead=1.42)
        p.line(M, self.y, M, sub_end - 0.04, color=INDIGO, lw=2.0, z=5)
        self.y = sub_end + 0.18

    def section(self, title: str, record_toc: bool = False, toc_label: Optional[str] = None) -> None:
        self.ensure_space(0.65)
        p = self.p
        clean_title = title.replace("`", "")
        if record_toc:
            self.entries.append({
                "level": 2,
                "number": "",
                "title": toc_label or clean_title,
                "subtitle": "",
                "page": self.current_page_1based(),
            })
        p.t(M, self.y, clean_title, size=12.0, bold=True, head=True, color=INK)
        self.y += 0.23
        p.line(M, self.y, W - M, self.y, color=RULE, lw=0.7, z=3)
        self.y += 0.11

    def command_banner(self, syntax: str, badge: str = "CLI COMMAND", record_toc: bool = True, toc_label: Optional[str] = None) -> None:
        self.ensure_space(0.85)
        p = self.p
        if record_toc:
            self.entries.append({
                "level": 2,
                "number": "",
                "title": toc_label or syntax.split("[")[0].strip(),
                "subtitle": "",
                "page": self.current_page_1based(),
            })
        bh = 0.36
        p.rrect(M, self.y, CW, bh, r=0.08, fc=NIGHT, z=2)
        fsize = 7.6 if len(syntax) > 64 else (8.1 if len(syntax) > 54 else 8.6)
        p.ax.text(M + 0.14, self.y + bh / 2, syntax, fontproperties=_fp_mono(fsize, bold=True),
                  color=SPARK, ha="left", va="center", zorder=6)
        bw = max(0.95, len(badge) * 0.066 + 0.22)
        p.rrect(W - M - bw - 0.08, self.y + 0.07, bw, bh - 0.14, r=0.05, fc=INDIGO, z=4)
        p.t(W - M - bw / 2 - 0.08, self.y + bh / 2, badge, size=6.5, bold=True, head=True,
            color="white", ha="center", va="center", z=6)
        self.y += bh + 0.11

    def para(self, text: str, size: float = 8.9, lead: float = 1.44, gap: float = 0.09) -> None:
        text = _clean_md(text)
        if self.p is None:
            self.new_page()
        h = self.p.para_height(text, CW, size=size, lead=lead)
        self.ensure_space(h + gap)
        self.y = self.p.flow(M, self.y, CW, text, size=size, lead=lead, color=TEXT) + gap

    def bullets(self, items: Sequence[str], size: float = 8.7, lead: float = 1.40) -> None:
        for item in items:
            item = _clean_md(item)
            if self.p is None:
                self.new_page()
            h = self.p.para_height(item, CW - 0.22, size=size, lead=lead)
            self.ensure_space(h + 0.07)
            self.p.t(M + 0.05, self.y + 0.01, "•", size=size + 0.8, bold=True, color=INDIGO)
            self.y = self.p.flow(M + 0.20, self.y, CW - 0.20, item, size=size, lead=lead, color=TEXT) + 0.055
        self.y += 0.04

    def callout(self, title: str, body: str, tone: str = "plain") -> None:
        title = title.replace("`", "")
        body = _clean_md(body)
        styles = {
            "plain": (BEIGE, INDIGO, INK),
            "tip": ("#EDF7F4", NPG_TEAL, "#0F5143"),
            "warn": ("#FDF4EA", NPG_AMBER, "#7C3E00"),
            "info": ("#EEF2FB", ACCENT, INK),
        }
        bg, accent, tcol = styles.get(tone, styles["plain"])
        if self.p is None:
            self.new_page()
        text_w = CW - 0.36
        bh = self.p.para_height(body, text_w, size=8.5, lead=1.42)
        total_h = 0.24 + bh + 0.20
        self.ensure_space(total_h + 0.10)
        p = self.p
        p.rrect(M, self.y, CW, total_h, r=0.10, fc=bg, z=1)
        p.rrect(M, self.y, 0.065, total_h, radii=(0.10, 0, 0, 0.10), fc=accent, z=2)
        p.t(M + 0.18, self.y + 0.11, title.upper(), size=7.6, bold=True, head=True, color=tcol, z=5)
        p.flow(M + 0.18, self.y + 0.30, text_w, body, size=8.5, lead=1.42, color=TEXT, z=5)
        self.y += total_h + 0.12

    def code_block(self, code: str, caption: Optional[str] = None) -> None:
        lines = code.strip("\n").splitlines()
        line_h = 0.158
        header_h = 0.22 if caption else 0.10
        total_h = header_h + len(lines) * line_h + 0.14
        self.ensure_space(total_h + 0.10)
        p = self.p
        p.rrect(M, self.y, CW, total_h, r=0.09, fc=NIGHT, z=2)
        if caption:
            # Subtle traffic dots + caption
            for i, col in enumerate(("#FF5F56", "#FFBD2E", "#27C93F")):
                p.ax.add_patch(plt.Circle((M + 0.16 + i * 0.12, self.y + 0.115), 0.032, fc=col, ec="none", zorder=5))
            p.t(M + 0.56, self.y + 0.115, caption, size=6.9, bold=True, head=True, color="#CBD5E1", va="center", z=5)
            p.line(M + 0.08, self.y + 0.21, W - M - 0.08, self.y + 0.21, color="#2C3352", lw=0.6, z=4)
        yy = self.y + header_h + 0.06
        for ln in lines:
            stripped = ln.lstrip()
            col = "#94A3B8" if stripped.startswith("#") else "#F8FAFC"
            p.ax.text(M + 0.14, yy, ln, fontproperties=_fp_mono(7.5), color=col, ha="left", va="top", zorder=6)
            yy += line_h
        self.y += total_h + 0.12

    def option_table(
        self,
        rows: Sequence[Tuple[str, str, str]],
        headers: Tuple[str, str, str] = ("Option / Flag", "Default", "Plain-English Meaning & When to Use It"),
        col_widths: Tuple[float, float, float] = (1.82, 1.12, CW - 2.94),
    ) -> None:
        """Auto-wrapping, auto-paginating Panda-family option table."""
        w0, w1, w2 = col_widths
        head_h = 0.27
        fsize = 7.8
        lead = 1.34
        clean_rows = [(r0.replace("`", ""), r1.replace("`", ""), _clean_md(r2)) for r0, r1, r2 in rows]

        def _draw_header():
            p = self.p
            p.shape(rect_path(M, self.y, CW, head_h), fc=NIGHT, z=2)
            p.t(M + 0.09, self.y + head_h / 2, headers[0], size=7.8, bold=True, color="white", va="center", z=6)
            p.t(M + w0 + 0.07, self.y + head_h / 2, headers[1], size=7.8, bold=True, color="white", va="center", z=6)
            p.t(M + w0 + w1 + 0.09, self.y + head_h / 2, headers[2], size=7.8, bold=True, color="white", va="center", z=6)
            self.y += head_h

        def _row_height(r0: str, r1: str, r2: str) -> float:
            h0 = self.p.para_height(f"**{r0}**", w0 - 0.16, size=fsize, lead=lead)
            h1 = self.p.para_height(r1, w1 - 0.14, size=fsize - 0.3, lead=lead)
            h2 = self.p.para_height(r2, w2 - 0.18, size=fsize, lead=lead)
            return max(h0, h1, h2, 0.18) + 0.11

        if self.p is None:
            self.new_page()
        first_h = _row_height(*clean_rows[0]) if clean_rows else 0.25
        self.ensure_space(head_h + first_h + 0.06)
        _draw_header()

        for ri, (opt, dflt, desc) in enumerate(clean_rows):
            rh = _row_height(opt, dflt, desc)
            if self.y + rh > Y_MAX:
                self.new_page()
                _draw_header()
            p = self.p
            if ri % 2 == 1:
                p.shape(rect_path(M, self.y, CW, rh), fc="#F7F8FC", z=1)
            pad_top = 0.055
            p.flow(M + 0.09, self.y + pad_top, w0 - 0.16, f"**{opt}**", size=fsize, lead=lead, color=ACCENT, z=6)
            p.flow(M + w0 + 0.07, self.y + pad_top, w1 - 0.14, dflt, size=fsize - 0.3, lead=lead, color=SOFT, z=6)
            p.flow(M + w0 + w1 + 0.09, self.y + pad_top, w2 - 0.18, desc, size=fsize, lead=lead, color=TEXT, z=6)
            self.y += rh
            p.line(M, self.y, W - M, self.y, color=RULE, lw=0.55, z=4)
        self.y += 0.12

    def draw_workflow_diagram(self) -> None:
        """Crisp 6-step vector pipeline diagram matching the PandaDock visual grammar."""
        bh = 1.24
        self.ensure_space(bh + 0.16)
        p = self.p
        p.rrect(M, self.y, CW, bh, r=0.14, fc=BEIGE, z=1)
        steps = [
            ("1", "Prepare", "Add H, pH 7.4\n& 3D ligand"),
            ("2", "Grid Box", "Click X-ray lig\nor cavities"),
            ("3", "Dock", "Spin bonds &\nL-BFGS relax"),
            ("4", "Audit", "11-pt PoseBusters\ngeometry"),
            ("5", "Inspect", "PandaLens 3D\n& report.pdf"),
            ("6", "MD Handoff", "OpenMM /\nGROMACS"),
        ]
        n = len(steps)
        gap = 0.13
        bw = (CW - 0.28 - gap * (n - 1)) / n
        y0 = self.y + 0.17
        for i, (num, title, sub) in enumerate(steps):
            bx = M + 0.14 + i * (bw + gap)
            p.grad(bx, y0, bw, 0.90, r=0.09, angle=55, c1=INDIGO if i % 2 == 0 else COBALT, c2=COBALT if i % 2 == 0 else "#2563EB", z=2)
            p.ax.add_patch(plt.Circle((bx + bw / 2, y0 + 0.18), 0.11, fc="white", ec="none", zorder=5))
            p.t(bx + bw / 2, y0 + 0.18, num, size=7.6, bold=True, head=True, color=INDIGO, ha="center", va="center", z=6)
            p.t(bx + bw / 2, y0 + 0.42, title, size=8.0, bold=True, head=True, color="white", ha="center", va="center", z=6)
            p.t(bx + bw / 2, y0 + 0.68, sub, size=6.4, color="#E2E8F0", ha="center", va="center", z=6)
            if i < n - 1:
                ax0 = bx + bw + 0.02
                ax1 = bx + bw + gap - 0.02
                ay = y0 + 0.45
                p.line(ax0, ay, ax1, ay, color=INDIGO, lw=1.4, z=5)
                p.t(ax1, ay, "▸", size=7.5, color=INDIGO, ha="center", va="center", z=6)
        self.y += bh + 0.14


def render_toc_page(
    p: Page,
    entries: Sequence[Dict[str, Any]],
) -> List[Tuple[Tuple[float, float, float, float], int]]:
    """
    Draw a modern 2-column clickable Table of Contents page.
    Returns a list of `((x, y, w, h), target_page_0based)` link rectangles in inches.
    """
    links: List[Tuple[Tuple[float, float, float, float], int]] = []

    # Header band
    p.grad(0, 0, W, 1.38, r=0, c1=NIGHT, c2=INDIGO, angle=18, z=1)
    p.t(M, 0.34, f"PANDADOCK {__version__}  ·  OFFICIAL USER MANUAL", size=7.8, bold=True, head=True, color=SPARK, z=5)
    p.t(M, 0.58, "Table of Contents", size=22.0, bold=True, head=True, color="white", z=5)
    p.t(M, 1.03, "Click any chapter or command below to jump straight to that page. Every page also includes a clickable ↑ Contents link.",
        size=8.2, color="#CBD5E1", z=5)

    # Group entries by chapter (level 1 + its level 2 children)
    chapters: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for e in entries:
        if e["level"] == 1:
            chapters.append((e, []))
        elif chapters:
            chapters[-1][1].append(e)

    # Balance left and right columns by estimated height
    def _ch_h(ch_pair) -> float:
        _, subs = ch_pair
        return 0.56 + len(subs) * 0.205 + 0.14

    heights = [_ch_h(c) for c in chapters]
    best_split, best_max = 1, 1e9
    for k in range(1, len(chapters)):
        mx = max(sum(heights[:k]), sum(heights[k:]))
        if mx < best_max:
            best_max, best_split = mx, k

    col_groups = [chapters[:best_split], chapters[best_split:]]
    gutter = 0.34
    col_w = (CW - gutter) / 2

    for ci, group in enumerate(col_groups):
        cx = M + ci * (col_w + gutter)
        yy = 1.62
        for ch, subs in group:
            # Chapter card header
            card_h = 0.44
            p.rrect(cx, yy, col_w, card_h, r=0.08, fc="#F3F5FC", ec="#D9E0F2", lw=0.6, z=2)
            # Chapter number badge
            p.grad(cx + 0.08, yy + 0.09, 0.34, 0.26, r=0.06, angle=35, z=3)
            p.t(cx + 0.25, yy + 0.22, ch["number"], size=8.0, bold=True, head=True, color="white", ha="center", va="center", z=6)
            # Chapter title + right page badge (shortened cleanly so it never touches p. XX)
            ch_title = ch["title"]
            if len(ch_title) > 31:
                ch_title = ch_title[:29].rstrip() + "…"
            p.t(cx + 0.50, yy + 0.135, ch_title, size=8.4, bold=True, head=True, color=INK, va="center", z=6)
            # Truncated subtitle
            sub_txt = ch["subtitle"].replace("**", "")
            if len(sub_txt) > 46:
                sub_txt = sub_txt[:44].rstrip() + "…"
            p.t(cx + 0.50, yy + 0.315, sub_txt, size=6.4, color=SOFT, va="center", z=6)
            p.t(cx + col_w - 0.10, yy + 0.135, f"p. {ch['page']}", size=7.8, bold=True, head=True, color=ACCENT, ha="right", va="center", z=6)

            links.append(((cx, yy, col_w, card_h), ch["page"] - 1))
            yy += card_h + 0.05

            for sub in subs:
                rh = 0.195
                p.t(cx + 0.16, yy + rh / 2, "›", size=8.0, bold=True, color=INDIGO, va="center", z=5)
                lbl = sub["title"]
                if len(lbl) > 40:
                    lbl = lbl[:38] + "…"
                p.t(cx + 0.28, yy + rh / 2, lbl, size=7.5, color=TEXT, va="center", z=5)
                p.t(cx + col_w - 0.10, yy + rh / 2, str(sub["page"]), size=7.4, bold=True, color=ACCENT, ha="right", va="center", z=5)
                p.line(cx + 0.28, yy + rh - 0.01, cx + col_w - 0.08, yy + rh - 0.01, color="#ECEEF5", lw=0.45, z=3)
                links.append(((cx + 0.12, yy, col_w - 0.14, rh), sub["page"] - 1))
                yy += rh
            yy += 0.12

    # Helpful interactive tip box at bottom of TOC
    tip_y = 10.22
    p.rrect(M, tip_y, CW, 0.56, r=0.10, fc=BEIGE, z=2)
    p.rrect(M, tip_y, 0.06, 0.56, radii=(0.10, 0, 0, 0.10), fc=INDIGO, z=3)
    p.t(M + 0.18, tip_y + 0.14, "HOW TO USE THIS MANUAL", size=7.4, bold=True, head=True, color=INK, z=5)
    p.flow(
        M + 0.18, tip_y + 0.30, CW - 0.30,
        "New to molecular docking? Read **Chapters 1–5** in order (15 minutes). Already docking and looking for a specific flag? Click any command above or open the PDF sidebar outline.",
        size=7.7, lead=1.32, color=TEXT, z=5,
    )
    return links


def build_manual(output_pdf: Path) -> Path:
    ca_xyz, lig_xyz = _extract_cover_coords()

    with plt.rc_context(PLOT_RC):
        doc = Doc(f"PandaDock {__version__} Official User Manual", kind="OFFICIAL USER MANUAL")

        # =====================================================================
        # PAGE 1: COVER (Exact Panda-family report.pdf layout)
        # =====================================================================
        art_fn = (lambda pg: art_trace(pg, ca_xyz, extra=lig_xyz)) if ca_xyz is not None else None
        cover(
            doc,
            title="PandaDock 5.0 User Manual",
            subtitle=(
                "A plain-language, step-by-step handbook to molecular docking — from preparing your "
                "protein and ligand, stripping or keeping pocket waters, and defining the binding site "
                "box, to covalent warheads, series SAR, PoseBusters audits, GNN scoring, MD handoff, "
                "and PandaDock Studio."
            ),
            highlights=[
                ("layers", "10 CLI Entry Points · 48 Commands Covered"),
                ("molecule", "Plain-English Guide for Every Option"),
                ("target", "Studio GUI · PandaLens 3D · Vector PDF"),
            ],
            quote="From raw PDB file to validated 3D binding poses, publication reports, and ready-to-run MD in one cohesive suite.",
            quote_by="PandaDock v5.0 Release Manual | Stanford University · MIT License",
            facts=[
                ("Release", f"PandaDock v{__version__} (2026)"),
                ("Scope", "All 10 CLI tools, 48 commands & Studio GUI"),
                ("Visuals", "PandaLens 3D, PandaMap 2D, A4 PDF & HTML"),
                ("Engines", "Vina, Vinardo, Physics, MM-GBSA, EGNN v5"),
                ("Platforms", "macOS (CPU/MPS), Linux (CPU/CUDA), Python 3.9+"),
            ],
            art=art_fn,
            kind="OFFICIAL USER MANUAL · RELEASE 5.0",
        )

        # =====================================================================
        # PAGE 2: RESERVE TABLE OF CONTENTS PAGE (drawn after content pages so
        # every page number is 100% exact!)
        # =====================================================================
        toc_page = doc.page(number_visible=True)
        mb = ManualBuilder(doc, toc_page_index=1)

        # =====================================================================
        # CHAPTER 1: GETTING STARTED & WHAT MOLECULAR DOCKING DOES
        # =====================================================================
        mb.chapter(
            "01",
            "Getting Started: Molecular Docking in Plain English",
            "What molecular docking actually does, how PandaDock 5.0 fits together, and how to run your very first docking job in five minutes.",
            start_new_page=True,
            toc_title="Getting Started",
        )
        mb.para(
            "Think of a **protein** as a complex 3D lock with a small carved pocket (the **binding site**), "
            "and a **drug molecule (ligand)** as a flexible key with hinges (rotatable chemical bonds). "
            "**Molecular docking** is a computer simulation that tries thousands of positions, rotations, "
            "and bond angles of your key inside the lock to answer two practical questions: "
            "**(1) How does the ligand sit inside the pocket?** and **(2) How tightly does it bind?** "
            "(measured as a binding energy score in kcal/mol — the more negative the number, the stronger the predicted binding)."
        )
        mb.draw_workflow_diagram()
        mb.callout(
            "In Plain English — The Six Steps of Every Docking Project",
            "1) **Prepare**: Take your raw protein (PDB) and ligand (SDF/MOL2/SMILES), add missing hydrogen atoms, and set realistic charges at body pH (7.4). Decide whether to strip bulk water or keep key pocket waters. "
            "2) **Define the Grid Box**: Draw an invisible 3D box around the binding pocket (for example by clicking the crystal ligand in a PDB complex, or letting PandaDock auto-detect cavities). "
            "3) **Dock**: Run flexible conformational search with physics scoring (CPU or GPU). "
            "4) **Audit**: Automatically verify every pose passes 11 PoseBusters chemistry & geometry checks. "
            "5) **Inspect**: View explicit hydrogens and 18 interaction classes in PandaDock Studio (powered by PandaLens), plus 300-DPI plots and A4 PDF/HTML reports. "
            "6) **Handoff**: Export your best pose directly to OpenMM or GROMACS molecular dynamics.",
            tone="plain",
        )
        mb.section("Installing PandaDock 5.0", record_toc=True, toc_label="Installation & Quick Start")
        mb.code_block(
            "# 1. Create a clean Python environment with RDKit (recommended via conda-forge)\n"
            "conda create -n pandadock python=3.10 -y && conda activate pandadock\n"
            "conda install -c conda-forge rdkit -y\n\n"
            "# 2. Install PandaDock 5.0 (+ optional Studio GUI, GNN, and PandaMap 2D maps)\n"
            "pip install pandadock\n"
            "pip install 'pandadock[gui]'    # Enables PandaDock Studio web GUI\n"
            "pip install 'pandadock[gnn]'    # Enables GPU search (CUDA/MPS) & GNN scoring\n"
            "pip install pandamap            # Enables 2D interaction diagrams",
            caption="Installation (macOS / Linux)",
        )
        mb.section("Complete Map of All 10 CLI Entry Points (48 Commands)", record_toc=True, toc_label="All 10 CLI Tools (48 Commands)")
        mb.option_table(
            [
                ("pandadock", "18 cmds", "Main hub: dock, screen, series, covalent, validate-poses, calibrate, to-md, gui, hybrid, list-algorithms, plus 8 gnn subcommands."),
                ("pandadock-gui", "1 cmd", "Launches PandaDock Studio (Apple-designed browser GUI with PandaLens 3D viewer, plots, PDF & HTML reports)."),
                ("pandadock-prepare", "1 cmd", "Cleans your protein and ligand: adds hydrogens, assigns pH 7.4 protonation, builds 3D coordinates, and minimizes strain."),
                ("pandadock-gridbox", "1 cmd", "Finds the binding site and writes box.json using a reference X-ray ligand, automatic cavity detection, residues, or manual coordinates."),
                ("pandadock-flex", "1 cmd", "4-phase induced-fit flexible docking where pocket side chains (and optionally loops) move to accommodate the ligand."),
                ("pandadock-metal", "5 cmds", "Specialized metalloprotein docking (Zn, Fe, Mg, Cu, Mn — 31 metals) with zero-order bonds and coordination geometry."),
                ("pandadock-tethered", "4 cmds", "Extracts an X-ray ligand from a PDB complex, runs tethered redocking validation, and scores crystal poses."),
                ("pandadock-report", "4 cmds", "Regenerates 300-DPI plots, PandaMap 2D/3D interaction maps, and detailed mathematical scoring breakdowns from any run."),
                ("pandadock-ml", "5 cmds", "Classical ML-enhanced docking pipelines: extract HDF5 features, train ranking models, and run ML + physics refinement."),
                ("pandadock-gnn", "8 cmds", "Standalone SE(3)-equivariant GNN CLI (mirrors `pandadock gnn *`): download-model, predict, rescore, train, train-sair, benchmark, compare."),
            ],
            headers=("CLI Entry Point", "Count", "What It Does in Plain English"),
            col_widths=(1.55, 0.72, CW - 2.27),
        )

        # =====================================================================
        # CHAPTER 2: PANDADOCK STUDIO GUI & PANDALENS 3D VIEWER
        # =====================================================================
        mb.chapter(
            "02",
            "PandaDock Studio GUI & PandaLens 3D Viewer",
            "Point-and-click molecular docking with guided 4-step preparation, X-ray ligand pocket selection, explicit-H binding site visualization, and one-click PDF/HTML reports.",
            start_new_page=True,
            toc_title="Studio GUI & PandaLens 3D",
        )
        mb.command_banner("pandadock gui [OPTIONS]   |   pandadock-gui [OPTIONS]", badge="STUDIO GUI · v5.0", toc_label="pandadock gui / pandadock-gui")
        mb.para(
            "If you prefer a visual desktop-style workflow over typing terminal flags, launch **PandaDock Studio**. "
            "It runs locally on your computer (your structures never leave your machine) and guides you cleanly through "
            "every stage of docking, live progress streaming, 3D pose inspection, 300-DPI plots, and embedded A4 PDF/HTML reports."
        )
        mb.option_table([
            ("--host TEXT", "127.0.0.1", "Local network address to bind the web server. Keep 127.0.0.1 for private local use, or 0.0.0.0 to share across a lab workstation."),
            ("--port INTEGER", "8765", "Port number in your browser URL (e.g. http://127.0.0.1:8765). Pick another port like 8785 if 8765 is busy."),
            ("--workspace DIR", ".", "Folder that Studio scans for your input structures and completed docking results (shown in the left Recent Results sidebar)."),
            ("--no-browser", "off", "Starts the server without automatically opening a browser tab (handy on remote SSH servers)."),
        ])
        mb.section("Guided 4-Step Studio Workflow (Step by Step)", record_toc=True, toc_label="4-Step Studio Workflow")
        mb.bullets([
            "**Step 1 — Select Workflow & Prepare Structures**: Choose Single Ligand, Ligand Series, or Covalent, then drop your receptor PDB and ligand SDF/MOL2/SMILES. Open the **Prepare Protein & Ligand** panel to add missing hydrogens, set physiological pH (default 7.4), run RDKit 3D MMFF94 ligand minimization, optionally enumerate pH protonation/tautomer states, and optionally strip crystallographic waters before docking.",
            "**Step 2 — Define Binding Site (Grid Box)**: Choose **Auto-Detect Pockets** to discover cavities automatically, **Protein Center** for blind docking, **Metal Ion** for zinc/iron enzymes, or toggle **Pick Residues** and click amino acids directly on the 3D protein to center the blue wireframe box around them.",
            "**Uploading a Protein–Ligand Complex (Schrödinger Maestro–Style Pocket Selection)**: If your uploaded PDB already contains a bound X-ray ligand (for example acetazolamide `AZM A:701` or a crystal inhibitor), Studio detects it automatically and displays a clickable **Co-Crystal Ligand** chip. Click that X-ray ligand chip and two things happen automatically: **(1)** the grid box centers tightly on that exact binding pocket, and **(2)** PandaDock **removes only that co-crystal ligand** from the receptor grid before docking starts — keeping all protein atoms, catalytic metals, and structural waters intact so your new ligand can dock freely into the empty pocket without colliding with the old X-ray ligand!",
            "**Step 3 — Docking Options**: Keep clean defaults for everyday runs, or expand the Apple disclosure drawers for Scoring (`vina` / `vinardo` / `physics_based`), Exhaustiveness, Structural Waters (`strip` / `keep` / `displace`), Differentiable Restraints, MM-GBSA Rescoring, Calibrated Weights, and Dock-to-MD Handoff.",
            "**Step 4 — Review & Launch**: Click **Run Docking & Reports**. Watch live terminal output stream in real time, then explore all 5 workspace tabs (**3D Structure**, **Plots & Analytics**, **PDF Report**, **HTML Report**, and **Console & Files**).",
        ])
        mb.callout(
            "Inspecting Docked Poses with PandaLens (Explicit Hydrogens & 18 Interaction Classes)",
            "When you open any completed run and click Pose #1..#N in the 3D viewer, Studio automatically opens in **Binding Pocket** mode powered by **PandaLens**: "
            "• **Show H** displays explicit 3D white hydrogens on **both** the binding-site protein residues/waters and your docked ligand, and feeds their exact 3D coordinates into PandaLens for true **D–H···A** hydrogen-bond geometry (H···A ≤ 2.7 Å, D–H···A ≥ 120°). "
            "• **Interactions** draws dashed/solid 3D vectors and distance pills (Å) across all 18 classes: Hydrogen bonds, Water bridges (2-leg), Salt bridges, Metal coordination, Covalent warhead bonds, π–π stacking, T-shaped π, Cation–π, Amide–π, Donor–π, Alkyl–π, Sulfur–π, Halogen bonds, Hydrophobic contacts, and Steric clashes. "
            "• **Draggable Residue Labels**: Drag any residue label clear of sticks (an SVG leader line follows it); double-click to snap back, or click **List (N)** to hover-highlight and click-center any individual contact.",
            tone="tip",
        )
        mb.code_block(
            "# Launch PandaDock Studio on your current project directory\n"
            "pandadock-gui --port 8765 --workspace ./my_docking_project\n"
            "# Or via the main CLI:\n"
            "pandadock gui --port 8785 --workspace . --no-browser",
            caption="Launching PandaDock Studio",
        )

        # =====================================================================
        # CHAPTER 3: PREPARING PROTEIN & LIGAND (pandadock-prepare)
        # =====================================================================
        mb.chapter(
            "03",
            "Preparing Your Protein & Ligand",
            "Why raw PDB and SMILES files must be cleaned, protonated at physiological pH, and minimized before docking — and how to handle waters cleanly.",
            start_new_page=True,
        )
        mb.command_banner("pandadock-prepare -r RECEPTOR.pdb -l LIGAND.sdf [OPTIONS]", badge="PREPARATION", toc_label="pandadock-prepare")
        mb.para(
            "Structures downloaded from the Protein Data Bank (RCSB PDB) are experimental X-ray snapshots: they almost "
            "never include **hydrogen atoms** (because hydrogens barely scatter X-rays), they often contain buffer salts "
            "and hundreds of floating crystal water molecules, and ligands drawn in ChemDraw or downloaded as SMILES are "
            "flat 2D sketches with no realistic 3D shape. **`pandadock-prepare`** fixes all of those issues in one step so "
            "the physics engine sees realistic chemistry before docking begins."
        )
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Input protein structure in PDB or mmCIF (.cif) format. Cleaned and saved as a ready-to-dock receptor PDB."),
            ("-l, --ligand PATH", "Required", "Input small molecule in SDF, MOL2, or PDB format. Receives explicit hydrogens, 3D conformer geometry, and charges."),
            ("-o, --output-dir PATH", "pandadock_output", "Folder where prepared receptor, prepared ligand(s), and protonation state summary JSON are written."),
            ("--ph FLOAT", "7.0", "Target pH used to decide which acidic/basic groups are charged. Use **7.4** for human blood/cytoplasm, **5.5–6.0** for lysosomes/endosomes, or **2.0** for stomach enzymes."),
            ("--add-hydrogens / --no-add-hydrogens", "True", "Adds explicit 3D hydrogen atoms to polar and nonpolar atoms. Always keep **True** for accurate H-bonding and steric volume."),
            ("--add-charges / --no-add-charges", "True", "Computes Gasteiger partial atomic charges needed for electrostatic and physics-based scoring."),
            ("--minimize-ligand / --no-minimize-ligand", "True", "Relaxes the ligand in 3D with the MMFF94 force field so bond lengths, angles, and rings start strain-free before docking."),
            ("--fix-missing-residues", "False", "Attempts to rebuild missing loops/side-chain atoms in the protein. Leave **False** unless missing atoms lie right inside your binding site, because rebuilding long disordered loops can be aggressive."),
            ("--enumerate-states", "False", "v5.0 feature: automatically generates all plausible protonation microspecies, tautomers, and unassigned chiral stereoisomers at `--ph`, each scored with a Henderson–Hasselbalch energy penalty."),
        ])
        mb.callout(
            "Stripping Water vs Keeping Water — Plain-English Rule of Thumb",
            "**Water stripping** removes floating crystallographic `HOH` water molecules from a PDB file so they do not accidentally block your ligand from entering the pocket. "
            "• **Strip waters (default `--waters strip`)**: Best for 80% of docking runs where bulk crystal waters are simply solvent noise. In PandaDock Studio Step 1 you can also tick **Strip crystal waters** to save a water-free receptor PDB immediately. "
            "• **Keep waters (`--waters keep`)**: Use when you know a specific buried water molecule acts as a permanent bridge holding the ligand to the protein (like conserved waters in kinases or HIV protease). "
            "• **Displace waters (`--waters displace`)**: The smartest v5.0 option for uncertain pockets — PandaDock keeps structured pocket waters during search, lets the ligand displace them if the binding energy gain beats the desolvation penalty, and automatically reports which waters were **bridging** vs **displaced**!",
            tone="info",
        )
        mb.code_block(
            "# Standard preparation at physiological pH 7.4\n"
            "pandadock-prepare -r raw_protein.pdb -l raw_ligand.sdf --ph 7.4 -o prepared/\n\n"
            "# Enumerate protonation states, tautomers & chiral isomers (e.g. basic amines / histidines)\n"
            "pandadock-prepare -r protein.pdb -l ketamine.sdf --ph 7.4 --enumerate-states -o prepared_states/",
            caption="Common Preparation Recipes",
        )

        # =====================================================================
        # CHAPTER 4: DEFINING THE BINDING SITE / GRID BOX (pandadock-gridbox)
        # =====================================================================
        mb.chapter(
            "04",
            "Defining the Binding Site (Grid Box)",
            "How to tell PandaDock exactly where the binding pocket is — from an X-ray crystal ligand, automatic cavity search, active-site residues, or manual coordinates.",
            start_new_page=True,
        )
        mb.command_banner("pandadock-gridbox -r RECEPTOR.pdb -m MODE [OPTIONS]", badge="BINDING SITE", toc_label="pandadock-gridbox")
        mb.para(
            "Before docking runs, PandaDock builds a 3D **grid box** — an invisible rectangular search room surrounding "
            "the binding pocket. Precomputing interaction energies at every point inside this grid makes docking "
            "hundreds of times faster than checking every protein atom from scratch on every step. "
            "A good grid box is **large enough to enclose your entire ligand plus ~5–6 Å of breathing room** on each side "
            "(typically **20 × 20 × 20 Å** to **24 × 24 × 24 Å**), without being so huge that the search wastes time exploring "
            "irrelevant surface grooves."
        )
        mb.section("Choosing the Right Grid Box Mode (`-m / --mode`)", record_toc=True, toc_label="5 Ways to Define the Pocket")
        mb.bullets([
            "**1. Co-Crystal / Reference Ligand (`-m similarity --reference-ligand xray_lig.sdf`)**: The gold standard whenever you have a crystal structure with a known binder. PandaDock centers the box on the X-ray ligand and adds `--padding` (default 5.0 Å). *Tip*: Make sure you dock into a receptor PDB with that old X-ray ligand removed — in **PandaDock Studio**, simply click the X-ray ligand chip on your complex PDB and Studio both defines the box AND removes that X-ray ligand automatically before docking!",
            "**2. Automatic Cavity Detection (`-m cavities`)**: Don't know where the pocket is? PandaDock scans the protein surface for buried geometric cavities larger than `--min-volume` and ranks the top `--max-sites` druggable pockets.",
            "**3. Active-Site Residues (`-m residues --residues A:145,A:41`)**: Know the catalytic residues or a resistance mutation? Give their `Chain:ResidueNumber` list and PandaDock tightly wraps the box around those residues plus `--padding`.",
            "**4. Manual Coordinates (`-m manual -c X Y Z -b DX DY DZ`)**: Already know exact box coordinates from PyMOL, ChimeraX, or a published paper? Type them directly.",
            "**5. Interactive / Pharmacophore (`-m interactive` or `--pharmacophore`)**: Guided terminal prompt or pharmacophore-biased site selection (e.g. `hydrophobic,donor,acceptor,aromatic`).",
        ])
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Prepared receptor PDB file whose binding pocket you want to locate."),
            ("-m, --mode TEXT", "interactive", "How to find the pocket: `similarity` (from reference ligand), `cavities` (auto blind pocket finder), `residues` (around named amino acids), `manual` (typed X Y Z), or `interactive`."),
            ("-c, --center X Y Z", "None", "Manual box center coordinates in Ångströms (used with `-m manual`)."),
            ("-b, --box DX DY DZ", "None", "Manual box width, height, and depth in Ångströms (e.g. `22 22 22`)."),
            ("-s, --spacing FLOAT", "0.375", "Distance between neighboring affinity grid points in Å. Keep 0.375 Å (AutoDock standard) for optimal accuracy/speed."),
            ("--reference-ligand PATH", "None", "3D X-ray or known ligand file (SDF/MOL2/PDB) used by `-m similarity` to center and size the box."),
            ("--residues TEXT", "None", "Comma-separated residue specs for `-m residues`, e.g. `A:41,A:145,A:163`."),
            ("--padding FLOAT", "5.0", "Extra margin (Å) added around the reference ligand or selected residues so slightly larger analogs still fit comfortably."),
            ("--pharmacophore TEXT", "None", "Optional comma-separated features (`hydrophobic,donor,acceptor,aromatic,positive,negative`) to prioritize pockets matching your pharmacophore."),
            ("--max-sites INTEGER", "3", "Maximum number of top candidate pockets returned by cavity detection."),
            ("--min-volume FLOAT", "50.0", "Minimum cavity volume (Å³) to consider — ignores tiny surface dimples too small for a drug molecule."),
            ("--confidence-threshold", "0.7", "Minimum confidence score (0–1) for AI/cavity pocket predictions."),
            ("-o, --output PATH", "None", "Exact JSON file path to save the grid box config (pass directly to `pandadock dock -g box.json`)."),
            ("--output-dir PATH", "gridbox_output", "Directory used when saving multiple detected site configs or visualization scripts."),
            ("--visualize", "False", "Generates 3D visualization files showing the detected box(es) over your protein."),
            ("-v, --verbose", "False", "Prints detailed cavity volumes, center coordinates, and diagnostic logs."),
        ])
        mb.code_block(
            "# Option A: Center box on a known X-ray ligand (5 Å padding)\n"
            "pandadock-gridbox -r protein_clean.pdb -m similarity \\\n"
            "  --reference-ligand xray_lig.sdf --padding 5.0 -o box.json\n\n"
            "# Option B: Automatic blind cavity detection (top 3 druggable pockets)\n"
            "pandadock-gridbox -r protein_clean.pdb -m cavities --max-sites 3 --min-volume 80 -o box.json\n\n"
            "# Option C: Wrap box around specific active-site residues\n"
            "pandadock-gridbox -r protein_clean.pdb -m residues --residues A:41,A:145,A:163 --padding 6.0 -o box.json",
            caption="Defining Your Binding Site Grid Box",
        )

        # =====================================================================
        # CHAPTER 5: CORE FLEXIBLE DOCKING (pandadock dock)
        # =====================================================================
        mb.chapter(
            "05",
            "Core Flexible Docking & Pocket Physics",
            "Running flexible-ligand docking with CPU/GPU acceleration, scoring functions, structural waters, geometric restraints, pH state enumeration, MM-GBSA rescoring, and automatic reports.",
            start_new_page=True,
            toc_title="Core Flexible Docking",
        )
        mb.command_banner("pandadock dock -r RECEPTOR.pdb -l LIGAND.sdf [OPTIONS]", badge="CORE ENGINE", toc_label="pandadock dock")
        mb.para(
            "**`pandadock dock`** is the workhorse command of PandaDock 5.0. It treats every rotatable bond in your "
            "ligand as a flexible hinge, explores positions uniformly inside your grid box, relaxes every candidate pose "
            "to a local energy minimum using **analytic-gradient L-BFGS optimization**, clusters distinct binding modes "
            "by RMSD, audits every pose with **PoseBusters**, and automatically generates 3D complexes (`complex1.pdb`), "
            "ranked poses (`poses.sdf`), 300-DPI diagnostic plots, PandaMap 2D/3D maps, `report.html`, and `report.pdf`."
        )
        mb.section("1. Inputs, Box & Scoring Options", record_toc=True, toc_label="Docking Inputs, Scoring & Speed")
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Prepared receptor PDB file (without the old co-crystal ligand blocking the pocket)."),
            ("-l, --ligand PATH", "Required", "Ligand file in SDF, MOL2, or PDB format."),
            ("-g, --grid-config PATH", "None", "Grid box JSON created by `pandadock-gridbox` (easiest way to specify the binding site)."),
            ("--center X Y Z", "None", "Three numbers (Å) giving the pocket center directly — alternative to `-g box.json`."),
            ("--box DX DY DZ", "None", "Three numbers (Å) giving the search box size (e.g. `22 22 22`) — alternative to `-g box.json`."),
            ("-s, --scoring CHOICE", "vina", "Physics scoring function: `vina` (fast, robust general standard), `vinardo` (smoother sterics, often great for buried/hydrophobic pockets), or `physics_based` (explicit electrostatics + desolvation)."),
            ("-n, --num-poses INT", "20", "Maximum number of distinct ranked binding poses returned in `poses.sdf` and `complex*.pdb`."),
            ("-e, --exhaustiveness INT", "auto (8–32)", "Number of independent Monte Carlo search trajectories. Higher = more thorough search (use 16–32 for very flexible drugs with 8+ rotatable bonds)."),
            ("-j, --workers INT", "1", "Number of CPU threads running independent search trajectories in parallel (e.g. `--workers 4` or `8`)."),
            ("--device TEXT", "cpu", "Compute hardware: `cpu` (default), `cuda` (NVIDIA GPU — up to 35× faster on flexible ligands), or `mps` (Apple Silicon M1/M2/M3/M4 GPU)."),
            ("--n-chains INT", "512", "Number of parallel batched search chains when using `--device cuda` or `mps`."),
            ("--grid-spacing FLOAT", "0.375", "Spacing between precomputed 3D grid points in Å. Smaller (0.25) is slightly finer; 0.375 is standard."),
            ("--seed INT", "random", "Fix an integer (e.g. `--seed 42`) so repeating the exact command reproduces identical poses."),
            ("--fast", "False", "Quick reduced-sampling smoke test to confirm your files and box are aligned before a full production run."),
            ("-o, --output-dir PATH", "docking_output", "Output directory where all poses, complexes, plots, JSON summaries, `report.html`, and `report.pdf` are saved."),
        ])
        mb.section("2. Chemistry, Waters, Restraints & Post-Processing (v5.0)", record_toc=True, toc_label="Waters, Restraints, States & MD")
        mb.option_table([
            ("--rigid-ligand", "False", "Freezes all rotatable bonds and only translates/rotates the ligand as a rigid block (useful for rigid fragments or pre-generated conformers)."),
            ("--sample-rings", "False", "Samples alternative 3D puckers (chair/boat/envelope) for flexible non-aromatic rings (e.g. piperidines, cyclohexanes, macrocycles) via ETKDGv3."),
            ("--enumerate-states", "False", "Automatically enumerates protonation states, tautomers, and unassigned chiral stereoisomers at `--ph`, docks each state, adds its Henderson–Hasselbalch state penalty, and picks the true biological winner."),
            ("--ph FLOAT", "7.4", "Target biological pH used when `--enumerate-states` is turned on."),
            ("--waters CHOICE", "strip", "How crystallographic pocket waters are treated: `strip` removes all waters; `keep` retains pocket waters as rigid bridging anchors; `displace` dynamically lets the ligand displace unfavorable waters while keeping bridging ones."),
            ("--restraint SPEC", "None", "Repeatable smooth flat-bottom distance restraint enforced during L-BFGS optimization! Syntax: `CHAIN:RESN:RESI:ATOM<DIST[:ROLE]` (e.g. `A:ZN:301:ZN<2.3:donor` or `A:ASP:128:OD1<3.2:hbond_donor`) or `point:X,Y,Z<DIST[:ROLE]`."),
            ("--restraints-file PATH", "None", "Load a JSON file containing multiple geometric, metal, H-bond, or pharmacophore restraints cleanly."),
            ("--rescoring CHOICE", "none", "Set `--rescoring mmgbsa` to rescore top poses with fast vectorized differential pocket MM-GBSA (polar/nonpolar solvation + electrostatics)."),
            ("--ensemble / --no-ensemble", "True", "Computes Boltzmann-weighted ensemble average binding energy at 298 K alongside the top-1 pose score."),
            ("--weights-file PATH", "None", "Apply target-specific calibrated Vina weights JSON trained with `pandadock calibrate`."),
            ("--export-md CHOICE", "none", "Automatically export your top pose as a ready-to-run molecular dynamics starter kit (`openmm`, `gromacs`, or `both`)."),
            ("--visualize / --no-visualize", "True", "Generates all 300-DPI diagnostic plots, PandaMap 2D/3D diagrams, `report.html`, and `report.pdf`."),
        ])
        mb.command_banner("pandadock list-algorithms", badge="UTILITY", toc_label="pandadock list-algorithms")
        mb.para(
            "Prints a clean table of every built-in search engine and scoring function (`vina`, `vinardo`, `physics_based`, "
            "metal-aware, covalent, induced-fit, and GNN models) available in your current installation."
        )
        mb.code_block(
            "# 1. Everyday flexible docking using a saved grid box\n"
            "pandadock dock -r receptor.pdb -l ligand.sdf -g box.json --workers 4 --seed 42 -o run_basic/\n\n"
            "# 2. Metalloprotein + bridging waters + catalytic zinc restraint + MM-GBSA + MD handoff\n"
            "pandadock dock -r 3hs4_zn_hoh.pdb -l azm.sdf --center -5.3 -0.5 14.8 --box 20 20 20 \\\n"
            "  --waters displace --restraint 'A:ZN:301:ZN<2.3:donor' --rescoring mmgbsa --export-md both -o run_3hs4/\n\n"
            "# 3. Apple Silicon GPU / NVIDIA CUDA accelerated docking with pH 7.4 state enumeration\n"
            "pandadock dock -r 6x3v.pdb -l ketamine.sdf --center 126.5 119.2 146.1 --box 22 22 22 \\\n"
            "  --device mps --n-chains 512 --enumerate-states --ph 7.4 --sample-rings -o run_gpu/",
            caption="Production Docking Recipes",
        )

        # =====================================================================
        # CHAPTER 6: VIRTUAL SCREENING & CONGENERIC SERIES SAR
        # =====================================================================
        mb.chapter(
            "06",
            "Virtual Screening & Congeneric Series SAR",
            "Docking hundreds of diverse library compounds on GPU/CPU (`screen`) vs docking a medicinal-chemistry analog series with MCS core tethering and SAR brochure PDFs (`series`).",
            start_new_page=True,
            toc_title="Screening & Series SAR",
        )
        mb.callout(
            "When to Use `screen` vs `series`",
            "• Use **`pandadock screen`** when you have a **large, chemically diverse library** (50 to 50,000+ compounds) and want maximum throughput by reusing affinity grids and batching ligands on a GPU or multi-core CPU.\n"
            "• Use **`pandadock series`** when you are a medicinal chemist optimizing **5 to 100 close analogs sharing a common chemical scaffold** — it aligns every analog to your reference scaffold via Maximum Common Substructure (MCS) tethering, scores every compound with multi-score consensus, computes 2D/3D SAR comparisons, and writes a publishable A4 **SAR Brochure PDF (`series_report.pdf`)**.",
            tone="plain",
        )
        mb.command_banner("pandadock screen -r RECEPTOR.pdb -l LIBRARY.sdf [OPTIONS]", badge="VIRTUAL SCREENING", toc_label="pandadock screen")
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Shared receptor PDB file — affinity grids are computed once and reused for every ligand in the library."),
            ("-l, --ligands PATH", "Required", "Either a single multi-molecule SDF library file OR a folder containing individual SDF/MOL2/PDB ligand files."),
            ("-g, --grid-config PATH", "None", "Grid box JSON from `pandadock-gridbox` (or pass `--center X Y Z` and `--box DX DY DZ`)."),
            ("--device TEXT", "auto", "Auto-selects `cuda`, `mps`, or `cpu`. Batching many ligands together saturates GPU cores for dramatic speedups."),
            ("--n-chains INT", "128", "Parallel search chains per ligand during batched screening."),
            ("--n-steps INT", "8", "Basin-hopping Monte Carlo steps per chain (increase to 12–16 for deeper screening)."),
            ("--max-batch INT", "64", "Maximum ligands processed simultaneously on the GPU — lower to 16 or 32 if you see out-of-memory errors."),
            ("--grid-spacing FLOAT", "0.375", "Affinity grid resolution in Å."),
            ("--rigid-ligand", "False", "Dock pre-generated conformers rigidly for ultra-fast screening."),
            ("--seed INT", "None", "Fixed seed for reproducible library rankings."),
            ("--top INT", "all", "Only write full 3D pose SDF files for the top N highest-scoring hits (keeps disk usage small on huge libraries)."),
            ("-o, --output-dir PATH", "screening_output", "Output folder with ranked hit list CSV/JSON and top hit pose files."),
        ])
        mb.command_banner("pandadock series -r REC -l ANALOGS --center X Y Z --box DX DY DZ", badge="SERIES SAR · v5.0", toc_label="pandadock series")
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Prepared receptor PDB file."),
            ("-l, --ligands PATH", "Required", "Multi-molecule SDF, SMILES file (`.smi`), or directory of congeneric analogs."),
            ("--center / --box", "Required", "Binding site center (`X Y Z`) and dimensions (`DX DY DZ`) in Å."),
            ("--reference-pose PATH", "None", "3D crystal or docked pose SDF of your parent/reference compound. PandaDock finds the Maximum Common Substructure (MCS) and gently tethers every analog's shared core onto this exact binding orientation!"),
            ("--core-smarts TEXT", "None", "Optional explicit SMARTS pattern specifying which shared scaffold atoms to tether (if omitted, MCS is detected automatically)."),
            ("--core-tolerance FLOAT", "0.75", "Flat-bottom tether radius (Å): zero penalty while core atoms stay within 0.75 Å of the reference, smooth restoring force outside."),
            ("-s, --scoring CHOICE", "vina", "Primary search scoring function (`vina` or `vinardo`)."),
            ("-e, --exhaustiveness INT", "8", "Independent search runs per analog."),
            ("--device / --workers", "cpu / 1", "Hardware (`cpu`, `cuda`, `mps`) and CPU worker threads per analog."),
            ("--weights-file PATH", "None", "Optional target-calibrated weights JSON from `pandadock calibrate` to sharpen SAR rank correlation."),
            ("--gnn-model PATH", "None", "Optional PandaDock-GNN checkpoint to add a neural affinity column alongside physics scores in the SAR brochure."),
            ("--seed INT", "42", "Random seed for reproducible analog rankings."),
            ("-o, --output-dir PATH", "series_output", "Folder containing `series_sar.csv`, `series_poses.sdf`, 2D MCS core grid image, `series_report.html`, and `series_report.pdf`."),
        ])
        mb.code_block(
            "# Congeneric analog series tethered to a crystal reference pose + A4 SAR brochure PDF\n"
            "pandadock series -r receptor.pdb -l analogs.smi --reference-pose parent_xray.sdf \\\n"
            "  --core-tolerance 0.75 --center 126.5 119.2 146.1 --box 22 22 22 -o series_sar_out/",
            caption="Congeneric Series SAR Docking",
        )

        # =====================================================================
        # CHAPTER 7: COVALENT WARHEAD DOCKING (pandadock covalent)
        # =====================================================================
        mb.chapter(
            "07",
            "Covalent Warhead Docking",
            "How PandaDock models irreversible and reversible covalent inhibitors — transforming the unreacted warhead, forming the protein–ligand bond, and generating PandaMap covalent reports.",
            start_new_page=True,
        )
        mb.command_banner("pandadock covalent -r REC -l LIG --residue A:CYS:145 [OPTIONS]", badge="COVALENT · v5.0", toc_label="pandadock covalent")
        mb.para(
            "Normal (non-covalent) docking assumes the drug only touches the protein through weak intermolecular forces. "
            "**Covalent drugs** (like Nirmatrelvir/Paxlovid targeting SARS-CoV-2 Mpro `Cys145`, or Osimertinib targeting "
            "EGFR `Cys797`) actually undergo a chemical reaction to form a real covalent bond to a nucleophilic residue "
            "(`Cys SG`, `Ser OG`, `Thr OG1`, `Lys NZ`, `Tyr OH`, or `His NE2`). **`pandadock covalent`** automates the "
            "entire chemistry: you simply supply the **unreacted** ligand (SDF or SMILES), name the reactive residue, and "
            "PandaDock detects the warhead, builds the reacted adduct, roots the torsion tree at the newly formed bond "
            "(excluding the bonded nucleophile atoms from steric clash), optimizes the remaining flexible tail, and outputs "
            "`covalent_complex1.pdb`, **PandaMap 2D/3D diagrams** showing the covalent bond, `report.html`, and `report.pdf`."
        )
        mb.section("All 8 Built-In Medicinal-Chemistry Warheads (`--warhead`)", record_toc=True, toc_label="8 Built-in Covalent Warheads")
        mb.option_table(
            [
                ("auto", "Default", "Automatically scans your ligand against all 8 warhead SMARTS patterns and selects the reactive electrophile."),
                ("michael_acceptor", "Acrylamide / Enone", "α,β-unsaturated carbonyls (e.g. Osimertinib, Ibrutinib): 1,4-conjugate addition to Cys SG."),
                ("chloroacetamide", "α-Haloacetamide", "Chloro-/bromo-acetamides: SN2 halide displacement forming a thioether bond to Cys."),
                ("nitrile", "Cyanamide / Nitrile", "Reversible covalent nitriles (e.g. Nirmatrelvir / Paxlovid, Saxagliptin): Pinner-type thioimidate adduct."),
                ("epoxide", "Oxirane", "Ring-opening nucleophilic attack by Cys/Ser/Thr (e.g. E-64 protease inhibitors)."),
                ("boronic_acid", "Boronate", "Reversible tetrahedral boronate adduct with catalytic Ser/Thr (e.g. Bortezomib)."),
                ("aldehyde", "Formyl", "Reversible hemithioacetal / hemiacetal formation with Cys or Ser proteases."),
                ("sulfonyl_fluoride", "SuFEx", "Fluoride displacement forming a sulfonate/sulfonamide adduct with Tyr, Lys, or Ser."),
                ("disulfide", "Disulfide Exchange", "Mixed disulfide S–S exchange with reactive cysteine thiols."),
            ],
            headers=("Warhead Name", "Chemistry", "Plain-English Reaction & Example Drugs"),
            col_widths=(1.50, 1.35, CW - 2.85),
        )
        mb.section("Command Options for `pandadock covalent`", record_toc=False)
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Receptor PDB containing the reactive residue (e.g. `6lu7.pdb`)."),
            ("-l, --ligand PATH", "Required", "Unreacted covalent inhibitor in SDF, MOL2, PDB, or SMILES (`.smi`) format."),
            ("--residue TEXT", "Required", "Target reactive residue as `Chain:ResName:ResNum` (e.g. `A:CYS:145`) or with explicit atom (`A:SER:200:OG`)."),
            ("--warhead TEXT", "auto", "Leave as `auto` for automatic detection, or force one of the 8 warhead types listed above."),
            ("--center X Y Z", "auto", "Optional search box center. If omitted, automatically centers in the binding cleft right next to `--residue`."),
            ("--box-size FLOAT", "22.0", "Cubic search box side length in Å around the covalent attachment pocket."),
            ("-s, --scoring CHOICE", "vina", "Non-covalent interaction scoring function (`vina` or `vinardo`) for the rest of the inhibitor."),
            ("-n / -e / --seed", "9 / 12 / 42", "Number of covalent binding poses, multi-start search depth, and reproducible seed."),
            ("-o, --output-dir PATH", "covalent_output", "Folder containing `covalent_poses.sdf`, `covalent_complex1.pdb`, PandaMap 2D/3D maps, plots, `report.html`, and `report.pdf`."),
        ])
        mb.code_block(
            "# Dock Nirmatrelvir covalently onto SARS-CoV-2 Mpro Cys145 with automatic nitrile detection\n"
            "pandadock covalent -r 6lu7_mpro.pdb -l nirmatrelvir.smi \\\n"
            "  --residue A:CYS:145 --warhead auto --num-poses 9 --exhaustiveness 12 -o mpro_covalent/",
            caption="Covalent Warhead Docking Example (6LU7 · Cys145)",
        )

        # =====================================================================
        # CHAPTER 8: QUALITY AUDIT, CALIBRATION & MD HANDOFF
        # =====================================================================
        mb.chapter(
            "08",
            "PoseBusters Audit, Weight Calibration & MD Handoff",
            "Verifying physical realism (validate-poses), tuning scoring weights to your target's assay data (calibrate), and launching molecular dynamics in one command (to-md).",
            start_new_page=True,
            toc_title="Audit, Calibration & MD",
        )
        mb.command_banner("pandadock validate-poses -p POSES.sdf [-r RECEPTOR.pdb] [OPTIONS]", badge="POSEBUSTERS · v5.0", toc_label="pandadock validate-poses")
        mb.para(
            "A negative docking score alone does **not** guarantee a physically valid structure — a pose could have "
            "twisted aromatic rings, snapped bond lengths, flipped chirality, or steric clashes with the protein. "
            "**`pandadock validate-poses`** runs a built-in **11-point PoseBusters chemistry & geometry audit** on any "
            "docked SDF (from PandaDock, Vina, Glide, or GOLD): RDKit sanitization, connectivity, bond lengths, bond "
            "angles, internal steric clashes, aromatic ring flatness, sp² double-bond flatness, stereochemistry/chirality "
            "preservation, MMFF94 internal strain energy, protein–ligand steric clashes, and protein volume overlap."
        )
        mb.option_table([
            ("-p, --poses PATH", "Required", "Multi-pose SDF file you want to audit (e.g. `poses.sdf`)."),
            ("-r, --receptor PATH", "None", "Receptor PDB file. Required to enable the protein–ligand clash and steric volume overlap checks."),
            ("--reference PATH", "None", "Original input ligand SDF — checks that docking never accidentally inverted a chiral center."),
            ("--max-strain FLOAT", "15.0", "Maximum allowed MMFF94/UFF internal strain energy (kcal/mol) above the ligand's relaxed conformer."),
            ("-o, --output PATH", "None", "Save the full per-pose 11-point pass/fail checklist as a structured JSON report."),
        ])

        mb.command_banner("pandadock calibrate -r RECEPTOR.pdb -p POSES.sdf -y LABELS.csv [OPTIONS]", badge="FEW-SHOT · v5.0", toc_label="pandadock calibrate")
        mb.para(
            "Every protein pocket has its own quirks — one pocket rewards hydrogen bonds heavily, another is driven by "
            "hydrophobic desolvation. If you already have **5 to 25 known compounds with experimental IC50, Ki, or ΔG "
            "measurements** for your target, **`pandadock calibrate`** fits custom Ridge-regularized Vina weights using "
            "Leave-One-Out Cross-Validation (LOOCV) so you don't overfit, and outputs `calibrated_weights.json` ready "
            "to plug into `pandadock dock --weights-file` or `pandadock series --weights-file`."
        )
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Target receptor PDB file."),
            ("-p, --poses PATH", "Required", "Multi-ligand 3D SDF containing one docked or crystal pose per known training compound."),
            ("-y, --labels PATH", "Required", "Simple two-column CSV (`name,dg` or `name,pIC50`) matching molecule titles in `--poses`."),
            ("--preset CHOICE", "vina", "Starting prior weights (`vina` or `vinardo`) that Ridge regularization shrinks toward when data is small."),
            ("--higher-is-better", "False", "Turn ON if your CSV labels are `pIC50` or `pKi` (bigger = stronger). Leave OFF if labels are ΔG in kcal/mol (more negative = stronger)."),
            ("-o, --output PATH", "calibrated_weights.json", "Output calibrated weights JSON file."),
        ])

        mb.command_banner("pandadock to-md -r RECEPTOR.pdb -p POSES.sdf [OPTIONS]", badge="MD HANDOFF · v5.0", toc_label="pandadock to-md")
        mb.para(
            "Docking gives a static snapshot; **Molecular Dynamics (MD)** tests whether the ligand actually stays stably "
            "bound when the protein wiggles in water at room temperature. **`pandadock to-md`** takes your chosen pose "
            "and generates a complete, ready-to-run simulation directory with clean complex PDB, ligand SDF, "
            "**OpenMM** Python script (`run_openmm.py`), **GROMACS** topology/MDP files (`em.mdp`, `nvt.mdp`, `npt.mdp`, "
            "`md.mdp`, `run_gromacs.sh`), and a **PandaInsight** post-MD trajectory analysis manifest."
        )
        mb.option_table([
            ("-r, --receptor PATH", "Required", "Receptor PDB file used for docking."),
            ("-p, --poses PATH", "Required", "Docked poses SDF file."),
            ("--pose-index INT", "0", "0-based index of which pose to export (`0` = top-ranked Pose #1, `1` = Pose #2, etc.)."),
            ("--engine CHOICE", "both", "Generate starter scripts for `openmm`, `gromacs`, or `both`."),
            ("--duration-ns FLOAT", "10.0", "Length of the production MD simulation in nanoseconds (e.g. `10.0`, `50.0`, or `100.0`)."),
            ("--temperature FLOAT", "300.0", "Simulation temperature in Kelvin (300 K ≈ 27 °C, 310 K ≈ body temperature 37 °C)."),
            ("-o, --output-dir PATH", "md_handoff", "Directory where all MD input structures, parameter files, and launch scripts are written."),
        ])
        mb.code_block(
            "# 1. Audit poses with PoseBusters\n"
            "pandadock validate-poses -p results/poses.sdf -r receptor.pdb -o audit.json\n\n"
            "# 2. Calibrate target-specific weights from 12 known pIC50 compounds\n"
            "pandadock calibrate -r receptor.pdb -p known_poses.sdf -y assay.csv --higher-is-better -o target_w.json\n\n"
            "# 3. Export Pose #1 to 50 ns OpenMM + GROMACS MD package\n"
            "pandadock to-md -r receptor.pdb -p results/poses.sdf \\\n"
            "  --pose-index 0 --engine both --duration-ns 50 -o md_pkg/",
            caption="Post-Docking Validation, Calibration & MD Handoff",
        )

        # =====================================================================
        # CHAPTER 9: SPECIALIZED PHYSICS MODES (flex, metal, tethered)
        # =====================================================================
        mb.chapter(
            "09",
            "Specialized Physics: Induced-Fit, Metals & Tethered Validation",
            "Three dedicated standalone suites for flexible binding pockets (`pandadock-flex`), catalytic metalloproteins (`pandadock-metal`), and crystal-pose redocking (`pandadock-tethered`).",
            start_new_page=True,
            toc_title="Induced-Fit, Metals & Tethered",
        )
        mb.command_banner("pandadock-flex -r RECEPTOR.pdb -l LIGAND.sdf --center X Y Z [OPTIONS]", badge="INDUCED-FIT", toc_label="pandadock-flex (Induced-Fit)")
        mb.para(
            "Sometimes a bulky drug cannot fit into a rigid crystal pocket until one or two protein side chains swing "
            "out of the way (an **induced fit**). **`pandadock-flex`** runs a classic 4-phase Induced-Fit Docking (IFD) "
            "protocol: **(1)** soft-potential docking (`--soft-potential-scale 0.7`) lets the ligand squeeze in gently, "
            "**(2)** nearby pocket side chains (χ₁/χ₂ rotamers, or loops) refine around each pose, **(3)** high-precision "
            "redocking restores full hard van der Waals radii (`1.0`), and **(4)** composite IFD scoring ranks the final complexes."
        )
        mb.option_table([
            ("-r / -l", "Required", "Cleaned receptor PDB (with hydrogens) and ligand file (SDF/MOL2/PDB)."),
            ("--center / --radius", "Req / 12.0 Å", "Pocket center (`X Y Z`) and spherical binding-site refinement radius in Å."),
            ("--flexible-residues", "auto", "Comma-separated flexible side chains (e.g. `A:123,A:145,B:67`). If omitted, automatically selects residues within `--refine-distance` (default 6.0 Å) of the ligand."),
            ("--refine-loops / --refine-ligand", "False / True", "Enable optional backbone loop flexibility (slower) and keep ligand refinement active during receptor relaxation."),
            ("--soft-potential-scale / --final-potential-scale", "0.7 / 1.0", "VDW softening factor for Phase 1 initial docking (0.5–0.9) and Phase 3 final precision redocking (1.0)."),
            ("--initial-poses-to-retain / --final-poses-to-retain", "20 / 10", "Number of poses carried from Phase 1 into refinement and kept in final output."),
            ("--refinement-method / --max-refinement-cycles", "minimization / 3", "Receptor relaxation engine and maximum refinement cycles."),
            ("--scoring-function", "ifd_composite", "Composite IFD ranking combining ligand binding score and receptor strain."),
            ("--use-gpu / --gpu-id / --cpu-workers", "False / 0 / auto", "Hardware acceleration and memory limit (`--max-memory-gb 8.0`)."),
            ("--energy-cutoff / --rmsd-clustering / -o", "50.0 / 2.0 / flex_docking", "Pose energy filter (kcal/mol), clustering threshold (Å), and output filename prefix."),
        ])

        mb.command_banner("pandadock-metal <analyze-protein|prepare-ligand|validate|dock|list-metals>", badge="METALLOPROTEINS", toc_label="pandadock-metal (5 commands)")
        mb.para(
            "Nearly one-third of all enzymes contain catalytic metal ions (`Zn²⁺`, `Fe²⁺/³⁺`, `Mg²⁺`, `Mn²⁺`, `Cu²⁺`, `Ca²⁺`, "
            "`Co²⁺`, etc.). Standard docking functions often underestimate strong metal–chelate coordination or misread "
            "organometallic bonds. **`pandadock-metal`** ships a bundled 31-metal parameter library (`AD4_parameters_full.dat`), "
            "supports Glide-style **zero-order metal bonds**, and enforces octahedral/tetrahedral/square-planar coordination geometry."
        )
        mb.option_table([
            ("pandadock-metal list-metals", "No args", "Displays all 31 supported metal ions, preferred coordination numbers, ideal bond lengths, and parameters."),
            ("pandadock-metal analyze-protein", "-r PDB [-o JSON]", "Inspects a metalloprotein PDB, locates every metal center, and lists its coordinating protein residues and geometry."),
            ("pandadock-metal prepare-ligand", "-i IN -o OUT [--validate]", "Prepares a metal-containing ligand by converting covalent metal bonds to zero-order bonds and adjusting formal charges."),
            ("pandadock-metal validate", "-l LIG [--metal-params]", "Audits coordination geometry, formal charges, and bond orders of a prepared metal ligand before docking."),
            ("pandadock-metal dock", "-r PDB -l LIG --center/--box", "Runs metal-aware docking (`--scoring metal_aware`, `--use-constraints`, `--constraint-weight 10.0`, `--prepare-ligand`, `--validate-geometry`, `--num-poses 20`, `-o DIR`)."),
        ])

        mb.command_banner("pandadock-tethered <extract|dock|validate|analyze>", badge="VALIDATION", toc_label="pandadock-tethered (4 commands)")
        mb.para(
            "When benchmarking a new protein target, computational chemists first **redock the known crystal ligand** to "
            "prove the setup reproduces the experimental X-ray pose (< 2.0 Å RMSD). **`pandadock-tethered`** handles "
            "crystal extraction, tethered local redocking within a defined radius, and scoring-function validation."
        )
        mb.option_table([
            ("pandadock-tethered extract", "-i COMPLEX.pdb -l LIG [-c CHAIN]", "Cleanly splits an X-ray ligand (e.g. `AZM`, `ATP`, `HEM`) out of a PDB complex into SDF/MOL2/PDB (`--output-format sdf`)."),
            ("pandadock-tethered dock", "-r REC -l LIG --ref REF.pdb", "Docks the ligand while softly constraining it within `--tether-radius` (default 2.0 Å) of the reference crystal pose (`-s physics_based`, `-a genetic`, `-n 10`)."),
            ("pandadock-tethered validate", "-i COMPLEX_OR_DIR [-l LIG]", "Scores crystallographic poses directly and optionally compares predictions against experimental `--reference-data` CSV/JSON."),
            ("pandadock-tethered analyze", "-i COMPLEX.pdb -l LIG", "One-shot validation pipeline: automatically runs **extract → tethered dock → validate** and writes a complete benchmark report."),
        ])

        # =====================================================================
        # CHAPTER 10: GNN SCORING & CLASSICAL ML PIPELINES
        # =====================================================================
        mb.chapter(
            "10",
            "SE(3)-Equivariant GNN Scoring & Machine Learning",
            "Using PandaDock-GNN (trained on 741,706 co-folded complexes) to predict binding affinity and rank ligands, plus classical ML docking pipelines (`pandadock-ml`).",
            start_new_page=True,
            toc_title="GNN Scoring & ML Pipelines",
        )
        mb.callout(
            "Important Scientific Best Practice — Separation of Concerns",
            "**Use physics (`pandadock dock`) to choose the 3D pose; use PandaDock-GNN to estimate binding affinity (pIC50/pKi) and rank different compounds against the same target.** "
            "Why? Across our PDBbind and pose-selection benchmarks, empirical Vina/Vinardo scoring picks accurate 3D poses far better than an affinity-trained GNN (median RMSD 2.09 Å vs 5.36 Å), whereas PandaDock-GNN excels at estimating potency (R = 0.53 on 4,640 held-out PDBbind complexes). Use `pandadock gnn predict` or `rescore` on physics-docked poses — avoid relying on `pandadock hybrid` alone for pose selection.",
            tone="warn",
        )
        mb.command_banner("pandadock gnn <SUBCOMMAND>   |   pandadock-gnn <SUBCOMMAND>", badge="GNN · 8 COMMANDS", toc_label="pandadock gnn / pandadock-gnn (8 cmds)")
        mb.para(
            "All 8 GNN commands work identically whether you type **`pandadock gnn <cmd>`** or **`pandadock-gnn <cmd>`**. "
            "Checkpoints automatically store training metadata (`site_radius`, `strip_hydrogens`), so you never mismatch preprocessing."
        )
        mb.option_table([
            ("download-model", "-o models/ [-v latest|sair] [--list]", "Downloads official pre-trained PandaDock-GNN weights (~82 MB) from GitHub Releases. Pass `--list` to see available models."),
            ("predict", "-m CKPT -p PROT -l LIG [-s SITE]", "Predicts binding affinity (pKd/pIC50 and v5 pose-RMSD head when trained) for a single protein–ligand complex (`--site-radius`, `--strip-hydrogens`, `-o JSON`)."),
            ("rescore", "-m CKPT -r REC -p POSES.sdf", "Universal rescorer for poses generated by **any** docking tool (PandaDock, Vina, Glide, GOLD). Writes ranked CSV (`-o`) and optional re-ranked SDF (`--output-sdf`)."),
            ("train", "-d ULVSH / -p PDBbind / -b BindingDB -o OUT", "Trains an SE(3)-equivariant EGNN from scratch (`--epochs 100`, `--batch-size 32`, `--lr 1e-4`, `--hidden-dim 256`, `--num-layers 6`, `--dropout 0.1`, `--split`, `--patience 20`, `--balanced`, `--gpu`, `--seed 42`)."),
            ("train-sair", "-c CACHE_DIR -o OUT", "Trains on large SAIR shard caches using strict target-disjoint splits, supporting v5 within-target ranking loss (`--rank-weight 1.0`, `--target-grouped`, `--ligands-per-target 4`, `--scheduler cosine_anneal`, `--target-centered`, `--block-shards 16`)."),
            ("benchmark", "-m CKPT -d DATASET -o OUT", "Evaluates Pearson R, Spearman ρ, RMSE, and MAE on the held-out `--split test` set (optionally `--compare-baselines`)."),
            ("compare", "-m CKPT -d DATASET -o OUT", "Head-to-head comparison report of PandaDock-GNN vs Vina, Vinardo, and physics-based baselines."),
            ("list-algorithms", "No args", "Lists available EGNN architectures, v5 dual-head capabilities, and supported commands."),
        ])

        mb.command_banner("pandadock hybrid -r REC -l LIG -m MODEL.pt [OPTIONS]", badge="HYBRID PIPELINE", toc_label="pandadock hybrid")
        mb.para(
            "Convenience pipeline that runs docking (`-n 50` poses) followed immediately by GNN rescoring and keeps "
            "`--top-k 10` poses. Supports `-g/--grid-config`, `--center`, `--box`, `--fast`, and `-o hybrid_output`. "
            "Remember the advisory above: for best accuracy, inspect the top physics poses alongside the GNN affinity estimates."
        )

        mb.command_banner("pandadock-ml <dock|hybrid-dock|extract-features|train|list-models>", badge="ML · 5 COMMANDS", toc_label="pandadock-ml (5 commands)")
        mb.option_table([
            ("pandadock-ml list-models", "No args", "Lists available classical ML and diffusion pose-proposal / ranking models."),
            ("pandadock-ml extract-features", "-c DIR -o features.h5", "Extracts volumetric & interaction features from a folder of complexes (`--feature-types`, `--grid-resolution 1.0`, `--pocket-radius 10.0`)."),
            ("pandadock-ml train", "-d DATA -m TYPE -o DIR", "Trains custom pose-ranking or affinity models (`--epochs 100`, `--batch-size 32`, `--learning-rate 1e-4`, `--validation-split 0.2`, `--pretrained-weights`, `--gpu`)."),
            ("pandadock-ml dock", "-r REC -l LIG --center/--box", "ML-guided pose generation (`-m diffusion`, `-n 50`, `--temperature 1.0`, `--confidence-threshold 0.7`, `--hybrid-scoring`, `--physics-refinement`)."),
            ("pandadock-ml hybrid-dock", "-r REC -l LIG --center/--box", "Generates `--ml-proposals 100` ML candidates and refines the `--top-k-refinement 20` best proposals with physics docking."),
        ])
        mb.code_block(
            "# Download pre-trained PandaDock-GNN model and predict binding affinity for Pose #1\n"
            "pandadock gnn download-model -o models/\n"
            "pandadock gnn predict -m models/pandadock_gnn.pt -p receptor.pdb -l results/pose1.pdb -o affinity.json\n\n"
            "# Universally rescore poses docked in Glide, Vina, GOLD, or PandaDock\n"
            "pandadock gnn rescore -m models/pandadock_gnn.pt -r receptor.pdb -p docked_poses.sdf -o gnn_scores.csv",
            caption="SE(3)-Equivariant GNN Affinity Prediction & Universal Rescoring",
        )

        # =====================================================================
        # CHAPTER 11: PUBLICATION REPORTS, PLOTS & PANDAMAP (pandadock-report)
        # =====================================================================
        mb.chapter(
            "11",
            "Publication Reports, 300-DPI Plots & PandaMap",
            "Understanding every output file PandaDock writes, and regenerating custom 300-DPI figures, 2D/3D PandaMap diagrams, and mathematical scoring breakdowns.",
            start_new_page=True,
            toc_title="Reports, Plots & PandaMap",
        )
        mb.para(
            "Every `pandadock dock`, `pandadock series`, and `pandadock covalent` run automatically builds a full "
            "publication bundle inside your `-o` folder — so you never have to write custom plotting scripts just to "
            "inspect your run or drop figures into a manuscript."
        )
        mb.section("What Every Output File Means (Plain-English Guide)", record_toc=True, toc_label="Understanding Output Files")
        mb.option_table(
            [
                ("poses.sdf", "3D Poses", "Multi-molecule SDF with exact bond orders and scores for all ranked binding poses (Pose #1 is best)."),
                ("complex1.pdb …", "3D Complexes", "Full receptor + docked ligand combined PDB files ready for PandaDock Studio, PyMOL, ChimeraX, or PandaLens."),
                ("report.pdf", "Vector A4 PDF", "Multi-page Panda-family publication brochure (energetics funnel, pose MDS landscape, contacts, hotspots, PoseBusters, PandaMap)."),
                ("report.html", "Interactive Web", "Self-contained editorial HTML report you can share with colleagues by email or view inside PandaDock Studio."),
                ("posebusters_audit.json", "Validity Audit", "Exact 11-point PoseBusters pass/fail results for every docked pose (strain, clashes, flatness, chirality)."),
                ("pose_scores.png & energy_components.png", "300-DPI Plots", "Score-vs-RMSD binding funnel, Boltzmann populations at 298 K, and stacked favorable/unfavorable energy decomposition."),
                ("pose_diversity.png & ligand_dynamics.png", "300-DPI Plots", "Pairwise RMSD matrix, 2D MDS conformational clustering, per-atom RMSF, and rotatable torsion distributions."),
                ("interaction_fingerprint.png & residue_hotspots.png", "300-DPI Plots", "Contact occupancy across poses (H-bond, hydrophobic, π-stacking, salt bridge, metal) and per-residue energy contributions."),
                ("pandamap_2d_complex_1.png", "2D/3D Map", "Discovery Studio–style 2D protein–ligand interaction diagram (plus covalent bonds on covalent runs) & 3D contact map."),
            ],
            headers=("Output File", "Type", "What It Shows You"),
            col_widths=(1.95, 0.95, CW - 2.90),
        )
        mb.command_banner("pandadock-report <combined|pandamap|plots|scoring> -i RUN_DIR [OPTIONS]", badge="REPORTS · 4 CMDS", toc_label="pandadock-report (4 commands)")
        mb.option_table([
            ("pandadock-report combined", "-i DIR [-o OUT] [-t TITLE] [-n 3]", "Regenerates BOTH publication plots (PNG/PDF) AND PandaMap 2D/3D interaction maps in one shot (`--simple`, `--with-3d`, `--skip-3d`)."),
            ("pandadock-report pandamap", "-i DIR [-o OUT] [-n 3]", "Generates Discovery Studio–style 2D interaction maps and interactive 3D HTML contact maps for the top `-n` poses (`--generate-3d`, `--skip-3d`)."),
            ("pandadock-report plots", "-i DIR [-o OUT] [-t TITLE]", "Creates the comprehensive 8-panel publication figure suite (or pass `--simple` for a compact 4-panel figure)."),
            ("pandadock-report scoring", "-i DIR [-o FILE] [-n 10]", "Writes a detailed text report breaking down VdW, H-bond, hydrophobic, electrostatic, and desolvation terms (`--detailed` includes full mathematical equations and `--include-references` adds citations)."),
        ])
        mb.code_block(
            "# Regenerate high-resolution plots + PandaMap 2D/3D diagrams for top 5 poses\n"
            "pandadock-report combined -i my_docking_run/ -t 'EGFR Inhibitor Series' --top-poses 5\n\n"
            "# Export full mathematical energy decomposition table for your manuscript Methods/SI\n"
            "pandadock-report scoring -i my_docking_run/ --detailed --top-poses 10 -o scoring_table.txt",
            caption="Regenerating Custom Reports & Scoring Tables",
        )

        # =====================================================================
        # CHAPTER 12: PRACTICAL RECIPES & TROUBLESHOOTING FAQ
        # =====================================================================
        mb.chapter(
            "12",
            "Practical Recipes & Plain-English Troubleshooting",
            "Quick answers to the most common docking questions — from co-crystal complexes and water stripping to hardware choice and PoseBusters fixes.",
            start_new_page=True,
            toc_title="Recipes & Troubleshooting",
        )
        mb.section("Frequently Asked Questions & Practical Solutions", record_toc=True, toc_label="Troubleshooting & FAQ")
        mb.option_table(
            [
                ("I uploaded a protein–ligand complex PDB. Will the old X-ray ligand block my new drug?", "Grid Setup", "In **PandaDock Studio**, click the detected **Co-Crystal Ligand** chip in Step 2: Studio centers the grid box on that pocket AND automatically strips only that X-ray ligand before docking (keeping protein, metals, and waters). On the CLI, first split the complex with `pandadock-tethered extract` or remove the old ligand `HETATM` lines before calling `pandadock dock`."),
                ("Should I strip crystallographic waters or keep them?", "Waters", "Start with `--waters strip` (default) for general screening. If you know the pocket contains conserved bridging waters (e.g. carbonic anhydrase, kinases, HIV protease), use `--waters displace` so PandaDock keeps bridging waters while automatically displacing clashing solvent waters."),
                ("My zinc or iron enzyme pose isn't coordinating the metal!", "Metals", "Add a smooth geometric restraint in `pandadock dock`, e.g. `--restraint 'A:ZN:301:ZN<2.3:donor'`, or use the dedicated `pandadock-metal dock` workflow with zero-order bonds and coordination geometry validation."),
                ("PoseBusters flagged high internal strain (> 15 kcal/mol) or a clash.", "Geometry", "Run `pandadock-prepare --minimize-ligand` first so your starting ligand has clean MMFF94 geometry, increase `--exhaustiveness 16`, or use `--sample-rings` for flexible saturated rings. If the pocket is tight, try `pandadock-flex` induced-fit docking."),
                ("Which scoring function should I choose?", "Scoring", "Use **`vina`** as your reliable all-around default; try **`vinardo`** for hydrophobic or buried pockets; use **`--rescoring mmgbsa`** when comparing charged ligands; and use **`pandadock gnn predict`** (or `calibrate --weights-file`) when ranking potency across a series."),
                ("When should I use CPU vs Apple Silicon MPS vs NVIDIA CUDA?", "Hardware", "For a single drug on a laptop, `--device cpu --workers 4` finishes in a few seconds. For very flexible ligands (10+ rotatable bonds) or virtual screening libraries (`pandadock screen`), `--device cuda` or `--device mps --n-chains 512` gives **8×–35× speedups**."),
                ("How do I see binding-site hydrogens and interactions after docking?", "Visualization", "Open the run in **PandaDock Studio** (`pandadock-gui`). It opens in **Binding Pocket** mode with explicit white H on both pocket residues and ligand (`Show H`), all 18 **PandaLens** interaction classes, hover-to-highlight rows (`List`), and draggable residue labels."),
            ],
            headers=("Common Question / Situation", "Topic", "Recommended PandaDock 5.0 Action"),
            col_widths=(2.10, 0.85, CW - 2.95),
        )
        mb.callout(
            "Complete 60-Second End-to-End Cheat Sheet",
            "1. **Prepare**: `pandadock-prepare -r raw.pdb -l lig.sdf --ph 7.4 -o prep/`\n"
            "2. **Grid Box**: `pandadock-gridbox -r prep/receptor.pdb -m similarity --reference-ligand xray.sdf -o box.json`\n"
            "3. **Dock + Audit + Reports**: `pandadock dock -r prep/receptor.pdb -l prep/ligand.sdf -g box.json --waters displace --workers 4 --seed 42 -o out/`\n"
            "4. **Inspect in 3D + Plots + PDF**: `pandadock-gui --workspace out/`  (opens Studio with PandaLens 3D pocket + H + report.pdf)\n"
            "5. **Launch MD**: `pandadock to-md -r prep/receptor.pdb -p out/poses.sdf --engine both --duration-ns 50 -o md_pkg/`",
            tone="tip",
        )

        # =====================================================================
        # END PAGE (Exact Panda-family brochure back cover)
        # =====================================================================
        end_page(
            doc,
            heading="Core Pillars of PandaDock 5.0",
            tiles=[
                ("Physics & GPU", "Analytic-gradient L-BFGS flexible docking across CPU threads, Apple MPS, and NVIDIA CUDA, with restraints & displaceable waters."),
                ("Med-Chem v5", "pH 7.4 state enumeration, 8 covalent warheads, MCS series SAR brochures, PoseBusters audits, and few-shot calibration."),
                ("Studio & 3D", "Apple-HIG local GUI with 4-step wizard, Maestro-style X-ray pocket selection, explicit-H PandaLens 3D, and PDF/HTML reports."),
                ("GNN & MD", "SE(3)-equivariant EGNN trained on 741k complexes plus one-command OpenMM, GROMACS & PandaInsight export."),
            ],
            methods_heading="Software Summary & Citation Details",
            methods_rows=[
                ("Version", f"PandaDock v{__version__} (MIT Open-Source License)"),
                ("Author", "Pritam Kumar Panda, Stanford University"),
                ("CLI Footprint", "10 entry points · 48 leaf commands (100% documented in this manual)"),
                ("Dependencies", "Python 3.9+, RDKit, NumPy, SciPy, Biopython, Matplotlib; optional PyTorch/PyG ([gnn]), FastAPI/Uvicorn ([gui]), PandaMap"),
                ("Source & Docs", "https://github.com/pritampanda15/PandaDock  ·  https://pandadock.readthedocs.io/"),
            ],
            references=[
                "Panda PK. PandaDock v5.0: Flexible Molecular Docking, Covalent & Series SAR, Studio GUI, and SE(3)-Equivariant GNN Scoring. **Stanford University**, 2026.",
                "Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. **J. Comput. Chem.** 2010, 31, 455–461.",
                "Quiroga R, Villarreal MA. Vinardo: a scoring function based on Vina improves scoring, docking, and virtual screening. **PLoS ONE** 2016, 11, e0155183.",
                "Buttenschoen M, Morris GM, Deane CM. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. **Chem. Sci.** 2024, 15, 3130–3139.",
                "Morris GM, et al. AutoDock4 and AutoDockTools4: automated docking with selective receptor flexibility. **J. Comput. Chem.** 2009, 30, 2785–2791.",
                "Landrum G. RDKit: Open-source cheminformatics. https://www.rdkit.org.",
            ],
            contact="https://github.com/pritampanda15/PandaDock",
        )

        # =====================================================================
        # NOW RENDER PAGE 2 (TABLE OF CONTENTS) WITH EXACT COMPUTED PAGE NUMBERS
        # =====================================================================
        toc_links = render_toc_page(toc_page, mb.entries)

        # Save raw vector PDF via report_kit.Doc
        raw_pdf = output_pdf.with_suffix(".raw.pdf")
        doc.save(raw_pdf, footer=True)

    # =========================================================================
    # POST-PROCESS WITH PYMUPDF: INJECT REAL CLICKABLE TOC LINKS, "↑ CONTENTS"
    # BACK-LINKS ON EVERY CONTENT PAGE, AND FULL SIDEBAR BOOKMARKS OUTLINE!
    # =========================================================================
    pdf_doc = fitz.open(str(raw_pdf))
    n_pages = len(pdf_doc)

    # 1. Clickable TOC links on Page 2 (0-based index 1)
    toc_p = pdf_doc[1]
    for (x, y, w, h), target_idx in toc_links:
        if 0 <= target_idx < n_pages:
            rect = fitz.Rect(x * 72.0, y * 72.0, (x + w) * 72.0, (y + h) * 72.0)
            toc_p.insert_link({
                "kind": fitz.LINK_GOTO,
                "from": rect,
                "page": target_idx,
                "to": fitz.Point(0, 0),
            })

    # 2. Clickable "↑ Contents" back-links on every content page
    for page_idx, (x, y, w, h) in mb.back_links:
        if 0 <= page_idx < n_pages:
            rect = fitz.Rect(x * 72.0, y * 72.0, (x + w) * 72.0, (y + h) * 72.0)
            pdf_doc[page_idx].insert_link({
                "kind": fitz.LINK_GOTO,
                "from": rect,
                "page": 1,  # jump straight back to Table of Contents (Page 2)
                "to": fitz.Point(0, 0),
            })

    # 3. Full clickable PDF Outline / Bookmarks tree (macOS Preview / Acrobat / Chrome)
    outline = [
        [1, f"Cover — PandaDock {__version__} User Manual", 1],
        [1, "Table of Contents", 2],
    ]
    for e in mb.entries:
        if e["level"] == 1:
            outline.append([1, f"Chapter {e['number']}: {e['title']}", e["page"]])
        else:
            outline.append([2, e["title"], e["page"]])
    outline.append([1, "Core Pillars, Methods & References", n_pages])
    pdf_doc.set_toc(outline)

    # Set clean PDF metadata
    pdf_doc.set_metadata({
        "title": f"PandaDock {__version__} Official User Manual",
        "author": "Pritam Kumar Panda (Stanford University)",
        "subject": "Plain-Language Molecular Docking Manual, CLI Reference & PandaDock Studio Guide",
        "keywords": "PandaDock, molecular docking, Vina, Vinardo, covalent docking, PoseBusters, PandaLens, GNN, OpenMM, GROMACS",
        "creator": f"PandaDock {__version__} Report Kit + PyMuPDF",
    })

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf_doc.save(str(output_pdf), garbage=4, deflate=True)
    pdf_doc.close()
    raw_pdf.unlink(missing_ok=True)

    print(f"Generated {output_pdf} ({n_pages} pages, {len(toc_links)} clickable TOC links, {len(mb.back_links)} back-to-TOC links, {len(outline)} PDF bookmarks)")
    return output_pdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    out_path = REPO_ROOT / "docs" / "PandaDock_v5.0_User_Manual.pdf"
    build_manual(out_path)
