"""Printable 1-page bedside risk card PDF (v3 — cleaner layout)."""
from __future__ import annotations
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).parent
OUT = HERE / "docs" / "bedside_card.pdf"

NAVY = HexColor("#374E55")
NAVY_DK = HexColor("#2A3C40")
GOLD = HexColor("#B98538")
RED = HexColor("#8C2F2D")
GREEN = HexColor("#496B57")
GREY = HexColor("#5C5A50")
PALE = HexColor("#F1EDE2")
WHITE = HexColor("#FFFFFF")
RULE = HexColor("#DCD7C9")
INK = HexColor("#222222")
WARN_BG = HexColor("#FBF1DA")
WARN_BD = HexColor("#C99B3A")
WARN_TX = HexColor("#5C3F00")

# Register a font that supports the full unicode set we use (×, ⁹, ≥, ⚠, →).
# DejaVu Sans (shipped with matplotlib) covers them all.
import matplotlib
_mpl_root = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
_candidates = {
    "UI":         _mpl_root / "DejaVuSans.ttf",
    "UI-Bold":    _mpl_root / "DejaVuSans-Bold.ttf",
    "UI-Italic":  _mpl_root / "DejaVuSans-Oblique.ttf",
}
try:
    for name, path in _candidates.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    FONT, FONT_B, FONT_I = "UI", "UI-Bold", "UI-Italic"
except Exception:
    FONT, FONT_B, FONT_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def main():
    c = canvas.Canvas(str(OUT), pagesize=letter)
    W, H = letter

    # ---- Header bar ----
    c.setFillColor(NAVY)
    c.rect(0, H - 0.85 * inch, W, 0.85 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT_B, 16)
    c.drawString(0.5 * inch, H - 0.42 * inch,
                 "MMA EMBOLIZATION  —  RESCUE SURGERY RISK SCORE")
    c.setFont(FONT, 9.5)
    c.drawString(0.5 * inch, H - 0.65 * inch,
                 "Pre-procedural bedside card  ·  v2  ·  Single-center derivation, n=214")

    # ---- Warning ribbon ----
    c.setFillColor(WARN_BG)
    c.rect(0, H - 1.20 * inch, W, 0.32 * inch, fill=1, stroke=0)
    c.setStrokeColor(WARN_BD)
    c.setLineWidth(0.6)
    c.line(0, H - 1.20 * inch, W, H - 1.20 * inch)
    c.line(0, H - 0.88 * inch, W, H - 0.88 * inch)
    c.setFillColor(WARN_TX)
    c.setFont(FONT_B, 9.5)
    c.drawString(0.5 * inch, H - 1.05 * inch, "⚠   Research use only.")
    c.setFont(FONT, 9.5)
    c.drawString(2.20 * inch, H - 1.05 * inch,
                 "Internally derived; NOT externally validated. Adjunct to clinical judgment, not a substitute.")

    # =================================================================
    # LEFT COLUMN — Model 1 score components
    # =================================================================
    y = H - 1.55 * inch
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 11.5)
    c.drawString(0.5 * inch, y, "MODEL 1  (full · max 8 pts)")
    c.setStrokeColor(NAVY)
    c.setLineWidth(0.7)
    c.line(0.5 * inch, y - 0.05 * inch, 4.6 * inch, y - 0.05 * inch)

    items = [
        ("Age   <65 / 65–80 / >80",            "0 / 1 / 2"),
        ("SDH volume ≥ 100 mL",           "+1"),
        ("Anticoagulation",                    "+1"),
        ("Absence of focal deficit",           "+1"),
        ("Platelets <150 ×10⁹/L",    "+1"),
        ("Antiplatelet therapy",               "+1"),
        ("Anterior + posterior embolization",  "+1"),
    ]
    y -= 0.35 * inch
    c.setFont(FONT, 10.5)
    for label, pts in items:
        c.setStrokeColor(NAVY)
        c.setLineWidth(1.0)
        c.rect(0.55 * inch, y - 0.04 * inch, 0.16 * inch, 0.16 * inch,
               fill=0, stroke=1)
        c.setFillColor(INK)
        c.setFont(FONT, 10.5)
        c.drawString(0.85 * inch, y, label)
        c.setFillColor(GOLD)
        c.setFont(FONT_B, 10.5)
        c.drawRightString(4.55 * inch, y, pts)
        y -= 0.30 * inch

    # ---- Score → rate lookup (Model 1) ----
    y -= 0.10 * inch
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 11.5)
    c.drawString(0.5 * inch, y, "TOTAL SCORE  →  RESCUE-RATE LOOKUP")
    c.line(0.5 * inch, y - 0.05 * inch, 4.6 * inch, y - 0.05 * inch)
    y -= 0.30 * inch

    rows = [
        ("Score", "n",  "Failures", "Rate"),
        ("0",     "1",  "0",        "0.0%"),
        ("1",     "11", "1",        "9.1%"),
        ("2",     "38", "3",        "7.9%"),
        ("3",     "56", "4",        "7.1%"),
        ("4",     "59", "7",        "11.9%"),
        ("5",     "38", "16",       "42.1%"),
        ("6",     "8",  "3",        "37.5%"),
        ("≥7", "3", "2",       "66.7%"),
    ]
    col_x = [0.55 * inch, 1.50 * inch, 2.45 * inch, 3.50 * inch]
    row_h = 0.21 * inch
    for i, row in enumerate(rows):
        is_high = i >= 6  # scores 5, 6, 7 are high
        if i == 0:
            c.setFillColor(NAVY)
            c.rect(0.50 * inch, y - 0.04 * inch, 4.10 * inch, row_h,
                   fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont(FONT_B, 10)
        else:
            if is_high:
                c.setFillColor(HexColor("#F4DDDC"))
                c.rect(0.50 * inch, y - 0.04 * inch, 4.10 * inch, row_h,
                       fill=1, stroke=0)
                c.setFillColor(RED)
                c.setFont(FONT_B, 10)
            else:
                if i % 2 == 0:
                    c.setFillColor(PALE)
                    c.rect(0.50 * inch, y - 0.04 * inch, 4.10 * inch, row_h,
                           fill=1, stroke=0)
                c.setFillColor(INK)
                c.setFont(FONT, 10)
        for j, cell in enumerate(row):
            c.drawString(col_x[j] + 0.05 * inch, y + 0.02 * inch, cell)
        y -= row_h

    # =================================================================
    # RIGHT COLUMN — Action triggers
    # =================================================================
    rx = 5.05 * inch
    rw = 3.20 * inch
    ry = H - 1.55 * inch
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 11.5)
    c.drawString(rx, ry, "RISK BAND  &  ACTION")
    c.line(rx, ry - 0.05 * inch, rx + rw, ry - 0.05 * inch)

    actions = [
        (GREEN, "Score 0–2  ·  Low",
         "Standard post-procedure care.\nRoutine clinical follow-up."),
        (GOLD,  "Score 3–4  ·  Intermediate",
         "Standard surveillance + interval\nimaging at 2–4 weeks."),
        (RED,   "Score ≥ 5  ·  High",
         "Tighter surveillance. Early CT\n(24–72 h). Low threshold for rescue."),
    ]
    ay = ry - 0.40 * inch
    for color, header, text in actions:
        # color stripe
        c.setFillColor(color)
        c.rect(rx, ay - 0.78 * inch, 0.10 * inch, 0.78 * inch,
               fill=1, stroke=0)
        # header
        c.setFillColor(color)
        c.setFont(FONT_B, 10.5)
        c.drawString(rx + 0.20 * inch, ay - 0.12 * inch, header)
        # body — wrap manually using \n
        c.setFillColor(INK)
        c.setFont(FONT, 9.5)
        for k, ln in enumerate(text.split("\n")):
            c.drawString(rx + 0.20 * inch,
                         ay - 0.32 * inch - k * 0.16 * inch, ln)
        ay -= 0.95 * inch

    # ---- Performance summary box ----
    ry2 = ay - 0.15 * inch
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 11.5)
    c.drawString(rx, ry2, "PERFORMANCE")
    c.line(rx, ry2 - 0.05 * inch, rx + rw, ry2 - 0.05 * inch)
    ry2 -= 0.30 * inch

    perf_lines = [
        ("AUC (Model 1)",          "0.73 (corrected)"),
        ("Cutoff ≥ 5",        "Sens 58%  ·  Spec 84%"),
        ("Low vs high",            "9.1% vs 42.9% rescue"),
        ("Calibration",            "Hosmer–Lemeshow OK"),
    ]
    c.setFont(FONT, 10)
    for label, value in perf_lines:
        c.setFillColor(GREY)
        c.setFont(FONT, 9.5)
        c.drawString(rx, ry2, label)
        c.setFillColor(NAVY)
        c.setFont(FONT_B, 9.5)
        c.drawRightString(rx + rw, ry2, value)
        ry2 -= 0.22 * inch

    # ---- Calculator URL ----
    ry2 -= 0.10 * inch
    c.setFillColor(NAVY)
    c.setFont(FONT_B, 9.5)
    c.drawString(rx, ry2, "Live calculator")
    ry2 -= 0.18 * inch
    c.setFillColor(GREY)
    c.setFont(FONT, 9)
    c.drawString(rx, ry2, "nielspac177.github.io/mmae-rescue-score/")

    # =================================================================
    # FOOTER
    # =================================================================
    c.setFillColor(NAVY)
    c.rect(0, 0.45 * inch, W, 0.30 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont(FONT, 8.5)
    c.drawString(0.5 * inch, 0.55 * inch,
        "Pre-procedural use only  ·  not for acute SDH triage  ·  external validation pending  ·  TRIPOD-compliant")
    c.drawRightString(W - 0.5 * inch, 0.55 * inch, "v2")

    c.setFillColor(GREY)
    c.setFont(FONT_I, 8.5)
    c.drawString(0.5 * inch, 0.25 * inch,
        "Score complements clinical judgment. Not a substitute. See live calculator for Model 2 (simple) and Model 3 alternatives.")

    c.save()
    print(f"Wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
