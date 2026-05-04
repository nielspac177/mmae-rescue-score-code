"""Printable 1-page bedside risk card PDF using reportlab."""
from __future__ import annotations
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

HERE = Path(__file__).parent
OUT = HERE / "docs" / "bedside_card.pdf"

NAVY = HexColor("#374E55")
GOLD = HexColor("#DF8F44")
RED = HexColor("#B24745")
GREEN = HexColor("#79AF97")
GREY = HexColor("#80796B")
PALE = HexColor("#F5F2EA")
WHITE = HexColor("#FFFFFF")
INK = HexColor("#222222")


def main():
    c = canvas.Canvas(str(OUT), pagesize=letter)
    W, H = letter

    # ---- Header bar ----
    c.setFillColor(NAVY)
    c.rect(0, H - 0.95 * inch, W, 0.95 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5 * inch, H - 0.45 * inch,
                 "MMA EMBOLIZATION  —  RESCUE SURGERY RISK SCORE")
    c.setFont("Helvetica", 10)
    c.drawString(0.5 * inch, H - 0.70 * inch,
                 "Pre-procedural bedside card  ·  Single-center derivation, n=214  ·  External validation pending")

    # ---- Section: Score components (Model 1 full) ----
    y = H - 1.30 * inch
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.5 * inch, y, "MODEL 1  (full, max 8 pts)")
    y -= 0.04 * inch
    c.setStrokeColor(NAVY)
    c.setLineWidth(0.6)
    c.line(0.5 * inch, y, W - 0.5 * inch, y)

    # Items with checkboxes
    items = [
        ("Age  <65 / 65–80 / >80", "0 / 1 / 2"),
        ("SDH volume ≥ 100 mL", "+1"),
        ("Anticoagulation", "+1"),
        ("Absence of focal deficit", "+1"),
        ("Platelets < 150 ×10⁹/L", "+1"),
        ("Antiplatelet therapy", "+1"),
        ("Anterior + posterior embolization", "+1"),
    ]
    y -= 0.30 * inch
    c.setFont("Helvetica", 11)
    c.setFillColor(INK)
    for label, pts in items:
        # checkbox
        c.setStrokeColor(NAVY)
        c.setLineWidth(1.0)
        c.rect(0.55 * inch, y - 0.04 * inch, 0.16 * inch, 0.16 * inch,
               fill=0, stroke=1)
        c.setFillColor(INK)
        c.drawString(0.85 * inch, y, label)
        c.setFillColor(GOLD)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(W - 1.2 * inch, y, pts)
        c.setFillColor(INK)
        c.setFont("Helvetica", 11)
        y -= 0.30 * inch

    # ---- Section: Total score band ----
    y -= 0.05 * inch
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.5 * inch, y, "TOTAL SCORE  →  RESCUE-RATE LOOKUP")
    c.line(0.5 * inch, y - 0.04 * inch, W - 0.5 * inch, y - 0.04 * inch)
    y -= 0.32 * inch

    # Score table
    rows = [
        ("Score", "n", "Failures", "Rate"),
        ("0", "1", "0", "0.0%"),
        ("1", "11", "1", "9.1%"),
        ("2", "38", "3", "7.9%"),
        ("3", "56", "4", "7.1%"),
        ("4", "59", "7", "11.9%"),
        ("5", "38", "16", "42.1%"),
        ("6", "8", "3", "37.5%"),
        ("≥7", "3", "2", "66.7%"),
    ]
    col_x = [0.55 * inch, 1.55 * inch, 2.55 * inch, 3.55 * inch]
    col_w = [1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]
    row_h = 0.21 * inch
    for i, row in enumerate(rows):
        if i == 0:
            c.setFillColor(NAVY)
            c.rect(0.5 * inch, y - 0.04 * inch, 4.0 * inch, row_h,
                   fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 10)
        else:
            score = int(row[0]) if row[0].isdigit() else 7
            if score >= 5:
                c.setFillColor(HexColor("#FBE9E5"))
                c.rect(0.5 * inch, y - 0.04 * inch, 4.0 * inch, row_h,
                       fill=1, stroke=0)
                c.setFillColor(RED)
                c.setFont("Helvetica-Bold", 10)
            else:
                if i % 2 == 0:
                    c.setFillColor(PALE)
                    c.rect(0.5 * inch, y - 0.04 * inch, 4.0 * inch, row_h,
                           fill=1, stroke=0)
                c.setFillColor(INK)
                c.setFont("Helvetica", 10)
        for j, cell in enumerate(row):
            c.drawString(col_x[j] + 0.05 * inch, y + 0.02 * inch, cell)
        y -= row_h

    # ---- Right column: Action triggers ----
    rx = 5.0 * inch
    ry = H - 1.65 * inch
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(rx, ry, "ACTION TRIGGERS")
    c.line(rx, ry - 0.04 * inch, W - 0.5 * inch, ry - 0.04 * inch)

    # Three action boxes
    actions = [
        (GREEN, "Score 0–2", "Standard post-procedure care."),
        (GOLD, "Score 3–4", "Standard surveillance + 2–4 wk CT."),
        (RED, "Score ≥ 5", "Tight surveillance; early CT 24–72 h; low threshold for rescue."),
    ]
    ay = ry - 0.40 * inch
    for color, label, text in actions:
        c.setFillColor(color)
        c.rect(rx, ay - 0.55 * inch, 0.10 * inch, 0.55 * inch,
               fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(rx + 0.18 * inch, ay - 0.10 * inch, label)
        c.setFillColor(INK)
        c.setFont("Helvetica", 9.5)
        # wrap
        words = text.split()
        line = ""
        lines = []
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 9.5) > 2.6 * inch:
                lines.append(line)
                line = w
            else:
                line = test
        lines.append(line)
        for k, ln in enumerate(lines):
            c.drawString(rx + 0.18 * inch, ay - 0.28 * inch - k * 0.16 * inch, ln)
        ay -= 0.80 * inch

    # ---- Bottom strip ----
    c.setFillColor(NAVY)
    c.rect(0, 0.45 * inch, W, 0.30 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica", 8.5)
    c.drawString(0.5 * inch, 0.55 * inch,
        "AUC 0.73 (corrected)  ·  HL P > 0.6  ·  External validation pending  ·  Score complements clinical judgment")
    c.drawRightString(W - 0.5 * inch, 0.55 * inch,
        "v2 · TRIPOD")

    # ---- Footer instruction ----
    c.setFillColor(GREY)
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawString(0.5 * inch, 0.25 * inch,
        "Use as decision aid, not as substitute for clinical judgment. Designed for pre-procedural use; not for acute SDH triage.")

    c.save()
    print(f"Wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
