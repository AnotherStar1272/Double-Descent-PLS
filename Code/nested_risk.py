"""Redraw the two smooth omitted-signal examples from the JRSSB manuscript."""
import csv
from math import log10, sqrt

from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont

from interpolation_peak import critical_risk
from reproduce import ROOT, figure_canvas, np, series, write_csv


PLS_GREYS = {1: "#101010", 2: "#383838", 5: "#606060", 10: "#828282", 20: "#a0a0a0"}
STYLES = {f"PLS{k}": (grey, ()) for k, grey in PLS_GREYS.items()}
STYLES.update({"OLS": ("#424242", (5, 3)), "Optimal ridge": ("#282828", (7, 2, 1, 2)),
               "Null": ("#707070", (1, 2))})


def limit_curves(alpha, grid):
    curves = {name: [] for name in STYLES}
    for gamma in grid:
        omitted = 5 * (1 + gamma)**(-alpha)
        signal, noise = 5 - omitted, 1 + omitted
        for k in PLS_GREYS:
            curves[f"PLS{k}"].append(omitted + noise * critical_risk(gamma, signal / noise, k))
        if gamma < 1:
            ols = omitted + noise * gamma / (1 - gamma)
        elif gamma > 1:
            ols = omitted + signal * (1 - 1 / gamma) + noise / (gamma - 1)
        else:
            ols = float("inf")
        curves["OLS"].append(ols)
        penalty = gamma * noise / signal
        delta = penalty + 1 - gamma
        transform = 2 / (sqrt(delta**2 + 4 * gamma * penalty) + delta)
        curves["Optimal ridge"].append(omitted + gamma * noise * transform)
        curves["Null"].append(5)
    return curves


def draw_legend(pdf, right, top):
    width, height = 143, 50
    left = right - width
    pdf.saveState()
    pdf.setDash()
    pdf.setStrokeGray(0.15)
    pdf.setFillColor(white)
    pdf.setLineWidth(0.6)
    pdf.rect(left, top - height, width, height, stroke=1, fill=1)
    for i, (method, (colour, dash)) in enumerate(STYLES.items()):
        x = left + 5 + (i // 4) * 69
        y = top - 8 - (i % 4) * 11
        pdf.setStrokeColor(HexColor(colour))
        pdf.setLineWidth(1.3)
        pdf.setDash(dash)
        pdf.line(x, y, x + 15, y)
        pdf.setDash()
        if method.startswith("PLS") or method == "OLS":
            pdf.setFillColor(white if method == "OLS" else HexColor(colour))
            pdf.setLineWidth(0.7)
            pdf.circle(x + 7.5, y, 1.7, fill=1, stroke=1)
        pdf.setFillGray(0)
        pdf.setFont("Times-Roman", 9)
        pdf.drawString(x + 18, y - 2.7, method.replace("PLS", "PLS k = "))
    pdf.restoreState()


def draw_panel(pdf, bottom, letter, alpha, grid, curves, samples, greek_font):
    left, width, height = 39, 343, 159
    log_min, log_max = log10(0.085), log10(12)

    def point(log_gamma, risk):
        return left + width * (log_gamma - log_min) / (log_max - log_min), bottom + height * risk / 12

    top = bottom + height
    pdf.setFillGray(0)
    pdf.setFont("Times-Bold", 11)
    pdf.drawString(left, top + 8, f"({letter})")
    pdf.setFont(greek_font or "Times-Italic", 11)
    pdf.drawString(left + 21, top + 8, f"α = {alpha}" if greek_font else f"Exponent {alpha}")
    pdf.setStrokeGray(0)
    pdf.setDash()
    pdf.setLineWidth(0.65)
    pdf.rect(left, bottom, width, height)
    pdf.setFont("Times-Roman", 10)
    for gamma in (0.1, 0.2, 0.5, 1, 2, 5, 10):
        x, _ = point(log10(gamma), 0)
        pdf.line(x, bottom, x, bottom - 3)
        pdf.drawCentredString(x, bottom - 13, str(gamma))
    for risk in range(0, 13, 2):
        _, y = point(0, risk)
        pdf.line(left - 3, y, left, y)
        pdf.drawRightString(left - 6, y - 3, str(risk))
    pdf.setFont(greek_font or "Times-Italic", 12)
    pdf.drawCentredString(left + width / 2, bottom - 25, "γ" if greek_font else "Aspect ratio")
    pdf.saveState()
    pdf.translate(left - 27, bottom + height / 2)
    pdf.rotate(90)
    pdf.setFont("Times-Roman", 11)
    pdf.drawCentredString(0, 0, "Generalisation error")
    pdf.restoreState()
    pdf.saveState()
    box = pdf.beginPath()
    box.rect(left, bottom, width, height)
    pdf.clipPath(box, stroke=0)
    pdf.setStrokeColor(HexColor("#b0b0b0"))
    pdf.setDash((1, 3))
    x, y = point(0, 0)
    pdf.line(x, y, x, y + height)
    for name, values in curves.items():
        colour, dash = STYLES[name]
        segments = [list(zip(grid, values))]
        if name == "OLS":
            segments = [[(g, v) for g, v in segments[0] if g < 1],
                        [(g, v) for g, v in segments[0] if g > 1]]
        for segment in segments:
            series(pdf, point, [(log10(g), v) for g, v in segment], colour, dash=dash)
    pdf.setDash()
    for row in samples:
        method = row["method"]
        px, py = point(log10(float(row["gamma"])), float(row["mean"]))
        colour = HexColor(STYLES[method][0])
        pdf.setStrokeColor(colour)
        pdf.setFillColor(white if method == "OLS" else colour)
        pdf.setLineWidth(0.7)
        pdf.circle(px, py, 1.7, fill=1, stroke=1)
    pdf.restoreState()
    draw_legend(pdf, left + width - 4, top - 4)


def main():
    figures, results = ROOT / "Figures", ROOT / "Code/results"
    figures.mkdir(exist_ok=True)
    grid = np.unique(np.r_[np.geomspace(0.1, 10, 1201), np.linspace(0.8, 1.2, 201), 1.0])
    with (results / "nested-risk-simulation.csv").open(newline="") as source:
        samples = list(csv.DictReader(source))
    pdf = figure_canvas(figures / "nested-risk.pdf", pagesize=(389, 419))
    # Embed the Greek glyphs so they survive inclusion in the LaTeX supplement.
    greek_font = None
    for filename in ("timesi.ttf", "LiberationSerif-Italic.ttf", "DejaVuSerif-Italic.ttf"):
        try:
            font = TTFont("PlotMathItalic", filename)
        except TTFError:
            continue
        pdfmetrics.registerFont(font)
        greek_font = font.fontName
        break
    records = []
    for alpha, letter, bottom in [(3, "A", 238), (5, "B", 30)]:
        curves = limit_curves(alpha, grid)
        draw_panel(pdf, bottom, letter, alpha, grid, curves,
                   [row for row in samples if row["model"] == f"a{alpha}"], greek_font)
        for name, values in curves.items():
            records.extend((alpha, gamma, name, value) for gamma, value in zip(grid, values))
    pdf.save()
    write_csv(results, "nested-risk.csv", ["exponent", "aspect_ratio", "method", "full_mean_risk_limit"],
              records)


if __name__ == "__main__":
    main()
