#!/usr/bin/env python3
"""Build the 3-slide CEI-GNN overview deck and the standalone clinical-graph diagram.

Docs-only builder: no patient data, no model code, no training. All numbers are
copied from docs/cei-v3-evidence-2026-10-01.json (aggregate-only).

Usage (from anywhere):
    python3 docs/presentations/cei-overview/build_cei_overview.py
Outputs next to this script:
    assets/clinical-graph.svg        standalone vector diagram (+ legend, caveats)
    cei-overview.pptx                editable 3-slide deck
PNG and PDF are produced separately (see README.md): the PNG by macOS Quick Look
from the SVG, the PDF by a genuine Keynote export of the PPTX.
Requires python-pptx.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FONT = "Arial"
INK, TEAL, NAVY, GREY, RULE = "111111", "0B7A75", "16324F", "555555", "BBBBBB"
FILL = {  # node kind -> (fill, stroke)
    "patient": ("DCE6F2", "16324F"),
    "complaint": ("FCE8CF", "9A5B13"),
    "vital": ("E1F2DE", "3C7A33"),
    "lab": ("D2EEF0", "0B7A75"),
    "dx": ("EBDDF1", "6B3A80"),
}

# ---------------------------------------------------------------- diagram spec
# Core coordinate space 1200 x 470 (units). Directions are those of graph.py.
REGIONS = [
    (0, 0, 390, 470, "Earlier visits (before decision time)"),
    (420, 0, 780, 470, "Index visit (decision time)"),
]
NODES = [  # x, y, w, h, text, kind, dashed
    (450, 50, 140, 40, "Patient", "patient", False),
    (450, 150, 140, 40, "Index visit", "patient", False),
    (740, 30, 180, 40, "Complaint", "complaint", False),
    (740, 110, 180, 40, "Complaint", "complaint", False),
    (740, 200, 180, 40, "Vital sign", "vital", False),
    (740, 330, 180, 40, "Lab result", "lab", False),
    (740, 415, 180, 40, "Analyte", "lab", False),
    (30, 150, 200, 40, "Prior diagnosis", "dx", True),
    (30, 225, 200, 40, "Prior lab result", "lab", False),
    (30, 330, 200, 40, "Prior lab result", "lab", False),
]
# points, both_heads, dashed, strong(teal accent)
ARROWS = [
    ([(520, 90), (520, 150)], False, False, False),            # has_visit
    ([(590, 170), (740, 50)], False, False, False),            # reports_complaint
    ([(590, 170), (740, 130)], False, False, False),
    ([(590, 170), (740, 220)], False, False, False),           # observed_vital
    ([(590, 170), (740, 350)], False, False, False),           # measured_in
    ([(830, 70), (830, 110)], True, False, False),             # co_complaint (symmetric)
    ([(830, 370), (830, 415)], False, False, False),           # instance_of
    ([(230, 170), (450, 170)], False, True, False),            # recurrence_of (optional)
    ([(130, 265), (130, 330)], False, False, False),           # trajectory_of
    ([(230, 350), (740, 350)], False, False, True),            # baseline_of
    ([(130, 370), (130, 435), (740, 435)], False, False, False),  # instance_of (prior)
]
LABELS = [  # x, y(center), text, anchor, bold
    (530, 120, "has_visit", "start", False),
    (665, 92, "reports_complaint", "middle", False),
    (665, 192, "observed_vital", "middle", False),
    (700, 268, "measured_in", "middle", False),
    (842, 90, "co_complaint (both ways)", "start", False),
    (842, 393, "instance_of", "start", False),
    (340, 154, "recurrence_of (optional)", "middle", False),
    (142, 298, "trajectory_of", "start", False),
    (485, 332, "baseline_of", "middle", True),
    (485, 372, "\u0394 value, \u0394 time on the edge", "middle", False),
    (420, 418, "instance_of", "middle", False),
]
NODE_FS, EDGE_FS, REGION_FS = 19, 16, 18


# ------------------------------------------------------------------------- SVG
def svg_core(ox, oy):
    o = []
    for x, y, w, h, t in REGIONS:
        o.append(f'<rect x="{ox+x}" y="{oy+y}" width="{w}" height="{h}" rx="14" fill="#FAFAFA" stroke="#{RULE}" stroke-width="1.5" stroke-dasharray="6 5"/>')
        o.append(f'<text x="{ox+x+16}" y="{oy+y+26}" font-size="{REGION_FS}" font-weight="700" fill="#{GREY}">{escape(t)}</text>')
    for pts, both, dashed, strong in ARROWS:
        col = TEAL if strong else INK
        d = " ".join(f"{ox+px},{oy+py}" for px, py in pts)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        ms = f' marker-start="url(#a{"t" if strong else "k"}s)"' if both else ""
        o.append(f'<polyline points="{d}" fill="none" stroke="#{col}" stroke-width="{3 if strong else 2}"{dash}{ms} marker-end="url(#a{"t" if strong else "k"})"/>')
    for x, y, w, h, t, kind, dashed in NODES:
        f, s = FILL[kind]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        o.append(f'<rect x="{ox+x}" y="{oy+y}" width="{w}" height="{h}" rx="9" fill="#{f}" stroke="#{s}" stroke-width="2"{dash}/>')
        o.append(f'<text x="{ox+x+w/2}" y="{oy+y+h/2+7}" font-size="{NODE_FS}" text-anchor="middle" fill="#{INK}">{escape(t)}</text>')
    for x, y, t, anchor, bold in LABELS:
        tw = len(t) * EDGE_FS * 0.54
        bx = x - tw / 2 if anchor == "middle" else x
        o.append(f'<rect x="{ox+bx-3}" y="{oy+y-12}" width="{tw+6}" height="22" fill="#FFFFFF" opacity="0.92"/>')
        o.append(f'<text x="{ox+x}" y="{oy+y+5}" font-size="{EDGE_FS}" text-anchor="{anchor}" fill="#{TEAL if bold else INK}" font-weight="{700 if bold else 400}">{escape(t)}</text>')
    return "\n".join(o)


def build_svg(path):
    W, H, ox, oy = 1280, 760, 40, 105
    marks = ""
    for name, col in (("k", INK), ("t", TEAL)):
        marks += (f'<marker id="a{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                  f'<path d="M0,0 L10,5 L0,10 z" fill="#{col}"/></marker>')
        marks += (f'<marker id="a{name}s" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                  f'<path d="M10,0 L0,5 L10,10 z" fill="#{col}"/></marker>')
    legend_y = oy + 470 + 30
    leg = []
    kinds = [("patient", "Patient / visit"), ("complaint", "Complaint"), ("vital", "Vital sign (numeric)"),
             ("lab", "Lab result / analyte (numeric)"), ("dx", "Prior diagnosis (optional)")]
    x = 40
    for k, t in kinds:
        f, s = FILL[k]
        leg.append(f'<rect x="{x}" y="{legend_y}" width="26" height="18" rx="4" fill="#{f}" stroke="#{s}" stroke-width="2"/>')
        leg.append(f'<text x="{x+34}" y="{legend_y+15}" font-size="16" fill="#{INK}">{escape(t)}</text>')
        x += 34 + len(t) * 9 + 30
    lines = [
        "Arrow directions follow the graph builder (graph.py). has_visit / index_visit_of and has_prior_diagnosis / recurrence_of have inverse pairs;",
        "co_complaint and comorbid_with are symmetric. Forward view in the core study; typed reverse edges (rev:*) are an optional mode, not used for reported results.",
        "Prior visits contribute through visit membership of their results and diagnoses. The index visit's own diagnosis is the target, never a graph input.",
        "Dashed = optional. Teal = numeric payload spanning two endpoints: trajectory_of chains prior results; baseline_of links the last prior result to the index result.",
    ]
    for i, t in enumerate(lines):
        leg.append(f'<text x="40" y="{legend_y+50+i*22}" font-size="14.5" fill="#{GREY}">{escape(t)}</text>')
    leg.append(f'<text x="{W-40}" y="{H-18}" font-size="18" font-weight="700" text-anchor="end" fill="#{INK}">Schematic \u2014 not patient data</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}, Helvetica, sans-serif">'
           f'<defs>{marks}</defs><rect width="{W}" height="{H}" fill="#FFFFFF"/>'
           f'<text x="40" y="46" font-size="30" font-weight="700" fill="#{NAVY}">Shared multi-visit clinical graph</text>'
           f'<text x="40" y="78" font-size="18" fill="#{GREY}">Reusable typed graph representation of one decision-time prediction (MedGNN / CEI-GNN)</text>'
           f'{svg_core(ox, oy)}{"".join(leg)}</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


# ------------------------------------------------------------------------ PPTX
def rgb(h):
    return RGBColor.from_string(h)


def tb(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    """paras: list of (text|[(text, {bold,color})...], size, color, bold)."""
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, (text, size, color, bold) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(6)
        runs = text if isinstance(text, list) else [(text, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = rgb(o.get("color", color))
    return s


def box(slide, x, y, w, h, fill, stroke, text="", size=14, dashed=False, bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE, lw=1.5, color=INK):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    s.line.color.rgb = rgb(stroke)
    s.line.width = Pt(lw)
    if dashed:
        s.line.dash_style = 4  # dash
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    if text:
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = text
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
        r.font.color.rgb = rgb(color)
    return s


def line(slide, p1, p2, color=INK, width=2.0, head_end=True, head_start=False, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(p1[0]), Inches(p1[1]), Inches(p2[0]), Inches(p2[1]))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dashed:
        c.line.dash_style = 4
    ln = c.line._get_or_add_ln()
    for tag, on in (("a:headEnd", head_start), ("a:tailEnd", head_end)):
        if on:
            e = etree.SubElement(ln, qn(tag))
            e.set("type", "triangle")
            e.set("w", "med")
            e.set("len", "med")
    return c


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def title(slide, text, sub=None):
    tb(slide, 0.6, 0.35, 12.1, 0.8, [(text, 32, NAVY, True)])
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.64), Inches(1.15), Inches(0.9), Inches(0.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(TEAL)
    bar.line.fill.background()
    bar.shadow.inherit = False
    if sub:
        tb(slide, 0.6, 1.28, 12.1, 0.75, [(sub, 18, GREY, False)])


def footer(slide, text, y=6.92):
    tb(slide, 0.6, y, 12.1, 0.45, [(text, 11, GREY, False)])


def draw_core(slide, x0, y0, scale):
    X = lambda v: x0 + v * scale
    Y = lambda v: y0 + v * scale
    pt = lambda u: u * scale * 72
    for x, y, w, h, t in REGIONS:
        box(slide, X(x), Y(y), w * scale, h * scale, "FAFAFA", RULE, dashed=True, lw=1.25)
        tb(slide, X(x) + 0.1, Y(y) + 0.03, w * scale - 0.2, 0.35, [(t, pt(REGION_FS), GREY, True)])
    for pts, both, dashed, strong in ARROWS:
        col = TEAL if strong else INK
        w = 2.5 if strong else 1.5
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            last = i == len(pts) - 2
            line(slide, (X(a[0]), Y(a[1])), (X(b[0]), Y(b[1])), col, w,
                 head_end=last, head_start=both and i == 0, dashed=dashed)
    for x, y, w, h, t, kind, dashed in NODES:
        f, s = FILL[kind]
        box(slide, X(x), Y(y), w * scale, h * scale, f, s, t, pt(NODE_FS), dashed=dashed, lw=1.5)
    for x, y, t, anchor, bold in LABELS:
        tw = (len(t) * EDGE_FS * 0.54 + 8) * scale
        bx = x * scale - tw / 2 if anchor == "middle" else x * scale - 0.03
        s = box(slide, X(0) + bx, Y(y) - 11 * scale, tw, 22 * scale, "FFFFFF", "FFFFFF", t, pt(EDGE_FS),
                bold=bold, shape=MSO_SHAPE.RECTANGLE, lw=0, color=TEAL if bold else INK)
        s.line.fill.background()
        s.text_frame.margin_left = s.text_frame.margin_right = 0
        s.text_frame.word_wrap = False
        s.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER if anchor == "middle" else PP_ALIGN.LEFT


def build_pptx(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---- Slide 1: shared clinical graph
    s1 = prs.slides.add_slide(blank)
    title(s1, "A shared clinical graph, not a flat patient record",
          "One typed graph per decision-time visit: patient, visits, complaints, labs and vitals, with numeric "
          "change between earlier and index results kept on the edge. Saved as a reusable project asset.")
    draw_core(s1, 0.66, 2.08, 0.0100)
    footer(s1, "Forward view in the core study; typed reverse edges are optional.  Schematic \u2014 not patient data.  "
               "Asset: docs/presentations/cei-overview/assets/clinical-graph.svg")
    notes(s1, "Contribution 1: the graph representation (comparison/standardized/clinical_graph_v2/graph.py). "
              "Typed nodes: patient, index visit, complaints, vitals, lab measurements/analytes, optional prior diagnoses. "
              "baseline_of / trajectory_of carry signed delta, elapsed hours and rate - a two-endpoint quantity a flat row cannot hold. "
              "Prior visits enter through visit membership of their measurements/diagnoses. The index diagnosis is the target, never an input. "
              "Directions: has_visit/index_visit_of and has_prior_diagnosis/recurrence_of are inverse pairs; co_complaint and comorbid_with are symmetric "
              "(tensorize.py lines 34-44). Forward view is the default (train.py ~213-219, 1046-1050) and is used by the core CEI v3 study (cei_v3_run.py ~210). "
              "Optional bidirectional mode adds rev:* edges for 9 relation types; only a bounded train/dev smoke (E6a) was run, so no reported result uses it. "
              "Caveat: temporal availability of several sources is assumed, not verified (temporal_clean=false); do not call the graph fully leakage-free. "
              "The figure is schematic and contains no patient records. No literature-novelty claim is made.")

    # ---- Slide 2: CEI decomposition
    s2 = prs.slides.add_slide(blank)
    title(s2, "CEI-GNN: a diagnosis score you can decompose",
          "Each class score is an exact sum of signed evidence terms: positive supports the class, negative opposes it.")
    rows = [
        ("Node evidence", "what each node is, its type and its numeric value (labs/vitals via PLE)", "dce6f2", NAVY),
        ("Edge evidence", "both endpoints plus relation payload, e.g. a rise since baseline", "d2eef0", TEAL),
        ("Within-visit pair evidence", "additive pair terms inside one visit (v2 mode, kept in v3)", "fce8cf", "9A5B13"),
        ("Absence evidence", "\u201cNo recorded result at the index visit\u201d", "ebddf1", "6B3A80"),
        ("Class bias", "a per-class constant", "f1f1f1", "777777"),
    ]
    y = 2.2
    for name, desc, f, st in rows:
        box(s2, 0.7, y, 6.3, 0.78, f.upper(), st, "", lw=1.5)
        tb(s2, 0.85, y + 0.04, 6.0, 0.72,
           [([(name, {"bold": True}), ("\n" if False else "  \u2013  " + desc, {"color": GREY})], 15, INK, False)],
           anchor=MSO_ANCHOR.MIDDLE)
        line(s2, (7.0, y + 0.39), (7.55, y + 0.39), INK, 1.5, head_end=False)
        y += 0.9
    line(s2, (7.55, 2.59), (7.55, 6.19), INK, 1.5, head_end=False)
    line(s2, (7.55, 4.39), (8.1, 4.39), TEAL, 3.0)
    box(s2, 8.15, 3.3, 4.55, 2.2, "D2EEF0", TEAL, "", lw=2.25)
    tb(s2, 8.25, 3.4, 4.35, 2.0,
       [("Class score (logit)", 24, NAVY, True),
        ("= bias + node + edge + pair + absence", 17, INK, False),
        ("Scores, not probabilities. One such decomposition per class.", 15, GREY, False)],
       anchor=MSO_ANCHOR.MIDDLE)
    tb(s2, 8.15, 2.1, 4.55, 1.0,
       [("Not a deep message-passing GNN: nodes and edges are scored and aggregated separately, so every term can be read off.", 15, INK, False)])
    tb(s2, 8.15, 5.7, 4.55, 0.9,
       [("Missing results reflect the recording and care process, not a negative finding or causal explanation.", 14, GREY, False)])
    footer(s2, "PLE = piecewise-linear encoding of numeric values, fitted on training data (K = 4 selected). CEI v3 builds on the additive v2 pair mode; "
               "the multiplicative pair interaction was not successful.")
    notes(s2, "Contribution 2: CEI-GNN, a class-specific exact signed evidence decomposition of the diagnosis logit "
              "(methods/cei_gnn.py, cei_gnn_v2.py, cei_gnn_v3.py). Logit(class) = bias + normalised node + edge + within-visit pair + absence contributions. "
              "Nodes carry identity, type and numeric encoding; edges inspect both endpoints plus relation payload and are aggregated separately. "
              "PLE: train-fitted piecewise-linear encoding for lab/vital values, K=4 selected. Absence block means 'no recorded result at the index visit' - "
              "not a negative finding and not proof that a test was not ordered; missingness reflects the recording/care process. "
              "The decomposition is exact on logits, not probabilities. The method name CEI-GNN is used without an invented expansion. "
              "v3 uses the v2 additive pair mode; the multiplicative pair interaction did not succeed.")

    # ---- Slide 3: evidence
    s3 = prs.slides.add_slide(blank)
    title(s3, "What the evidence supports")
    tb(s3, 0.6, 1.4, 6.3, 0.4, [("Primary screen, macro-F1 (3-seed mean)", 16, GREY, True)])
    data = [("CEI-GNN v3 (arm C)", "0.6418"), ("CEI-GNN v2 (arm A)", "0.6420"),
            ("ProtGNN", "0.6371"), ("XGBoost", "0.6371")]
    tbl = s3.shapes.add_table(len(data) + 1, 2, Inches(0.6), Inches(1.95), Inches(6.3), Inches(2.9)).table
    tbl.columns[0].width, tbl.columns[1].width = Inches(4.3), Inches(2.0)
    for r in range(len(data) + 1):
        for c in range(2):
            cell = tbl.cell(r, c)
            txt = (("Method", "Macro-F1") if r == 0 else data[r - 1])[c]
            cell.text = txt
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if c else PP_ALIGN.LEFT
            f = p.runs[0].font
            f.name, f.size = FONT, Pt(20)
            f.bold = r == 0 or r == 1
            f.color.rgb = rgb("FFFFFF" if r == 0 else INK)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if r == 0 else ("D2EEF0" if r == 1 else ("FFFFFF" if r % 2 else "F4F4F4")))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    box(s3, 0.6, 5.05, 6.3, 1.25, "FFFFFF", TEAL, "", lw=2.25)
    tb(s3, 0.75, 5.1, 6.0, 1.15,
       [("No decisive screen gain yet", 24, TEAL, True),
        ("No contrast passes the preregistered rule; v3 over v2: \u22120.0002, 95% CI [\u22120.0047, +0.0042].", 14, INK, False)],
       anchor=MSO_ANCHOR.MIDDLE)
    tb(s3, 7.4, 1.4, 5.4, 4.9,
       [("What we can say", 20, NAVY, True),
        ([("Interpretable by construction. ", {"bold": True}),
          ("Every CEI score splits exactly into signed evidence terms.", {})], 18, INK, False),
        ([("Explanation evidence is mixed. ", {"bold": True}),
          ("Using Grad and IG on 500 screen visits, CEI outperforms ProtGNN on fidelity-minus and sparsity, "
           "not uniformly on fidelity-plus. Native ProtGNN attribution was unavailable in this implementation.", {})], 18, INK, False),
        ([("Superiority is not established. ", {"bold": True}),
          ("The screen is inconclusive; this does not prove equivalence.", {})], 18, INK, False)])
    footer(s3, "Max6 cohort, train-derived Top-10 classes; 10,000 train / 5,000 screen visits; seeds 1234, 2025, 7; patient-disjoint splits; "
               "rounded seed means. Held-out test never accessed.  Source: docs/cei-v3-evidence-2026-10-01.json", y=6.75)
    notes(s3, "Source: docs/cei-v3-delivery-2026-10-01.md and docs/cei-v3-evidence-2026-10-01.json (aggregate only). "
              "Cohort: max6 (patients with more than 6 total visits excluded), train-derived Top-10 classes; 10,000 train, 5,000 dev (selection), "
              "5,000 screen visits, 4,254 validation. Seeds 1234/2025/7, patient-disjoint; key weighted_macro_f1 (patient-equal). Held-out test untouched. "
              "Screen seed means: CEI v3 C 0.6418, v2 A 0.6420, ProtGNN 0.6371, XGBoost 0.6371. No screen contrast passes the preregistered rule; v3 benefit over v2 not demonstrated. "
              "Validation second-look (opened after the inconclusive screen, no multiplicity correction, 3 seeds): C 0.6395 vs ProtGNN 0.6303, delta 0.0092 (95% CI 0.0024-0.0153); "
              "not a broad winner claim, and no XGBoost victory is claimed. A non-significant interval is not evidence of equivalence. "
              "GraphXAI: 500 screen visits / 498 subjects, 3 seeds; comparison using Grad and IG: CEI outperforms ProtGNN on fidelity-minus and sparsity, not uniformly on fidelity-plus; "
              "native ProtGNN attribution unavailable in this implementation. Faithfulness to the model, not clinical validity. "
              "Arm C absence evidence share is about 1.7-1.8%, averaged over visits; arm B without absence is above C on screen, so no benefit of absence evidence is shown.")
    prs.save(path)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    build_svg(out / "assets" / "clinical-graph.svg")
    build_pptx(out / "cei-overview.pptx")
    print("built", out)
