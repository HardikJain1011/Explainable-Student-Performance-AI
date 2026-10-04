"""Generate a compact PDF student performance report."""
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

from .data import FEATURE_LABELS


def build_report(student_name, inputs, prediction, confidence, probabilities, positive, negative, suggestions, counterfactuals=None):
    """Return PDF bytes for a single student assessment."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=22, leading=26,
        textColor=colors.HexColor("#111827"), alignment=TA_CENTER, spaceAfter=8
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9, leading=13,
        textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=18
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, leading=16,
        textColor=colors.HexColor("#111827"), spaceBefore=12, spaceAfter=7
    )
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=14)
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.HexColor("#64748b"))

    story = [
        Paragraph("Student Performance AI", title),
        Paragraph(f"Performance report for <b>{student_name}</b>", subtitle),
        Paragraph(
            f"Explainable performance assessment | Generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
            subtitle,
        ),
    ]

    result_table = Table([
        ["Predicted performance", "Model probability"],
        [prediction, f"{confidence:.0%}"],
    ], colWidths=[250, 250])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#d1d5db")),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [result_table, Spacer(1, 8)]

    story.append(Paragraph("Prediction probabilities", h2))
    prob_rows = [["Class", "Probability"]] + [[k, f"{v:.1%}"] for k, v in probabilities.items()]
    prob_table = Table(prob_rows, colWidths=[250, 250])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [prob_table]

    story.append(Paragraph("Student profile", h2))
    profile_rows = [["Factor", "Value"]]
    for feature, value in inputs.items():
        profile_rows.append([FEATURE_LABELS.get(feature, feature), f"{value:g}"])
    profile_table = Table(profile_rows, colWidths=[350, 150], repeatRows=1)
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(profile_table)

    def factor_rows(items):
        rows = [["Factor", "Value", "Model impact"]]
        for r in items:
            rows.append([FEATURE_LABELS.get(r.feature, r.feature), f"{r.value:g}", f"{r.shap:+.3f}"])
        return rows

    story.append(Paragraph("What influenced the prediction", h2))
    if positive:
        story.append(Paragraph("Positive contributors", body))
        t = Table(factor_rows(positive), colWidths=[270, 90, 140], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcfce7")),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    if negative:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Negative contributors", body))
        t = Table(factor_rows(negative), colWidths=[270, 90, 140], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d1d5db")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

    story.append(Paragraph("Personalized improvement plan", h2))
    if suggestions:
        for _, title_text, target, description in suggestions:
            story.append(KeepTogether([
                Paragraph(f"<b>{title_text}</b> — {target}", body),
                Paragraph(description, small),
                Spacer(1, 5),
            ]))
    else:
        story.append(Paragraph("No strong actionable negative contributor was identified for this profile.", body))

    if counterfactuals:
        story.append(Paragraph("Model-based what-if options", h2))
        for i, item in enumerate(counterfactuals[:3], 1):
            changes = ", ".join(item["changes"])
            story.append(Paragraph(
                f"Option {i}: change {changes}; simulated High probability {item['confidence']:.0%}.", body
            ))

    story += [
        Spacer(1, 12),
        Paragraph(
            "Important: This report describes model predictions and what-if simulations. It is not a diagnosis, "
            "a causal claim, or a substitute for academic advising. The portfolio version uses a synthetic dataset.",
            small,
        ),
    ]
    doc.build(story)
    return buffer.getvalue()
