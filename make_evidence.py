"""Generate evidence declaration and test-log PDFs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
EV = ROOT / "evidence"
EV.mkdir(exist_ok=True)

VERIFICATION = "AI-A2-7483"
NAME = "KHALIL ABDEEN HAMID MUSA"
REG = "25/27483"


def base_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "INES Ruhengeri - SWE 3513 - Individual Assignment 2", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    return pdf


# Declaration
pdf = base_pdf()
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Student Declaration", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Helvetica", "", 11)
for line in [
    f"Student: {NAME}",
    f"Registration number: {REG}",
    f"Institutional username: ug2527483@ines.ac.rw",
    f"Verification code: {VERIFICATION}",
    "",
    "I hereby declare that this submission (AI_A2_2527483) is my individual work.",
    "All sources and AI assistance used during development are disclosed in",
    "evidence/AI_USE.md. I understand the entire system, can explain and modify",
    "every part of the code, and I verified all datasets, metrics, plots and",
    "API behaviour reported in the technical report. The dataset is synthetic",
    "and no real farm or personal data is included. This prototype is a",
    "classroom decision-support tool only and must not replace professional",
    "agronomic judgement or determine access to finance, insurance or public",
    "services.",
    "",
    "Signature: ______________________________",
    "Date: ______________________________",
]:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, line)
pdf.output(str(EV / "AI_A2_2527483_DECLARATION.pdf"))

# Test log
pdf = base_pdf()
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Test Log - pytest -q", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Helvetica", "", 10)
pytest_out = (EV / "pytest_output.txt").read_text(encoding="utf-8-sig").strip()
pdf.set_x(pdf.l_margin)
pdf.multi_cell(0, 5.5, pytest_out)
pdf.ln(3)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 7, "Manual verification (19 October 2026)", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
for line in [
    "- run_all.py --student-id ug2527483@ines.ac.rw --seed 7483: SUCCESS",
    "  SHA-256 fingerprint b9a0067604ae61a7add8d7e174a95e588645899008b97feac17f468bb1b8aff2",
    "  reproduced identically on a second full run.",
    "- predict.py --json samples/valid.json: SUCCESS",
    "  -> yield_tons 11.2163, high_risk 0, probability 0.12, cluster 0, code AI-A2-7483",
    "- predict.py --json samples/invalid.json: rejected with exit code 1 and field-level errors.",
    "- FastAPI: GET /health 200, GET /model-info 200, POST /predict valid 200,",
    "  POST /predict out-of-range rainfall 422 (structured Pydantic error).",
]:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5.5, line)
pdf.output(str(EV / "AI_A2_2527483_TEST_LOG.pdf"))
print("Declaration and test log written.")
