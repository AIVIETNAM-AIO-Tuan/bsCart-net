"""Chuyen bao cao Markdown sang .docx. Khong can pandoc, chi can python-docx.

    cd bsc && python docs/md_to_docx.py                       # mac dinh: bao_cao_14_09.md
    cd bsc && python docs/md_to_docx.py docs/khac.md          # hoac chi dinh file

Quy trinh lam viec: sua `report_data.py` -> chay `make_figs.py` -> sua van ban trong file .md
-> chay file nay. File .docx la BAN XUAT, dung sua truc tiep neu con muon sinh lai.

Ho tro dung nhung gi bao cao dang dung: tieu de 1-3, doan van, danh sach gach dau dong va
danh sach danh so, bang co duong ke, anh kem chu thich in nghieng, trich dan (`>`), duong ke
ngang, va dinh dang trong dong **dam** / *nghieng* / `ma`.
"""
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent))
from md_blocks import blocks, spans                                     # noqa: E402

BODY_FONT, MONO_FONT = "Calibri", "Consolas"
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5E, 0x6B, 0x78)
ACCENT = RGBColor(0x1F, 0x4D, 0x66)
HDR_FILL = "DCE6EC"          # nen hang tieu de cua bang
QUOTE_FILL = "F2F5F7"        # nen khoi trich dan


# ------------------------------------------------------------------ tien ich XML
def _shade(cell_or_par, hexfill):
    """To nen cho mot o bang hoac mot doan van (python-docx khong co API san)."""
    el = cell_or_par._tc if hasattr(cell_or_par, "_tc") else cell_or_par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexfill)
    (el.get_or_add_tcPr() if hasattr(cell_or_par, "_tc") else el).append(shd)


def _left_bar(par, hexcolor="1F4D66"):
    """Vach dung ben trai - dung cho khoi trich dan."""
    pPr = par._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:space"), "8")
    left.set(qn("w:color"), hexcolor)
    borders.append(left)
    pPr.append(borders)


# ------------------------------------------------------------------ dinh dang trong dong
def _run(par, txt, size, italic, bold, color, code=False):
    if not txt:
        return
    r = par.add_run(txt)
    r.bold, r.italic = bold, italic
    r.font.name = MONO_FONT if code else BODY_FONT
    if size:
        r.font.size = Pt(size)
    if code:
        r.font.size = Pt((size or 10.5) - 1)
        r.font.color.rgb = ACCENT
    elif color:
        r.font.color.rgb = color


def add_runs(par, text, *, size=None, italic=False, bold=False, color=None):
    """Them van ban co **dam** / *nghieng* / `ma` (long nhau van dung) - xem md_blocks.spans."""
    for txt, b, it, code in spans(text, bold, italic):
        _run(par, txt, size, it, b, color, code=code)


# ------------------------------------------------------------------ khoi
def add_table(doc, rows):
    head, body = rows[0], rows[1:]          # md_blocks da bo dong `---|---`
    t = doc.add_table(rows=1, cols=len(head))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = True
    for c, txt in zip(t.rows[0].cells, head):
        c.paragraphs[0].paragraph_format.space_before = Pt(3)
        c.paragraphs[0].paragraph_format.space_after = Pt(3)
        add_runs(c.paragraphs[0], f"**{txt}**" if txt else "", size=9.5)
        _shade(c, HDR_FILL)
    for row in body:
        cells = t.add_row().cells
        for c, txt in zip(cells, row):
            p = c.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, txt, size=9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_image(doc, path, max_in=6.2):
    if not path.exists():
        p = doc.add_paragraph()
        add_runs(p, f"[THIEU HINH: {path.name}]", size=10, color=RGBColor(0xA9, 0x32, 0x26))
        return
    doc.add_picture(str(path), width=Inches(max_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_before = Pt(6)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(2)


def convert(md_path: Path, docx_path: Path):
    lines = md_path.read_text(encoding="utf-8").split("\n")
    doc = Document()

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)      # A4
    for side in ("left", "right"):
        setattr(sec, f"{side}_margin", Inches(1.0))
    sec.top_margin = sec.bottom_margin = Inches(0.9)

    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(10.5)
    st.font.color.rgb = INK
    st.paragraph_format.space_after = Pt(7)
    st.paragraph_format.line_spacing = 1.12

    for kind, data in blocks(lines):
        if kind == "hr":                    # duong ke ngang
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            pPr = p._p.get_or_add_pPr()
            bd = OxmlElement("w:pBdr")
            bt = OxmlElement("w:bottom")
            bt.set(qn("w:val"), "single")
            bt.set(qn("w:sz"), "6")
            bt.set(qn("w:space"), "1")
            bt.set(qn("w:color"), "C3C2B7")
            bd.append(bt)
            pPr.append(bd)
        elif kind == "table":
            add_table(doc, data)
        elif kind == "image":
            add_image(doc, (md_path.parent / data).resolve())
        elif kind == "heading":
            lv, txt = data
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt({1: 0, 2: 16, 3: 11}[lv])
            p.paragraph_format.space_after = Pt({1: 6, 2: 5, 3: 4}[lv])
            p.paragraph_format.keep_with_next = True
            add_runs(p, f"**{txt}**", size={1: 20, 2: 14, 3: 11.5}[lv],
                     color=INK if lv == 1 else ACCENT)
        elif kind == "quote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.16)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(8)
            _left_bar(p)
            _shade(p, QUOTE_FILL)
            add_runs(p, data, size=10)
        elif kind == "caption":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(11)
            add_runs(p, data, size=9, italic=True, color=MUTED)
        elif kind == "list":
            numbered, level, txt = data
            p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            p.paragraph_format.left_indent = Inches(0.28 + 0.22 * level)
            p.paragraph_format.space_after = Pt(4)
            add_runs(p, txt)
        else:                               # "para"
            add_runs(doc.add_paragraph(), data)

    # chan trang
    ftr = sec.footer.paragraphs[0]
    ftr.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_runs(ftr, f"{md_path.stem}  ·  sinh tu {md_path.name}", size=8, color=MUTED)

    # Canh gac: dau sao con sot NGOAI cac run ma nghia la mot khoi markdown khong duoc nhan
    # dien. Bo qua run ma, vi ten cot kieu `fcl_*_pct` co dau sao la noi dung that.
    def _leftover(par):
        bad = "".join(r.text for r in par.runs if r.font.name != MONO_FONT)
        return par.text[:70] if "*" in bad or "`" in bad else None

    leftover = [t for t in (_leftover(p) for p in doc.paragraphs) if t]
    leftover += [t for t in (_leftover(p) for tb in doc.tables for row in tb.rows
                             for c in row.cells for p in c.paragraphs) if t]
    if leftover:
        print(f"CANH BAO: {len(leftover)} doan con dau markdown chua duoc xu ly:")
        for x in leftover[:8]:
            print("   ", repr(x))
    else:
        print("khong con dau markdown sot lai")

    doc.save(docx_path)
    return docx_path


if __name__ == "__main__":
    here = Path(__file__).parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "bao_cao_14_09.md"
    out = src.with_suffix(".docx")
    convert(src, out)
    print(f"da tao {out}  ({out.stat().st_size / 1024:.0f} KB)")
