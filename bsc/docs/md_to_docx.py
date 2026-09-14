"""Chuyen bao cao Markdown sang .docx. Khong can pandoc, chi can python-docx.

    cd bsc && python docs/md_to_docx.py                       # mac dinh: bao_cao_14_09.md
    cd bsc && python docs/md_to_docx.py docs/khac.md          # hoac chi dinh file

Quy trinh lam viec: sua `report_data.py` -> chay `make_figs.py` -> sua van ban trong file .md
-> chay file nay. File .docx la BAN XUAT, dung sua truc tiep neu con muon sinh lai.

Ho tro dung nhung gi bao cao dang dung: tieu de 1-3, doan van, danh sach gach dau dong va
danh sach danh so, bang co duong ke, anh kem chu thich in nghieng, trich dan (`>`), duong ke
ngang, va dinh dang trong dong **dam** / *nghieng* / `ma`.
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

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
CODE_RE = re.compile(r"`([^`]+?)`", re.S)
#: `(?!\*)` o dau dong la BAT BUOC. Voi `**dam chua *nghieng***`, cum dong la ba dau sao
#: lien nhau: mot dau dong cua nghieng roi hai dau dong cua dam. Khong co chan nay thi regex
#: an hai dau SAM NHAT, cat mat dau dong cua nghieng va de lai mot dau sao thua.
BOLD_RE = re.compile(r"\*\*(.+?)\*\*(?!\*)", re.S)
ITAL_RE = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)", re.S)
_PATS = (("code", CODE_RE), ("bold", BOLD_RE), ("ital", ITAL_RE))


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
    """Tach **dam** / *nghieng* / `ma`, DE QUY nen long nhau van dung.

    Ban truoc dung mot regex tach phang nen `**dam chua *nghieng* ben trong**` bi vo:
    `.+?\\*\\*` an mat mot dau sao cua cum `***` o cuoi, de lai dau sao thua trong file .docx.
    O day lay cum khop SOM NHAT trong ba mau roi de quy vao ruot no.
    """
    while text:
        found = [(m.start(), i, kind, m)
                 for i, (kind, pat) in enumerate(_PATS) if (m := pat.search(text))]
        if not found:
            _run(par, text, size, italic, bold, color)
            return
        pos, _, kind, m = min(found, key=lambda t: (t[0], t[1]))
        _run(par, text[:pos], size, italic, bold, color)
        if kind == "code":
            _run(par, m.group(1), size, italic, bold, color, code=True)
        elif kind == "bold":
            add_runs(par, m.group(1), size=size, italic=italic, bold=True, color=color)
        else:
            add_runs(par, m.group(1), size=size, italic=True, bold=bold, color=color)
        text = text[m.end():]


# ------------------------------------------------------------------ khoi
def add_table(doc, rows):
    head, body = rows[0], rows[2:]          # rows[1] la dong `---|---`
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

    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()

        if not s:
            i += 1
            continue

        # --- duong ke ngang -> ngat trang mem
        if re.fullmatch(r"-{3,}", s):
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
            i += 1
            continue

        # --- bang: gom cac dong lien tiep bat dau bang |
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            if len(rows) >= 2:
                add_table(doc, rows)
            continue

        # --- anh
        m = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if m:
            add_image(doc, (md_path.parent / m.group(1)).resolve())
            i += 1
            continue

        # --- tieu de
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            lv, txt = len(m.group(1)), m.group(2)
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt({1: 0, 2: 16, 3: 11}[lv])
            p.paragraph_format.space_after = Pt({1: 6, 2: 5, 3: 4}[lv])
            p.paragraph_format.keep_with_next = True
            add_runs(p, f"**{txt}**", size={1: 20, 2: 14, 3: 11.5}[lv],
                     color=INK if lv == 1 else ACCENT)
            i += 1
            continue

        # --- trich dan (co the nhieu dong)
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.16)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(8)
            _left_bar(p)
            _shade(p, QUOTE_FILL)
            add_runs(p, " ".join(buf), size=10)
            continue

        # --- chu thich hinh/bang: ca khoi nam trong *...*, CO THE NHIEU DONG
        # Ban truoc doi dong DAU vua mo vua dong bang dau sao, nen chu thich dai nhieu dong
        # roi thang xuong nhanh doan van thuong va de lo dau sao trong file .docx.
        if s.startswith("*") and not s.startswith("**"):
            buf, closed = [], False
            while i < n and lines[i].strip():
                cur = lines[i].strip()
                buf.append(cur)
                i += 1
                if cur.endswith("*") and not cur.endswith("**"):
                    closed = True
                    break
            txt = " ".join(buf)[1:]                 # bo dau sao MO
            if closed:
                txt = txt[:-1]                      # bo dau sao DONG
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(11)
            add_runs(p, txt, size=9, italic=True, color=MUTED)
            continue

        # --- danh sach gach dau dong / danh so
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", raw)
        if m:
            indent, marker, txt = m.group(1), m.group(2), m.group(3)
            while i + 1 < n and lines[i + 1].strip() and not re.match(
                    r"^(\s*)([-*]|\d+\.)\s|^[#>|]|^!\[", lines[i + 1]) and lines[i + 1].startswith(" "):
                i += 1
                txt += " " + lines[i].strip()
            style = "List Number" if marker[0].isdigit() else "List Bullet"
            p = doc.add_paragraph(style=style)
            p.paragraph_format.left_indent = Inches(0.28 + 0.22 * (len(indent) // 2))
            p.paragraph_format.space_after = Pt(4)
            add_runs(p, txt)
            i += 1
            continue

        # --- doan van thuong: gom cac dong tiep theo cho den dong trong
        buf = [s]
        while i + 1 < n and lines[i + 1].strip() and not re.match(
                r"^([#>|]|!\[|\s*([-*]|\d+\.)\s|-{3,})", lines[i + 1].strip()):
            i += 1
            buf.append(lines[i].strip())
        add_runs(doc.add_paragraph(), " ".join(buf))
        i += 1

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
