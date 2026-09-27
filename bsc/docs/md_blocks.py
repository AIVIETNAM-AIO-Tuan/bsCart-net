"""Bo tach Markdown DUNG CHUNG cho md_to_docx.py va md_to_pdf.py.

Chi ho tro dung nhung gi bao cao dang dung (xem docstring cua md_to_docx.py). Hai bo xuat chi
khac nhau o khau VE; cach nhan dien khoi va dinh dang trong dong nam o day, mot noi duy nhat,
de .docx va .pdf khong bao gio doc cung mot file .md thanh hai cau truc khac nhau.

    blocks(lines) -> [(kind, data), ...]
        "hr"      None
        "table"   [[o, ...], ...]          hang dau la tieu de; dong `---|---` da bo
        "image"   duong dan tuong doi, nhu ghi trong .md
        "heading" (cap 1-3, van ban)
        "quote"   van ban (cac dong `>` noi lai)
        "caption" van ban (khoi *...* nhieu dong, da bo dau sao mo/dong)
        "list"    (danh_so: bool, muc_lui: int, van ban)
        "para"    van ban

    spans(text) -> [(van ban, dam, nghieng, ma), ...]
"""
import re

CODE_RE = re.compile(r"`([^`]+?)`", re.S)
#: `(?!\*)` o dau dong la BAT BUOC. Voi `**dam chua *nghieng***`, cum dong la ba dau sao
#: lien nhau: mot dau dong cua nghieng roi hai dau dong cua dam. Khong co chan nay thi regex
#: an hai dau SAM NHAT, cat mat dau dong cua nghieng va de lai mot dau sao thua.
BOLD_RE = re.compile(r"\*\*(.+?)\*\*(?!\*)", re.S)
ITAL_RE = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)", re.S)
_PATS = (("code", CODE_RE), ("bold", BOLD_RE), ("ital", ITAL_RE))


def spans(text, bold=False, italic=False):
    """Tach **dam** / *nghieng* / `ma`, DE QUY nen long nhau van dung.

    Ban truoc dung mot regex tach phang nen `**dam chua *nghieng* ben trong**` bi vo:
    `.+?\\*\\*` an mat mot dau sao cua cum `***` o cuoi, de lai dau sao thua.
    O day lay cum khop SOM NHAT trong ba mau roi de quy vao ruot no. Doan rong bi bo.
    """
    out = []
    while text:
        found = [(m.start(), i, kind, m)
                 for i, (kind, pat) in enumerate(_PATS) if (m := pat.search(text))]
        if not found:
            out.append((text, bold, italic, False))
            break
        pos, _, kind, m = min(found, key=lambda t: (t[0], t[1]))
        if pos:
            out.append((text[:pos], bold, italic, False))
        if kind == "code":
            out.append((m.group(1), bold, italic, True))
        elif kind == "bold":
            out += spans(m.group(1), True, italic)
        else:
            out += spans(m.group(1), bold, True)
        text = text[m.end():]
    return out


def leftover_markup(span_list):
    """Dau sao / backtick con sot NGOAI doan ma nghia la mot khoi markdown khong duoc nhan
    dien. Bo qua doan ma, vi ten cot kieu `fcl_*_pct` co dau sao la noi dung that."""
    bad = "".join(t for t, _, _, code in span_list if not code)
    return "*" in bad or "`" in bad


def blocks(lines):
    """Tach danh sach dong thanh cac khoi. Thu tu kiem tra la mot phan cua dinh nghia."""
    out = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()

        if not s:
            i += 1
            continue

        # --- duong ke ngang
        if re.fullmatch(r"-{3,}", s):
            out.append(("hr", None))
            i += 1
            continue

        # --- bang: gom cac dong lien tiep bat dau bang |
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) >= 2:
                out.append(("table", [rows[0]] + rows[2:]))
            continue

        # --- anh
        m = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if m:
            out.append(("image", m.group(1)))
            i += 1
            continue

        # --- tieu de
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            out.append(("heading", (len(m.group(1)), m.group(2))))
            i += 1
            continue

        # --- trich dan (co the nhieu dong)
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(("quote", " ".join(buf)))
            continue

        # --- chu thich hinh/bang: ca khoi nam trong *...*, CO THE NHIEU DONG
        # Ban truoc doi dong DAU vua mo vua dong bang dau sao, nen chu thich dai nhieu dong
        # roi thang xuong nhanh doan van thuong va de lo dau sao o ban xuat.
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
            out.append(("caption", txt))
            continue

        # --- danh sach gach dau dong / danh so
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", raw)
        if m:
            indent, marker, txt = m.group(1), m.group(2), m.group(3)
            while i + 1 < n and lines[i + 1].strip() and not re.match(
                    r"^(\s*)([-*]|\d+\.)\s|^[#>|]|^!\[", lines[i + 1]) and lines[i + 1].startswith(" "):
                i += 1
                txt += " " + lines[i].strip()
            out.append(("list", (marker[0].isdigit(), len(indent) // 2, txt)))
            i += 1
            continue

        # --- doan van thuong: gom cac dong tiep theo cho den dong trong
        buf = [s]
        while i + 1 < n and lines[i + 1].strip() and not re.match(
                r"^([#>|]|!\[|\s*([-*]|\d+\.)\s|-{3,})", lines[i + 1].strip()):
            i += 1
            buf.append(lines[i].strip())
        out.append(("para", " ".join(buf)))
        i += 1
    return out
