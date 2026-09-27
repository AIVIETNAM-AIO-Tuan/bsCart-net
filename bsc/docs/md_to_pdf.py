"""Chuyen bao cao Markdown sang .pdf qua Chrome/Edge headless. Khong can pandoc hay LaTeX.

    cd bsc && python docs/md_to_pdf.py docs/bao_cao_14_09.md                   # -> docs/bao_cao_14_09.pdf
    cd bsc && python docs/md_to_pdf.py docs/bao_cao_14_09.md docs/report/x.pdf

Doc .md bang CUNG bo tach voi md_to_docx.py (md_blocks.py), nen hai ban xuat khong the hieu
mot file .md theo hai cach. Trinh duyet: bien moi truong BROWSER neu co, roi Chrome, roi Edge.
Chan trang: ben trai la tieu de cap 1 dau tien, ben phai la so trang (CSS @page, Chrome >= 131).
File .pdf la BAN XUAT, dung sua truc tiep.
"""
import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from md_blocks import blocks, leftover_markup, spans                    # noqa: E402

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]

CSS = """
@page {
  size: A4; margin: 20mm 19mm 18mm 19mm;
  @bottom-left  { content: "%(footer)s"; font: 7.5pt "Segoe UI", Arial, sans-serif; color: #9b9a97; }
  @bottom-right { content: counter(page); font: 7.5pt "Segoe UI", Arial, sans-serif; color: #9b9a97; }
}
:root { --ink: #1f1f1d; --muted: #6b6a66; --line: #e4e2dc; --head: #f5f4f0; --accent: #1f4d66; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 10.3pt/1.55 "Segoe UI", Inter, Arial, sans-serif; color: var(--ink); margin: 0; }
h1 { font-size: 25pt; line-height: 1.2; margin: 0 0 8pt; font-weight: 700; }
h2 { font-size: 14.5pt; margin: 20pt 0 6pt; font-weight: 700; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; font-weight: 700; break-after: avoid; }
p { margin: 0 0 7pt; }
p:has(+ ul), p:has(+ ol), p:has(+ table) { break-after: avoid; }   /* dong dan khong bi bo lai cuoi trang */
p.subtitle { font-style: italic; color: var(--ink); margin-bottom: 12pt; }
p.meta { margin: 0 0 3pt; }
code { font: 0.9em Consolas, "Cascadia Mono", monospace; color: #b3423a;
       background: #f3f2ee; padding: 0.5pt 3pt; border-radius: 3pt; }
ul, ol { margin: 0 0 8pt; padding-left: 20pt; }
li { margin: 0 0 3pt; }
li > ul, li > ol { margin: 3pt 0 0; }
table { border-collapse: collapse; margin: 6pt 0 5pt; font-size: 9pt; line-height: 1.4;
        width: auto; max-width: 100%%; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 0.75pt solid var(--line); padding: 4pt 7pt; text-align: left; vertical-align: top; }
th { background: var(--head); font-weight: 600; }
figure { margin: 10pt 0 8pt; break-inside: avoid; }
figure img { display: block; max-width: 100%%; margin: 0 auto 5pt; }
.caption { font-size: 9pt; line-height: 1.5; color: var(--muted); margin: 0 0 11pt; }
.caption b { color: #4a4945; }
blockquote { margin: 6pt 0 10pt; padding: 6pt 10pt; background: #f2f5f7;
             border-left: 3pt solid var(--accent); break-inside: avoid; }
hr { border: 0; border-top: 0.75pt solid var(--line); margin: 14pt 0; }
.missing { color: #a93226; font-weight: 600; }
"""


def find_browser():
    for cand in ([os.environ["BROWSER"]] if os.environ.get("BROWSER") else []) + BROWSERS:
        path = cand if Path(cand).is_file() else shutil.which(cand)
        if path:
            return path
    sys.exit("khong tim thay Chrome/Edge - dat bien moi truong BROWSER tro toi file chay")


class Renderer:
    def __init__(self, md_path: Path):
        self.md_path = md_path
        self.leftover = []

    def inline(self, text):
        sp = spans(text)
        if leftover_markup(sp):
            self.leftover.append(text[:70])
        out = []
        for txt, b, it, code in sp:
            s = html.escape(txt)
            if code:
                s = f"<code>{s}</code>"
            if it:
                s = f"<i>{s}</i>"
            if b:
                s = f"<b>{s}</b>"
            out.append(s)
        return "".join(out)

    def image(self, rel):
        path = (self.md_path.parent / rel).resolve()
        if not path.exists():
            return f'<p class="missing">[THIEU HINH: {html.escape(path.name)}]</p>'
        return f'<img src="{path.as_uri()}">'

    def table(self, rows):
        head, body = rows[0], rows[1:]
        th = "".join(f"<th>{self.inline(c)}</th>" for c in head)
        trs = "".join("<tr>" + "".join(f"<td>{self.inline(c)}</td>" for c in r) + "</tr>"
                      for r in body)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"

    def lists(self, items):
        """items: [(danh_so, muc_lui, van ban)] lien tiep -> ul/ol long nhau theo muc lui."""
        out, stack = [], []                 # stack: [(tag, level)]
        for numbered, level, txt in items:
            tag = "ol" if numbered else "ul"
            while stack and stack[-1][1] > level:
                out.append(f"</li></{stack.pop()[0]}>")
            if stack and stack[-1][1] == level and stack[-1][0] != tag:
                out.append(f"</li></{stack.pop()[0]}>")
            if not stack or stack[-1][1] < level:
                out.append(f"<{tag}>")
                stack.append((tag, level))
            else:
                out.append("</li>")
            out.append(f"<li>{self.inline(txt)}")
        while stack:
            out.append(f"</li></{stack.pop()[0]}>")
        return "".join(out)

    def body(self, blks):
        out, i, n = [], 0, len(blks)
        prev, header = None, True           # header: vung truoc tieu de cap 2 / duong ke dau tien
        while i < n:
            kind, data = blks[i]
            if kind == "list":
                j = i
                while j < n and blks[j][0] == "list":
                    j += 1
                out.append(self.lists([d for _, d in blks[i:j]]))
                prev, i = "list", j
                continue
            if kind == "image":
                # anh + chu thich ngay sau no = mot khoi khong bi ngat trang
                cap = ""
                if i + 1 < n and blks[i + 1][0] == "caption":
                    cap = f'<div class="caption">{self.inline(blks[i + 1][1])}</div>'
                    i += 1
                out.append(f"<figure>{self.image(data)}{cap}</figure>")
            elif kind == "heading":
                lv, txt = data
                header = header and lv == 1
                out.append(f"<h{lv}>{self.inline(txt)}</h{lv}>")
            elif kind == "caption":
                cls = "subtitle" if prev == "h1" else "caption"
                tag = "p" if cls == "subtitle" else "div"
                out.append(f'<{tag} class="{cls}">{self.inline(data)}</{tag}>')
            elif kind == "table":
                out.append(self.table(data))
            elif kind == "quote":
                out.append(f"<blockquote>{self.inline(data)}</blockquote>")
            elif kind == "hr":
                header = False
                out.append("<hr>")
            else:                                           # para
                # dong thong tin dau bao cao (**Nhan:** gia tri) -> xep sat nhau
                cls = ' class="meta"' if header and data.startswith("**") else ""
                out.append(f"<p{cls}>{self.inline(data)}</p>")
            prev = f"h{data[0]}" if kind == "heading" else kind
            i += 1
        return "\n".join(out)


def convert(md_path: Path, pdf_path: Path):
    lines = md_path.read_text(encoding="utf-8").split("\n")
    blks = blocks(lines)
    title = next((t for k, d in blks if k == "heading" and d[0] == 1 for t in [d[1]]),
                 md_path.stem)
    r = Renderer(md_path)
    body = r.body(blks)
    footer = title.replace("\\", "\\\\").replace('"', '\\"')
    doc = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
           f"<title>{html.escape(title)}</title><style>{CSS % {'footer': footer}}</style>"
           f"</head><body>{body}</body></html>")

    if r.leftover:
        print(f"CANH BAO: {len(r.leftover)} doan con dau markdown chua duoc xu ly:")
        for x in r.leftover[:8]:
            print("   ", repr(x))
    else:
        print("khong con dau markdown sot lai")

    pdf_path = pdf_path.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()                   # de kiem duoc la lan chay NAY da ghi file
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(doc, encoding="utf-8")
        cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--allow-file-access-from-files",
               f"--user-data-dir={Path(tmp) / 'profile'}",   # tach khoi phien trinh duyet dang mo
               "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", page.as_uri()]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        sys.exit(f"trinh duyet khong tao duoc PDF (ma {res.returncode}):\n{res.stderr[-2000:]}")
    return pdf_path


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")      # console cp1252 khong in duoc tieng Viet
    here = Path(__file__).parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "bao_cao_14_09.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    convert(src, out)
    print(f"da tao {out}  ({out.stat().st_size / 1024:.0f} KB)")
