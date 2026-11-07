#!/bin/bash
# Build the report as a web page (site/report/report.html) and as a PDF (results/REPORT.pdf).
# Needs pandoc, wkhtmltopdf, and python packages pypdf + reportlab (for page numbers).
set -e
cd "$(dirname "$0")/.."
B=$(mktemp -d)
mkdir -p site/report/img
cp results/learning_curves.png results/attention_examples.png site/report/img/
# web version
sed -E 's#\]\((learning_curves|attention_examples)\.png\)#](img/\1.png)#' results/REPORT.md > "$B/web.md"
pandoc -f gfm -t html5 -s --metadata pagetitle="NL2Regex technical report" --metadata author-meta="Mohamed Ali Aoun" -c report.css "$B/web.md" -o site/report/report.html
python3 - <<'P'
import re
p = "site/report/report.html"; x = open(p).read()
x = re.sub(r'<header id="title-block-header">.*?</header>', '', x, flags=re.S)
x = x.replace("<body>", "<body>\n<main>").replace("</body>", "</main>\n</body>")
open(p, "w").write(x)
P
# PDF version
sed -E "s#\]\((learning_curves|attention_examples)\.png\)#]($PWD/results/\1.png)#" results/REPORT.md > "$B/pdf.md"
cp tools/report_src/pdf.css "$B/style.css"
pandoc -f gfm -t html5 -s --metadata pagetitle="NL2Regex report" -c style.css "$B/pdf.md" -o "$B/report.html"
python3 - "$B/report.html" <<'P'
import re, sys
p = sys.argv[1]; x = open(p).read()
x = re.sub(r'<header id="title-block-header">.*?</header>', '', x, flags=re.S)
open(p, "w").write(x)
P
wkhtmltopdf --enable-local-file-access -q --page-size A4 -T 16mm -B 20mm -L 15mm -R 15mm "$B/report.html" "$B/raw.pdf" 2>/dev/null || true
python3 - "$B/raw.pdf" <<'P'
import io, sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
r = PdfReader(sys.argv[1]); n = len(r.pages); w = PdfWriter()
for i, pg in enumerate(r.pages):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); c.setFont("Helvetica", 8); c.setFillGray(0.4)
    c.drawCentredString(A4[0] / 2, 28, f"{i + 1} / {n}"); c.save(); buf.seek(0)
    pg.merge_page(PdfReader(buf).pages[0]); w.add_page(pg)
w.add_metadata({"/Title": "From English to regular expressions: an encoder-decoder with attention, trained on a verified dataset",
                "/Author": "Mohamed Ali Aoun"})
w.write("results/REPORT.pdf"); print("REPORT.pdf pages:", n)
P
cp results/REPORT.pdf site/report/REPORT.pdf
rm -rf "$B"
