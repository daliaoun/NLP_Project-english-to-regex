"""Render the slide viewer in print mode to a PDF (one 1920x1080 page per slide).
    python tools/slides_pdf.py http://localhost:8765/slides/index.html site/slides/NL2Regex-slides.pdf
"""
import sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1:3]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(url + "?print=1", wait_until="networkidle")
    pg.evaluate("document.fonts.ready")
    pg.pdf(path=out, width="1920px", height="1080px", print_background=True,
           margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
    b.close()
