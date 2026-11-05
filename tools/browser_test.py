"""End-to-end browser test of the site. Usage: python tools/browser_test.py http://localhost:8765 <screenshot_dir>"""
import sys, json
from playwright.sync_api import sync_playwright
base, shots = sys.argv[1], sys.argv[2]
errors, fails = [], []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1280, "height": 900}, permissions=["clipboard-read", "clipboard-write"])
    pg = ctx.new_page()
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("requestfailed", lambda r: errors.append("requestfailed " + r.url))
    pg.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)
    pg.goto(base + "/", wait_until="networkidle")
    pg.wait_for_function("!document.getElementById('go').disabled", timeout=20000)
    check("Model ready" in pg.inner_text("#status-text"), "model loads: " + pg.inner_text("#status-text")[:60])
    pg.screenshot(path=f"{shots}/01_home.png", full_page=True)

    # chip -> result
    pg.click("#chips-good .chip:nth-child(1)")
    check(pg.inner_text("#rx-attn").strip() == "(.*cat){2,}.*", "chip 1 regex = " + pg.inner_text("#rx-attn"))
    rows = pg.locator("#tester-out .t-row")
    texts = [rows.nth(i).inner_text().replace("\n", " ") for i in range(rows.count())]
    print("   tester:", texts)
    check(any(t.startswith("cat and cat") and "MATCH" in t and "no match" not in t for t in texts), "'cat and cat' matches")
    check(any(t.startswith("cat no match") or t == "cat no match" for t in texts), "'cat' does not match")
    pg.click("#heat-box summary")
    check(pg.locator("#heat td").count() > 10, "heatmap cells: %d" % pg.locator("#heat td").count())
    pg.screenshot(path=f"{shots}/02_result.png", full_page=True)

    # compare baseline
    pg.check("#compare", force=True) if False else pg.click(".switch")
    pg.wait_for_function("!document.getElementById('rx-base').textContent.includes('loading')", timeout=20000)
    print("   baseline:", pg.inner_text("#rx-base"), "|", pg.inner_text("#cmp-verdict"))
    check(pg.inner_text("#rx-base").strip() != "", "baseline output shown")
    pg.screenshot(path=f"{shots}/03_compare.png", full_page=True)

    # free text with unknown word + dialect note
    pg.fill("#q", "lines that kick off with 'Hello'")
    pg.press("#q", "Enter")
    check(pg.locator("#unk-note").is_visible(), "unknown-word note visible")
    check(pg.locator(".tok.unk").count() >= 1, "unknown token highlighted")
    pg.fill("#q", "lines that contain 'error' and do not contain 'warning'")
    pg.click("#go")
    check(pg.locator("#dialect-note").is_visible(), "dialect note for ~: " + pg.inner_text("#rx-attn"))

    # copy
    pg.click("[data-copy=rx-attn]")
    clip = pg.evaluate("navigator.clipboard.readText()")
    check(clip == pg.inner_text("#rx-attn"), "copy button")

    # invalid input / empty
    pg.fill("#q", "")
    pg.click("#go")  # should just focus
    # random example
    pg.click("#rand")
    check(pg.locator("#rand-out").is_visible() and pg.locator("#rand-out code").count() == 3, "random example shown")
    check("New phrasings" in pg.inner_text("#stats"), "stats shown: " + pg.inner_text("#stats").replace("\n", " | ")[:300])
    pg.select_option("#rand-split", "h"); pg.click("#rand")
    check("Handwritten" in pg.inner_text("#rand-out"), "random handwritten example")
    pg.screenshot(path=f"{shots}/04_test.png", full_page=True)

    # tabs
    for tab in ["report", "slides", "downloads"]:
        pg.click(f"#t-{tab}")
        check(pg.evaluate(f"location.hash") == "#" + tab, f"hash #{tab}")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{shots}/05_{tab}.png", full_page=False)
    # report iframe content
    fr = pg.frame_locator("#report-frame")
    check(fr.locator("h1").count() == 1, "report h1: " + fr.locator("h1").inner_text()[:50])
    check(fr.locator("img").count() == 2, "report images")
    imgs_ok = pg.evaluate("[...document.getElementById('report-frame').contentDocument.images].every(i=>i.naturalWidth>0)")
    check(imgs_ok, "report images loaded")
    # slides iframe
    pg.click("#t-slides"); pg.wait_for_timeout(500)
    sf = pg.frame_locator("#slides-frame")
    check("1 / 19" in sf.locator("#counter").inner_text(), "slides counter")
    sf.locator("#next").click(); sf.locator("#next").click()
    check("3 / 19" in sf.locator("#counter").inner_text(), "slides next")
    sf.locator("#notes-btn").click()
    check(len(sf.locator("#notes").inner_text()) > 40, "slides notes")
    # downloads
    for href in ["downloads/nl2regex-source-code.zip", "report/REPORT.pdf", "slides/NL2Regex-slides.pdf"]:
        r = ctx.request.get(f"{base}/{href}")
        check(r.status == 200 and len(r.body()) > 10000, f"download {href} {len(r.body())//1024} KB")

    # standalone slides: every slide screenshot + overflow check
    sp = ctx.new_page(); sp.goto(base + "/slides/index.html", wait_until="networkidle")
    for i in range(19):
        sp.goto(f"{base}/slides/index.html#{i+1}"); sp.reload(wait_until="networkidle")
        over = sp.evaluate("""()=>{const s=document.querySelector('.slide.active');const out=[];
          s.querySelectorAll('*').forEach(e=>{const r=e.getBoundingClientRect(),q=s.getBoundingClientRect();
          if(r.width&&(r.right>q.right+2||r.bottom>q.bottom+2||r.left<q.left-2)) out.push(e.tagName+':'+(e.textContent||'').slice(0,30))});
          const sc=[...s.querySelectorAll('*')].filter(e=>e.scrollHeight>e.clientHeight+3&&getComputedStyle(e).overflow!='visible').map(e=>e.tagName);
          return {out:out.slice(0,3),sc:sc.slice(0,3),sh:s.scrollHeight}}""")
        if over["out"] or over["sc"] or over["sh"] > 1081: print("   OVERFLOW slide", i + 1, over); fails.append(f"slide {i+1} overflow")
        sp.screenshot(path=f"{shots}/slide_{i+1:02d}.png")

    # mobile
    m = b.new_context(viewport={"width": 390, "height": 800}, is_mobile=True, has_touch=True).new_page()
    m.on("pageerror", lambda e: errors.append("mobile " + str(e)))
    m.goto(base + "/", wait_until="networkidle")
    m.wait_for_function("!document.getElementById('go').disabled", timeout=20000)
    m.click("#chips-good .chip:nth-child(2)")
    sw = m.evaluate("document.documentElement.scrollWidth"); cw = m.evaluate("document.documentElement.clientWidth")
    check(sw <= cw + 1, f"mobile no horizontal scroll ({sw} vs {cw})")
    m.screenshot(path=f"{shots}/06_mobile.png", full_page=True)
    for t in ["report", "slides", "downloads"]:
        m.click(f"#t-{t}"); m.wait_for_timeout(800)
        sw = m.evaluate("document.documentElement.scrollWidth")
        check(sw <= 391, f"mobile {t} no horizontal scroll ({sw})")
        m.screenshot(path=f"{shots}/07_mobile_{t}.png")
    b.close()

print("console/network errors:", errors)
print("FAILS:", fails)
sys.exit(1 if fails or errors else 0)
