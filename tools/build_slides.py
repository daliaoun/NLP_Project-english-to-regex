"""Build the standalone slide viewer (site/slides/index.html) from the slide sources in tools/slides_src.

    python tools/build_slides.py
The viewer scales a 1920x1080 stage to the window, has keyboard/touch navigation, speaker notes,
full screen, and a print mode (?print=1) that lays out one slide per page for PDF export.
"""
import json, re, html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "slides_src"
OUT = ROOT / "site" / "slides" / "index.html"
deck = json.loads((SRC / "deck.json").read_text())

BLOBS = {
    "/_blob/aee40b7a44ed013204da9b44d69b4c9a": "img/learning_curves.png",
    "/_blob/4805ce116864ff27f218f6b647cc45fd": "img/attention_examples.png",
}

slides, notes = [], []
for name in deck["order"]:
    s = (SRC / f"{name}.html").read_text()
    m = re.search(r"<aside>(.*?)</aside>", s, re.S)
    notes.append(html.unescape(m.group(1).strip()) if m else "")
    s = re.sub(r"<aside>.*?</aside>", "", s, flags=re.S)
    for k, v in BLOBS.items():
        s = s.replace(k, v)
    # <x-shape kind="arrow-right" style="..."> -> clipped div
    s = re.sub(r'<x-shape kind="(arrow-right|arrow-down)"', lambda mm: f'<div class="shape {mm.group(1)}"', s)
    s = s.replace("</x-shape>", "</div>")
    s = s.replace("<section ", '<section class="slide" ', 1)
    slides.append(s.strip())

assert not any("—" in x for x in slides + notes), "em dash found"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NL2Regex: presentation</title>
<style>
@font-face{font-family:"IBM Plex Sans";font-weight:400;font-style:normal;font-display:swap;src:url(../fonts/ibm-plex-sans-latin-400-normal.woff2) format("woff2")}
@font-face{font-family:"IBM Plex Sans";font-weight:500;font-style:normal;font-display:swap;src:url(../fonts/ibm-plex-sans-latin-500-normal.woff2) format("woff2")}
@font-face{font-family:"IBM Plex Sans";font-weight:600;font-style:normal;font-display:swap;src:url(../fonts/ibm-plex-sans-latin-600-normal.woff2) format("woff2")}
@font-face{font-family:"IBM Plex Sans";font-weight:700;font-style:normal;font-display:swap;src:url(../fonts/ibm-plex-sans-latin-700-normal.woff2) format("woff2")}
@font-face{font-family:"JetBrains Mono";font-weight:400;font-style:normal;font-display:swap;src:url(../fonts/jetbrains-mono-latin-400-normal.woff2) format("woff2")}
@font-face{font-family:"JetBrains Mono";font-weight:600;font-style:normal;font-display:swap;src:url(../fonts/jetbrains-mono-latin-600-normal.woff2) format("woff2")}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#0B1220;font-family:"IBM Plex Sans",system-ui,Arial,sans-serif;color:#F1EEE5;overflow:hidden}
p,h1,h2,h3,ul,ol,li,figure{margin:0}
ul,ol{padding:0;list-style:none}
.app{display:flex;flex-direction:column;height:100%}
.viewport{flex:1;min-height:0;position:relative;overflow:hidden;display:grid;place-items:center}
.stage{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform-origin:0 0;background:#111A2C;box-shadow:0 8px 40px rgba(0,0,0,.45)}
.slide{position:absolute;left:0;top:0;width:1920px;height:1080px;overflow:hidden;box-sizing:border-box}
.slide:not(.active){display:none!important}
.shape{flex:none}
.slide table{border-collapse:collapse;width:100%}
.slide th,.slide td{padding:11px 20px;border-bottom:1px solid #DDD7C6;text-align:left;vertical-align:top}
.slide th{font-weight:600;color:#4B5568;border-bottom:2px solid #C9C2AE}
.slide tr:last-child td{border-bottom:none}
.shape.arrow-right{clip-path:polygon(0 36%,62% 36%,62% 8%,100% 50%,62% 92%,62% 64%,0 64%)}
.shape.arrow-down{clip-path:polygon(36% 0,64% 0,64% 62%,92% 62%,50% 100%,8% 62%,36% 62%)}
.bar{display:flex;align-items:center;justify-content:center;gap:10px;padding:8px 12px;background:#111A2C;border-top:1px solid #2C3B5C;flex-wrap:wrap}
.bar button{font:inherit;font-size:15px;font-weight:600;color:#F1EEE5;background:#1B2740;border:1px solid #2C3B5C;border-radius:9px;padding:7px 14px;cursor:pointer}
.bar button:hover{border-color:#F2B544}
.bar button:focus-visible{outline:3px solid #F2B544;outline-offset:2px}
.bar button[aria-pressed="true"]{background:#F2B544;color:#111A2C;border-color:#F2B544}
.counter{font-size:15px;color:#B7C0D0;min-width:84px;text-align:center}
.notes{max-height:34vh;overflow:auto;padding:14px 22px;background:#0F1626;border-top:1px solid #2C3B5C;font-size:17px;line-height:1.55;color:#E5E1D4}
.notes strong{color:#F2B544}
.embed .bar{padding:5px 8px}
.embed .bar button{font-size:13.5px;padding:5px 10px}
.embed .notes{font-size:15px;max-height:30vh}
[hidden]{display:none!important}
/* print: one slide per page */
.print html,.print body{overflow:visible;height:auto;background:#fff}
.print .app{display:block;height:auto}
.print .viewport{display:block;overflow:visible}
.print .stage{position:static;transform:none!important;box-shadow:none}
.print .slide:not(.active){display:flex!important}
.print .slide{position:relative;page-break-after:always;break-after:page}
.print .bar,.print .notes{display:none!important}
@page{size:1920px 1080px;margin:0}
@media print{html,body{background:#fff;overflow:visible;height:auto}.bar,.notes{display:none!important}}
</style>
</head>
<body>
<div class="app" id="app">
  <div class="viewport" id="viewport">
    <div class="stage" id="stage">
__SLIDES__
    </div>
  </div>
  <div class="notes" id="notes" hidden aria-live="polite"></div>
  <div class="bar" role="toolbar" aria-label="Slide controls">
    <button type="button" id="prev" aria-label="Previous slide">&larr; Prev</button>
    <span class="counter" id="counter" aria-live="polite">1 / __N__</span>
    <button type="button" id="next" aria-label="Next slide">Next &rarr;</button>
    <button type="button" id="notes-btn" aria-pressed="false">Speaker notes</button>
    <button type="button" id="fs">Full screen</button>
  </div>
</div>
<script>
const NOTES = __NOTES__;
const params = new URLSearchParams(location.search);
const printMode = params.has("print");
if (params.has("embed")) document.getElementById("app").classList.add("embed");
if (printMode) document.documentElement.classList.add("print");
const slides = Array.from(document.querySelectorAll(".slide"));
const stage = document.getElementById("stage"), viewport = document.getElementById("viewport");
const notesEl = document.getElementById("notes"), counter = document.getElementById("counter");
let idx = 0;

function fit() {
  if (printMode) return;
  const s = Math.min(viewport.clientWidth / 1920, viewport.clientHeight / 1080);
  stage.style.transform = `translate(${-960 * s}px, ${-540 * s}px) scale(${s})`;
}
function show(i, updateHash = true) {
  idx = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, k) => s.classList.toggle("active", k === idx));
  counter.textContent = `${idx + 1} / ${slides.length}`;
  notesEl.textContent = NOTES[idx] || "No notes for this slide.";
  if (updateHash) { try { history.replaceState(null, "", "#" + (idx + 1)); } catch (e) {} }
}
document.getElementById("prev").onclick = () => show(idx - 1);
document.getElementById("next").onclick = () => show(idx + 1);
const nbtn = document.getElementById("notes-btn");
nbtn.onclick = () => { notesEl.hidden = !notesEl.hidden; nbtn.setAttribute("aria-pressed", String(!notesEl.hidden)); fit(); };
const fsBtn = document.getElementById("fs");
fsBtn.onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
};
document.addEventListener("fullscreenchange", () => { fsBtn.textContent = document.fullscreenElement ? "Exit full screen" : "Full screen"; setTimeout(fit, 50); });
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === "ArrowRight" || k === "PageDown" || k === " " || k === "Enter") { e.preventDefault(); show(idx + 1); }
  else if (k === "ArrowLeft" || k === "PageUp" || k === "Backspace") { e.preventDefault(); show(idx - 1); }
  else if (k === "Home") show(0);
  else if (k === "End") show(slides.length - 1);
  else if (k === "f" || k === "F") fsBtn.click();
  else if (k === "n" || k === "N") nbtn.click();
});
let x0 = null;
viewport.addEventListener("touchstart", (e) => { x0 = e.touches[0].clientX; }, { passive: true });
viewport.addEventListener("touchend", (e) => {
  if (x0 === null) return;
  const dx = e.changedTouches[0].clientX - x0; x0 = null;
  if (Math.abs(dx) > 50) show(idx + (dx < 0 ? 1 : -1));
});
viewport.addEventListener("click", (e) => {
  if (e.target.closest("a")) return;
  const r = viewport.getBoundingClientRect();
  show(idx + (e.clientX - r.left > r.width / 2 ? 1 : -1));
});
window.addEventListener("resize", fit);
window.addEventListener("hashchange", () => show((parseInt(location.hash.slice(1), 10) || 1) - 1, false));
if (printMode) slides.forEach((s) => s.classList.add("active"));
show((parseInt(location.hash.slice(1), 10) || 1) - 1, false);
fit();
window.__ready = true;
</script>
</body>
</html>
"""
out = PAGE.replace("__SLIDES__", "\n".join(slides)).replace("__NOTES__", json.dumps(notes, ensure_ascii=False)).replace("__N__", str(len(slides)))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(out)
print(f"wrote {OUT} ({len(out)//1024} KB, {len(slides)} slides)")
