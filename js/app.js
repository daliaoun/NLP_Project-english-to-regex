import { loadModel } from "./model.js";
import { compileRegex, fullMatch } from "./regex.js";

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ tiny DOM helper */
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}

/* ------------------------------------------------------------------ tabs and hash routing */
const TABS = ["try", "report", "slides", "downloads"];
const lazyLoaded = new Set();

function showTab(name, push = true) {
  if (!TABS.includes(name)) name = "try";
  for (const t of TABS) {
    const on = t === name;
    $("tab-" + t).hidden = !on;
    const btn = $("t-" + t);
    btn.setAttribute("aria-selected", String(on));
    btn.tabIndex = on ? 0 : -1;
  }
  const frameId = { report: "report-frame", slides: "slides-frame" }[name];
  if (frameId && !lazyLoaded.has(frameId)) {
    const f = $(frameId);
    f.src = f.dataset.src;
    lazyLoaded.add(frameId);
  }
  if (push && location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  window.scrollTo({ top: 0 });
}

document.querySelectorAll(".tabs button").forEach((b) => {
  b.addEventListener("click", () => showTab(b.dataset.tab));
  b.addEventListener("keydown", (e) => {
    const i = TABS.indexOf(b.dataset.tab);
    let j = null;
    if (e.key === "ArrowRight") j = (i + 1) % TABS.length;
    if (e.key === "ArrowLeft") j = (i + TABS.length - 1) % TABS.length;
    if (j !== null) { e.preventDefault(); showTab(TABS[j]); $("t-" + TABS[j]).focus(); }
  });
});
window.addEventListener("hashchange", () => showTab(location.hash.slice(1), false));
showTab(location.hash.slice(1) || "try", false);

/* ------------------------------------------------------------------ examples */
const GOOD = [
  { q: "lines containing the string 'dog' at least 2 times", lines: ["dogdog", "dog", "dog dog", "xdogdogx", "cat"] },
  { q: "lines that start with a capital letter and end with a number", lines: ["Hello 5", "hello 5", "A1", "Abc", "7up"] },
  { q: "lines that contain a number", lines: ["abc", "abc1", "42", "hello world"] },
  { q: "lines with a capital letter", lines: ["hello", "Hello", "hELLO", "123"] },
  { q: "lines that begin with the string 'dog'", lines: ["dog", "dogs and cats", "a dog", "Dog"] },
  { q: "lines with the string 'dog' before the string 'truck'", lines: ["dog truck", "truck dog", "dog and truck", "dogtruck", "dog"] },
  { q: "lines without any digit", lines: ["hello", "abc1", "", "2024"] },
  { q: "lines with a capital letter followed by a lower-case letter", lines: ["Hello", "hello", "HELLO", "aB", "Ab"] },
];
const HARD = [
  { q: "lines ending with the string 'truck'", lines: ["truck", "big truck", "truck stop", "dog"] },
  { q: "lines having either the string 'dog' or the string 'truck'", lines: ["dog", "truck", "dog truck", "cat"] },
  { q: "lines that do not contain the string 'dog'", lines: ["cat", "dog", "hotdog", "a dog here"] },
];
const DEFAULT_LINES = ["Hello 5", "hello", "dog", "42", "dog truck", ""];

/* ------------------------------------------------------------------ state */
let attn = null, base = null, baseLoading = null, testset = null;
let current = null;   // { q, a, b }
const status = $("status");

function setStatus(kind, text) {
  status.className = "status " + kind;
  $("status-text").textContent = text;
}

function makeChips(list, host, hard) {
  for (const ex of list) {
    host.append(el("button", {
      class: "chip" + (hard ? " hard" : ""), type: "button", text: ex.q,
      onclick: () => { $("q").value = ex.q; run(ex.lines); $("q").focus({ preventScroll: true }); },
    }));
  }
}
makeChips(GOOD, $("chips-good"), false);
makeChips(HARD, $("chips-hard"), true);

/* ------------------------------------------------------------------ loading */
async function boot() {
  try {
    const t0 = performance.now();
    const [m, ts] = await Promise.all([
      loadModel("attention"),
      fetch("data/testset.json").then((r) => r.json()),
    ]);
    attn = m;
    testset = ts;
    $("go").disabled = false;
    $("rand").disabled = false;
    setStatus("ready", `Model ready (${(m.nParams / 1e6).toFixed(2)}M parameters, loaded in ${Math.round(performance.now() - t0)} ms). Runs in your browser.`);
    renderStats();
  } catch (e) {
    console.error(e);
    setStatus("error", "Could not load the model files. If you opened this page as a local file, serve the folder with a web server (for example: python3 -m http.server).");
  }
}

function ensureBaseline() {
  if (base) return Promise.resolve(base);
  baseLoading ??= loadModel("baseline").then((m) => (base = m));
  return baseLoading;
}

/* ------------------------------------------------------------------ generate */
function run(lines) {
  if (!attn) return;
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }
  const a = attn.translate(q);
  current = { q, a, b: null };
  $("result-card").hidden = false;
  $("rx-attn").textContent = a.text || "(empty output)";

  // tokens the model read
  const tok = $("tokens");
  tok.replaceChildren();
  a.inputTokens.forEach((t, i) => tok.append(el("span", { class: "tok" + (a.inputIds[i] === 1 ? " unk" : ""), text: t })));
  const unk = a.unknown;
  const note = $("unk-note");
  if (unk.length) {
    note.hidden = false;
    note.textContent = `The model has never seen ${unk.map((u) => `“${u}”`).join(", ")}, so it treats ${unk.length > 1 ? "them" : "it"} as unknown (highlighted above). Try the dataset’s own wording, or the stand-in words 'dog', 'truck', 'ring', 'lake'.`;
  } else note.hidden = true;

  $("dialect-note").hidden = !/[&~]/.test(a.text);
  $("lines").value = (lines ?? DEFAULT_LINES).join("\n");
  renderHeat(a);
  if ($("compare").checked) runBaseline();
  renderTester();
}

async function runBaseline() {
  $("out-base").hidden = false;
  $("rx-base").textContent = "loading the baseline model…";
  try {
    const m = await ensureBaseline();
    if (!current) return;
    current.b = m.translate(current.q);
    $("rx-base").textContent = current.b.text || "(empty output)";
    renderTester();
  } catch (e) {
    console.error(e);
    $("rx-base").textContent = "could not load the baseline model";
  }
}

$("compare").addEventListener("change", () => {
  if ($("compare").checked) { if (current) runBaseline(); else $("out-base").hidden = false; }
  else $("out-base").hidden = true;
  renderTester();
});
$("go").addEventListener("click", () => run());
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

/* ------------------------------------------------------------------ line tester */
const MAX_LINES = 40, MAX_LEN = 200;

function renderTester() {
  const host = $("tester-out");
  host.replaceChildren();
  if (!current) return;
  const showBase = $("compare").checked && current.b;
  const rxA = compileRegex(current.a.text);
  const rxB = showBase ? compileRegex(current.b.text) : null;
  const lines = $("lines").value.split("\n").slice(0, MAX_LINES);
  let allSame = true;

  const test = (ast, line) => {
    if (!ast) return null;
    try { return fullMatch(ast, line.slice(0, MAX_LEN)); } catch { return null; }
  };
  const pill = (v, label) => {
    const p = el("span", { class: "pill " + (v === null ? "bad" : v ? "yes" : "no"), text: v === null ? "invalid" : v ? "MATCH" : "no match" });
    return label ? [p, el("span", { class: "pill-label", text: label })] : [p];
  };

  lines.forEach((line) => {
    const row = el("div", { class: "t-row" });
    row.append(el("span", { class: "t-line", text: line === "" ? "(empty line)" : line }));
    const ra = test(rxA, line);
    pill(ra, showBase ? "attn" : null).forEach((n) => row.append(n));
    if (showBase) {
      const rb = test(rxB, line);
      pill(rb, "base").forEach((n) => row.append(n));
      if (ra !== rb) allSame = false;
    }
    host.append(row);
  });
  if (!rxA) host.prepend(el("p", { class: "note warn", text: "The generated text is not a valid regular expression, so no line can be tested." }));

  // verdict of the comparison
  if (showBase) {
    const same = current.a.text === current.b.text;
    $("cmp-verdict").textContent = same
      ? "Both models produced the same regex."
      : allSame
        ? "Different regexes, but they give the same answer on the lines below."
        : "Different regexes, and they disagree on at least one of the lines below.";
  } else $("cmp-verdict").textContent = "";
}
$("lines").addEventListener("input", renderTester);

/* ------------------------------------------------------------------ attention heatmap */
function renderHeat(a) {
  const host = $("heat");
  host.replaceChildren();
  if (!a.attention.length || !a.tokens.length) return;
  const table = el("table", { class: "heat" });
  const head = el("tr", {}, el("th"));
  a.inputTokens.forEach((t) => head.append(el("th", {}, el("span", { text: t }))));
  table.append(el("thead", {}, head));
  const body = el("tbody");
  a.tokens.forEach((tk, r) => {
    const w = a.attention[r];
    if (!w) return;
    const tr = el("tr", {}, el("th", { text: tk }));
    w.forEach((v, c) => {
      const td = el("td", { title: `${tk} ← ${a.inputTokens[c]}: ${v.toFixed(2)}` });
      td.style.opacity = String(0.06 + 0.94 * v);
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(body);
  host.append(table);
}

/* ------------------------------------------------------------------ copy buttons */
document.querySelectorAll("[data-copy]").forEach((b) => {
  b.addEventListener("click", async () => {
    const text = $(b.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const r = document.createRange();
      r.selectNodeContents($(b.dataset.copy));
      const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    }
    const old = b.textContent;
    b.textContent = "Copied";
    setTimeout(() => (b.textContent = old), 1200);
  });
});

/* ------------------------------------------------------------------ held-out test explorer */
function renderStats() {
  const n = testset.length;
  const pct = (k) => (100 * k / n).toFixed(1) + "%";
  const count = (flag, idx) => testset.filter((r) => r[flag][idx]).length;
  const nd = testset.filter((r) => !r.a[3]);
  const ndFunc = (flag) => (100 * nd.filter((r) => r[flag][2]).length / nd.length).toFixed(1) + "%";
  const card = (title, flag) => el("div", { class: "stat" },
    el("strong", { text: title }),
    el("span", { text: `Exact match: ${pct(count(flag, 1))}` }),
    el("span", { text: `Same behaviour as the reference: ${pct(count(flag, 2))}` }),
    el("span", { text: `Same behaviour, excluding degenerate references: ${ndFunc(flag)}` }));
  $("stats").replaceChildren(card("Attention model (best)", "a"), card("Baseline (no attention)", "b"));
}

const pickPill = (v) => v[1] ? el("span", { class: "pill exact", text: "EXACT MATCH" })
  : v[2] ? el("span", { class: "pill same", text: "DIFFERENT BUT EQUIVALENT" })
  : el("span", { class: "pill diff", text: "DIFFERENT" });

$("rand").addEventListener("click", () => {
  if (!testset) return;
  const r = testset[Math.floor(Math.random() * testset.length)];
  const out = $("rand-out");
  out.hidden = false;
  out.replaceChildren(
    el("p", { class: "rand-desc", text: r.s }),
    el("div", { class: "rand-grid" },
      el("span", { class: "k", text: "Reference" }), el("div", { class: "v" }, el("code", { text: r.r })),
      el("span", { class: "k", text: "Attention model" }), el("div", { class: "v" }, el("code", { text: r.a[0] || "(empty)" }), pickPill(r.a)),
      el("span", { class: "k", text: "Baseline" }), el("div", { class: "v" }, el("code", { text: r.b[0] || "(empty)" }), pickPill(r.b))),
    el("button", { class: "btn small", type: "button", text: "Try this description above", onclick: () => {
      $("q").value = r.s; run(); window.scrollTo({ top: 0, behavior: "smooth" });
    } }));
});

boot();
