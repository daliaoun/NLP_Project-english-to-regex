import { loadModel, anonymize, deanonymize } from "./model.js";
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
const TABS = ["try", "notebook", "report", "slides", "downloads"];
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
  const frameId = { notebook: "notebook-frame", report: "report-frame", slides: "slides-frame" }[name];
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
// filled in from the evaluated model (see the report); GOOD = generated correctly, HARD = known failures
const GOOD = [
  { q: "lines containing 'cat' at least 2 times", lines: ["cat", "cat and cat", "catcat", "concatenate cat"] },
  { q: "lines containing 'cat' 2 times in a row", lines: ["catcat", "cat cat", "xcatcatx", "cat"] },
  { q: "lines that start with a capital letter and end with a number", lines: ["Hello 5", "hello 5", "A1", "Abc", "7up"] },
  { q: "lines that contain 'error' and do not contain 'warning'", lines: ["error 42", "warning: error", "all good", "ERROR"] },
  { q: "lines that are at least 8 characters long", lines: ["password", "pass", "12345678", "a b c d"] },
  { q: "lines made only of digits", lines: ["2024", "20a4", "", "007"] },
  { q: "lines where 'price' comes before 'total'", lines: ["price 10, total 12", "total then price", "pricetotal", "price"] },
  { q: "lines that begin with 'Dear' or end with '!'", lines: ["Dear Sir", "Hello!", "dear me", "Hi."] },
  { q: "lines without any vowel", lines: ["rhythm", "hello", "123", ""] },
];
const HARD = [
  { q: "lines with no digits at all", lines: ["hello", "abc1", "", "2024"] },
  { q: "lines that contain 'x' no fewer than 3 times", lines: ["xxx", "x x x x", "xx", "box"] },
  { q: "lines that have 'com' at the end", lines: ["example.com", "come here", "com", "dotcom!"] },
  { q: "lines that are not empty", lines: ["a", "", "hello world", " "] },
];
const DEFAULT_LINES = ["Hello 5", "hello", "cat", "42", "cat and dog", ""];
// test lines built from the quoted literals of a free-text description
function defaultLines(literals) {
  if (!literals.length) return DEFAULT_LINES;
  const [x, y] = literals;
  const out = [x, `${x} ${x}`, `${x}${x}`, `Hello ${x} 5`, "hello"];
  if (y) out.splice(3, 0, `${x} ${y}`, `${y} ${x}`, y);
  return [...new Set(out)];
}

/* ------------------------------------------------------------------ state */
let attn = null, base = null, baseLoading = null, testset = null;
let current = null;   // { q, a, b }
const status = $("status");

function setStatus(kind, text) {
  status.className = "status " + kind;
  $("status-text").textContent = text;
}

function makeChips(list, host) {
  for (const ex of list) {
    host.append(el("li", {}, el("button", {
      class: "ex chip", type: "button", text: ex.q,
      onclick: () => { $("q").value = ex.q; run(ex.lines); $("result-card").scrollIntoView({ behavior: "smooth", block: "start" }); },
    })));
  }
}
makeChips(GOOD, $("chips-good"));
makeChips(HARD, $("chips-hard"));

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
    // show a first result straight away
    if (!$("q").value) { $("q").value = GOOD[0].q; run(GOOD[0].lines); }
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

/* ------------------------------------------------------------------ regex display */
// split a regex into coloured pieces: literal text, character class, repetition, operator, grouping, "anything"
function regexPieces(rx) {
  const out = [];
  for (let i = 0; i < rx.length;) {
    const c = rx[i];
    if (c === "[") {
      const j = rx.indexOf("]", i + 2);
      const end = j < 0 ? rx.length : j + 1;
      const body = rx.slice(i, end);
      out.push([body, body.length === 3 ? "lit" : "cls"]);   // [.] is an escaped literal
      i = end;
    } else if (c === "{") {
      const j = rx.indexOf("}", i);
      const end = j < 0 ? rx.length : j + 1;
      out.push([rx.slice(i, end), "q"]); i = end;
    } else if (c === "." ) {
      if (rx[i + 1] === "*" || rx[i + 1] === "+") { out.push([rx.slice(i, i + 2), "any"]); i += 2; }
      else { out.push([c, "any"]); i++; }
    } else if (c === "*" || c === "+") { out.push([c, "q"]); i++; }
    else if (c === "&" || c === "|" || c === "~") { out.push([c, "op"]); i++; }
    else if (c === "(" || c === ")") { out.push([c, "grp"]); i++; }
    else if (c === "\\" && rx[i + 1] === "b") { out.push(["\\b", "cls"]); i += 2; }
    else { out.push([c, "lit"]); i++; }
  }
  return out;
}

let typingTimer = null;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function renderRegex(host, rx, animate) {
  host.replaceChildren();
  if (!rx) { host.textContent = "(empty output)"; return; }
  const spans = regexPieces(rx).map(([t, k]) => el("span", { class: `r-tok r-${k}`, text: t }));
  host.append(...spans);
  if (!animate || reduceMotion) return;
  // the decoder writes one symbol at a time: reveal the regex in the same order
  clearInterval(typingTimer);
  host.classList.add("typing");
  let i = 0;
  typingTimer = setInterval(() => {
    if (i < spans.length) spans[i++].classList.add("on");
    else { clearInterval(typingTimer); host.classList.remove("typing"); }
  }, Math.max(12, Math.min(40, 700 / spans.length)));
}

/* ------------------------------------------------------------------ generate */
function placeholder(tok, literals) {
  const m = /^__w(\d)__$/.exec(tok);
  return m && Number(m[1]) < literals.length ? literals[Number(m[1])] : null;
}

function run(lines) {
  if (!attn) return;
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }
  const a = attn.translate(q);
  current = { q, a, b: null };
  $("result-card").hidden = false;
  renderRegex($("rx-attn"), a.regex, true);

  // tokens the model read (quoted literals are shown as the placeholder slot they became)
  const tok = $("tokens");
  tok.replaceChildren();
  a.inputTokens.forEach((t, i) => {
    const lit = placeholder(t, a.literals);
    tok.append(el("span", {
      class: "tok" + (a.inputIds[i] === 1 ? " unk" : "") + (lit !== null ? " lit" : ""),
      text: lit !== null ? `${t} = '${lit}'` : t,
    }));
  });
  const unk = a.unknown;
  const note = $("unk-note");
  if (unk.length) {
    note.hidden = false;
    note.textContent = `The model has never seen ${unk.map((u) => `“${u}”`).join(", ")}, so it treats ${unk.length > 1 ? "them" : "it"} as unknown (highlighted above). Rephrase with more common words, and put literal text in quotes.`;
  } else if (!a.literals.length && /\b(word|string|text)\b/i.test(q)) {
    note.hidden = false;
    note.textContent = "Tip: put literal text in quotes, for example 'cat' or \"2024\". Without quotes the model cannot see it as a literal.";
  } else note.hidden = true;

  $("dialect-note").hidden = !/[&~]/.test(a.regex);
  $("lines").value = (lines ?? defaultLines(a.literals)).join("\n");
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
    renderRegex($("rx-base"), current.b.regex, false);
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
  const rxA = compileRegex(current.a.regex);
  const rxB = showBase ? compileRegex(current.b.regex) : null;
  const lines = $("lines").value.split("\n").slice(0, MAX_LINES);
  let allSame = true;

  const test = (ast, line) => {
    if (!ast) return null;
    try { return fullMatch(ast, line.slice(0, MAX_LEN)); } catch { return null; }
  };
  const flag = (v, label) => el("span", { class: "flag " + (v === null ? "bad" : v ? "yes" : "no") },
    label ? el("small", { text: label }) : null, v === null ? "invalid" : v ? "match" : "no match");
  let hits = 0;
  lines.forEach((line) => {
    const ra = test(rxA, line);
    if (ra) hits++;
    const text = el("span", { class: "t-line" + (ra ? " hit" : "") },
      line === "" ? el("span", { class: "empty", text: "(empty line)" }) : (ra ? el("mark", { text: line }) : line));
    const flags = el("span", { class: "t-flags" }, flag(ra, showBase ? "attn" : null));
    if (showBase) {
      const rb = test(rxB, line);
      flags.append(flag(rb, "base"));
      if (ra !== rb) allSame = false;
    }
    host.append(el("div", { class: "t-row" }, text, flags));
  });
  $("t-summary").textContent = rxA ? `${hits} of ${lines.length} line${lines.length === 1 ? "" : "s"} match` : "Result";
  if (!rxA) host.prepend(el("p", { class: "note warn", text: "The generated text is not a valid regular expression, so no line can be tested." }));

  // verdict of the comparison
  if (showBase) {
    const same = current.a.regex === current.b.regex;
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
  const label = (t) => { const l = placeholder(t, a.literals); return l !== null ? `'${l}'` : t; };
  a.inputTokens.forEach((t) => head.append(el("th", {}, el("span", { text: label(t) }))));
  table.append(el("thead", {}, head));
  const body = el("tbody");
  a.tokens.forEach((tk, r) => {
    const w = a.attention[r];
    if (!w) return;
    const tr = el("tr", {}, el("th", { text: label(tk) }));
    w.forEach((v, c) => {
      const td = el("td", { title: `${label(tk)} ← ${label(a.inputTokens[c])}: ${v.toFixed(2)}` });
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
const SPLITS = { t: "New regexes", p: "New phrasings", h: "Handwritten" };

function renderStats() {
  const pct = (rows, f) => (100 * rows.filter(f).length / rows.length).toFixed(1) + "%";
  const table = el("table", { class: "stats-table" });
  table.append(el("thead", {}, el("tr", {},
    el("th", { text: "Test set" }), el("th", { text: "n" }),
    el("th", { text: "Attention: exact" }), el("th", { text: "Attention: same behaviour" }),
    el("th", { text: "Baseline: exact" }), el("th", { text: "Baseline: same behaviour" }))));
  const body = el("tbody");
  for (const [k, name] of Object.entries(SPLITS)) {
    const rows = testset.filter((r) => r.k === k);
    if (!rows.length) continue;
    body.append(el("tr", {}, el("th", { text: name }), el("td", { text: String(rows.length) }),
      el("td", { class: "hi", text: pct(rows, (r) => r.a[1]) }), el("td", { class: "hi", text: pct(rows, (r) => r.a[2]) }),
      el("td", { text: pct(rows, (r) => r.b[1]) }), el("td", { text: pct(rows, (r) => r.b[2]) })));
  }
  table.append(body);
  $("stats").replaceChildren(el("div", { class: "table-scroll" }, table));
}

const pickPill = (v) => v[1] ? el("span", { class: "pill exact", text: "Exact match" })
  : v[2] ? el("span", { class: "pill same", text: "Equivalent, written differently" })
  : el("span", { class: "pill diff", text: "Different" });

/* ------------------------------------------------------------------ report version */
document.querySelectorAll("[data-report]").forEach((b) => b.addEventListener("click", () => {
  const full = b.dataset.report === "full";
  document.querySelectorAll("[data-report]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  const f = $("report-frame");
  f.src = full ? "report/report.html" : "report/short.html";
  lazyLoaded.add("report-frame");
  $("report-pdf").href = full ? "report/REPORT.pdf" : "report/REPORT_short.pdf";
}));

$("rand").addEventListener("click", () => {
  if (!testset) return;
  const k = $("rand-split").value;
  const pool = testset.filter((r) => k === "all" || r.k === k);
  const r = pool[Math.floor(Math.random() * pool.length)];
  const lits = anonymize(r.s).literals;
  const out = $("rand-out");
  out.hidden = false;
  out.replaceChildren(
    el("p", { class: "small muted", text: "From the " + SPLITS[r.k].toLowerCase() + " test set" }),
    el("p", { class: "rand-desc", text: r.s }),
    el("div", { class: "rand-grid" },
      el("span", { class: "k", text: "Reference" }), el("div", { class: "v" }, el("code", { text: r.r })),
      el("span", { class: "k", text: "Attention model" }), el("div", { class: "v" }, el("code", { text: deanonymize(r.a[0], lits) || "(empty)" }), pickPill(r.a)),
      el("span", { class: "k", text: "Baseline" }), el("div", { class: "v" }, el("code", { text: deanonymize(r.b[0], lits) || "(empty)" }), pickPill(r.b))),
    el("button", { class: "btn small", type: "button", text: "Try this description above", onclick: () => {
      $("q").value = r.s; run(); window.scrollTo({ top: 0, behavior: "smooth" });
    } }));
});

boot();
