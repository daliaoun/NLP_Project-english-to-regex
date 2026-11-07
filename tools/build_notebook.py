"""
Build NL2Regex_notebook.ipynb: the whole project in one self-contained notebook (dataset generation,
EDA, preprocessing, model, training, evaluation, analysis, demo). The code cells are extracted from
src/*.py with the `ast` module, so the notebook runs exactly the same code as the project.

    python tools/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace NL2Regex_notebook.ipynb --ExecutePreprocessor.timeout=-1
"""
import ast
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def extract(module, keep=None, drop=()):
    """Source of the top-level definitions of src/<module>.py, in file order.
    keep: names to keep (functions, classes, assigned variables); None = everything but imports/docstring/main."""
    text = (SRC / f"{module}.py").read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    out = []
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue
        if isinstance(node, ast.If):          # if __name__ == "__main__"
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = set()
            for t in node.targets:
                names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            names = set()
        if keep is not None and not (names & set(keep)):
            continue
        if names & set(drop):
            continue
        start = node.lineno - 1
        if getattr(node, "decorator_list", None):
            start = node.decorator_list[0].lineno - 1
        # keep comment lines directly above the definition
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        seg = "\n".join(lines[start:node.end_lineno])
        seg = "\n".join(l for l in seg.splitlines()
                        if not re.match(r"\s*from (data|regex_engine|models|evaluate) import", l)
                        and "sys.path.insert" not in l)
        out.append(seg)
    return "\n\n\n".join(out)


cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# =====================================================================================================
md("""
# NL2Regex: from English to regular expressions with an encoder-decoder

**Author:** Mohamed Ali Aoun, Université Paris Dauphine Tunis

NLP course project, text generation with encoder-decoder architectures.

This notebook contains the whole project in one place and runs from top to bottom:

| # | Section | What it does |
|---|---|---|
| 1 | Setup | imports, seeds, configuration |
| 2 | The task and why we built our own dataset | problem statement, the label problem of the NL-RX benchmark |
| 3 | The regex dialect and its matcher | a parser and whole-line matcher that supports `&` (and) and `~` (not) |
| 4 | Dataset: VeriRegex generator | meaning trees, English phrasings, regex, independent verification, splits |
| 5 | Exploratory data analysis | sizes, kinds, lengths, vocabulary, literals, numbers, diversity, leakage checks |
| 6 | Preprocessing | literal slots, normalisation, tokenisation, vocabularies, encoding, padding, batching |
| 7 | Model | LSTM encoder, bridge, LSTM decoder, additive attention, parameter counts |
| 8 | Training | baseline (no attention) and attention model, learning curves |
| 9 | Evaluation | exact match, behavioural equivalence, BLEU, perplexity on three test sets; greedy versus beam |
| 10 | Analysis | attention heatmaps, generated examples, error analysis |
| 11 | Interactive demo | type a description, get a regex, test it on lines |
| 12 | Discussion and conclusion | |

**Constraints of the assignment respected:** encoder-decoder implemented from scratch in PyTorch (only basic layers such as `nn.Embedding`, `nn.LSTM`, `nn.Linear`); no pretrained model, no API.

**How to run.** Put this notebook at the root of the project folder (it only needs `torch`, `numpy`, `pandas` and `matplotlib`). With the flags below set to `True`, it regenerates the dataset (about 5 minutes) and trains both models on the CPU (about one hour in total). Set `RETRAIN = False` to load the trained checkpoints from `results/` instead and run in a few minutes.
""")

md("## 1. Setup")
code("""
import json, math, random, re, sys, time, os
from collections import Counter, defaultdict
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

REGENERATE = True   # rebuild the dataset with the generator (about 5 min); False = read data/veriregex
RETRAIN = True      # train both models here (about 1 h on CPU); False = load results/base_long and results/attn
RUN_BEAM = True     # also evaluate with beam search (a few minutes per model)

PROJECT = Path(".").resolve()
NB_OUT = PROJECT / "notebook_runs"          # everything this notebook writes goes here
NB_OUT.mkdir(exist_ok=True)
print("torch", torch.__version__, "| python", sys.version.split()[0], "| CPU threads available:", os.cpu_count())
""")

# =====================================================================================================
md("""
## 2. The task, and why we built our own dataset

**Task.** The input is a short English description of a set of text lines; the output is a regular expression that matches exactly those lines (the regex must match the **whole** line).

```
lines containing 'cat' at least 2 times        ->  (.*cat){2,}.*
lines containing 'cat' 2 times in a row        ->  .*(cat){2,}.*
lines that start with a capital letter and end with a number   ->  ([A-Z].*)&(.*[0-9])
```

**Formulation.** Given a source token sequence $x = (x_1, \\dots, x_n)$, generate a target token sequence $y = (y_1, \\dots, y_m)$ by maximising $p(y \\mid x) = \\prod_t p(y_t \\mid y_{<t}, x)$. Source tokens are words; target tokens are regex symbols. The output language is formal, so every generated regex can be **run** to check it, which allows an evaluation beyond string comparison.

**Why not the standard benchmark.** Our first version used NL-RX-Turk (Locascio et al., 2016) and reached 45.3% exact match. Reading the outputs showed that some labels contradict their English. The dataset contains

```
lines with the string 'dog' at least 2 times   ->   .*(dog){2,}.*
```

but this regex only accepts two **adjacent** copies (`dogdog`) and rejects `dog and dog`. The templated version NL-RX-Synth uses the same convention, so the problem comes from the original grammar. We therefore built **VeriRegex**, a dataset in which every label is checked by an independent implementation (section 4). The cell after the matcher below demonstrates the problem.
""")

# =====================================================================================================
md("""
## 3. The regex dialect and its matcher

The dialect is the one of NL-RX: `.`, character classes `[A-Z]`, `*`, `+`, `{n}`, `{n,}`, `{n,m}`, grouping, alternation `|`, and two operators that Python's `re` does not have: **intersection** `A&B` (the line matches A and B) and **complement** `~A` (the line does not match A). They make combinations of conditions easy to write but force us to implement our own parser and matcher.

The matcher computes, for each sub-expression and start position, the **set of end positions** it can reach (memoised), which handles `&` (intersect the sets) and `~` (complement within the line) naturally.
""")
code(extract("regex_engine"))
code("""
# sanity checks of the matcher, including & and ~
tests = [(".*(dog){2,}.*", "dogdog", True), (".*(dog){2,}.*", "dog and dog", False),
         ("(.*dog){2,}.*", "dog and dog", True), ("([A-Z].*)&(.*[0-9])", "Hello 5", True),
         ("([A-Z].*)&(.*[0-9])", "hello 5", False), ("~(.*[0-9].*)", "abc", True), ("~(.*[0-9].*)", "a1", False),
         (".{0,3}", "abcd", False), ("[0-9]+", "", False)]
for rx, s, expected in tests:
    got = full_match(parse(rx), s)
    assert got == expected, (rx, s, got)
    print(f"{rx:24s} {s!r:14s} -> {'MATCH' if got else 'no match'}")
print("\\nThe NL-RX label .*(dog){2,}.* for 'the string dog at least 2 times' rejects 'dog and dog':",
      not full_match(parse(".*(dog){2,}.*"), "dog and dog"))
""")

# =====================================================================================================
md("""
## 4. Dataset: the VeriRegex generator

Every example is built from one **meaning tree**, from which everything else is derived:

1. **Meaning tree.** 13 kinds of condition (contains, starts with, ends with, is exactly, at least N times, fewer than N times, N times in a row, before, immediately followed by, length at least / at most / exactly N, made only of), applied to a quoted literal or to a character type (digit, letter, capital, lowercase, vowel, space), and combined with `not`, `and` (two or three conditions) and `or`.
2. **English.** One description written from the tree, choosing among about 216 phrasing templates, synonyms, subjects and openers.
3. **Regex.** One canonical regex per tree.
4. **Verification.** The tree is also evaluated by a plain Python predicate that never looks at the regex (`holds`). On about 300 probe strings per example, the regex (through our matcher) and the predicate must agree on every string: a single disagreement raises an error and stops the build. Trees that accept every probe or none (contradictions, tautologies) are thrown away.

The next cells contain the generator code, unchanged from `src/generate.py`.
""")
md("### 4.1 Vocabulary of the generated world, character classes and held-out phrasings")
code(extract("generate", keep=["WORDS", "CAPS", "NUMS", "MULTI", "CHARS", "random_literal", "CLASSES", "ONLY_CLASSES",
                               "NUMBER_WORDS", "HELD_OUT", "Phraser", "num_text"]))
md("### 4.2 Meaning trees: canonical regex and independent Python meaning")
code(extract("generate", keep=["item_regex", "regex_of", "occurrences", "count_disjoint", "holds"]))
md("### 4.3 From a tree to English")
code(extract("generate", keep=["quote", "item_np", "atom_phrases", "wh_phrases", "wh_ok", "wh", "_SING", "conjugate",
                               "fill", "vp", "SUBJECTS", "SING_SUBJECTS", "PREFIXES", "tidy", "describe", "_describe"]))
md("### 4.4 Sampling trees, verification, literal slots")
md("""Literal slots (explained in section 6) are needed here already, because two examples count as the same if their descriptions and regexes are equal once the literals are replaced by `__w0__`, `__w1__`, ...""")
code(extract("data", keep=["PAD", "SPECIALS", "PAD_ID", "MAX_SRC_LEN", "MAX_TGT_LEN", "_QUOTED", "_straighten",
                           "literal_spans", "anonymize", "escape_literal", "deanonymize"]))
code(extract("generate", keep=["sample_item", "sample_atom", "sample_literal_side", "sample_tree", "literal_order",
                               "FILLERS", "probe_strings", "verify", "placeholder_regex", "make_example", "_worker",
                               "generate", "anon_target"]))
code("""
# one example, step by step
rng = random.Random(2026)
tree, lits = sample_tree(rng)
P = Phraser(rng, allow_held_out=False)
print("meaning tree :", tree)
print("literals     :", lits)
print("English      :", describe(P, tree, lits))
print("regex        :", regex_of(tree, lits))
probes = probe_strings(tree, lits, random.Random(0))
agree = all(full_match(parse(regex_of(tree, lits)), s) == holds(tree, s, lits) for s in probes)
print(f"regex and Python meaning agree on all {len(probes)} probe strings:", agree)
print("matching probes:", [s for s in probes if holds(tree, s, lits)][:6])
""")
md("### 4.5 Handwritten test set\n\nFifty descriptions written by hand in free wording; their meaning tree is written by hand too, so the regex is computed and verified like the generated ones.")
code(extract("generate", keep=["handwritten"]))
md("""### 4.6 Building the dataset and the splits

- 42,000 examples for train/validation/test, 30,000 candidates for the held-out-phrasing test.
- Duplicates of (anonymised description, regex) are removed.
- **Split by regex**: regex structures (with literal slots) are shuffled and assigned to train (86%), validation (7%) or test (7%), so no test regex appears in training. Single conditions all go to training.
- **Held-out phrasings**: generated with 11 phrasing patterns that are never used in training, kept only if one was used, the regex is new and the anonymised description is not in training.
""")
code('''
def build_dataset(out_dir):
    """Same procedure as src/generate.py main(), writing to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "fork" in mp.get_all_start_methods():          # Linux: 2 worker processes
        ctx = mp.get_context("fork")
        with ctx.Pool(2) as pool:
            main_set, rej1 = generate(42000, 1_000_000, False, pool)
            phr_set, rej2 = generate(30000, 9_000_000, True, pool)
    else:                                              # Windows / macOS notebooks: one process
        class Seq:
            def map(self, f, xs, chunksize=1):
                return list(map(f, xs))
        main_set, rej1 = generate(42000, 1_000_000, False, Seq())
        phr_set, rej2 = generate(30000, 9_000_000, True, Seq())
    for ex in main_set + phr_set:
        ex["src_anon"] = anonymize(ex["text"])[0]
        ex["regex_anon"] = anon_target(ex)
    rng = random.Random(7)
    seen, uniq = set(), []
    for ex in main_set:
        key = (ex["src_anon"], ex["regex_anon"])
        if key not in seen:
            seen.add(key)
            uniq.append(ex)
    by_rx = {}
    for ex in uniq:
        by_rx.setdefault(ex["regex_anon"], []).append(ex)
    rxs = sorted(by_rx)
    rng.shuffle(rxs)
    train, val, test = [], [], []
    for rx in rxs:
        exs = by_rx[rx]
        r = rng.random()
        if exs[0]["kind"] == "atom" or r < 0.86:
            train += exs
        elif r < 0.93:
            val += exs
        else:
            test += exs
    train_rx = {ex["regex_anon"] for ex in train}
    train_src = {ex["src_anon"] for ex in train}
    phrasing = [ex for ex in phr_set if ex["held_out_phrasing"] and ex["regex_anon"] not in train_rx
                and ex["src_anon"] not in train_src]
    rng.shuffle(phrasing)
    val, test, phrasing = val[:2000], test[:3000], phrasing[:1500]
    hand = handwritten()
    for name, rows in [("train", train), ("val", val), ("test", test), ("test_phrasing", phrasing),
                       ("test_handwritten", hand)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\\n")
    stats = {"sampled_main": len(main_set) + rej1, "rejected_main": rej1, "kept_main": len(main_set),
             "after_dedup": len(uniq), "train": len(train), "val": len(val), "test": len(test),
             "test_phrasing": len(phrasing), "test_handwritten": len(hand),
             "unique_regex_structures_train": len(train_rx), "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if REGENERATE:
    DATA = NB_OUT / "veriregex"
    print(json.dumps(build_dataset(DATA), indent=2))
    shipped = PROJECT / "data" / "veriregex"
    if shipped.exists():
        same = all((DATA / f"{s}.jsonl").read_text() == (shipped / f"{s}.jsonl").read_text()
                   for s in ["train", "val", "test", "test_phrasing", "test_handwritten"])
        print("identical to the dataset shipped in data/veriregex:", same)
else:
    DATA = PROJECT / "data" / "veriregex"
print("using", DATA.relative_to(PROJECT).as_posix())
''')

# =====================================================================================================
md("""
## 5. Exploratory data analysis

We look at the raw pairs before any model-specific preprocessing: sizes, kinds of meaning, kinds of condition, lengths, vocabulary, literals, numbers, phrasing diversity, and checks that the splits do not leak.
""")
code("""
SPLITS = ["train", "val", "test", "test_phrasing", "test_handwritten"]
raw = {s: [json.loads(l) for l in (DATA / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in SPLITS}
for s in SPLITS:
    for ex in raw[s]:
        ex["split"] = s
df = pd.DataFrame([ex for s in SPLITS for ex in raw[s]])
df["tree_t"] = df["tree"].map(eval)           # the meaning tree as a Python tuple (our own repr)
print(df.shape)
display(df[["split", "text", "regex", "kind"]].groupby("split").head(2).reset_index(drop=True))
""")
md("### 5.1 Split sizes and kinds of meaning")
code("""
size = df.groupby("split").size().reindex(SPLITS)
kinds = pd.crosstab(df["split"], df["kind"]).reindex(SPLITS)[["atom", "not", "and", "or"]]
kinds_pct = (kinds.div(kinds.sum(axis=1), axis=0) * 100).round(1)
display(pd.concat([size.rename("pairs"), kinds, kinds_pct.add_suffix(" %")], axis=1))
ax = kinds_pct.plot(kind="barh", stacked=True, figsize=(9, 3.2), color=["#0B6465", "#5FA8A3", "#E0B04A", "#9A5B00"])
ax.set_xlabel("% of the split"); ax.set_title("Kind of meaning per split (atom = one condition)"); ax.invert_yaxis()
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout(); plt.show()
""")
md("**Reading.** Training contains the single conditions plus combinations. Because single conditions go to training (the few single-condition rows in validation and test are \"fewer than N\" conditions, whose regex is a negation), the test set of new regexes is essentially made of combinations and negations, which is the harder, compositional part of the task.")
md("### 5.2 Which conditions and character types occur")
code("""
def atoms(t):
    if t[0] in ("and", "or"):
        return [a for c in t[1:] for a in atoms(c)]
    if t[0] == "not":
        return atoms(t[1])
    return [t]

def items(a):
    return [x for x in a[1:] if isinstance(x, tuple)]

tr = df[df.split == "train"]
atom_kinds = Counter(a[0] for t in tr.tree_t for a in atoms(t))
item_kinds = Counter(("literal" if it[0] == "lit" else it[1]) for t in tr.tree_t for a in atoms(t) for it in items(a))
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
pd.Series(atom_kinds).sort_values().plot(kind="barh", ax=ax[0], color="#0B6465")
ax[0].set_title("Conditions in the training set (13 kinds)")
pd.Series(item_kinds).sort_values().plot(kind="barh", ax=ax[1], color="#E0B04A")
ax[1].set_title("What the conditions apply to")
plt.tight_layout(); plt.show()
n_cond = tr.tree_t.map(lambda t: len(atoms(t)))
print("conditions per training example:", dict(sorted(Counter(n_cond).items())))
""")
md("### 5.3 Lengths of descriptions and regexes")
code(extract("data", keep=["_SRC_TOKEN", "normalize_src", "tokenize_src_veri", "_TGT_TOKEN", "tokenize_tgt", "detokenize_tgt"]))
code("""
df["src_tok"] = df.text.map(tokenize_src_veri)
df["tgt_tok"] = df.regex_anon.map(tokenize_tgt)
df["src_len"] = df.src_tok.map(len)
df["tgt_len"] = df.tgt_tok.map(len)
display(df.groupby("split")[["src_len", "tgt_len"]].describe().round(1).reindex(SPLITS))
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for s, c in zip(["train", "test", "test_phrasing"], ["#0B6465", "#E0B04A", "#9A5B00"]):
    d = df[df.split == s]
    ax[0].hist(d.src_len, bins=range(0, 40), alpha=0.5, density=True, label=s, color=c)
    ax[1].hist(d.tgt_len, bins=range(0, 70, 2), alpha=0.5, density=True, label=s, color=c)
ax[0].set_title("Description length (tokens)"); ax[1].set_title("Regex length (tokens)")
ax[0].legend(); ax[1].legend()
ax[2].scatter(tr_len := df[df.split == "train"].src_len, df[df.split == "train"].tgt_len, s=3, alpha=0.15, color="#0B6465")
ax[2].set_xlabel("description tokens"); ax[2].set_ylabel("regex tokens"); ax[2].set_title("Train: description vs regex length")
plt.tight_layout(); plt.show()
t = df[df.split == "train"]
print("Pearson correlation of description and regex lengths (train): %.2f" % np.corrcoef(t.src_len, t.tgt_len)[0, 1])
print("regexes longer than the 59-token limit (all splits):", int((df.tgt_len > MAX_TGT_LEN - 1).sum()), "of", len(df))
""")
md("**Reading.** Descriptions are short (about 14 words) and regexes about 17 tokens, with a long tail for three-condition descriptions. The two lengths are clearly correlated: more conditions means a longer description and a longer regex. Only 11 regexes exceed the output limit of 60 tokens (including the end token).")
md("### 5.4 Vocabulary of the descriptions")
code("""
train_counts = Counter(w for toks in df[df.split == "train"].src_tok for w in toks)
print("distinct words in training descriptions:", len(train_counts), "| tokens:", sum(train_counts.values()))
top = pd.Series(dict(train_counts.most_common(30)))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
top.plot(kind="bar", ax=ax[0], color="#0B6465"); ax[0].set_title("30 most frequent words (train)")
freqs = np.array(sorted(train_counts.values(), reverse=True))
ax[1].loglog(np.arange(1, len(freqs) + 1), freqs, color="#9A5B00"); ax[1].set_xlabel("rank"); ax[1].set_ylabel("frequency")
ax[1].set_title("Rank-frequency (Zipf) plot")
plt.tight_layout(); plt.show()
cum = np.cumsum(freqs) / freqs.sum()
print("share of tokens covered by the 20 / 50 / 100 most frequent words: %.1f%% / %.1f%% / %.1f%%"
      % (100 * cum[19], 100 * cum[49], 100 * cum[99]))
for s in SPLITS[1:]:
    toks = [w for t_ in df[df.split == s].src_tok for w in t_]
    unk = [w for w in toks if w not in train_counts]
    print(f"{s:17s} words never seen in training: {100 * len(unk) / len(toks):.2f}% of tokens  {Counter(unk).most_common(6)}")
""")
md("**Reading.** The vocabulary is small (about 200 words) because the language is generated, and a few function words (`lines`, `that`, `__w0__`, `the`) dominate. The two test sets of new regexes have no unknown words, the held-out phrasings almost none (`each`), and the handwritten set about 1%, which is where robustness to new wording is really tested.")
md("### 5.5 Literals, numbers and target symbols")
code("""
n_lits = df[df.split == "train"].words.map(len)
lit_words = Counter(w for ws in df[df.split == "train"].words for w in ws)
nums = Counter(int(n) for t in df[df.split == "train"].regex_anon for n in re.findall(r"\\{(\\d+)", t))
tgt_counts = Counter(tok for toks in df[df.split == "train"].tgt_tok for tok in toks)
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
pd.Series(Counter(n_lits)).sort_index().plot(kind="bar", ax=ax[0], color="#0B6465"); ax[0].set_title("Quoted literals per description (train)")
pd.Series(nums).sort_index().plot(kind="bar", ax=ax[1], color="#E0B04A"); ax[1].set_title("Numbers inside {n,m} in training regexes")
pd.Series(dict(tgt_counts.most_common())).plot(kind="bar", ax=ax[2], color="#9A5B00"); ax[2].set_title("Target symbols (train)")
plt.tight_layout(); plt.show()
print("distinct literal values:", len(lit_words), "| most common:", lit_words.most_common(8))
""")
md("**Reading.** Most descriptions have zero to two literals; numbers range from 1 to 20 (counts 2 to 6, lengths 1 to 20). The target alphabet is tiny: `.`, `*`, parentheses, brackets, braces, class ranges, `&`, `|`, `~` and the slots.")
md("### 5.6 Phrasing diversity")
code("""
t = df[df.split == "train"].copy()
t["pattern"] = t.text.map(lambda s: re.sub(r"\\b\\d+\\b", "N", anonymize(s)[0].lower()))
per_rx = t.groupby("regex_anon").size()
print("training pairs:", len(t), "| distinct descriptions:", t.text.nunique(),
      "| distinct sentence patterns (literals and numbers abstracted):", t.pattern.nunique())
print("distinct regex structures:", t.regex_anon.nunique(), "| descriptions per structure: mean %.2f, max %d" % (per_rx.mean(), per_rx.max()))
rx = t.regex_anon.value_counts().index[0]
display(pd.DataFrame({"descriptions of the most frequent structure " + rx: t[t.regex_anon == rx].text.head(8).values}))
""")
md("**Reading.** Nearly every training sentence is different, and one regex is expressed in many different ways (the table shows one structure), so the model cannot simply memorise sentence-to-regex pairs.")
md("### 5.7 Leakage checks")
code("""
sets = {s: set(df[df.split == s].regex_anon) for s in SPLITS}
srcs = {s: set(df[df.split == s].src_tok.map(" ".join)) for s in SPLITS}
rows = []
for s in SPLITS[1:]:
    rows.append({"split": s, "regexes also in train": len(sets[s] & sets["train"]),
                 "distinct regexes": len(sets[s]), "descriptions also in train": len(srcs[s] & srcs["train"])})
display(pd.DataFrame(rows))
""")
md("**Reading.** Validation, test and held-out-phrasing regexes never occur in training (0 overlap), and none of their descriptions does either. The handwritten set is different by design: 41 of its 45 distinct regexes (46 of its 50 examples) are known from training, and 4 of its simplest descriptions (such as \"lines containing 'ice cream'\" once the literal is replaced by a slot) can also be produced by the generator. It tests free wording, not new structure.")

# =====================================================================================================
md("""
## 6. Preprocessing

| Step | What we do | Why |
|---|---|---|
| Literal slots | each quoted span (`'cat'`, `"2024"`, `'ice cream'`) becomes `__w0__`, `__w1__`, ... in order of first appearance, in the description **and** in the regex; slots are replaced by the literals after decoding (special characters escaped as `[.]`) | the model works with any word, including words never seen in training |
| Normalisation | lowercase, remaining quotes removed (so `don't` becomes `don t`) | smaller vocabulary |
| Source tokenisation | words (hyphenated words kept whole) and punctuation | |
| Target tokenisation | one token per regex character; `__wK__` and `\\b` are single tokens | regex syntax is character-level |
| Vocabularies | built on the training split only; `<PAD>`=0, `<UNK>`=1, `<SOS>`=2, `<EOS>`=3 | no information from the test sets |
| Encoding | token ids, then a learned embedding layer (dimension 128) | |
| Padding and batching | batches padded with `<PAD>`, grouped by target length; the encoder uses packed sequences, attention masks padding, the loss ignores it | variable-length batches, less padding |
| Truncation | at most 40 source and 60 target tokens | |
| Unknown words | `<UNK>`, plus 5% word dropout during training (random words replaced by `<UNK>`) | robustness to unseen words |
""")
md("### 6.1 Literal slots")
code("""
for s in ["lines that don't contain 'dog' or \\"2024\\"",
          "lines with \\u2018ice cream\\u2019 and 'dog', then 'dog' again",
          "lines where 'blue' comes somewhere after 'red'",
          "lines containing 'a.b'"]:
    anon, lits = anonymize(s)
    print(f"{s}\\n   -> {anon.strip()!r}   literals={lits}")
print()
print("put back after decoding:", deanonymize(".*__w0__.*__w1__.*", ["red", "blue"]), "|", deanonymize(".*__w0__.*", ["a.b"]))
""")
md("### 6.2 Normalisation and tokenisation")
code("""
ex = raw["train"][3]
print("description :", ex["text"])
print("tokens      :", tokenize_src_veri(ex["text"]))
print("regex       :", ex["regex"])
print("slot regex  :", ex["regex_anon"])
print("tokens      :", tokenize_tgt(ex["regex_anon"]))
bad = [r for r in df.regex_anon if detokenize_tgt(tokenize_tgt(r)) != r]
print("tokenisation round-trip failures on all", len(df), "regexes:", len(bad))
""")
md("### 6.3 Vocabularies, encoding, padding and batching")
code(extract("data", keep=["Vocab", "build_veri", "collate", "batches"]))
code("""
def get_data(dataset="veriregex", seed=42, **_):
    return build_veri(seed)

data = get_data()
src_vocab, tgt_vocab = data["src_vocab"], data["tgt_vocab"]
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in data["stats"].items()}, indent=1))
print("\\nspecial tokens:", src_vocab.itos[:4], "| first words:", src_vocab.itos[4:20])
print("target vocabulary (%d):" % len(tgt_vocab), tgt_vocab.itos)
r = data["splits"]["train"][3]
print("\\nencoded source:", r["src_ids"])
print("encoded target:", r["tgt_ids"], "->", tgt_vocab.decode(r["tgt_ids"]))
""")
code("""
# a padded batch, as fed to the model
batch_rows, src, src_len, tgt = next(batches(data["splits"]["train"][:8], 4))
print("src", tuple(src.shape), "src_len", src_len.tolist(), "tgt", tuple(tgt.shape))
print(src)
print("padding share in shuffled, length-bucketed training batches:",
      round(float(np.mean([(t == PAD_ID).float().mean().item() for _, _, _, t in batches(data["splits"]["train"], 64, shuffle=True, rng=random.Random(0))])), 3),
      "vs. unbucketed:",
      round(float(np.mean([(t == PAD_ID).float().mean().item() for _, _, _, t in batches(data["splits"]["train"], 64, shuffle=True, rng=random.Random(0), bucket=False)])), 3))
""")

# =====================================================================================================
md("""
## 7. Model

- **Encoder:** `Embedding(198, 128)` -> dropout -> `LSTM(128 -> 256)`, packed sequences.
- **Bridge:** the encoder's final $(h, c)$, each through a linear layer and $\\tanh$, initialise the decoder.
- **Decoder, baseline:** `Embedding(44, 128)` -> `LSTM(256)` -> `Linear(256 -> 44)` -> softmax. Its only link to the input is the initial state: one fixed-size vector must summarise the whole description.
- **Decoder with attention (Bahdanau):** at each step $e_{t,i} = v^\\top \\tanh(W_s s_t + W_h h_i)$, $\\alpha_t = \\mathrm{softmax}(e_t)$ (padding masked), context $c_t = \\sum_i \\alpha_{t,i} h_i$. The context is fed to the LSTM with the previous token, and the output layer sees $[s_t; c_t; \\mathrm{emb}(y_{t-1})]$.
- **Training:** teacher forcing (the gold previous token is fed). **Inference:** greedy arg-max or beam search, stopping at `<EOS>` or after 60 tokens.
""")
code(extract("models"))
code("""
def param_table(model):
    rows = [{"module": n, "shape": tuple(p.shape), "parameters": p.numel()} for n, p in model.named_parameters()]
    return pd.DataFrame(rows)

for att in (False, True):
    m = Seq2Seq(len(src_vocab), len(tgt_vocab), 128, 256, 1, "lstm", False, att, 0.3, 0.05)
    t_ = param_table(m)
    print(("WITH" if att else "WITHOUT") + f" attention: {t_.parameters.sum():,} parameters")
    t_["block"] = t_.module.str.split(".").str[0] + "." + t_.module.str.split(".").str[1]
    display(t_.groupby("block", sort=False).parameters.sum().to_frame().T)
""")

# =====================================================================================================
md("""
## 8. Training

Same procedure as `src/train.py`: token-level cross-entropy (padding ignored), Adam (learning rate 1e-3, halved when the validation loss stops improving for 2 epochs), gradient clipping at 1.0, batch size 64, dropout 0.3, word dropout 0.05, teacher forcing ratio 1.0. After each epoch we compute validation loss, perplexity and greedy exact match; the checkpoint with the best validation exact match is kept, with early stopping.

- **Baseline (no attention):** up to 30 epochs, patience 6 (it is still improving at 15 epochs).
- **Attention:** up to 15 epochs, patience 4.
""")
code(extract("evaluate", keep=["predict", "corpus_bleu", "teacher_forced_stats", "PROBE_WORDS", "probe_form",
                               "evaluate_split", "full_report"]))
code('''
def train_model(name, attention, epochs, patience, seed=1, dropout=0.3, word_dropout=0.05, batch=64, lr=1e-3, clip=1.0):
    """Training loop of src/train.py (same seeds and order of random draws, so the results are reproducible)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    train, val = data["splits"]["train"], data["splits"]["val"]
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), 128, 256, 1, "lstm", False, attention, dropout, word_dropout)
    run_dir = NB_OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    rng = random.Random(seed)
    history, best_em, best_epoch, bad, t_start = [], -1.0, 0, 0, time.time()
    print(f"[{name}] {sum(p.numel() for p in model.parameters()):,} parameters, {len(train)} training pairs")
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _, src, src_len, tgt in batches(train, batch, shuffle=True, rng=rng):
            logits = model(src, src_len, tgt, 1.0)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item(); n += 1
        vtf = teacher_forced_stats(model, val)
        vres, *_ = evaluate_split(model, val, tgt_vocab, "greedy", functional=False)
        sched.step(vtf["loss"])
        rec = {"epoch": epoch, "train_loss": tot / n, "val_loss": vtf["loss"], "val_ppl": vtf["perplexity"],
               "val_token_acc": vtf["token_acc"], "val_em": vres["exact_match"], "val_bleu": vres["bleu"],
               "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        improved = vres["exact_match"] > best_em
        if improved:
            best_em, best_epoch, bad = vres["exact_match"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "model_cfg": model.cfg, "epoch": epoch,
                        "data_seed": 42, "dataset": "veriregex"}, run_dir / "best.pt")
        else:
            bad += 1
        print(f"[{name}] ep {epoch:02d} train loss {rec['train_loss']:.4f} | val loss {rec['val_loss']:.4f} "
              f"ppl {rec['val_ppl']:.3f} tok-acc {rec['val_token_acc']:.4f} EM {rec['val_em']:.4f} "
              f"BLEU {rec['val_bleu']:.1f} | {rec['sec']:.0f}s{'  *' if improved else ''}", flush=True)
        if bad >= patience:
            print(f"[{name}] early stop: no validation exact-match gain for {patience} epochs")
            break
    summary = {"best_epoch": best_epoch, "best_val_em": best_em, "epochs_run": len(history),
               "train_minutes": round((time.time() - t_start) / 60, 1)}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name}] done: {summary}")
    return history


def load_run(path):
    ck = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    m = Seq2Seq(**ck["model_cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, json.loads((path / "history.json").read_text()), ck["epoch"]
''')
code("""
RUNS = {"baseline": dict(attention=False, epochs=30, patience=6, shipped="base_long"),
        "attention": dict(attention=True, epochs=15, patience=4, shipped="attn")}
models, histories, best_epochs = {}, {}, {}
for name, cfg in RUNS.items():
    if RETRAIN:
        train_model(name, cfg["attention"], cfg["epochs"], cfg["patience"])
        path = NB_OUT / name
    else:
        path = PROJECT / "results" / cfg["shipped"]
    models[name], histories[name], best_epochs[name] = load_run(path)
    print(f"{name}: best epoch {best_epochs[name]}\\n")
""")
code("""
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
colors = {"baseline": "#9A5B00", "attention": "#0B6465"}
for name, h in histories.items():
    ep = [x["epoch"] for x in h]
    ax[0].plot(ep, [x["train_loss"] for x in h], "--", color=colors[name], label=f"{name} train")
    ax[0].plot(ep, [x["val_loss"] for x in h], "-", color=colors[name], label=f"{name} validation")
    ax[1].plot(ep, [100 * x["val_em"] for x in h], "o-", ms=3, color=colors[name], label=name)
    ax[2].plot(ep, [x["val_bleu"] for x in h], "o-", ms=3, color=colors[name], label=name)
ax[0].set_yscale("log"); ax[0].set_title("Loss (log scale)"); ax[1].set_title("Validation exact match (%)"); ax[2].set_title("Validation BLEU-4")
for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(NB_OUT / "learning_curves.png", dpi=130); plt.show()
for name, h in histories.items():
    first = next((x["epoch"] for x in h if x["val_em"] >= 0.88), None)
    print(f"{name}: validation EM after epoch 1 = {100 * h[0]['val_em']:.1f}%, first epoch >= 88%: {first}, "
          f"best = {100 * max(x['val_em'] for x in h):.2f}%")
""")
md("""**Reading.** The attention model is near its best after very few epochs, while the baseline improves slowly for about 30 epochs. Training and validation losses stay close (validation is even lower because dropout is only active in training), so there is no sign of overfitting on this data.""")

# =====================================================================================================
md("""
## 9. Evaluation

Three test sets, all with regexes the models have not been trained on (except 46 of the 50 handwritten ones):

- **New regexes** (1,884): new combinations and parameters, written in the training style.
- **New phrasings** (1,500): 11 sentence patterns held out of training.
- **Handwritten** (50): free wording.

Metrics: **exact match** (identical regex, in slot form), **same behaviour** (the generated regex accepts and rejects the same ~600 probe strings as the reference; a sampling test, not a proof), **BLEU-4** on regex tokens, **valid-regex rate**, and teacher-forced **perplexity** and **token accuracy**.
""")
code("""
TESTS = {"test": "New regexes", "test_phrasing": "New phrasings", "test_handwritten": "Handwritten"}
reports, per_example = {}, {}
for name, m in models.items():
    for split in TESTS:
        res, pe = full_report(m, data, split, "greedy")
        reports[(name, split, "greedy")] = res
        per_example[(name, split)] = pe
rows = []
for (name, split, mode), r in reports.items():
    rows.append({"model": name, "test set": TESTS[split], "n": r["n"], "exact match %": 100 * r["exact_match"],
                 "same behaviour %": 100 * r["functional_acc"], "BLEU-4": r["bleu"], "valid %": 100 * r["valid_rate"],
                 "perplexity": r["perplexity"], "token acc %": 100 * r["token_acc"]})
results = pd.DataFrame(rows)
display(results.round(2))
""")
code("""
kind_rows = []
for name in models:
    r = reports[(name, "test", "greedy")]
    for k, label in [("atom", "one condition"), ("not", "negated condition"), ("and", "AND of 2 or 3"), ("or", "OR of 2")]:
        if f"em_kind_{k}" in r:
            kind_rows.append({"model": name, "kind": label, "n": r[f"n_kind_{k}"], "exact match %": 100 * r[f"em_kind_{k}"]})
kt = pd.DataFrame(kind_rows).pivot(index="kind", columns="model", values="exact match %").round(1)
display(kt)
ax = results.pivot(index="test set", columns="model", values="exact match %").reindex(TESTS.values()).plot(
    kind="bar", figsize=(8, 3.8), color=["#0B6465", "#9A5B00"], rot=0)
ax.set_ylabel("exact match (%)"); ax.set_ylim(60, 100); ax.set_title("Exact match per test set")
plt.tight_layout(); plt.show()
""")
md("### 9.1 Greedy versus beam search")
code("""
if RUN_BEAM:
    rows = []
    for name, m in models.items():
        for split in TESTS:
            t0 = time.time()
            res, _ = full_report(m, data, split, "beam", beam=5)
            g = reports[(name, split, "greedy")]
            rows.append({"model": name, "test set": TESTS[split], "greedy EM %": 100 * g["exact_match"],
                         "beam-5 EM %": 100 * res["exact_match"], "greedy same %": 100 * g["functional_acc"],
                         "beam-5 same %": 100 * res["functional_acc"], "beam seconds": round(time.time() - t0)})
    display(pd.DataFrame(rows).round(1))
""")
md("""**Reading.** Beam search barely changes the results: the models, especially the attention one, are so confident that the best sequence is almost always the greedy one. (Our beam search stops only when no open hypothesis can still beat the best finished one; a first version that stopped as soon as five hypotheses had ended discarded good outputs.)""")

# =====================================================================================================
md("## 10. Analysis\n\n### 10.1 Attention heatmaps")
code("""
def translate(model, text, return_attention=False):
    toks = tokenize_src_veri(text)
    lits = anonymize(text)[1]
    row = {"src_ids": src_vocab.encode(toks[:MAX_SRC_LEN]), "tgt_ids": [SOS_ID, EOS_ID]}
    src, src_len, _ = collate([row])
    res = model.greedy_decode(src, src_len, return_attention=return_attention)
    ids, att = res if isinstance(res, tuple) else (res, None)
    out = tgt_vocab.decode(ids[0].tolist())
    rx = deanonymize(detokenize_tgt(out), lits)
    return (rx, toks, out, att[0, :len(out)].numpy(), lits) if return_attention else rx

texts = ["lines containing 'cat' at least 3 times", "lines where a vowel comes before a digit",
         "lines that do not start with 'the' and end with a number", "strings with 'blue' somewhere after 'red'"]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, text in zip(axes.ravel(), texts):
    rx, toks, out, a, lits = translate(models["attention"], text, return_attention=True)
    show = lambda t: f"'{lits[int(t[3])]}'" if re.fullmatch(r"__w\\d__", t) else t
    ax.imshow(a, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(toks))); ax.set_xticklabels([show(t) for t in toks], rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(out))); ax.set_yticklabels([show(t) for t in out], fontsize=8)
    ax.set_title(f"{text}\\n-> {rx}", fontsize=10); ax.grid(False)
plt.tight_layout(); plt.savefig(NB_OUT / "attention_examples.png", dpi=130); plt.show()
""")
md("""**Reading.** Each generated token attends to the words that justify it: counts to the number in the sentence, each slot to its quoted literal, `~` to "not", `&` to "and", `[AEIOUaeiou]` to "vowel". For "'blue' somewhere after 'red'" the regex must start with `red`, and the decoder indeed attends to the last word of the sentence first: a reordering that a single summary vector cannot do well.""")
md("### 10.2 Generated examples")
code("""
def show_examples(name, split, kind, k=6, seed=0):
    pe = per_example[(name, split)]
    if kind == "correct":
        sel = [x for x in pe if x["em"]]
    elif kind == "equivalent":
        sel = [x for x in pe if x["functional"] and not x["em"]]
    else:
        sel = [x for x in pe if not x["functional"]]
    sel = random.Random(seed).sample(sel, min(k, len(sel)))
    rows = []
    for x in sel:
        lits = anonymize(x["src"])[1]
        rows.append({"description": x["src"], "reference": deanonymize(x["ref"], lits), "generated": deanonymize(x["pred"], lits)})
    return pd.DataFrame(rows)

for split in TESTS:
    display(Markdown(f"**{TESTS[split]}: correct (attention model)**"))
    display(show_examples("attention", split, "correct", 4))
    display(Markdown(f"**{TESTS[split]}: wrong (attention model)**"))
    display(show_examples("attention", split, "wrong", 5))
""")
md("### 10.3 Error analysis")
code("""
def error_type(x):
    p, g = x["pred"], x["ref"]
    if not x["valid"]:
        return "invalid regex"
    if re.sub(r"\\d", "N", p) == re.sub(r"\\d", "N", g):
        return "wrong number only"
    if re.sub(r"__w\\d__", "W", p) == re.sub(r"__w\\d__", "W", g):
        return "literals swapped"
    if p.count("~") != g.count("~"):
        return "negation missing or extra"
    if ("&" in p) != ("&" in g) or ("|" in p) != ("|" in g):
        return "wrong connective (and/or)"
    if re.sub(r"\\[[^\\]]*\\]", "C", p) == re.sub(r"\\[[^\\]]*\\]", "C", g):
        return "wrong character class"
    return "other structure error"

rows = []
for (name, split), pe in per_example.items():
    c = Counter(error_type(x) for x in pe if not x["functional"])
    c["exact"] = sum(x["em"] for x in pe)
    c["equivalent, written differently"] = sum(1 for x in pe if x["functional"] and not x["em"])
    rows.append({"model": name, "test set": TESTS[split], **c})
cols = ["exact", "equivalent, written differently", "other structure error", "negation missing or extra",
        "wrong connective (and/or)", "wrong character class", "wrong number only", "literals swapped", "invalid regex"]
err = pd.DataFrame(rows).reindex(columns=["model", "test set"] + cols).fillna(0).set_index(["model", "test set"]).astype(int)
display(err)
""")
md("""**Reading.** Almost every output is a valid regex: the syntax is learned. Errors are about meaning. On new regexes the attention model makes only a handful of mistakes, most of them on one structure absent from training ("ends with X or Y", where the second ending becomes "contains") and on outputs cut at the 60-token limit. On held-out phrasings the dominant errors are a condition read as another one (for example the held-out "have X at the end" read as "starts with" or "contains") and flipped negations ("no fewer than N" read as "fewer than N"): the model has learned constructions rather than the meaning of each word independently.""")

# =====================================================================================================
md("## 11. Interactive demo\n\nPut literal text in quotes. The regex must match the whole line; `&` means and, `~` means not.")
code("""
def demo(text, lines):
    print("English :", text)
    for name in ("attention", "baseline"):
        rx = translate(models[name], text)
        ast_ = compile_regex(rx)
        res = ["MATCH" if (ast_ and full_match(ast_, l)) else "no" for l in lines]
        print(f"  {name:9s}: {rx:40s}", " ".join(f"{l!r}:{r}" for l, r in zip(lines, res)))
    print()

demo("lines containing 'cat' at least 2 times", ["cat", "cat and cat", "catcat"])
demo("lines containing 'cat' 2 times in a row", ["cat", "cat and cat", "catcat"])
demo("lines that start with a capital letter and end with a number", ["Hello 5", "hello 5", "Abc"])
demo("lines that contain 'error' and do not contain 'warning'", ["error 42", "warning: error", "all good"])
demo("lines where 'price' comes before 'total'", ["price 10, total 12", "total then price"])
demo("lines made only of digits", ["2024", "20a4", ""])
demo("lines with no digits at all", ["hello", "abc1"])                  # a known failure
demo("lines that contain 'x' no fewer than 3 times", ["xxx", "xx"])     # a known failure
""")
code("""
# try your own description here
demo("lines that begin with 'Dear' or end with '!'", ["Dear Sir", "Hello!", "Hi."])
""")

# =====================================================================================================
md("""
## 12. Discussion and conclusion

**Does the model learn the task?** On the distribution it was trained on, yes: the attention model generates almost all regexes it has never seen exactly right, with valid syntax and near-perfect BLEU. Attention is the decisive component for combinations of conditions: the baseline, even trained twice as long, makes several times more errors on new regexes, concentrated in AND combinations where one vector has to carry two or three conditions with their literals and numbers. Attention also learns several times faster.

**Where it fails.** New wording. On sentence patterns held out of training, exact match drops by about 17 points, and here the long-trained baseline is close behind: what limits both models is the variety of the language they were trained on, not the architecture. The errors are interpretable (a held-out construction mapped to a known one, negation words in unfamiliar constructions).

**Strengths.** Small model (1.4M parameters) trained on a CPU, interpretable attention, no pretrained component, any literal word thanks to slots, and labels that are verified rather than assumed.

**Limitations.** The English is generated from templates, so the in-distribution score is optimistic; the held-out-phrasing and handwritten tests are the honest estimates, and the handwritten set is small (50). Behavioural equivalence is a sampling test (we found one false "equivalent" in the project analysis). Outputs are capped at 60 tokens.

**Possible improvements.** Human paraphrases of the generated sentences to diversify the training language; more constructions per condition, especially for negation; a copy mechanism for unquoted literals; grammar-constrained decoding; exact DFA-based equivalence.

**Conclusion.** A from-scratch LSTM encoder-decoder with additive attention translates English descriptions into regular expressions. Training it on verified labels, rather than on a benchmark whose labels sometimes contradict their English, turns a 45% exact-match system into one that is nearly perfect on new regexes and solid on new wording and handwritten requests. The main lessons: check the labels before trusting a metric, and always evaluate outside the distribution the data generator produces.

The full technical report (`results/REPORT.pdf`), the slides and an in-browser demo (`site/`) accompany this notebook.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"authors": [{"name": "Mohamed Ali Aoun"}], "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "NL2Regex_notebook.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
