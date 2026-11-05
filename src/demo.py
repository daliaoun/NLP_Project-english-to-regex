"""
Live demo: type an English description, get a regular expression, and test it on your own lines.

    python src/demo.py --run results/attn                          # interactive
    python src/demo.py --run results/attn --text "lines containing 'cat' at least 2 times" \
                       --lines "cat" "cat and cat" "catcat"
    python src/demo.py --run results/attn --showcase               # prepared list of examples

Interactive commands:  just type a description | ':lines a|b|c' sets the lines to test | ':att' saves a heatmap | ':q'

Put literal text in quotes: 'cat', "2024", 'ice cream'. A regex must match the WHOLE line.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import anonymize, collate, deanonymize, get_data, tokenize_src, tokenize_src_veri  # noqa: E402
from evaluate import load_model  # noqa: E402
from regex_engine import compile_regex, full_match  # noqa: E402

SHOWCASE = [
    "lines containing 'cat' at least 2 times",
    "lines containing 'cat' 2 times in a row",
    "lines that start with a capital letter and end with a number",
    "lines with a vowel before a digit",
    "lines that do not contain 'error'",
    "lines containing either 'yes' or 'no'",
]
DEFAULT_LINES = ["cat", "cat and cat", "catcat", "A1", "Hello 5", "a9", "yes", "error 42", "Zebra", "7up"]


class Translator:
    def __init__(self, run_dir):
        self.model, ck = load_model(run_dir)
        self.veri = ck.get("dataset", "NL-RX-Turk") == "veriregex"
        d = get_data(ck.get("dataset", "NL-RX-Turk"), seed=ck["data_seed"], augment_synth=ck.get("augment_synth", False))
        self.src_vocab, self.tgt_vocab = d["src_vocab"], d["tgt_vocab"]

    def _row(self, text):
        toks = tokenize_src_veri(text) if self.veri else tokenize_src(text)
        return {"src_tok": toks, "src_ids": self.src_vocab.encode(toks), "tgt_ids": [2, 3]}, toks

    def translate(self, text, beam=0):
        rx, extra, toks, unk = self._translate(text, beam)
        if self.veri:
            rx = deanonymize(rx, anonymize(text)[1])
        return rx, extra, toks, unk

    def _translate(self, text, beam=0):
        row, toks = self._row(text)
        src, src_len, _ = collate([row])
        unk = [t for t, i in zip(toks, row["src_ids"]) if i == 1]
        if beam:
            ids = self.model.beam_search(src, src_len, beam=beam)
            return "".join(self.tgt_vocab.decode(ids)), None, toks, unk
        res = self.model.greedy_decode(src, src_len, return_attention=self.model.decoder.use_attention)
        tokens, att = res if isinstance(res, tuple) else (res, None)
        out = self.tgt_vocab.decode(tokens[0].tolist())
        if att is not None:
            att = att[0, :len(out) + 1].cpu().numpy()
        return "".join(out), (out, att), toks, unk


def heatmap(tr, text, path):
    rx, extra, toks, _ = tr.translate(text)
    if extra is None or extra[1] is None:
        print("(this model has no attention)")
        return
    out, att = extra
    att = att[:len(out)]
    plt.figure(figsize=(max(6, 0.55 * len(toks)), max(4, 0.28 * len(out))))
    plt.imshow(att, aspect="auto", cmap="viridis")
    plt.xticks(range(len(toks)), toks, rotation=60, ha="right")
    plt.yticks(range(len(out)), out)
    plt.title(text[:80])
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"attention heatmap saved to {path}")


def show(tr, text, lines, beam=5):
    greedy, _, toks, unk = tr.translate(text)
    print(f"\n  English : {text}")
    print(f"  Regex   : {greedy}    (greedy)")
    if beam:
        b, *_ = tr.translate(text, beam=beam)
        if b != greedy:
            print(f"            {b}    (beam {beam})")
    if unk:
        print(f"  warning : unknown words ignored by the model: {unk}")
    ast = compile_regex(greedy)
    if ast is None:
        print("  (the generated string is not a valid regex)")
        return
    print("  Test on lines:")
    for ln in lines:
        print(f"     {'MATCH   ' if full_match(ast, ln) else 'no match'}  {ln!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/attn")
    ap.add_argument("--text")
    ap.add_argument("--lines", nargs="*")
    ap.add_argument("--showcase", action="store_true")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--heatmap", help="save an attention heatmap for --text to this png path")
    a = ap.parse_args()
    torch.set_num_threads(1)
    tr = Translator(a.run)
    lines = a.lines or DEFAULT_LINES
    if a.text:
        show(tr, a.text, lines, a.beam)
        if a.heatmap:
            heatmap(tr, a.text, a.heatmap)
        return
    if a.showcase:
        for t in SHOWCASE:
            show(tr, t, lines, a.beam)
        return
    print(__doc__)
    last = None
    while True:
        try:
            s = input("\nEnglish> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s:
            continue
        if s in (":q", ":quit"):
            break
        if s.startswith(":lines"):
            lines = [x for x in s[6:].strip().split("|") if x != ""] or lines
            print("lines set to", lines)
            continue
        if s == ":att" and last:
            heatmap(tr, last, "attention.png")
            continue
        last = s
        show(tr, s, lines, a.beam)


if __name__ == "__main__":
    main()
