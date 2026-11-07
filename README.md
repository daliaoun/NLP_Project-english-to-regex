# NL2Regex: English to regular expressions with an encoder-decoder (from scratch)

NLP course project (text generation with encoder-decoder architectures), Mohamed Ali Aoun.

A sequence-to-sequence model written in PyTorch, with no pretrained component, reads an English
description such as

    lines containing 'cat' at least 2 times

and generates a regular expression that matches exactly those lines:

    (.*cat){2,}.*

It is trained on **VeriRegex**, a dataset we generate and verify ourselves (see below), and runs in a
web page without any server (`site/`).

## Results (greedy decoding)

| Test set | n | Baseline LSTM, 30 epochs: exact / same behaviour (%) | LSTM + attention: exact / same behaviour (%) |
|---|---|---|---|
| New regexes (never seen in training) | 1,884 | 95.6 / 96.2 | **99.4 / 99.4** |
| New phrasings (sentence patterns held out of training) | 1,500 | 81.2 / 83.3 | **82.6 / 84.3** |
| Handwritten descriptions | 50 | 92.0 / 92.0 | **96.0 / 96.0** |

Details, analysis and limitations: `results/REPORT.md` (PDF: `results/REPORT.pdf`); two-page version: `results/REPORT_short.pdf`.

## Why our own dataset

The standard benchmark, NL-RX (Locascio et al., 2016), has labels that sometimes contradict their
English: "lines with the string 'dog' at least 2 times" is labelled `.*(dog){2,}.*`, which rejects
`dog and dog`. Our first version on NL-RX reached 45.3% exact match; it is archived in
`results/nlrx_v1.zip`. `src/generate.py` builds every example from one meaning tree, writes the English
(216 phrasing templates, 135 literal words) and the regex from it, and checks the regex against an
independent Python predicate on about 300 strings per example. Any disagreement stops the build.

## Notebook

`NL2Regex_notebook.ipynb` is the whole project in one executed notebook: dataset generation, exploratory
data analysis, preprocessing, model, training, evaluation, analysis and a demo. Its code cells are
extracted from `src/` (`python tools/build_notebook.py`), so it runs exactly the same code. With
`REGENERATE` and `RETRAIN` set to `True` it takes about 1.5 hours on a CPU; with `RETRAIN = False` it
loads the trained models from `results/` and runs in a few minutes.

## Layout

    src/generate.py      VeriRegex generator and verifier, handwritten test set
    src/data.py          preprocessing (literal slots __w0__, __w1__), vocabularies, batching
    src/models.py        Encoder, additive Attention, Decoder, Seq2Seq (greedy + beam search)
    src/train.py         training loop (teacher forcing, early stopping on validation exact match)
    src/evaluate.py      exact match, behavioural equivalence, BLEU-4, perplexity, token accuracy
    src/regex_engine.py  parser and whole-line matcher for the dialect (supports & and ~)
    src/analyze.py       tables, learning curves, attention heatmaps, error analysis
    src/demo.py          terminal demo: English in, regex out, test on your own lines
    data/veriregex/      the generated dataset (train, val, test, test_phrasing, test_handwritten)
    results/<run>/       config, history, best checkpoint, metrics and predictions per test set
    results/*.md|png     comparison table, curves, attention figure, error analysis, examples, report
    site/                static website (browser inference, report, slides)
    tools/               export to the website, JavaScript parity tests, slide and report builders

## Reproduce

    pip install torch matplotlib
    python src/generate.py                                     # about 5 minutes on 2 cores
    python src/train.py --name base_long --dropout 0.3 --word_dropout 0.05 --epochs 30 --patience 6
    python src/train.py --name attn --attention --dropout 0.3 --word_dropout 0.05 --epochs 15 --patience 4
    python src/evaluate.py results/attn --beam 5               # all three test sets
    python src/evaluate.py results/base_long
    python src/analyze.py base_long attn
    python src/demo.py --run results/attn

Training uses the CPU only (about 25 minutes for the attention model, 50 for the 30-epoch baseline).

## Website

`site/` runs both trained models in the browser, in plain JavaScript, with the report and slides
embedded. Deploy it as a static site (for example on Vercel, root directory `site`, no build command).
`python tools/export_site_data.py` regenerates its data, `node tools/parity_test.mjs` checks that the
JavaScript models give the same output as PyTorch on every test sentence, `node tools/regex_parity.mjs`
checks the JavaScript regex matcher, and `python tools/build_slides.py` rebuilds the slide viewer.
