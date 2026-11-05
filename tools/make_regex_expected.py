"""Expected matcher results from the Python engine, for tools/regex_parity.mjs.
    python tools/make_regex_expected.py out.json && node tools/regex_parity.mjs out.json
Uses the regexes of the VeriRegex test sets and the attention model's predictions (with the probe
words in place of slots), plus a few escaped literals, on the standard probe strings."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import probe_form  # noqa: E402
from regex_engine import compile_regex, full_match, make_test_strings  # noqa: E402

rxs = set()
for split in ("test", "test_phrasing", "test_handwritten"):
    for line in (ROOT / "data" / "veriregex" / f"{split}.jsonl").read_text().splitlines():
        rxs.add(json.loads(line)["regex"])
    p = ROOT / "results" / "attn" / f"{split}_greedy_predictions.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            rxs.add(probe_form(json.loads(line)["pred"]))
rxs |= {".*a[.]b.*", "[*]+", ".*[(]x[)].*", "(.*dog){2,}.*", ".{0,3}", "((a"}
strings = make_test_strings() + ["a.b", "a*b", "(x)", "***"]
cases = []
for rx in sorted(rxs):
    ast = compile_regex(rx)
    cases.append({"rx": rx, "valid": ast is not None,
                  "res": [int(full_match(ast, s)) for s in strings] if ast is not None else None})
Path(sys.argv[1]).write_text(json.dumps({"strings": strings, "cases": cases}))
print(len(cases), "regexes,", len(strings), "strings")
