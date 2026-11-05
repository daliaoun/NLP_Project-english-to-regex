"""
Export the trained PyTorch models to a static format that the website reads in the browser.

    python tools/export_site_data.py

Writes into site/data/:
    models.json       configs, tensor manifests (name, shape, byte offset) and vocabularies of both models
    attention.bin     float32 weights of the best attention model, tensors concatenated in manifest order
    baseline.bin      float32 weights of the baseline model
    testset.json      the held-out test examples (3 test splits) with the reference regex and the Python
                      predictions of both models (used by the "random test example" feature and parity tests)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import anonymize, deanonymize, get_data  # noqa: E402

OUT = ROOT / "site" / "data"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    # key: (run folder, label shown in the interface)
    "attention": ("attn", "LSTM + attention (best model)"),
    "baseline": ("base_long", "LSTM baseline (no attention, 30 epochs)"),
}

manifest = {}
for key, (run, label) in MODELS.items():
    ck = torch.load(ROOT / "results" / run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["model_cfg"]
    assert cfg["rnn"] == "lstm" and cfg["layers"] == 1 and not cfg["bidirectional"], cfg
    data = get_data(ck.get("dataset", "veriregex"), seed=ck["data_seed"])
    assert len(data["src_vocab"]) == cfg["src_vocab"] and len(data["tgt_vocab"]) == cfg["tgt_vocab"]

    tensors, chunks, offset = [], [], 0
    for name, t in ck["state_dict"].items():
        arr = t.detach().cpu().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset, "length": int(arr.size)})
        chunks.append(arr.reshape(-1))
        offset += arr.size
    blob = np.concatenate(chunks)
    (OUT / f"{key}.bin").write_bytes(blob.tobytes())
    manifest[key] = {
        "label": label, "run": run, "epoch": ck["epoch"],
        "config": {k: cfg[k] for k in ("emb_dim", "hid_dim", "layers", "attention", "src_vocab", "tgt_vocab")},
        "n_params": int(blob.size), "file": f"{key}.bin", "tensors": tensors,
        "src_vocab": data["src_vocab"].itos, "tgt_vocab": data["tgt_vocab"].itos,
    }
    print(f"{key}: run={run} epoch={ck['epoch']} params={blob.size:,} bytes={blob.nbytes:,}")

(OUT / "models.json").write_text(json.dumps(manifest, separators=(",", ":")))

# ---- held-out test sets with the Python predictions of both models (greedy)
def load_preds(run, split):
    p = ROOT / "results" / run / f"{split}_greedy_predictions.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


rows = []
for split, code in (("test", "t"), ("test_phrasing", "p"), ("test_handwritten", "h")):
    a = load_preds(MODELS["attention"][0], split)
    b = load_preds(MODELS["baseline"][0], split)
    assert len(a) == len(b) and all(x["src"] == y["src"] and x["ref"] == y["ref"] for x, y in zip(a, b))
    for x, y in zip(a, b):
        lits = anonymize(x["src"])[1]
        rows.append({
            "k": code, "s": x["src"], "r": deanonymize(x["ref"], lits), "ra": x["ref"], "kind": x["kind"],
            "a": [x["pred"], int(x["em"]), int(bool(x["functional"]))],
            "b": [y["pred"], int(y["em"]), int(bool(y["functional"]))],
        })
(OUT / "testset.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
for code in "tph":
    rs = [r for r in rows if r["k"] == code]
    print("testset", code, len(rs), "attention EM %.3f" % (sum(r["a"][1] for r in rs) / len(rs)),
          "baseline EM %.3f" % (sum(r["b"][1] for r in rs) / len(rs)))
