// Parity test: the JavaScript models must give the same regexes as the PyTorch models.
//   node tools/parity_test.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { modelFromFiles } from "../site/js/model.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "site", "data");
const manifest = JSON.parse(readFileSync(path.join(root, "models.json"), "utf8"));
const rows = JSON.parse(readFileSync(path.join(root, "testset.json"), "utf8"));

let failed = 0;
for (const [key, flag] of [["attention", "a"], ["baseline", "b"]]) {
  const file = readFileSync(path.join(root, manifest[key].file));
  const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const model = modelFromFiles(manifest[key], buf);
  let same = 0, em = 0;
  const diffs = [];
  const t0 = performance.now();
  for (const r of rows) {
    const out = model.translate(r.s).text;
    if (out === r[flag][0]) same++; else diffs.push({ src: r.s, js: out, py: r[flag][0] });
    if (out === r.ra) em++;
  }
  const ms = (performance.now() - t0) / rows.length;
  console.log(`${key}: JS == Python on ${same}/${rows.length} test examples; JS exact match ${(100 * em / rows.length).toFixed(1)}%; ${ms.toFixed(1)} ms per example`);
  for (const d of diffs.slice(0, 5)) console.log("   DIFF", JSON.stringify(d));
  if (same < rows.length) failed += rows.length - same;
}
process.exit(failed > 2 ? 1 : 0);   // tolerate float32-vs-float64 near ties
