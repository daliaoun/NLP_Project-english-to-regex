// Prints what the browser models generate for candidate example sentences.
//   node tools/chip_check.mjs "sentence" ...
import { readFileSync } from "node:fs";
import { modelFromFiles } from "../site/js/model.js";
const root = new URL("../site/data/", import.meta.url);
const man = JSON.parse(readFileSync(new URL("models.json", root)));
const load = (k) => { const f = readFileSync(new URL(man[k].file, root)); return modelFromFiles(man[k], f.buffer.slice(f.byteOffset, f.byteOffset + f.byteLength)); };
const A = load("attention"), B = load("baseline");
for (const s of process.argv.slice(2)) {
  const a = A.translate(s), b = B.translate(s);
  console.log(JSON.stringify(s), "\n   attn:", a.regex, "\n   base:", b.regex, a.unknown.length ? "  UNK " + a.unknown : "");
}
