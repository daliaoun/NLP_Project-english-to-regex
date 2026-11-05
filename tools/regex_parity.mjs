// Compares the JavaScript matcher with the Python matcher on regexes x strings (expected values come from Python).
//   node tools/regex_parity.mjs <expected.json>
import { readFileSync } from "node:fs";
import { compileRegex, fullMatch } from "../site/js/regex.js";
const { strings, cases } = JSON.parse(readFileSync(process.argv[2], "utf8"));
let bad = 0, checks = 0;
for (const c of cases) {
  const ast = compileRegex(c.rx);
  if (!c.valid) { if (ast !== null) { bad++; console.log("should be invalid:", JSON.stringify(c.rx)); } continue; }
  if (ast === null) { bad++; console.log("wrongly invalid:", JSON.stringify(c.rx)); continue; }
  strings.forEach((s, i) => {
    checks++;
    if (Number(fullMatch(ast, s)) !== c.res[i]) { bad++; if (bad < 8) console.log("MISMATCH", JSON.stringify(c.rx), JSON.stringify(s), "js", fullMatch(ast, s), "py", c.res[i]); }
  });
}
console.log(`${cases.length} regexes, ${checks} (regex, string) checks, mismatches: ${bad}`);
process.exit(bad ? 1 : 0);
