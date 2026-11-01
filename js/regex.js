/*
 * Matcher for the NL-RX regex dialect, used to test generated regexes on lines of text.
 * Port of src/regex_engine.py (same grammar, same semantics).
 *
 * The dialect has two operators that ordinary regex engines lack:
 *     A&B   intersection (the same text matches A and B)
 *     ~A    complement   (any text that A does not match)
 * so the browser's RegExp cannot be used. A regex must match the WHOLE line.
 *
 * Grammar (lowest to highest precedence):
 *   alt     := cat ('|' cat)*
 *   cat     := inter+
 *   inter   := unary ('&' unary)*
 *   unary   := '~' unary | postfix
 *   postfix := atom ('*' | '+' | '{n}' | '{n,}' | '{n,m}')*
 *   atom    := '(' alt ')' | '[' class ']' | '.' | '\b' | literal
 */

export class RegexParseError extends Error {}

let nextId = 1;
const node = (kind, props = {}) => ({ id: nextId++, kind, ...props });

class Parser {
  constructor(text) {
    this.t = Array.from(text);
    this.i = 0;
  }
  peek() { return this.i < this.t.length ? this.t[this.i] : null; }
  eat(c = null) {
    const ch = this.peek();
    if (ch === null || (c !== null && ch !== c)) throw new RegexParseError(`expected ${c ?? "a character"} at ${this.i}`);
    this.i += 1;
    return ch;
  }
  parse() {
    const n = this.alt();
    if (this.i !== this.t.length) throw new RegexParseError(`unexpected input at ${this.i}`);
    return n;
  }
  alt() {
    const parts = [this.cat()];
    while (this.peek() === "|") { this.eat("|"); parts.push(this.cat()); }
    return parts.length === 1 ? parts[0] : node("alt", { parts });
  }
  cat() {
    const parts = [];
    while (this.peek() !== null && this.peek() !== "|" && this.peek() !== ")") parts.push(this.inter());
    if (parts.length === 0) return node("eps");
    return parts.length === 1 ? parts[0] : node("cat", { parts });
  }
  inter() {
    const parts = [this.unary()];
    while (this.peek() === "&") { this.eat("&"); parts.push(this.unary()); }
    return parts.length === 1 ? parts[0] : node("and", { parts });
  }
  unary() {
    if (this.peek() === "~") { this.eat("~"); return node("not", { child: this.unary() }); }
    return this.postfix();
  }
  postfix() {
    let n = this.atom();
    for (;;) {
      const c = this.peek();
      if (c === "*") { this.eat(); n = node("rep", { child: n, lo: 0, hi: null }); }
      else if (c === "+") { this.eat(); n = node("rep", { child: n, lo: 1, hi: null }); }
      else if (c === "{") {
        this.eat("{");
        const lo = this.int();
        let hi = lo;
        if (this.peek() === ",") {
          this.eat(",");
          hi = this.peek() !== "}" ? this.int() : null;
        }
        this.eat("}");
        n = node("rep", { child: n, lo, hi });
      } else return n;
    }
  }
  int() {
    let s = "";
    while (this.peek() !== null && /[0-9]/.test(this.peek())) s += this.eat();
    if (!s) throw new RegexParseError("integer expected");
    return parseInt(s, 10);
  }
  atom() {
    const c = this.peek();
    if (c === null) throw new RegexParseError("unexpected end");
    if (c === "(") { this.eat("("); const n = this.alt(); this.eat(")"); return n; }
    if (c === "[") return this.charClass();
    if (c === ".") { this.eat(); return node("any"); }
    if (c === "\\") {
      this.eat();
      if (this.eat("b") !== "b") throw new RegexParseError("only \\b is supported");
      return node("wb");
    }
    if (")*+{}|&~]".includes(c)) throw new RegexParseError(`unexpected ${c} at ${this.i}`);
    this.eat();
    return node("lit", { ch: c });
  }
  charClass() {
    this.eat("[");
    const chars = new Set();
    const ranges = [];
    let first = true;
    while (this.peek() !== "]") {
      if (this.peek() === null) throw new RegexParseError("unterminated class");
      const a = this.eat();
      const after = this.t[this.i + 1];
      if (this.peek() === "-" && after !== undefined && after !== "]") {
        this.eat("-");
        ranges.push([a, this.eat()]);
      } else chars.add(a);
      first = false;
    }
    this.eat("]");
    if (first) throw new RegexParseError("empty class");
    return node("cls", { chars, ranges });
  }
}

/** Parse a regex string. Returns an AST, or null when the text is not a valid NL-RX regex. */
export function compileRegex(text) {
  try {
    return new Parser(text).parse();
  } catch (e) {
    if (e instanceof RegexParseError || e instanceof RangeError) return null;
    throw e;
  }
}

const isWord = (c) => /[\p{L}\p{N}_]/u.test(c);
const EMPTY = new Set();

class Matcher {
  constructor(line) {
    this.s = Array.from(line);
    this.n = this.s.length;
    this.cache = new Map();
  }
  ends(nd, i) {
    const key = nd.id * (this.n + 2) + i;
    let r = this.cache.get(key);
    if (r === undefined) { r = this.compute(nd, i); this.cache.set(key, r); }
    return r;
  }
  step(nd, positions) {
    const out = new Set();
    for (const j of positions) for (const e of this.ends(nd, j)) out.add(e);
    return out;
  }
  compute(nd, i) {
    const { s, n } = this;
    switch (nd.kind) {
      case "lit": return i < n && s[i] === nd.ch ? new Set([i + 1]) : EMPTY;
      case "any": return i < n ? new Set([i + 1]) : EMPTY;
      case "cls": {
        if (i < n) {
          const c = s[i];
          if (nd.chars.has(c) || nd.ranges.some(([a, b]) => a <= c && c <= b)) return new Set([i + 1]);
        }
        return EMPTY;
      }
      case "eps": return new Set([i]);
      case "wb": {
        const left = i > 0 ? isWord(s[i - 1]) : false;
        const right = i < n ? isWord(s[i]) : false;
        return left !== right ? new Set([i]) : EMPTY;
      }
      case "cat": {
        let cur = new Set([i]);
        for (const child of nd.parts) {
          cur = this.step(child, cur);
          if (cur.size === 0) break;
        }
        return cur;
      }
      case "alt": {
        const out = new Set();
        for (const child of nd.parts) for (const e of this.ends(child, i)) out.add(e);
        return out;
      }
      case "and": {
        let out = null;
        for (const child of nd.parts) {
          const e = this.ends(child, i);
          out = out === null ? new Set(e) : new Set([...out].filter((x) => e.has(x)));
          if (out.size === 0) break;
        }
        return out;
      }
      case "not": {
        const inner = this.ends(nd.child, i);
        const out = new Set();
        for (let j = i; j <= n; j++) if (!inner.has(j)) out.add(j);
        return out;
      }
      case "rep": {
        const { child, lo, hi } = nd;
        let cur = new Set([i]);
        for (let k = 0; k < lo; k++) {
          cur = this.step(child, cur);
          if (cur.size === 0) return EMPTY;
        }
        if (hi === null) {
          const seen = new Set(cur);
          let frontier = new Set(cur);
          while (frontier.size) {
            const nxt = this.step(child, frontier);
            const fresh = new Set();
            for (const e of nxt) if (!seen.has(e)) { seen.add(e); fresh.add(e); }
            frontier = fresh;
          }
          return seen;
        }
        const result = new Set(cur);
        for (let k = lo; k < hi; k++) {
          cur = this.step(child, cur);
          if (cur.size === 0) break;
          for (const e of cur) result.add(e);
        }
        return result;
      }
      default: throw new RegexParseError(`unknown node ${nd.kind}`);
    }
  }
}

/** True when the whole line matches the compiled regex. */
export function fullMatch(ast, line) {
  const m = new Matcher(line);
  return m.ends(ast, 0).has(m.n);
}
