"""
Small regex engine for the NL-RX dialect, used to measure FUNCTIONAL correctness.

NL-RX regexes are not plain Python regexes: they also use intersection (A&B) and
complement (~A). Python's `re` cannot evaluate them, so we parse them ourselves and
match with a memoised "set of end positions" algorithm.

Grammar (lowest to highest precedence):
    alt     := cat ('|' cat)*
    cat     := inter+
    inter   := unary ('&' unary)*
    unary   := '~' unary | postfix
    postfix := atom ('*' | '+' | '{n}' | '{n,}' | '{n,m}')*
    atom    := '(' alt ')' | '[' class ']' | '.' | '\\b' | literal

Matching semantics: the regex must match the WHOLE line (full match), as in the
NL-RX paper where equivalence is defined on the language of the regex.
"""
import random
import re
import sys

sys.setrecursionlimit(5000)


class RegexParseError(Exception):
    pass


# --------------------------------------------------------------------------- parser
class _Parser:
    def __init__(self, text):
        self.t = text
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, c=None):
        ch = self.peek()
        if ch is None or (c is not None and ch != c):
            raise RegexParseError(f"expected {c!r} at {self.i} in {self.t!r}")
        self.i += 1
        return ch

    def parse(self):
        node = self.alt()
        if self.i != len(self.t):
            raise RegexParseError(f"trailing input at {self.i} in {self.t!r}")
        return node

    def alt(self):
        parts = [self.cat()]
        while self.peek() == "|":
            self.eat("|")
            parts.append(self.cat())
        return parts[0] if len(parts) == 1 else ("alt", parts)

    def cat(self):
        parts = []
        while self.peek() is not None and self.peek() not in "|)":
            parts.append(self.inter())
        if not parts:
            return ("eps",)
        return parts[0] if len(parts) == 1 else ("cat", parts)

    def inter(self):
        parts = [self.unary()]
        while self.peek() == "&":
            self.eat("&")
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def unary(self):
        if self.peek() == "~":
            self.eat("~")
            return ("not", self.unary())
        return self.postfix()

    def postfix(self):
        node = self.atom()
        while True:
            c = self.peek()
            if c == "*":
                self.eat()
                node = ("rep", node, 0, None)
            elif c == "+":
                self.eat()
                node = ("rep", node, 1, None)
            elif c == "{":
                self.eat("{")
                lo = self._int()
                hi = lo
                if self.peek() == ",":
                    self.eat(",")
                    hi = self._int() if self.peek() != "}" else None
                self.eat("}")
                node = ("rep", node, lo, hi)
            else:
                return node

    def _int(self):
        s = ""
        while self.peek() is not None and self.peek().isdigit():
            s += self.eat()
        if not s:
            raise RegexParseError("integer expected")
        return int(s)

    def atom(self):
        c = self.peek()
        if c is None:
            raise RegexParseError("unexpected end")
        if c == "(":
            self.eat("(")
            node = self.alt()
            self.eat(")")
            return node
        if c == "[":
            return self.char_class()
        if c == ".":
            self.eat()
            return ("any",)
        if c == "\\":
            self.eat()
            if self.eat("b") != "b":
                raise RegexParseError("only \\b is supported")
            return ("wb",)
        if c in ")*+{}|&~]":
            raise RegexParseError(f"unexpected {c!r} at {self.i}")
        self.eat()
        return ("lit", c)

    def char_class(self):
        self.eat("[")
        chars = set()
        ranges = []
        first = True
        while self.peek() != "]":
            if self.peek() is None:
                raise RegexParseError("unterminated class")
            a = self.eat()
            if self.peek() == "-" and self.t[self.i + 1:self.i + 2] not in ("]", ""):
                self.eat("-")
                b = self.eat()
                ranges.append((a, b))
            else:
                chars.add(a)
            first = False
        self.eat("]")
        if first:
            raise RegexParseError("empty class")
        return ("cls", frozenset(chars), tuple(ranges))


def parse(text):
    return _Parser(text).parse()


# -------------------------------------------------------------------------- matcher
def _is_word(c):
    return c.isalnum() or c == "_"


class _Matcher:
    def __init__(self, s):
        self.s = s
        self.n = len(s)
        self.cache = {}

    def ends(self, node, i):
        key = (id(node), i)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        res = self._ends(node, i)
        self.cache[key] = res
        return res

    def _step(self, node, positions):
        out = set()
        for j in positions:
            out |= self.ends(node, j)
        return out

    def _ends(self, node, i):
        kind = node[0]
        s, n = self.s, self.n
        if kind == "lit":
            return frozenset([i + 1]) if i < n and s[i] == node[1] else frozenset()
        if kind == "any":
            return frozenset([i + 1]) if i < n else frozenset()
        if kind == "cls":
            if i < n:
                c = s[i]
                if c in node[1] or any(a <= c <= b for a, b in node[2]):
                    return frozenset([i + 1])
            return frozenset()
        if kind == "eps":
            return frozenset([i])
        if kind == "wb":
            left = _is_word(s[i - 1]) if i > 0 else False
            right = _is_word(s[i]) if i < n else False
            return frozenset([i]) if left != right else frozenset()
        if kind == "cat":
            cur = {i}
            for child in node[1]:
                cur = self._step(child, cur)
                if not cur:
                    break
            return frozenset(cur)
        if kind == "alt":
            out = set()
            for child in node[1]:
                out |= self.ends(child, i)
            return frozenset(out)
        if kind == "and":
            out = None
            for child in node[1]:
                e = self.ends(child, i)
                out = set(e) if out is None else out & e
                if not out:
                    break
            return frozenset(out)
        if kind == "not":
            inner = self.ends(node[1], i)
            return frozenset(j for j in range(i, n + 1) if j not in inner)
        if kind == "rep":
            _, child, lo, hi = node
            cur = {i}
            for _ in range(lo):
                cur = self._step(child, cur)
                if not cur:
                    return frozenset()
            if hi is None:
                seen = set(cur)
                frontier = set(cur)
                while frontier:
                    nxt = self._step(child, frontier) - seen
                    seen |= nxt
                    frontier = nxt
                return frozenset(seen)
            result = set(cur)
            for _ in range(lo, hi):
                cur = self._step(child, cur)
                if not cur:
                    break
                result |= cur
            return frozenset(result)
        raise RegexParseError(f"unknown node {kind}")


def full_match(ast, s):
    return len(s) in _Matcher(s).ends(ast, 0)


def compile_regex(text):
    """Return AST or None if the text is not a valid NL-RX regex."""
    try:
        return parse(text)
    except (RegexParseError, IndexError, RecursionError):
        return None


# ------------------------------------------------------- functional-equivalence test
_POOL = ["dog", "truck", "ring", "lake", "a", "b", "e", "i", "o", "u", "A", "E", "U",
         "Q", "z", "Z", "x", "0", "1", "5", "9", " ", ".", "-", "Dog", "g"]


def make_test_strings(n=600, seed=0):
    """Deterministic probe strings used to compare two regexes behaviourally."""
    rng = random.Random(seed)
    out = {""}
    # repeated tokens so that counters like {5,} or {7,} can be exercised
    for tok in _POOL:
        for r in (2, 5, 8):
            out.add(tok * r)
    while len(out) < n:
        k = rng.randint(1, 9)
        out.add("".join(rng.choice(_POOL) for _ in range(k)))
    return sorted(out)


def signature(text, strings):
    """Tuple of match booleans on `strings`, or None if `text` does not parse."""
    ast = compile_regex(text)
    if ast is None:
        return None
    try:
        return tuple(full_match(ast, s) for s in strings)
    except RecursionError:
        return None


def functionally_equal(pred, ref_signature, strings):
    sig = signature(pred, strings)
    return sig is not None and sig == ref_signature


if __name__ == "__main__":
    tests = [
        (r".*(dog){2,}.*", "dogdog", True),
        (r".*(dog){2,}.*", "dog", False),
        (r"([A-Z].*[0-9].*)&(~(dog))", "A1", True),
        (r"~(dog)", "dog", False),
        (r"~(dog)", "dogs", True),
        (r"\b([A-Z])&([AEIOUaeiou])\b", "E", True),
        (r"\b([A-Z])&([AEIOUaeiou])\b", "A1", False),
        (r"~(((dog)(.*))&([AEIOUaeiou]))", "dog", True),
        (r"(.*)((dog.*[A-Z].*)|(truck))", "xdogA", True),
    ]
    for rx, s, want in tests:
        got = full_match(parse(rx), s)
        print("OK " if got == want else "BAD", rx, repr(s), got)
