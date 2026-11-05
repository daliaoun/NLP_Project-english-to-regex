"""
VeriRegex: a generated, verified dataset of (English description, regular expression) pairs.

Why: in NL-RX the English and the regex sometimes disagree (for example "the string 'dog' at least
2 times" is labelled .*(dog){2,}.*, which only accepts two ADJACENT copies). Here every pair is built
from one meaning tree, and every label is checked by two independent implementations:

    meaning tree --> regular expression --(our regex matcher)--> accept / reject
    meaning tree --> Python predicate   ----------------------> accept / reject

The two must agree on every probe string (about 300 per example). A pair is kept only if they agree
everywhere and the probes contain both a matching and a non-matching line (no contradictions such as
"starts with a digit and starts with a letter", no tautologies).

Literals are real quoted words ('cat', "2024", 'ice cream'). For the model they are replaced by
placeholders __w0__, __w1__, ... (see data.py), so the model works with any word at test time.

    python src/generate.py            # writes data/veriregex/{train,val,test,test_phrasing,test_handwritten}.jsonl
"""
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regex_engine import full_match, parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "veriregex"

# ============================================================================ vocabulary of the world
WORDS = """cat dog sun car tree book river stone apple pizza coffee music water light house city
mouse horse train phone paper table chair window garden bridge island forest planet rocket robot
data code test error file user admin login email price order ticket report invoice sale offer
red blue green yellow black white orange purple pink gray gold silver
alpha beta gamma delta omega zero one two ten hello world yes no ok foo bar baz qux
tunis paris london rome cairo tokyo madrid berlin dubai lyon
ing tion th qu ab xy zz abc xyz ion pre ed er est ly un
""".split()
CAPS = ["Paris", "Tunis", "NASA", "Hello", "OK", "USA", "Bob", "Alice", "Python", "Monday"]
NUMS = ["2024", "42", "007", "123", "99", "1", "5", "10", "3000"]
MULTI = ["ice cream", "new york", "big data", "hot dog", "red car", "good morning"]
CHARS = ["x", "a", "z", "q", "e", "s", "k"]


def random_literal(rng):
    r = rng.random()
    if r < 0.70:
        return rng.choice(WORDS)
    if r < 0.80:
        return rng.choice(CAPS)
    if r < 0.88:
        return rng.choice(NUMS)
    if r < 0.94:
        return rng.choice(MULTI)
    return rng.choice(CHARS)


# character classes: regex, membership test (independent of the regex), English (singular, plural)
CLASSES = {
    "digit": ("[0-9]", lambda c: c in "0123456789",
              ["a digit", "a number", "a numeral", "a numeric character"],
              ["digits", "numeric characters", "numerals"]),
    "letter": ("[A-Za-z]", lambda c: ("a" <= c <= "z") or ("A" <= c <= "Z"),
               ["a letter", "an alphabetic character", "a letter of the alphabet"],
               ["letters", "alphabetic characters"]),
    "upper": ("[A-Z]", lambda c: "A" <= c <= "Z",
              ["a capital letter", "an uppercase letter", "an upper-case letter", "a capital"],
              ["capital letters", "uppercase letters", "upper-case letters", "capitals"]),
    "lower": ("[a-z]", lambda c: "a" <= c <= "z",
              ["a lowercase letter", "a lower-case letter", "a small letter"],
              ["lowercase letters", "lower-case letters", "small letters"]),
    "vowel": ("[AEIOUaeiou]", lambda c: c in "AEIOUaeiou",
              ["a vowel"], ["vowels"]),
    "space": (" ", lambda c: c == " ",
              ["a space", "a blank space"], ["spaces", "blank spaces"]),
}
ONLY_CLASSES = ["digit", "letter", "upper", "lower"]

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty"}

# Phrasings that never occur in training; they make up the "unseen phrasings" test split.
# Every word in them does occur in training, only in other combinations.
HELD_OUT = {
    "have {X} at the start", "have {X} at the end", "contain {X} no fewer than {N} times",
    "show me all ", "are made up of only {P}", "have {X} somewhere before {Y}",
    "contain {X} with {Y} right after", "are no longer than {N} characters", "all strings",
    "{Y} comes right after {X}", "each line",
}


class Phraser:
    """Chooses phrasings; in 'train' mode never uses HELD_OUT ones, records whether one was used."""

    def __init__(self, rng, allow_held_out):
        self.rng, self.allow, self.used_held_out = rng, allow_held_out, False

    def pick(self, options):
        opts = options if self.allow else [o for o in options if o not in HELD_OUT]
        o = self.rng.choice(opts)
        if o in HELD_OUT:
            self.used_held_out = True
        return o


def num_text(rng, n):
    return NUMBER_WORDS[n] if n in NUMBER_WORDS and rng.random() < 0.4 else str(n)


# ============================================================================ meaning trees
# item  = ("lit", k)  literal number k (0, 1, 2) | ("cls", name)
# node  = (kind, ...) ; kinds: contains, starts, ends, exactly, atleast, fewer, run, len_ge, len_le,
#                              len_eq, only, before, adjacent, not, and, or


def item_regex(it, lits):
    return lits[it[1]] if it[0] == "lit" else CLASSES[it[1]][0]


def regex_of(node, lits):
    """Canonical regex. `lits` maps literal number -> text written in the regex (word or placeholder)."""
    k = node[0]
    if k == "contains":
        return f".*{item_regex(node[1], lits)}.*"
    if k == "starts":
        return f"{item_regex(node[1], lits)}.*"
    if k == "ends":
        return f".*{item_regex(node[1], lits)}"
    if k == "exactly":
        return item_regex(node[1], lits)
    if k == "atleast":
        return f"(.*{item_regex(node[1], lits)}){{{node[2]},}}.*"
    if k == "fewer":
        return f"~((.*{item_regex(node[1], lits)}){{{node[2]},}}.*)"
    if k == "run":
        return f".*({item_regex(node[1], lits)}){{{node[2]},}}.*"
    if k == "len_ge":
        return f".{{{node[1]},}}"
    if k == "len_le":
        return f".{{0,{node[1]}}}"
    if k == "len_eq":
        return f".{{{node[1]}}}"
    if k == "only":
        return f"{CLASSES[node[1]][0]}+"
    if k == "before":
        return f".*{item_regex(node[1], lits)}.*{item_regex(node[2], lits)}.*"
    if k == "adjacent":
        return f".*{item_regex(node[1], lits)}{item_regex(node[2], lits)}.*"
    if k == "not":
        return f"~({regex_of(node[1], lits)})"
    if k in ("and", "or"):
        op = "&" if k == "and" else "|"
        return op.join(f"({regex_of(c, lits)})" for c in node[1:])
    raise ValueError(k)


# ------------------------------------------------------------- the independent meaning (Python predicates)
def occurrences(it, s, words):
    """start positions and lengths of every occurrence of an item"""
    if it[0] == "lit":
        w = words[it[1]]
        return [(i, len(w)) for i in range(len(s) - len(w) + 1) if s.startswith(w, i)]
    test = CLASSES[it[1]][1]
    return [(i, 1) for i, c in enumerate(s) if test(c)]


def count_disjoint(it, s, words):
    if it[0] == "lit":
        return s.count(words[it[1]])          # str.count counts non-overlapping occurrences (maximal)
    test = CLASSES[it[1]][1]
    return sum(test(c) for c in s)


def holds(node, s, words):
    k = node[0]
    if k == "contains":
        return bool(occurrences(node[1], s, words))
    if k == "starts":
        return any(i == 0 for i, _ in occurrences(node[1], s, words))
    if k == "ends":
        return any(i + n == len(s) for i, n in occurrences(node[1], s, words))
    if k == "exactly":
        return s == words[node[1][1]]
    if k == "atleast":
        return count_disjoint(node[1], s, words) >= node[2]
    if k == "fewer":
        return count_disjoint(node[1], s, words) < node[2]
    if k == "run":
        return words[node[1][1]] * node[2] in s
    if k == "len_ge":
        return len(s) >= node[1]
    if k == "len_le":
        return len(s) <= node[1]
    if k == "len_eq":
        return len(s) == node[1]
    if k == "only":
        test = CLASSES[node[1]][1]
        return len(s) > 0 and all(test(c) for c in s)
    if k == "before":
        ends_x = [i + n for i, n in occurrences(node[1], s, words)]
        return bool(ends_x) and any(j >= min(ends_x) for j, _ in occurrences(node[2], s, words))
    if k == "adjacent":
        ys = {j for j, _ in occurrences(node[2], s, words)}
        return any(i + n in ys for i, n in occurrences(node[1], s, words))
    if k == "not":
        return not holds(node[1], s, words)
    if k == "and":
        return all(holds(c, s, words) for c in node[1:])
    if k == "or":
        return any(holds(c, s, words) for c in node[1:])
    raise ValueError(k)


# ============================================================================ English
def quote(rng, w):
    return f'"{w}"' if rng.random() < 0.2 else f"'{w}'"


def item_np(P, it, words, plural=False):
    rng = P.rng
    if it[0] == "lit":
        q = quote(rng, words[it[1]])
        return P.pick([q, q, f"the string {q}", f"the text {q}", f"the substring {q}", f"the sequence {q}"])
    _, _, sing, plur = CLASSES[it[1]]
    return P.pick(plur if plural else sing)


# Each atom gives: list of finite verb phrases (plural subject), participle phrases, negated forms.
def atom_phrases(P, node, words):
    """Return (positive VPs, negative VPs, positive participles, negative participles) for an atom.
    Templates use {X}, {Y}, {N}, {P}; they are filled later so that held-out templates can be tracked."""
    k = node[0]
    if k == "contains":
        pos = ["contain {X}", "include {X}", "have {X}", "contain {X} somewhere", "have {X} in them",
               "mention {X}" if node[1][0] == "lit" else "contain {X} anywhere"]
        neg = ["do not contain {X}", "don't contain {X}", "do not include {X}", "never contain {X}",
               "do not have {X}"]
        ppos = ["containing {X}", "with {X}", "having {X}", "including {X}"]
        pneg = ["not containing {X}", "without {X}", "lacking {X}"]
    elif k == "starts":
        pos = ["start with {X}", "begin with {X}", "have {X} at the start", "have {X} at the beginning",
               "open with {X}"]
        neg = ["do not start with {X}", "don't begin with {X}", "do not begin with {X}"]
        ppos = ["starting with {X}", "beginning with {X}"]
        pneg = ["not starting with {X}", "not beginning with {X}"]
    elif k == "ends":
        pos = ["end with {X}", "finish with {X}", "have {X} at the end", "terminate with {X}"]
        neg = ["do not end with {X}", "don't end with {X}", "do not finish with {X}"]
        ppos = ["ending with {X}", "ending in {X}", "finishing with {X}"]
        pneg = ["not ending with {X}", "not ending in {X}"]
    elif k == "exactly":
        pos = ["are exactly {X}", "are equal to {X}", "consist of exactly {X}", "match {X} exactly"]
        neg = ["are not exactly {X}", "are not equal to {X}", "are anything other than {X}"]
        ppos = ["equal to {X}", "identical to {X}"]
        pneg = ["not equal to {X}", "different from {X}"]
    elif k == "atleast":
        pos = ["contain {X} at least {N} times", "contain {X} {N} or more times", "have {X} at least {N} times",
               "contain {X} no fewer than {N} times", "include {X} {N} times or more"]
        if node[1][0] == "cls":
            pos += ["contain at least {N} {P}", "have {N} or more {P}", "contain {N} or more {P}"]
        if node[2] == 2:
            pos += ["contain {X} at least twice", "contain {X} more than once", "have {X} twice or more"]
        neg = ["contain {X} fewer than {N} times", "contain {X} less than {N} times"]
        ppos = ["containing {X} at least {N} times", "with {X} at least {N} times", "with {X} {N} or more times"]
        if node[1][0] == "cls":
            ppos += ["with at least {N} {P}", "having {N} or more {P}"]
        pneg = ["containing {X} fewer than {N} times", "with {X} less than {N} times"]
    elif k == "fewer":
        pos = ["contain {X} fewer than {N} times", "contain {X} less than {N} times",
               "have {X} fewer than {N} times"]
        if node[1][0] == "cls":
            pos += ["contain fewer than {N} {P}", "have less than {N} {P}"]
        neg = ["contain {X} at least {N} times", "do not contain {X} fewer than {N} times"]
        ppos = ["with fewer than {N} occurrences of {X}", "containing {X} fewer than {N} times"]
        if node[1][0] == "cls":
            ppos += ["with fewer than {N} {P}"]
        pneg = ["with {X} at least {N} times"]
    elif k == "run":
        pos = ["contain {X} {N} times in a row", "contain {X} repeated {N} times in a row",
               "have {N} consecutive copies of {X}", "contain {X} {N} times back to back",
               "contain {X} at least {N} times consecutively"]
        neg = ["do not contain {X} {N} times in a row", "never have {N} consecutive copies of {X}"]
        ppos = ["with {X} {N} times in a row", "containing {N} consecutive copies of {X}"]
        pneg = ["without {N} consecutive copies of {X}"]
    elif k == "len_ge":
        pos = ["are at least {N} characters long", "have at least {N} characters",
               "contain {N} or more characters", "have a length of at least {N}", "are {N} or more characters long"]
        neg = ["are shorter than {N} characters", "have fewer than {N} characters"]
        ppos = ["with at least {N} characters", "of length at least {N}", "at least {N} characters long"]
        pneg = ["shorter than {N} characters", "with fewer than {N} characters"]
    elif k == "len_le":
        pos = ["are at most {N} characters long", "have at most {N} characters", "are no longer than {N} characters",
               "have no more than {N} characters", "are {N} characters long or shorter"]
        neg = ["are longer than {N} characters", "have more than {N} characters"]
        ppos = ["with at most {N} characters", "of length at most {N}", "at most {N} characters long"]
        pneg = ["longer than {N} characters", "with more than {N} characters"]
    elif k == "len_eq":
        pos = ["are exactly {N} characters long", "have exactly {N} characters", "have a length of exactly {N}",
               "are {N} characters long"]
        neg = ["are not exactly {N} characters long", "do not have exactly {N} characters"]
        ppos = ["with exactly {N} characters", "of length {N}", "exactly {N} characters long"]
        pneg = ["not exactly {N} characters long"]
    elif k == "only":
        pos = ["contain only {P}", "consist only of {P}", "are made of {P} only", "are made up of only {P}",
               "are made up entirely of {P}",
               "are entirely {P}", "have nothing but {P}"]
        neg = ["do not consist only of {P}", "are not made of {P} only"]
        ppos = ["made only of {P}", "consisting only of {P}", "containing only {P}"]
        pneg = ["not made only of {P}"]
    elif k == "before":
        pos = ["contain {X} before {Y}", "contain {X} followed later by {Y}", "have {X} somewhere before {Y}",
               "contain {X} and later {Y}", "have {Y} somewhere after {X}", "contain {Y} after {X}"]
        neg = ["do not contain {X} before {Y}", "never have {X} before {Y}"]
        ppos = ["with {X} before {Y}", "containing {X} and later {Y}", "with {Y} somewhere after {X}"]
        pneg = ["without {X} before {Y}"]
    elif k == "adjacent":
        pos = ["contain {X} immediately followed by {Y}", "contain {X} directly followed by {Y}",
               "have {X} right before {Y}", "contain {X} with {Y} right after",
               "contain {Y} immediately after {X}"]
        neg = ["do not contain {X} immediately followed by {Y}", "never have {X} right before {Y}"]
        ppos = ["with {X} immediately followed by {Y}", "containing {X} directly followed by {Y}"]
        pneg = ["without {X} immediately followed by {Y}"]
    else:
        raise ValueError(k)
    return pos, neg, ppos, pneg


def wh_phrases(node):
    """clauses for 'lines where ...' / 'lines in which ...' (singular subject = the item). None if unsupported."""
    k = node[0]
    cls = len(node) > 1 and isinstance(node[1], tuple) and node[1][0] == "cls"
    if k == "contains":
        return (["{X} appears", "{X} occurs", "{X} shows up", "{X} appears somewhere"],
                ["{X} never appears", "{X} does not appear", "{X} doesn't occur", "{X} never shows up"])
    if k == "starts":
        pos = ["the line starts with {X}", "the text begins with {X}"] + (["the first character is {X}"] if cls else [])
        return pos, ["the line does not start with {X}", "the text doesn't begin with {X}"]
    if k == "ends":
        pos = ["the line ends with {X}", "the text finishes with {X}"] + (["the last character is {X}"] if cls else [])
        return pos, ["the line does not end with {X}", "the text doesn't end with {X}"]
    if k == "atleast":
        pos = ["{X} appears at least {N} times", "{X} occurs {N} or more times", "{X} shows up at least {N} times"]
        if node[2] == 2:
            pos += ["{X} appears at least twice", "{X} occurs more than once"]
        return pos, ["{X} appears fewer than {N} times", "{X} occurs less than {N} times"]
    if k == "fewer":
        return (["{X} appears fewer than {N} times", "{X} occurs less than {N} times"],
                ["{X} appears at least {N} times"])
    if k == "run":
        return (["{X} appears {N} times in a row", "{X} is repeated {N} times in a row",
                 "{X} occurs {N} times consecutively"], ["{X} never appears {N} times in a row"])
    if k == "before":
        return (["{X} comes before {Y}", "{X} appears before {Y}", "{Y} comes after {X}", "{X} is followed later by {Y}"],
                ["{X} never comes before {Y}", "{X} does not appear before {Y}"])
    if k == "adjacent":
        return (["{X} is immediately followed by {Y}", "{X} is directly followed by {Y}", "{Y} comes right after {X}"],
                ["{X} is never immediately followed by {Y}"])
    if k == "len_ge":
        return ["the length is at least {N}", "the length is {N} or more"], ["the length is less than {N}"]
    if k == "len_le":
        return ["the length is at most {N}", "the length is {N} or less"], ["the length is more than {N}"]
    if k == "len_eq":
        return ["the length is exactly {N}", "the length is {N}"], ["the length is not {N}"]
    return None


def wh_ok(node):
    return wh_phrases(node[1] if node[0] == "not" else node) is not None


def wh(P, node, words):
    neg = node[0] == "not"
    atom = node[1] if neg else node
    pos, negs = wh_phrases(atom)
    t = fill(P, P.pick(negs if neg else pos), atom, words)
    return t[0].lower() + t[1:] if t.startswith("{") else t


_SING = {"contain": "contains", "include": "includes", "have": "has", "mention": "mentions", "start": "starts",
         "begin": "begins", "open": "opens", "end": "ends", "finish": "finishes", "terminate": "terminates",
         "are": "is", "consist": "consists", "match": "matches", "do": "does", "don't": "doesn't"}


def conjugate(phrase):
    """plural verb phrase -> singular ('contain X' -> 'contains X', 'never have' -> 'never has')"""
    w = phrase.split(" ")
    i = 1 if w[0] in ("never", "both", "either") and len(w) > 1 else 0
    if w[i] in _SING:
        w[i] = _SING[w[i]]
    return " ".join(w).replace(" in them", " in it")


def fill(P, template, node, words):
    out = template
    if "{X}" in out:
        out = out.replace("{X}", item_np(P, node[1], words))
    if "{Y}" in out:
        out = out.replace("{Y}", item_np(P, node[2], words))
    if "{P}" in out:
        cls = node[1] if node[0] == "only" else node[1][1]
        out = out.replace("{P}", P.pick(CLASSES[cls][3]))
    if "{N}" in out:
        n = node[2] if node[0] in ("atleast", "fewer", "run") else node[1]
        out = out.replace("{N}", num_text(P.rng, n))
    return out


def vp(P, node, words, form):
    """form: 'vp' (finite, plural subject) or 'pp' (participle / prepositional, after the noun)."""
    neg = node[0] == "not"
    atom = node[1] if neg else node
    pos, negs, ppos, pneg = atom_phrases(P, atom, words)
    lst = (negs if neg else pos) if form == "vp" else (pneg if neg else ppos)
    return fill(P, P.pick(lst), atom, words)


SUBJECTS = ["lines", "lines", "lines", "strings", "lines of text", "text lines", "all lines", "the lines",
            "rows", "all strings"]
SING_SUBJECTS = ["any line", "a line", "every line", "each line", "any string", "a string", "any text"]
PREFIXES = ["", "", "", "find ", "match ", "select ", "show me ", "give me ", "I want ", "I need ", "show me all ",
            "a regex for ", "a regular expression for ", "regex matching ", "get "]


def tidy(text):
    """singular forms for the number one"""
    for a, b in [(" 1 characters", " 1 character"), (" one characters", " one character"),
                 ("length of at least 1", "length of at least 1"), ("than 1 characters", "than 1 character")]:
        text = text.replace(a, b)
    return text


def describe(P, tree, words):
    return tidy(_describe(P, tree, words))


def _describe(P, tree, words):
    rng = P.rng
    pre = P.pick(PREFIXES)
    singular = rng.random() < 0.25 and pre != "show me all "
    subj = P.pick(SING_SUBJECTS if singular else SUBJECTS)
    if pre == "show me all " and subj.startswith(("all ", "the ")):
        subj = subj.split(" ", 1)[1]
    kind = tree[0]
    parts = list(tree[1:]) if kind in ("and", "or") else [tree]
    r = rng.random()
    form = "wh" if r < 0.2 and all(wh_ok(p) for p in parts) else ("pp" if r < 0.5 else "vp")
    rel = P.pick(["that ", "that ", "which "])
    conj = "and" if kind == "and" else "or"
    if form == "wh":
        head = P.pick(["where ", "where ", "in which "])
        cl = [wh(P, p, words) for p in parts]
        if len(cl) == 1:
            body = cl[0]
        elif len(cl) == 2:
            body = cl[0] + (" but " if kind == "and" and parts[1][0] == "not" and rng.random() < 0.5 else f" {conj} ") + cl[1]
        else:
            body = ", ".join(cl[:-1]) + f" {conj} " + cl[-1]
        return f"{pre}{subj} {head}{body}"

    # shared-verb coordination: "lines that contain 'a' and 'b'", "lines starting with a digit or a vowel"
    same = kind in ("and", "or") and len(parts) == 2 and parts[0][0] == parts[1][0] and \
        parts[0][0] in ("contains", "starts", "ends") and rng.random() < 0.5
    if same:
        x1, x2 = item_np(P, parts[0][1], words), item_np(P, parts[1][1], words)
        verb = {"contains": ["contain", "include", "have"], "starts": ["start with", "begin with"],
                "ends": ["end with", "finish with"]}[parts[0][0]]
        pverb = {"contains": ["containing", "with"], "starts": ["starting with", "beginning with"],
                 "ends": ["ending with", "ending in"]}[parts[0][0]]
        both = ("both " if kind == "and" else "either ") if rng.random() < 0.4 else ""
        if form != "pp":
            v = P.pick(verb)
            body = f"{rel}{conjugate(v) if singular else v} {both}{x1} {conj} {x2}"
        else:
            body = f"{P.pick(pverb)} {both}{x1} {conj} {x2}"
        return f"{pre}{subj} {body}"

    if form == "wh":
        form = "vp"
    phrases = [vp(P, p, words, form) for p in parts]
    if singular and form == "vp":
        phrases = [conjugate(x) for x in phrases]
    if kind in ("and", "or"):
        if len(phrases) == 2:
            if kind == "and":
                joiner = P.pick([" and ", " and ", " and also ", " but " if parts[1][0] == "not" else " and "])
            else:
                joiner = " or "
            lead = ""
            if rng.random() < 0.25:
                lead = "both " if kind == "and" else "either "
                joiner = f" {conj} "
            body = lead + phrases[0] + joiner + phrases[1]
        else:
            body = ", ".join(phrases[:-1]) + P.pick([f" {conj} ", f", {conj} "]) + phrases[-1]
    else:
        body = phrases[0]
    if form == "vp":
        return f"{pre}{subj} {rel}{body}"
    return f"{pre}{subj} {body}"


# ============================================================================ sampling meaning trees
def sample_item(rng, lits, allow_lit=True, p_lit=0.5):
    if allow_lit and rng.random() < p_lit:
        k = len(lits) if len(lits) < 3 and rng.random() < 0.85 or not lits else rng.randrange(len(lits))
        if k == len(lits):
            lits.append(random_literal(rng))
        return ("lit", k)
    return ("cls", rng.choice(list(CLASSES)))


def sample_atom(rng, lits):
    kinds = ["contains"] * 5 + ["starts"] * 4 + ["ends"] * 4 + ["exactly"] + ["atleast"] * 3 + ["fewer"] * 2 + \
            ["run"] * 2 + ["len_ge"] * 2 + ["len_le"] * 2 + ["len_eq"] + ["only"] * 2 + ["before"] * 3 + ["adjacent"] * 2
    k = rng.choice(kinds)
    if k in ("contains", "starts", "ends"):
        return (k, sample_item(rng, lits))
    if k == "exactly":
        return (k, sample_item(rng, lits, p_lit=1.0))
    if k in ("atleast", "fewer"):
        return (k, sample_item(rng, lits), rng.randint(2, 6))
    if k == "run":
        return (k, sample_item(rng, lits, p_lit=1.0), rng.randint(2, 5))
    if k in ("len_ge", "len_le", "len_eq"):
        return (k, rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]))
    if k == "only":
        return (k, rng.choice(ONLY_CLASSES))
    a = sample_item(rng, lits)
    b = sample_item(rng, lits)
    if a == b:
        b = ("cls", rng.choice([c for c in CLASSES if ("cls", c) != a]))
    return (k, a, b)


def sample_literal_side(rng, lits, p_not):
    a = sample_atom(rng, lits)
    if a[0] not in ("fewer",) and rng.random() < p_not:
        return ("not", a)
    return a


def sample_tree(rng):
    lits = []
    r = rng.random()
    if r < 0.30:
        t = sample_atom(rng, lits)
    elif r < 0.42:
        a = sample_atom(rng, lits)
        t = ("not", a) if a[0] != "fewer" else a
    elif r < 0.75:
        t = ("and", sample_literal_side(rng, lits, 0.25), sample_literal_side(rng, lits, 0.3))
    elif r < 0.93:
        t = ("or", sample_literal_side(rng, lits, 0.15), sample_literal_side(rng, lits, 0.15))
    else:
        t = ("and", sample_atom(rng, lits), sample_atom(rng, lits), sample_literal_side(rng, lits, 0.3))
    return t, lits


def literal_order(tree):
    """literal numbers in the order the regex mentions them"""
    out = []

    def walk(n):
        for x in n[1:]:
            if isinstance(x, tuple):
                if x[0] == "lit":
                    out.append(x[1])
                elif x[0] != "cls":
                    walk(x)
    walk(tree)
    return out


# ============================================================================ verification
FILLERS = ["a", "e", "b", "Z", "E", "x", "0", "7", " ", ".", "-", "Q", "u"]


def probe_strings(tree, words, rng, n=300):
    pieces = list(words) + [w[: max(1, len(w) // 2)] for w in words] + [w.upper() for w in words] + FILLERS
    out = {"", " "}
    for w in words:
        for r in range(1, 8):
            out.add(w * r)
            out.add("x".join([w] * r))
            out.add("A" + " ".join([w] * r) + "1")
    for k in range(0, 23):
        out.add("a" * k)
        out.add(("A1 b" * 6)[:k])
    for _ in range(80):                       # line = one literal surrounded by random pieces
        out.add("".join(rng.choice(pieces) for _ in range(rng.randint(0, 3))) + rng.choice(words or FILLERS) +
                "".join(rng.choice(pieces) for _ in range(rng.randint(0, 3))))
    while len(out) < n:
        out.add("".join(rng.choice(pieces) for _ in range(rng.randint(0, 10))))
    return sorted(out)


def verify(tree, words, seed):
    """True if regex and predicate agree on all probes and the probes include both outcomes."""
    rng = random.Random(seed)
    rx = regex_of(tree, words)
    ast = parse(rx)
    pos = neg = 0
    for s in probe_strings(tree, words, rng):
        a, b = full_match(ast, s), holds(tree, s, words)
        if a != b:
            raise AssertionError(f"MISMATCH regex={rx!r} tree={tree} words={words} string={s!r}: regex {a}, meaning {b}")
        pos += a
        neg += not a
    return pos > 0 and neg > 0


# ============================================================================ building the dataset
def placeholder_regex(tree):
    return regex_of(tree, {k: f"__w{k}__" for k in range(3)})


def make_example(seed, allow_held_out):
    rng = random.Random(seed)
    tree, lits = sample_tree(rng)
    if len(set(lits)) != len(lits):          # two literal slots got the same word
        return None
    P = Phraser(rng, allow_held_out)
    text = describe(P, tree, lits)
    if not verify(tree, lits, seed):
        return {"rejected": True}
    return {"text": text, "regex": regex_of(tree, lits), "tree": repr(tree), "words": lits,
            "kind": tree[0] if tree[0] in ("and", "or", "not") else "atom",
            "held_out_phrasing": P.used_held_out}


def _worker(args):
    seed, allow = args
    return make_example(seed, allow)


def generate(n_target, seed0, allow_held_out, pool):
    out, rejected, s = [], 0, seed0
    while len(out) < n_target:
        batch = [(s + i, allow_held_out) for i in range(4000)]
        s += 4000
        for ex in pool.map(_worker, batch, chunksize=100):
            if ex is None:
                continue
            if ex.get("rejected"):
                rejected += 1
                continue
            out.append(ex)
    return out[:n_target], rejected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(2) as pool:
        main_set, rej1 = generate(42000, 1_000_000, False, pool)
        phr_set, rej2 = generate(30000, 9_000_000, True, pool)
    # anonymised description + placeholder regex decide identity
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import anonymize  # noqa: E402
    for ex in main_set + phr_set:
        src, _ = anonymize(ex["text"])
        ex["src_anon"] = src
        ex["regex_anon"] = anon_target(ex)
    rng = random.Random(7)
    # de-duplicate identical (anonymised description, regex) pairs
    seen, uniq = set(), []
    for ex in main_set:
        key = (ex["src_anon"], ex["regex_anon"])
        if key not in seen:
            seen.add(key)
            uniq.append(ex)
    # split by REGEX: a regex structure in val/test never occurs in training
    by_rx = {}
    for ex in uniq:
        by_rx.setdefault(ex["regex_anon"], []).append(ex)
    rxs = sorted(by_rx)
    rng.shuffle(rxs)
    # atoms (one condition) all go to training; test checks new combinations and new parameters
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
        with open(OUT / f"{name}.jsonl", "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats = {"generated_main": len(main_set), "rejected_contradiction_or_tautology_main": rej1,
             "after_dedup": len(uniq), "train": len(train), "val": len(val), "test": len(test),
             "test_phrasing": len(phrasing), "test_handwritten": len(hand),
             "unique_regex_structures_train": len(train_rx),
             "kinds_train": Counter(ex["kind"] for ex in train),
             "kinds_test": Counter(ex["kind"] for ex in test)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def anon_target(ex):
    """placeholder regex where __wK__ is numbered by order of appearance in the DESCRIPTION"""
    from data import literal_spans  # noqa: E402
    tree = eval(ex["tree"])  # noqa: S307  (our own repr)
    words = ex["words"]
    order = []
    for w in literal_spans(ex["text"]):
        if w not in order:
            order.append(w)
    names = {k: f"__w{order.index(w)}__" for k, w in enumerate(words)}
    return regex_of(tree, names)


# ============================================================================ handwritten test set
def handwritten():
    """Descriptions written by hand, in wordings the generator does not produce. The regex is computed
    from the meaning tree, so the label is exactly as reliable as the generated ones."""
    L = lambda k: ("lit", k)  # noqa: E731
    C = lambda n: ("cls", n)  # noqa: E731
    items = [
        ("lines with the word 'error' in them", ("contains", L(0)), ["error"]),
        ("any line that mentions 'invoice'", ("contains", L(0)), ["invoice"]),
        ("text that begins with 'Dear'", ("starts", L(0)), ["Dear"]),
        ("lines finishing with a digit", ("ends", C("digit")), []),
        ("lines whose first character is a capital letter", ("starts", C("upper")), []),
        ("lines with no digits at all", ("not", ("contains", C("digit"))), []),
        ("lines that contain both 'dog' and 'cat'", ("and", ("contains", L(0)), ("contains", L(1))), ["dog", "cat"]),
        ("lines containing 'yes' or 'no'", ("or", ("contains", L(0)), ("contains", L(1))), ["yes", "no"]),
        ("the line must be exactly 'admin'", ("exactly", L(0)), ["admin"]),
        ("lines made up of digits only", ("only", "digit"), []),
        ("lines made only of lower-case letters", ("only", "lower"), []),
        ("lines that are 8 or more characters long", ("len_ge", 8), []),
        ("lines with at most 5 characters", ("len_le", 5), []),
        ("lines that have exactly 4 characters", ("len_eq", 4), []),
        ("lines where 'ab' is repeated 3 times in a row", ("run", L(0), 3), ["ab"]),
        ("lines that contain 'login' but not 'failed'", ("and", ("contains", L(0)), ("not", ("contains", L(1)))), ["login", "failed"]),
        ("lines that start with 'Paris' and end with a number", ("and", ("starts", L(0)), ("ends", C("digit"))), ["Paris"]),
        ("lines where a vowel appears before a digit", ("before", C("vowel"), C("digit")), []),
        ("lines in which 'price' comes before 'total'", ("before", L(0), L(1)), ["price", "total"]),
        ("lines with a capital letter directly followed by a digit", ("adjacent", C("upper"), C("digit")), []),
        ("lines containing 'ice cream'", ("contains", L(0)), ["ice cream"]),
        ("lines with fewer than 3 spaces", ("fewer", C("space"), 3), []),
        ("lines containing at least 5 digits", ("atleast", C("digit"), 5), []),
        ("lines that do not start with a space", ("not", ("starts", C("space"))), []),
        ("lines not ending with 'ing'", ("not", ("ends", L(0))), ["ing"]),
        ("lines that contain 'tunis' or begin with 'Tunis'", ("or", ("contains", L(0)), ("starts", L(1))), ["tunis", "Tunis"]),
        ("lines having two or more capital letters", ("atleast", C("upper"), 2), []),
        ("lines where 'qu' is directly followed by a vowel", ("adjacent", L(0), C("vowel")), ["qu"]),
        ("lines that begin with a digit and contain 'kg'", ("and", ("starts", C("digit")), ("contains", L(0))), ["kg"]),
        ("lines ending in a vowel or a space", ("or", ("ends", C("vowel")), ("ends", C("space"))), []),
        ("lines of at most 10 characters that contain a digit", ("and", ("len_le", 10), ("contains", C("digit"))), []),
        ("lines with '2024' somewhere in them", ("contains", L(0)), ["2024"]),
        ("lines where 'na' appears 3 or more times", ("atleast", L(0), 3), ["na"]),
        ("lines that contain 'id' less than 2 times", ("fewer", L(0), 2), ["id"]),
        ("lines that never contain 'password'", ("not", ("contains", L(0))), ["password"]),
        ("lines starting with 'dear' that do not contain a digit", ("and", ("starts", L(0)), ("not", ("contains", C("digit")))), ["dear"]),
        ("lines other than 'none'", ("not", ("exactly", L(0))), ["none"]),
        ("lines consisting entirely of capital letters", ("only", "upper"), []),
        ("lines shorter than 3 characters", ("not", ("len_ge", 3)), []),
        ("lines longer than 12 characters", ("not", ("len_le", 12)), []),
        ("lines containing a space, a digit and a capital letter", ("and", ("contains", C("space")), ("contains", C("digit")), ("contains", C("upper"))), []),
        ("lines where 'red' is followed later by 'blue'", ("before", L(0), L(1)), ["red", "blue"]),
        ("lines where 'blue' comes somewhere after 'red'", ("before", L(1), L(0)), ["blue", "red"]),
        ("lines with 'zz' twice in a row", ("run", L(0), 2), ["zz"]),
        ("lines starting with a lowercase letter or ending with a digit", ("or", ("starts", C("lower")), ("ends", C("digit"))), []),
        ("lines without any vowel", ("not", ("contains", C("vowel"))), []),
        ("lines with at least 3 vowels", ("atleast", C("vowel"), 3), []),
        ("lines that begin with 'www'", ("starts", L(0)), ["www"]),
        ("lines that end in 'com'", ("ends", L(0)), ["com"]),
        ("lines that contain 'foo', 'bar' and 'baz'", ("and", ("contains", L(0)), ("contains", L(1)), ("contains", L(2))), ["foo", "bar", "baz"]),
    ]
    rows = []
    for text, tree, words in items:
        assert verify(tree, words, 0), text
        ex = {"text": text, "regex": regex_of(tree, words), "tree": repr(tree), "words": words,
              "kind": tree[0] if tree[0] in ("and", "or", "not") else "atom", "held_out_phrasing": True}
        from data import anonymize  # noqa: E402
        ex["src_anon"] = anonymize(text)[0]
        ex["regex_anon"] = anon_target(ex)
        rows.append(ex)
    return rows


if __name__ == "__main__":
    main()
