# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""OmniDocBench v1.5 end-to-end page parsing, scored by edit distance.

Follows opendatalab/OmniDocBench's end2end evaluation where it can without the
toolkit's LaTeX/CDM system dependencies:
- the prompt is the toolkit's Gemini img2md prompt (tools/model_infer);
- the Markdown is split into display formulas, HTML tables and text paragraphs;
- each element type is matched to the annotations by minimum normalized edit
  distance (Hungarian), with the toolkit's text / formula / table
  normalizations (simplified: inline LaTeX is not rendered to Unicode);
- text matched to the toolkit's ignored categories (headers, footers, page
  numbers, captions, footnotes) is dropped, so neither transcribing nor
  skipping them is penalized;
- each page's edit distance per type is sum(Levenshtein) / sum(max length),
  unmatched elements counting in full, and the reading-order edit distance
  compares the matched text blocks' predicted order with the annotated one.
omnidocbench_edit is the mean of the text, formula, table and reading-order
page averages ("overall edit distance", lower is better). The toolkit's
truncated-paragraph span search and its timeouts are not reproduced.
"""

import random
import re
import unicodedata
from functools import lru_cache

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    from editdistance import eval as _levenshtein
except ImportError:  # pure-Python fallback; the [audio] extra installs editdistance

    def _levenshtein(a, b):
        if len(a) < len(b):
            a, b = b, a
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + (ca != cb),
                    )
                )
            previous = current
        return previous[-1]


REPO_ID = "opendatalab/OmniDocBench"
# v1.5: the last commit before "add v1.6" (2026-04-10) changed the annotations.
REVISION = "91fe284bbfacfa687959ae3eb00846ca852aa907"
SHUFFLE_SEED = 1234

# tools/model_infer/gemini25_img2md.py in the OmniDocBench toolkit.
PROMPT = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

        1. Text Processing:
        - Accurately recognize all text content in the PDF image without guessing or inferring.
        - Convert the recognized text into Markdown format.
        - Maintain the original document structure, including headings, paragraphs, lists, etc.

        2. Mathematical Formula Processing:
        - Convert all mathematical formulas to LaTeX format.
        - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
        - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

        3. Table Processing:
        - Convert tables to HTML format.
        - Wrap the entire table with <table> and </table>.

        4. Figure Handling:
        - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

        5. Output Format:
        - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
        - For complex layouts, try to maintain the original document's structure and format as closely as possible.

        Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
        """

FORMULA_CATEGORY = "equation_isolated"
TABLE_CATEGORY = "table"
# Matched for text, then dropped from the score (end2end_dataset.py).
IGNORED_TEXT_CATEGORIES = {
    "figure_caption",
    "figure_footnote",
    "table_caption",
    "table_footnote",
    "code_algorithm",
    "code_algorithm_caption",
    "header",
    "footer",
    "page_footnote",
    "page_number",
    "equation_caption",
}
COMPONENTS = ("text", "formula", "table", "reading_order")


def process_docs(dataset):
    """The annotation file is grouped by source; a fixed shuffle makes any
    --limit prefix a mix of document types."""
    order = list(range(len(dataset)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    return dataset.select(order)


@lru_cache(maxsize=4)
def _page_image(image_path: str) -> Image.Image:
    path = hf_hub_download(
        REPO_ID, f"images/{image_path}", repo_type="dataset", revision=REVISION
    )
    return Image.open(path).convert("RGB")


def doc_to_visual(doc):
    return [_page_image(doc["page_info"]["image_path"])]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return PROMPT


def doc_to_target(doc):
    return ""


# ---------------------------------------------------------------------------
# Normalization (src/core/preprocess/data_preprocess.py, simplified)
# ---------------------------------------------------------------------------

_FORMULA_FILTERS = [
    "\\mathbf", "\\mathrm", "\\mathnormal", "\\mathit", "\\mathbb", "\\mathcal",
    "\\mathscr", "\\mathfrak", "\\mathsf", "\\mathtt", "\\textbf", "\\text",
    "\\boldmath", "\\boldsymbol", "\\operatorname", "\\bm", "\\symbfit",
    "\\mathbfcal", "\\symbf", "\\scriptscriptstyle", "\\notag", "\\setlength",
    "\\coloneqq", "\\space", "\\thickspace", "\\thinspace", "\\medspace",
    "\\nobreakspace", "\\negmedspace", "\\quad", "\\qquad", "\\enspace",
    "\\substackw", " ", "$$", "\\left", "\\right", "\\displaystyle",
]  # fmt: skip


def normalize_text(text: str) -> str:
    """clean_string: keep word characters and CJK only."""
    text = (text or "").replace("\\t", "").replace("\\n", "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text)


def _strip_delimiters(text: str) -> str:
    text = text.strip()
    changed = True
    while text and changed:
        changed = False
        for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
            if (
                text.startswith(left)
                and text.endswith(right)
                and len(text) >= len(left) + len(right)
            ):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
    return text


def normalize_formula(text: str) -> str:
    text = _strip_delimiters(text or "")
    text = re.sub(r"\\tag\*?\{[^{}]*\}", "", text).replace("\\nonumber", "")
    text = re.sub(r"\\[!,;:]", "", text)
    text = re.sub(r"(?<!\\)&", "", text)
    text = text.replace("\\mid", "|").replace("\\vert", "|")
    text = text.replace(r"\{", "").replace(r"\}", "").replace("~", "")
    text = re.sub(r"\\hspace\{.*?\}", "", text)
    text = re.sub(r"\\begin\{.*?\}", "", text)
    text = re.sub(r"\\end\{.*?\}", "", text)
    text = re.sub(r"\\arraycolsep.*?\}", "", text)
    text = re.sub(r"\{[lcr| ]+\}", "", text)
    text = text.strip(".")
    for token in _FORMULA_FILTERS:
        text = text.replace(token, "")
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"\{([A-Za-z0-9.+\-]+)\}", r"\1", text).replace("{}", "")
    return re.sub(r"\s+", "", text.lower())


def normalize_table(html: str) -> str:
    """normalized_html_table without BeautifulSoup: th -> td, drop thead /
    tbody / span wrappers and presentational attributes, collapse spaces."""
    html = unicodedata.normalize("NFKC", (html or "").replace("\n", ""))
    match = re.search(r"<table\b[^>]*>(.*)</table>", html, re.DOTALL | re.IGNORECASE)
    body = match.group(1) if match else html
    body = re.sub(r"<(/?)th\b", r"<\1td", body, flags=re.IGNORECASE)
    body = re.sub(r"</?(thead|tbody|span)\b[^>]*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r'\s(style|height|width|align|class)="[^"]*"', "", body)
    return re.sub(r"\s+", "", body)


# ---------------------------------------------------------------------------
# Ground truth and prediction elements
# ---------------------------------------------------------------------------


def _merge_truncated(doc):
    """Text blocks joined by a "truncated" relation are one paragraph (split
    across columns or pages); returns the layout items with those merged."""
    items = [d for d in doc["layout_dets"] if not d.get("ignore")]
    by_id = {d["anno_id"]: d for d in items}
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for rel in (doc.get("extra") or {}).get("relation") or []:
        if rel.get("relation_type") != "truncated":
            continue
        a, b = rel.get("source_anno_id"), rel.get("target_anno_id")
        if a in by_id and b in by_id:
            parent[find(a)] = find(b)

    groups = {}
    for d in items:
        groups.setdefault(find(d["anno_id"]), []).append(d)
    merged = []
    for group in groups.values():
        group.sort(key=lambda d: d.get("order") or 0)
        head = dict(group[0])
        if len(group) > 1:
            head["text"] = "".join(d.get("text") or "" for d in group)
        merged.append(head)
    return merged


def gt_elements(doc):
    """{"text": [(category, text, order)], "formula": [latex], "table": [html]}."""
    text, formula, table = [], [], []
    for d in _merge_truncated(doc):
        category = d["category_type"]
        if category == FORMULA_CATEGORY and d.get("latex"):
            formula.append(d["latex"])
        elif category == TABLE_CATEGORY and d.get("html"):
            table.append(d["html"])
        elif d.get("text"):
            text.append((category, d["text"], d.get("order")))
    return {"text": text, "formula": formula, "table": table}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?")
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_MD_TABLE_RE = re.compile(
    r"((?:^\|.+\|[ \t]*\n)(?:^\|[\s:|-]+\|[ \t]*\n)(?:^\|.+\|[ \t]*(?:\n|$))+)",
    re.MULTILINE,
)


def _md_table_to_html(md: str) -> str:
    rows = [line.strip().strip("|").split("|") for line in md.strip().splitlines()]
    rows = [rows[0]] + rows[2:]
    return (
        "<table>"
        + "".join(
            "<tr>" + "".join(f"<td>{c.strip()}</td>" for c in row) + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def pred_elements(response: str):
    """Split the Markdown into tables, display formulas and text paragraphs."""
    text = _FENCE_RE.sub("", _THINK_RE.sub("", response or "")).replace("```", "")
    tables = _TABLE_RE.findall(text)
    text = _TABLE_RE.sub("\n\n", text)
    tables += [_md_table_to_html(m) for m in _MD_TABLE_RE.findall(text)]
    text = _MD_TABLE_RE.sub("\n\n", text)
    formulas = [
        (a or b).strip() for a, b in _DISPLAY_RE.findall(text) if (a or b).strip()
    ]
    text = _DISPLAY_RE.sub("\n\n", text)
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"^\s*#{1,6}\s*", "", para.strip())
        if normalize_text(para):
            paragraphs.append(para)
    return {"text": paragraphs, "formula": formulas, "table": tables}


# ---------------------------------------------------------------------------
# Matching and per-page scores
# ---------------------------------------------------------------------------


def _match(gt, pred):
    """Hungarian matching on normalized edit distance. Returns (gt_idx,
    pred_idx) pairs with an edit distance below 1 (a pair that shares
    nothing counts as two unmatched elements)."""
    if not gt or not pred:
        return []
    cost = np.ones((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            longest = max(len(g), len(p))
            if longest:
                cost[i, j] = _levenshtein(g, p) / longest
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1.0]


def _edit(gt, pred, pairs):
    """sum(Levenshtein) / sum(max length) over matched pairs plus every
    unmatched element; None when the page has neither."""
    dist = total = 0
    for i, j in pairs:
        dist += _levenshtein(gt[i], pred[j])
        total += max(len(gt[i]), len(pred[j]))
    matched_gt = {i for i, _ in pairs}
    matched_pred = {j for _, j in pairs}
    for i, g in enumerate(gt):
        if i not in matched_gt:
            dist += len(g)
            total += len(g)
    for j, p in enumerate(pred):
        if j not in matched_pred:
            dist += len(p)
            total += len(p)
    return dist / total if total else None


def score_page(doc, response: str):
    gt = gt_elements(doc)
    pred = pred_elements(response)
    scores = {}

    gt_text = [normalize_text(t) for _, t, _ in gt["text"]]
    pred_text = [normalize_text(t) for t in pred["text"]]
    pairs = _match(gt_text, pred_text)
    ignored_gt = {
        i for i, (cat, _, _) in enumerate(gt["text"]) if cat in IGNORED_TEXT_CATEGORIES
    }
    ignored_pred = {j for i, j in pairs if i in ignored_gt}
    keep_gt = [i for i in range(len(gt_text)) if i not in ignored_gt]
    keep_pred = [j for j in range(len(pred_text)) if j not in ignored_pred]
    gt_index = {i: k for k, i in enumerate(keep_gt)}
    pred_index = {j: k for k, j in enumerate(keep_pred)}
    kept_pairs = [(gt_index[i], pred_index[j]) for i, j in pairs if i not in ignored_gt]
    scores["text"] = _edit(
        [gt_text[i] for i in keep_gt], [pred_text[j] for j in keep_pred], kept_pairs
    )

    # Reading order: the annotated order of the matched text blocks, listed in
    # predicted order, against the same blocks in annotated order.
    orders = [gt["text"][keep_gt[i]][2] for i, _ in kept_pairs]
    if kept_pairs and all(o is not None for o in orders):
        annotated = sorted(orders)
        predicted = [o for _, o in sorted(zip((j for _, j in kept_pairs), orders))]
        scores["reading_order"] = _levenshtein(annotated, predicted) / len(annotated)
    elif keep_gt:
        scores["reading_order"] = 1.0 if not kept_pairs else None
    else:
        scores["reading_order"] = None

    for kind, norm in (("formula", normalize_formula), ("table", normalize_table)):
        g = [norm(x) for x in gt[kind]]
        p = [norm(x) for x in pred[kind]]
        scores[kind] = _edit(g, p, _match(g, p)) if g else None
    return scores


def process_results(doc, results):
    scores = score_page(doc, results[0])
    payload = {"page": doc["page_info"]["image_path"], **scores}
    return {
        "omnidocbench_edit_score": payload,
        "omnidocbench_edit": payload,
        "text_edit": payload,
        "formula_edit": payload,
        "table_edit": payload,
        "reading_order_edit": payload,
    }


def _page_average(results, component):
    values = [r[component] for r in results if r[component] is not None]
    return sum(values) / len(values) if values else float("nan")


def aggregate_text_edit(results):
    return _page_average(results, "text")


def aggregate_formula_edit(results):
    return _page_average(results, "formula")


def aggregate_table_edit(results):
    return _page_average(results, "table")


def aggregate_reading_order_edit(results):
    return _page_average(results, "reading_order")


def aggregate_overall_edit(results):
    """Mean of the four page-averaged edit distances (types absent from the
    sampled pages are left out)."""
    values = [_page_average(results, c) for c in COMPONENTS]
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def aggregate_overall_edit_score(results):
    """(1 - overall edit distance) * 100: the higher-is-better percent the
    release report's ratio check expects."""
    return 100.0 * (1.0 - aggregate_overall_edit(results))
