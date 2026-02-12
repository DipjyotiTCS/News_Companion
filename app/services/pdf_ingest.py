import os
import re
import hashlib
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI

LABEL_RE = re.compile(r"^(Author Name|Date|IsHeadline|Topic|Domain|Country|Location|References)\s*:\s*(.*)$", re.IGNORECASE)

def _read_pdf_pages(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages: List[str] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pages.append(page.get_text("text"))
    return pages

def _doc_id(pdf_path: str) -> str:
    h = hashlib.sha1()
    with open(pdf_path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _extract_details_and_summery(page_text: str) -> Tuple[str, str, str]:
    """Returns (title, details, summery) from an 'article content' page."""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return "", "", ""
    title = lines[0]

    details = ""
    summery = ""

    m_details = re.search(r"(?is)\bDetails\s*:\s*(.*?)(?=\bSummery\s*:|\bSummary\s*:|\Z)", page_text)
    m_sum = re.search(r"(?is)\bSummery\s*:\s*(.*?)(?=\bAuthor Name\s*:|\Z)", page_text)
    if not m_sum:
        m_sum = re.search(r"(?is)\bSummary\s*:\s*(.*?)(?=\bAuthor Name\s*:|\Z)", page_text)

    if m_details:
        details = re.sub(r"\s+", " ", m_details.group(1)).strip()
    if m_sum:
        summery = re.sub(r"\s+", " ", m_sum.group(1)).strip()

    return title.strip(), details, summery

def _extract_meta(page_text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    refs: List[str] = []

    for raw in page_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LABEL_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()

        if key == "references":
            parts = [p.strip() for p in re.split(r"\s*,\s*", val) if p.strip()]
            for p in parts:
                if p.startswith("http"):
                    refs.append(p)
            for u in re.findall(r"https?://\S+", page_text):
                u = u.strip().rstrip(",")
                if u.startswith("http") and u not in refs:
                    refs.append(u)
        else:
            meta[key] = val

    is_headline = (meta.get("isheadline", "") or "").strip().lower()
    meta["isheadline"] = True if is_headline in ("yes", "true", "1") else False
    meta["references"] = refs
    return meta

def _parse_articles_deterministic(pages: List[str]) -> List[Dict[str, Any]]:
    """Parse articles from PDFs that follow the pattern:
    - A 'content' page starting with a Title line and containing 'Details:'
    - 'Summery:' may be on the same page OR spill onto the next page(s)
    - Metadata lines (Author Name, Date, Topic, Domain, References, etc.) may appear on the same page as the summary
      and/or on subsequent page(s).

    This parser is intentionally conservative: it only starts an article on a page that contains 'Details:'.
    """
    articles: List[Dict[str, Any]] = []
    i = 0

    def _looks_like_meta_page(t: str) -> bool:
        return any(k in t for k in (
            "Author Name", "Date", "IsHeadline", "Topic", "Domain", "Country", "Location", "References"
        ))

    while i < len(pages):
        page_text = pages[i]
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        title = lines[0] if lines else ""

        # We only treat pages containing Details as potential article starts.
        if ("Details:" not in page_text) or not title:
            i += 1
            continue

        # Articles sometimes have 'Summery:' (or 'Summary:') on a following page, and metadata can spill onto
        # subsequent pages. To robustly capture *all* articles in the PDF, we merge continuation pages until we hit the
        # next article start (a page containing 'Details:') or we reach the end.
        combined_text = page_text
        consumed = 1

        def _has_summary(t: str) -> bool:
            return ("Summery:" in t) or ("Summary:" in t)

        # Pull in continuation pages (including meta-only pages) until the next article starts.
        j = i + 1
        while j < len(pages):
            nxt = pages[j]
            if "Details:" in nxt:
                break
            combined_text = combined_text + "\n" + nxt
            consumed += 1
            j += 1

        # Extract title/details/summery from the combined text.
        title_ex, details, summery = _extract_details_and_summery(combined_text)
        if not title_ex:
            title_ex = title

        # If we couldn't extract any details, treat this as non-article content.
        if not details:
            i += 1
            continue

        # If we couldn't find a summary before the next article, keep the article (some PDFs omit it or place it irregularly).

        # Collect metadata from any of the consumed pages (summary pages often also contain Author/Date/etc.)
        meta: Dict[str, Any] = {}
        for j in range(i, i + consumed):
            t = pages[j]
            if _looks_like_meta_page(t):
                meta.update(_extract_meta(t))

        # Some PDFs place Topic/Domain/etc. on the page AFTER the author/date page.
        k = i + consumed
        while k < len(pages):
            t = pages[k]
            if ("Details:" in t):
                break  # start of next article
            if _looks_like_meta_page(t):
                meta.update(_extract_meta(t))
                k += 1
                continue
            break

        refs = meta.get("references") or []
        ref1 = refs[0] if len(refs) > 0 else ""
        ref2 = refs[1] if len(refs) > 1 else ""

        article = {
            "title": title_ex,
            "details": details,
            "summery": summery,
            "author": meta.get("author name", ""),
            "date": meta.get("date", ""),
            "isheadline": bool(meta.get("isheadline", False)),
            "topic": meta.get("topic", ""),
            "domain": meta.get("domain", ""),
            "country": meta.get("country", ""),
            "location": meta.get("location", ""),
            "reference1": ref1,
            "reference2": ref2,

            # backward compatible
            "description": summery,
            "body": details,
            "references": refs,
        }
        articles.append(article)

        # Advance: skip consumed content pages + any meta-only pages we merged.
        i = k

    return articles

def _extract_articles_with_llm(full_text: str) -> List[Dict[str, Any]]:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    llm = ChatOpenAI(model=model, temperature=0.0)

    prompt = (
        "Extract structured news articles from this PDF text dump. "
        "Return ONLY valid JSON (no markdown).\\n"
        "Schema: {\"articles\": ["
        "{\"title\": str, \"details\": str, \"summery\": str, \"author\": str, \"date\": str, "
        "\"isheadline\": bool, \"topic\": str, \"domain\": str, \"country\": str, \"location\": str, "
        "\"reference1\": str, \"reference2\": str}"
        "]}\\n"
        "Text:\\n"
        + full_text[:140000]
    )

    raw = llm.invoke(prompt).content

    import json
    data = json.loads(raw)
    return data.get("articles", [])


def ingest_pdf_to_articles(pdf_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pages = _read_pdf_pages(pdf_path)
    did = _doc_id(pdf_path)

    mode = os.environ.get("INGEST_MODE", "deterministic").strip().lower()
    if mode == "llm":
        full_text = "\n\n".join(pages)
        raw_articles = _extract_articles_with_llm(full_text)
        articles = raw_articles
    else:
        articles = _parse_articles_deterministic(pages)

    out: List[Dict[str, Any]] = []
    for idx, a in enumerate(articles):
        refs_list = a.get("references") or []
        if isinstance(refs_list, str):
            refs_list = [refs_list]

        out.append({
            "id": f"{did}:{idx}",

            "title": (a.get("title") or "").strip(),
            "details": (a.get("details") or "").strip(),
            "summery": (a.get("summery") or "").strip(),
            "author": (a.get("author") or "").strip(),
            "date": (a.get("date") or "").strip(),
            "isheadline": bool(a.get("isheadline", False)),
            "topic": (a.get("topic") or "").strip(),
            "domain": (a.get("domain") or "").strip(),
            "country": (a.get("country") or "").strip(),
            "location": (a.get("location") or "").strip(),
            "reference1": (a.get("reference1") or "").strip(),
            "reference2": (a.get("reference2") or "").strip(),

            # backward-compatible fields
            "description": (a.get("description") or a.get("summery") or "").strip(),
            "body": (a.get("body") or a.get("details") or "").strip(),
            "references": refs_list,
        })

    return out
