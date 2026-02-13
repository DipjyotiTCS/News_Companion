import os
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_openai import ChatOpenAI


class SerpApiService:
    """
    Thin wrapper around SerpAPI (https://serpapi.com/) using REST calls.

    Env:
      - SERPAPI_API_KEY: required
      - SERPAPI_ENGINE: default "google_news" (fallback "google")
      - SERPAPI_GL / SERPAPI_HL: optional locale hints (default gl="us", hl="en")
    """

    def __init__(self) -> None:
        self.api_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
        self.engine = (os.environ.get("SERPAPI_ENGINE") or "google_news").strip()
        self.gl = (os.environ.get("SERPAPI_GL") or "us").strip()
        self.hl = (os.environ.get("SERPAPI_HL") or "en").strip()
        self.base_url = "https://serpapi.com/search.json"

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set")

    def search(self, query: str, num: int = 10) -> Dict[str, Any]:
        self._require_key()
        params = {
            "engine": self.engine,
            "q": query,
            "api_key": self.api_key,
            "gl": self.gl,
            "hl": self.hl,
        }
        # some engines support "num"
        if num:
            params["num"] = int(num)

        r = requests.get(self.base_url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()


# -------------------------
# LLM helpers (preferred)
# -------------------------

def _llm(temp: float = 0.0) -> ChatOpenAI:
    """Shared LLM constructor (keeps consistent with rest of the app)."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    return ChatOpenAI(model=model, temperature=temp)


def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    """Best-effort extraction of a top-level JSON array from an LLM response."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty LLM response")

    # Prefer exact array
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        raise ValueError("Could not find JSON array in LLM response")

    arr = json.loads(m.group(0))
    if not isinstance(arr, list):
        raise ValueError("LLM response is not a JSON array")
    out: List[Dict[str, Any]] = []
    for it in arr:
        if isinstance(it, dict):
            out.append(it)
    if not out:
        raise ValueError("LLM response array contained no objects")
    return out


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort extraction of a top-level JSON object from an LLM response."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty LLM response")

    # Prefer exact object
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("Could not find JSON object in LLM response")

    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM response is not a JSON object")
    return obj


def llm_longform_analysis(title: str, description: str, topic: str) -> Dict[str, Any]:
    """Generate a long-form analysis JSON for a news article using LLM."""
    llm = _llm(0.2)
    prompt = (
        "You are an expert news analyst. Produce a long-form analysis based on the given article details.\n"
        "Perform internal websearch to fetch suitable details for each section. Do NOT limit yourself to the provided details only.\n"
        "Provide only proven facts citing the source. Do not assume or provide informaiton without fact checking."
        "Return ONLY valid JSON (no markdown, no extra text) with exactly these keys:\n"
        "{\n"
        "  \"what_happened\": \"...\",\n"
        "  \"impact_analysis\": \"...\",\n"
        "  \"historical_context\": \"...\",\n"
        "  \"why_it_matters\": \"...\",\n"
        "  \"future_outlook\": \"...\"\n"
        "}\n\n"
        "Guidelines:\n"
        "- Add basic HTML tags like paragraph, bold, new line wherever necesarry.\n"
        "- Be balanced and avoid speculation; if uncertain, clearly say so.\n"
        "- Keep each section concise but substantive (roughly 80-180 words).\n\n"
        f"Title: {title}\n"
        f"Topic: {topic}\n"
        f"Description: {description}\n"
    )
    raw = llm.invoke(prompt).content
    obj = _extract_json_object(raw)

    # Normalize: ensure required keys exist and values are strings
    keys = [
        "what_happened",
        "impact_analysis",
        "historical_context",
        "why_it_matters",
        "future_outlook",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        v = obj.get(k, "")
        v = re.sub(r"\s+", " ", str(v or "").strip())
        out[k] = v
    return out


def llm_timeline_events(title: str, summary: str, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Generate timeline events using LLM. Must return ISO timestamps."""
    llm = _llm(0.0)

    prompt = (
        "You are a news timeline generator.\n"
        "Given an article title, and topic, produce a concise timeline of how the story unfolded.\n\n"
        "Return ONLY valid JSON (no markdown, no extra text), as an array with this schema:\n"
        "[\n"
        "  {\"order\": 1, \"time\": \"ISO-8601 UTC timestamp\", \"short_details\": \"1 sentence\"}\n"
        "]\n\n"
        "Rules:\n"
        f"- Return at most {int(limit)} items.\n"
        "- order must start at 1 and increment by 1.\n"
        "- time MUST be ISO-8601 in UTC and must include 'Z' (e.g., 2026-02-11T04:15:22Z).\n"
        "- short_details must be <= 160 characters, factual, and not speculative.\n"
        "- If exact times are unknown, approximate using reasonable ordering and set times spaced apart on the same date; still output valid ISO UTC.\n\n"
        f"Title: {title}\n"
        f"Topic: {topic}\n"
    )

    raw = llm.invoke(prompt).content
    arr = _extract_json_array(raw)

    # Normalize, enforce limit and shape
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(arr[: max(1, int(limit))], start=1):
        t = str(it.get("time", "")).strip()
        d = str(it.get("short_details", "")).strip()
        if not t:
            # If model missed time, set a safe ISO timestamp now
            t = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if t.endswith("+00:00"):
            t = t.replace("+00:00", "Z")
        if not t.endswith("Z") and "+" not in t:
            # best-effort: treat as UTC
            t = t + "Z"
        d = re.sub(r"\s+", " ", d)
        if len(d) > 160:
            d = d[:157].rstrip() + "..."
        out.append({"order": idx, "time": t, "short_details": d})
    return out


def llm_interesting_facts(topic: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Generate interesting facts using LLM."""
    llm = _llm(0.0)
    prompt = (
        "You are an assistant that produces interesting, accurate facts.\n"
        "Return ONLY valid JSON (no markdown, no extra text) as an array with this schema:\n"
        "[\n"
        "  {\"fact_id\": \"F1\", \"details\": \"1-2 lines\"}\n"
        "]\n\n"
        "Rules:\n"
        f"- Return at most {int(limit)} facts.\n"
        "- fact_id must be unique and sequential: F1, F2, ...\n"
        "- details must be concise (<= 200 characters) and avoid speculation.\n"
        f"Topic: {topic}\n"
    )
    raw = llm.invoke(prompt).content
    arr = _extract_json_array(raw)

    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(arr[: max(1, int(limit))], start=1):
        d = str(it.get("details", "")).strip()
        d = re.sub(r"\s+", " ", d)
        if len(d) > 200:
            d = d[:197].rstrip() + "..."
        out.append({"fact_id": f"F{idx}", "details": d})
    return out


# -------------------------
# Helpers (parsing timeline)
# -------------------------

_REL_RE = re.compile(r"^(?P<n>\d+)\s+(?P<u>minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago$", re.I)


def _parse_relative_time(s: str, now: datetime) -> Optional[datetime]:
    m = _REL_RE.match((s or "").strip())
    if not m:
        return None
    n = int(m.group("n"))
    unit = m.group("u").lower()
    if unit.startswith("minute"):
        return now - timedelta(minutes=n)
    if unit.startswith("hour"):
        return now - timedelta(hours=n)
    if unit.startswith("day"):
        return now - timedelta(days=n)
    if unit.startswith("week"):
        return now - timedelta(weeks=n)
    if unit.startswith("month"):
        # approx month as 30 days
        return now - timedelta(days=30 * n)
    if unit.startswith("year"):
        return now - timedelta(days=365 * n)
    return None


_DATE_FORMATS = [
    "%b %d, %Y",   # Jan 02, 2026
    "%B %d, %Y",   # January 02, 2026
    "%d %b %Y",    # 02 Jan 2026
    "%Y-%m-%d",    # 2026-01-02
    "%b %d, %Y %I:%M %p",  # Jan 02, 2026 09:30 AM
    "%B %d, %Y %I:%M %p",
]


def parse_event_time(raw: str) -> Tuple[Optional[datetime], str]:
    """
    Returns (dt, display_str). dt may be None if parsing fails.
    We keep display_str as the original raw string (trimmed).
    """
    raw = (raw or "").strip()
    if not raw:
        return None, ""

    now = datetime.now(timezone.utc)

    rel = _parse_relative_time(raw, now)
    if rel:
        # keep the raw relative string for display
        return rel, raw

    # Try stripping leading source separators like " · "
    cleaned = raw.replace("•", " ").replace("·", " ").strip()

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            return dt, raw
        except Exception:
            pass

    # Sometimes SerpAPI returns something like "Jan 2" without year
    for fmt in ("%b %d", "%B %d"):
        try:
            dt = datetime.strptime(cleaned, fmt).replace(year=now.year, tzinfo=timezone.utc)
            return dt, raw
        except Exception:
            pass

    return None, raw


def build_timeline_events(serp_json: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Builds a normalized list of events from SerpAPI response.
    """
    items: List[Dict[str, Any]] = []

    # Preferred: google_news engine commonly returns "news_results"
    for it in (serp_json.get("news_results") or []):
        title = (it.get("title") or "").strip()
        snippet = (it.get("snippet") or it.get("summary") or "").strip()
        date = (it.get("date") or "").strip()
        if not (title or snippet):
            continue
        items.append({"time": date, "short_details": (title + (": " if snippet else "") + snippet).strip()})

    # Fallback: google engine returns "organic_results"
    if not items:
        for it in (serp_json.get("organic_results") or []):
            title = (it.get("title") or "").strip()
            snippet = (it.get("snippet") or "").strip()
            date = (it.get("date") or it.get("published_date") or "").strip()
            if not (title or snippet):
                continue
            items.append({"time": date, "short_details": (title + (": " if snippet else "") + snippet).strip()})

    # Clip + sort by parsed datetime if possible
    now = datetime.now(timezone.utc)
    enriched = []
    for it in items[: max(limit, 1) * 2]:
        dt, display = parse_event_time(it.get("time") or "")
        enriched.append((dt, display, it.get("short_details") or ""))

    # Determine if we have enough parseable datetimes to sort
    if sum(1 for dt, _, _ in enriched if dt is not None) >= 2:
        enriched.sort(key=lambda x: (x[0] is None, x[0] or now))
    # else preserve original order

    out: List[Dict[str, Any]] = []
    for idx, (dt, _display, details) in enumerate(enriched[:limit], start=1):
        # Keep it short-ish
        details = re.sub(r"\s+", " ", (details or "").strip())
        if len(details) > 240:
            details = details[:237].rstrip() + "..."

        # Always return ISO time (UTC). If we can't parse, use 'now' as a safe fallback.
        when = (dt or now).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        out.append({"order": idx, "time": when, "short_details": details})

    return out


def build_interesting_facts(serp_json: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    """
    Extracts short fact-like snippets from SerpAPI response.
    """
    facts: List[str] = []

    # Try answer box / knowledge graph first
    ab = serp_json.get("answer_box") or {}
    for k in ("snippet", "answer", "title"):
        v = (ab.get(k) or "").strip()
        if v:
            facts.append(v)

    kg = serp_json.get("knowledge_graph") or {}
    for k in ("description", "title", "type"):
        v = (kg.get(k) or "").strip()
        if v:
            facts.append(v)

    # Then organic/news snippets
    for it in (serp_json.get("organic_results") or []):
        sn = (it.get("snippet") or "").strip()
        if sn:
            facts.append(sn)

    for it in (serp_json.get("news_results") or []):
        sn = (it.get("snippet") or it.get("summary") or "").strip()
        if sn:
            facts.append(sn)

    # De-dup (case-insensitive) while preserving order
    seen = set()
    uniq: List[str] = []
    for f in facts:
        f2 = re.sub(r"\s+", " ", f).strip()
        if not f2:
            continue
        key = f2.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f2)
        if len(uniq) >= limit:
            break

    return [{"fact_id": f"F{idx}", "details": d if len(d) <= 220 else (d[:217].rstrip() + "...")} for idx, d in enumerate(uniq, start=1)]
