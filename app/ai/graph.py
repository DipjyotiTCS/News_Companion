import os
from typing import TypedDict, List, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.services.neo4j_service import Neo4jService
from app.services.vector_service import VectorService

import json
import re

class AgentState(TypedDict, total=False):
    question: str
    intent: str
    domain: str
    topic: str
    keywords: List[str]
    query_analyse_intent: str
    query_analyse_confidence: float
    query_analyse_rationale: str
    kb_query: str
    kb_hits: List[Dict[str, Any]]
    kb_best: Dict[str, Any]
    kb_confidence: float
    kb_context: str
    kb_answer: str

    # Neo4j graph retrieval
    graph_schema: str
    cypher: str
    cypher_params: Dict[str, Any]
    graph_rows_raw: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    graph_error: str

    retrieval_query: str
    results: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    guardrail_ok: bool
    guardrail_reason: str
    confidence: float
    final_answer: str
    response: Dict[str, Any]

def _llm(temp: float = 0.0) -> ChatOpenAI:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    return ChatOpenAI(model=model, temperature=temp)

# 1) Intent

def node_intent(state: AgentState) -> AgentState:
    llm = _llm(0.0)
    q = state["question"]

    prompt = (
        "You are a news assistant NLU classifier.\n"
        "Extract and return ONLY valid JSON (no markdown, no extra text) with this schema:\n"
        "{\n"
        '  "intent": "select one: latest_news|summarize_topic|explain_article|find_related|fact_check|advice|other",\n'
        '  "domain": "select one: politics|business|technology|sports|health|science|entertainment|world|us|local|crime|weather|other",\n'
        '  "topic": "select one: NFL|NBA|Olympics|Federal Budget|Retail|Technology|Sports Economy",\n'
        '  "keywords": ["3-10 lowercase keywords/phrases, no duplicates"]\n'
        '  "confidence": the confidence score of the LLM response. Ranging from 0 to 1. Decimal number.\n'
        '  "rationale": The explanation from LLM about the response thats generated.\n'
        "}\n\n"
        "Rules:\n"
        "- intent must be exactly one of the enum values.\n"
        "- domain must be exactly one of the enum values.\n"
        "- topic should be a concise label inferred from the question.\n"
        "- keywords should help retrieval (entities, places, issues, orgs).\n\n"
        f"Question: {q}\n"
    )

    raw = llm.invoke(prompt).content.strip()

    # Robust JSON extraction (in case model adds leading/trailing text)
    m = re.search(r"\{[\s\S]*\}$", raw)
    if not m:
        # fallback: minimal safe defaults
        return {"intent": "other", "domain": "other", "topic": "", "keywords": []}

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"intent": "other", "domain": "other", "topic": "", "keywords": []}

    # Normalize + guard
    intent = str(data.get("intent", "other")).strip().lower()
    domain = str(data.get("domain", "other")).strip().lower()
    topic = str(data.get("topic", "")).strip()
    keywords = data.get("keywords", [])
    confidence_score = str(data.get("confidence", "0")).strip().lower()
    rationale = str(data.get("rationale", "")).strip().lower()
    if not isinstance(keywords, list):
        keywords = []

    keywords = [str(k).strip().lower() for k in keywords if str(k).strip()]
    # de-dupe while preserving order
    seen = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))]

    supported_intents = {"latest_news","summarize_topic","explain_article","find_related","fact_check","advice","other"}
    supported_domains = {"politics","business","sports","climate","other"}

    if intent not in supported_intents:
        intent = "other"
    if domain not in supported_domains:
        domain = "other"

    print("Intent", intent)
    print("Domain", domain)
    print("Confidance score", confidence_score)
    print("rationale", rationale)
    return {"intent": intent, "domain": domain, "topic": topic, "keywords": keywords}

def node_analysis(state: AgentState) -> AgentState:
    llm = _llm(0.0)
    q = state["question"]

    prompt = (
        "You are an user query analyser.\n"
        "Analyse the provided user query and categorise it based on the type of the request.\n"
        "Return ONLY the correct category based on user's query analysis:\n"
        "{\n"
        '  "query_analyse_intent": "Return one of the two available categories mentioned bellow. either user_query or relational_query",\n'
        '  "query_analyse_confidence": the confidence score of the LLM response. Ranging from 0 to 1. Decimal number.\n'
        '  "query_analyse_rationale": The explanation from LLM about the response thats generated.\n'
        "}\n\n"
        "Available categories:\n"
        "user_query: Select this category only if the user is looking for some specific information. Some example are as follows.\n"
        "- What happened at the Super Bowl.\n"
        "- Who won the NFL.\n"
        "- How is today's weather\n\n"
        "relational_query: Select this category only if the user's query is indicating towards a collection of news articles.\n"
        "- What is happening in my locality?\n"
        "- What are today's top stories?\n"
        "- What are the headlines today?\n"
        "- What is going on in US politics?\n"
        "- What is happening in the NBA.\n"
        "Rules:\n"
        "- intent must be exactly one of the Available categories values.\n"
        "- Do NOT return anyting except the keywords user_query or relational_query.\n"
        f"Question: {q}\n"
    )

    raw = llm.invoke(prompt).content.strip()

    # Robust JSON extraction (in case model adds leading/trailing text)
    m = re.search(r"\{[\s\S]*\}$", raw)
    if not m:
        # fallback: minimal safe defaults
        return {"query_analyse_intent": "other"}

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"query_analyse_intent": "other"}

    # Normalize + guard
    query_analyse_intent = str(data.get("query_analyse_intent", "other")).strip().lower()
    query_analyse_confidence = str(data.get("query_analyse_confidence", "other")).strip().lower()
    query_analyse_rationale = str(data.get("query_analyse_rationale", "other")).strip().lower()
    

    print("query_analyse_intent", query_analyse_intent)
    return {
        "query_analyse_intent": query_analyse_intent,
        "query_analyse_confidence": query_analyse_confidence,
        "query_analyse_rationale": query_analyse_rationale,
    }

def _distance_to_confidence(score: float) -> float:
    """Chroma usually returns a distance score (lower is better). Convert to [0..1] confidence."""
    try:
        s = float(score)
    except Exception:
        return 0.0
    conf = 1.0 / (1.0 + max(0.0, s))
    return round(max(0.0, min(1.0, conf)), 4)

# 1.5) KB Semantic Retrieval (Chroma)
def node_kb_semantic(state: AgentState) -> AgentState:
    q = (state.get("question") or "").strip()
    keywords = state.get("keywords") or []
    kw_query = " ".join([str(k).strip() for k in keywords if str(k).strip()]).strip()
    kb_query = kw_query if kw_query else q

    vector = VectorService()

    hits: List[Dict[str, Any]] = []
    best_context = ""
    best_conf = 0.0
    best_meta: Dict[str, Any] = {}

    # Prefer similarity_search_with_score if available (gives distance score)
    if hasattr(vector, "db") and hasattr(vector.db, "similarity_search_with_score"):
        pairs = vector.db.similarity_search_with_score(kb_query, k=5)
        # distance: lower is better
        pairs = sorted(pairs, key=lambda x: x[1])
        for doc, score in pairs:
            meta = doc.metadata or {}
            conf = _distance_to_confidence(score)
            hits.append({
                "article_id": meta.get("article_id", ""),
                "title": meta.get("title", ""),
                "topic": meta.get("topic", ""),
                "domain": meta.get("domain", ""),
                "score_raw": float(score),
                "confidence": conf,
            })
        if pairs:
            best_doc, best_score = pairs[0]
            best_context = best_doc.page_content or ""
            best_meta = best_doc.metadata or {}
            best_conf = _distance_to_confidence(best_score)
    else:
        # fallback without scores
        docs = vector.search(kb_query, k=5)
        for doc in docs:
            meta = doc.metadata or {}
            hits.append({
                "article_id": meta.get("article_id", ""),
                "title": meta.get("title", ""),
                "topic": meta.get("topic", ""),
                "domain": meta.get("domain", ""),
                "score_raw": 999.0,
                "confidence": 0.0,
            })
        if docs:
            best_context = docs[0].page_content or ""
            best_meta = docs[0].metadata or {}
            best_conf = 0.0

    print("kb hits", hits)
    print("kb query", kb_query)
    print("kb confidence", best_conf)

    return {
        "kb_query": kb_query,
        "kb_hits": hits,
        "kb_best": {
            "article_id": best_meta.get("article_id", ""),
            "title": best_meta.get("title", ""),
            "topic": best_meta.get("topic", ""),
            "domain": best_meta.get("domain", ""),
        },
        "kb_confidence": best_conf,
        "kb_context": best_context,
    }

# 1.6) KB Answer (only if kb_confidence >= 0.65)
def node_kb_answer(state: AgentState) -> AgentState:
    conf = float(state.get("kb_confidence", 0.0))
    if conf < 0.65:
        return {"kb_answer": ""}

    llm = _llm(0.2)
    question = state.get("question", "")
    context = state.get("kb_context", "")

    prompt = (
        "You are a news assistant. Use the provided knowledge base excerpt to answer the user's question.\n"
        "If the excerpt does not contain enough information DO NOT mention that in the answer.\n\n"
        f"User question: {question}\n\n"
        f"Knowledge base excerpt:\n{context}\n"
    )
    kb_answer = llm.invoke(prompt).content.strip()

    print("kb_answer", kb_answer)

    return {
        "kb_answer": kb_answer
    }


# -------------------------
# Neo4j schema -> Cypher -> execute
# -------------------------
from typing import Optional

_SCHEMA_CACHE: Optional[str] = None

def node_graph_schema(state: AgentState) -> AgentState:
    """Extract Neo4j schema text once per process and store in state."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE:
        return {"graph_schema": _SCHEMA_CACHE}

    neo = Neo4jService()
    try:
        schema_text = neo.get_schema_text()
    finally:
        neo.close()

    _SCHEMA_CACHE = schema_text
    return {"graph_schema": schema_text}


def node_generate_cypher(state: AgentState) -> AgentState:
    llm = _llm(0.0)

    question = state.get("question", "")
    domain = state.get("domain", "")
    topic = state.get("topic", "")
    qintent = state.get("query_analyse_intent", "")
    schema = state.get("graph_schema", "")

    # Keep this prompt *very* restrictive: read-only, return JSON only.
    prompt = (
        "You are an expert Neo4j Cypher generator for a local news graph database.\n"
        "You MUST generate a READ-ONLY Cypher query. Do NOT use CREATE, MERGE, DELETE, SET, DROP, APOC, LOAD CSV, or any admin procedures.\n"
        "Return ONLY valid JSON (no markdown, no extra text) with this schema:\n"
        "{\n"
        "  \"cypher\": \"...\",\n"
        "  \"params\": { ... },\n"
        "  \"rationale\": \"short explanation\"\n"
        "}\n\n"
        "The query should return a list of Article-like rows (id, title, description/details, topic, domain, references)\n"
        "and should include a LIMIT (<= 10).\n\n"
        "Neo4j schema (compact):\n"
        f"{schema}\n\n"
        "User inputs:\n"
        f"- question: {question}\n"
        f"- domain: {domain}\n"
        f"- topic: {topic}\n"
        f"- query_analyse_intent: {qintent}\n\n"
        "Guidelines:\n"
        "- If query_analyse_intent is 'relational_query', prefer traversals: Article -> Topic/Domain/Location/Author/Reference, and related articles via SAME_* relationships.\n"
        "- If query_analyse_intent is 'user_query', prefer directly matching Article properties, fulltext index 'articleIndex' if relevant, or Topic/Domain nodes.\n"
        "- Always project fields as: id, title, description, topic, domain, references.\n"
        "- If the graph uses property names like details/summery instead of description, coalesce them.\n"
    )

    raw = llm.invoke(prompt).content.strip()
    m = re.search(r"\{[\s\S]*\}$", raw)
    if not m:
        # safe fallback to fulltext query
        q = question
        cypher = (
            "CALL db.index.fulltext.queryNodes('articleIndex', $q) YIELD node, score "
            "RETURN node.id AS id, node.title AS title, "
            "coalesce(node.description, node.details, node.body, '') AS description, "
            "coalesce(node.topic,'') AS topic, coalesce(node.domain,'') AS domain, "
            "coalesce(node.references, []) AS references, score "
            "ORDER BY score DESC LIMIT 6"
        )
        return {"cypher": cypher, "cypher_params": {"q": q}}

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"cypher": "", "cypher_params": {}, "graph_error": "Failed to parse cypher JSON from LLM."}

    cypher = (data.get("cypher") or "").strip()
    params = data.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    return {"cypher": cypher, "cypher_params": params}


def node_run_cypher(state: AgentState) -> AgentState:
    cypher = (state.get("cypher") or "").strip()
    params = state.get("cypher_params") or {}

    if not cypher:
        return {"results": [], "graph_error": state.get("graph_error", "No cypher generated.")}

    neo = Neo4jService()
    try:
        rows = neo.run_readonly(cypher, params)
    except Exception as e:
        # Fallback to the existing fulltext search to keep the app useful.
        try:
            q = state.get("question", "")
            rows = neo.search(q, limit=int(os.environ.get("MAX_RETRIEVAL", "6")))
        except Exception:
            rows = []
        return {"graph_rows_raw": [], "results": _normalize_article_rows(rows), "graph_error": str(e)}
    finally:
        neo.close()

    return {"graph_rows_raw": rows, "results": _normalize_article_rows(rows)}


def _normalize_article_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize arbitrary Cypher result rows into the article schema used by node_summarize."""
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        # Rows could either be flat projections (id/title/...) or nested under 'node'
        node = r.get("node") if isinstance(r, dict) else None
        if node and isinstance(node, dict):
            base = node
        else:
            base = r or {}

        out.append({
            "id": base.get("id") or base.get("article_id") or r.get("id"),
            "title": base.get("title") or r.get("title") or "",
            "description": (
                base.get("description")
                or base.get("details")
                or base.get("body")
                or r.get("description")
                or r.get("details")
                or r.get("body")
                or ""
            ),
            "topic": base.get("topic") or r.get("topic") or "",
            "domain": base.get("domain") or r.get("domain") or "",
            "references": base.get("references") or r.get("references") or [],
            "score": float(r.get("score") or 0.0),
        })
    return out

def _route_after_kb(state: AgentState) -> str:
    return "kb_answer" if float(state.get("kb_confidence", 0.0)) > 0.7 else "domain_node"

# 3) Retrieval
def node_retrieve(state: AgentState) -> AgentState:
    """Backwards-compatible fulltext retrieval (kept as a fallback path)."""
    max_k = int(os.environ.get("MAX_RETRIEVAL", "6"))
    q = state.get("question", "")
    domain = state.get("domain", "")
    rq = q if not domain or domain.lower() == "other" else f"{q} {domain}"

    neo = Neo4jService()
    try:
        rows = neo.search(rq, limit=max_k)
    finally:
        neo.close()

    return {"retrieval_query": rq, "results": _normalize_article_rows(rows)}

# 4) Summarize
def node_summarize(state: AgentState) -> AgentState:
    llm = _llm(0.2)
    summaries: List[Dict[str, Any]] = []
    for r in (state.get("results") or []):
        prompt = (
            "Summarize this news article in 4-6 bullet points.\n"
            "Be factual, avoid speculation. If details are missing, say so.\n"
            f"Title: {r.get('title','')}\n"
            f"Description: {r.get('description','')}\n"
            f"Topic: {r.get('topic','')}\n"
            f"Domain: {r.get('domain','')}\n"
            f"References: {r.get('references',[])}\n"
        )
        summary = llm.invoke(prompt).content.strip()
        summaries.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "topic": r.get("topic"),
            "domain": r.get("domain"),
            "references": r.get("references") or [],
            "summary": summary,
            "score": float(r.get("score") or 0.0),
        })
    return {"summaries": summaries}

# 5) Guardrails
def node_guardrails(state: AgentState) -> AgentState:
    q = (state.get("question") or "").lower()
    blocked = ["how to make a bomb", "buy drugs", "self harm", "kill myself"]
    for b in blocked:
        if b in q:
            return {"guardrail_ok": False, "guardrail_reason": "Request is not allowed."}
    return {"guardrail_ok": True, "guardrail_reason": ""}

# 6) Confidence
def node_confidence(state: AgentState) -> AgentState:
    summaries = state.get("summaries") or []
    if not summaries:
        return {"confidence": 0.2}
    mx = max([s.get("score", 0.0) for s in summaries] or [0.0])
    conf = 1.0 - (1.0 / (1.0 + max(mx, 0.0)))
    conf = max(0.0, min(1.0, conf))
    return {"confidence": round(conf, 2)}

# 7) Response
def node_response(state: AgentState) -> AgentState:
    if not state.get("guardrail_ok", True):
        return {"response": {"type": "blocked", "message": state.get("guardrail_reason", "Blocked.")}}

    # Optionally produce a single "final" answer string that blends the KB answer and
    # the retrieved graph summaries. The UI can show this above the items.
    kb_answer = (state.get("kb_answer") or "").strip()
    items = state.get("summaries", []) or []

    final_msg = ""
    if kb_answer and items:
        llm = _llm(0.2)
        top = items[:3]
        prompt = (
            "You are a news assistant. Combine the knowledge-base answer with the article summaries.\n"
            "Write a concise final response (5-10 sentences). Be factual.\n\n"
            f"User question: {state.get('question','')}\n\n"
            f"Knowledge-base answer:\n{kb_answer}\n\n"
            "Top related articles (summaries):\n"
            + "\n".join([f"- {t.get('title','')}: {t.get('summary','')}" for t in top])
        )
        final_msg = llm.invoke(prompt).content.strip()
    elif kb_answer:
        final_msg = kb_answer

    return {
        "response": {
            "type": "results",
            "intent": state.get("intent", ""),
            "domain": state.get("domain", ""),
            "confidence": state.get("confidence", 0.0),
            "message": final_msg,
            "kb_confidence": state.get("kb_confidence", 0.0),
            "graph_error": state.get("graph_error", ""),
            "items": items,
        }
    }

_GRAPH = None

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("intent_node", node_intent)
    g.add_node("analysis_node", node_analysis)
    g.add_node("kb_semantic", node_kb_semantic)
    # NOTE: LangGraph node names must NOT collide with state keys.
    # Keep the state key as `kb_answer`, but use a different node id.
    g.add_node("kb_answer_node", node_kb_answer)

    # Neo4j Cypher pipeline
    g.add_node("graph_schema_node", node_graph_schema)
    g.add_node("graph_cypher", node_generate_cypher)
    g.add_node("graph_query", node_run_cypher)

    # Fallback fulltext retrieval node is kept, but graph_query is the primary.
    g.add_node("retrieve", node_retrieve)
    g.add_node("summarize", node_summarize)
    g.add_node("guardrails", node_guardrails)
    g.add_node("confidence_node", node_confidence)
    g.add_node("respond", node_response)

    g.set_entry_point("intent_node")
    g.add_edge("intent_node", "analysis_node")
    g.add_edge("analysis_node", "kb_semantic")

    # Always try to produce a KB answer if confidence is high, but continue to graph search
    g.add_edge("kb_semantic", "kb_answer_node")
    g.add_edge("kb_answer_node", "graph_schema_node")
    g.add_edge("graph_schema_node", "graph_cypher")
    g.add_edge("graph_cypher", "graph_query")

    # If graph query yields nothing, summarize will just return empty.
    g.add_edge("graph_query", "retrieve")
    g.add_edge("retrieve", "summarize")
    g.add_edge("summarize", "guardrails")
    g.add_edge("guardrails", "confidence_node")
    g.add_edge("confidence_node", "respond")
    g.add_edge("respond", END)
    return g.compile()

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH

def run_agent(question: str) -> Dict[str, Any]:
    graph = get_graph()
    out = graph.invoke({"question": question})
    return out["response"]
