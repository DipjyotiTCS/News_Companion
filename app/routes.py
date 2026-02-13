from flask import Blueprint, render_template, request, jsonify, abort
from app.ai.graph import run_agent
from app.services.neo4j_service import Neo4jService
from app.services.serpapi_service import (
    SerpApiService,
    build_timeline_events,
    build_interesting_facts,
    llm_timeline_events,
    llm_interesting_facts,
    llm_longform_analysis,
)

bp = Blueprint("main", __name__)

@bp.get("/")
def home():
    return render_template("home.html")

@bp.get("/chat")
def chat():
    return render_template("chat.html")


@bp.get("/article/<path:article_id>")
def article(article_id: str):
    # Render a professional, standalone article page.
    neo = Neo4jService()
    try:
        a = neo.get_article(article_id)
    finally:
        neo.close()

    if not a:
        abort(404)
    return render_template("article.html", article=a)

@bp.get("/article_json/<path:article_id>")
def article_json(article_id: str):
    # Return article data as JSON
    neo = Neo4jService()
    try:
        a = neo.get_article(article_id)
    finally:
        neo.close()

    if not a:
        abort(404)

    return jsonify(a)

@bp.post("/api/query")
def api_query():
    data = request.get_json(silent=True) or {}
    q = (data.get("message") or "").strip()
    if not q:
        return jsonify({"error": "Empty message"}), 400
    try:
        return jsonify(run_agent(q))
    except Exception as e:
        return jsonify({"error": f"Agent failed: {e}"}), 500


@bp.post("/api/timeline")
def api_timeline():
    """
    Payload:
      {
        "title": "...",
        "summary": "...",
        "topic": "..."
      }

    Response:
      [
        {"order": 1, "time": "...", "short_details": "..."},
        ...
      ]
    """
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    summary = (data.get("summary") or data.get("summery") or "").strip()  # allow common misspelling
    topic = (data.get("topic") or "").strip()

    if not (title or summary or topic):
        return jsonify({"error": "Provide at least one of: title, summary, topic"}), 400

    q = " ".join([p for p in [title, topic] if p]).strip()
    # include a hint that we want chronology / unfolding
    if summary:
        q = f"{q} {summary}"
    q = f"{q} key events timeline"

    # Prefer LLM output; fallback to SerpAPI if LLM fails.
    try:
        events = llm_timeline_events(title=title, summary=summary, topic=topic, limit=5)
        return jsonify(events)
    except Exception:
        try:
            serp = SerpApiService()
            serp_json = serp.search(q, num=10)
            events = build_timeline_events(serp_json, limit=5)
            return jsonify(events)
        except Exception as e:
            return jsonify({"error": f"Timeline generation failed: {e}"}), 500


@bp.route("/api/interesting-facts", methods=["GET", "POST"])
def api_interesting_facts():
    """
    GET: /api/interesting-facts?topic=...
    POST payload:
      {"topic": "..."}
    Response:
      [
        {"fact_id": "F1", "details": "..."},
        ...
      ]
    """
    topic = ""
    if request.method == "GET":
        topic = (request.args.get("topic") or "").strip()
    else:
        data = request.get_json(silent=True) or {}
        topic = (data.get("topic") or "").strip()

    if not topic:
        return jsonify({"error": "Missing topic"}), 400

    q = f"interesting facts about {topic}"

    # Prefer LLM output; fallback to SerpAPI if LLM fails.
    try:
        facts = llm_interesting_facts(topic=topic, limit=8)
        return jsonify(facts)
    except Exception:
        try:
            # Facts tend to come from organic results/answer box; use google engine if configured
            serp = SerpApiService()
            serp_json = serp.search(q, num=10)
            facts = build_interesting_facts(serp_json, limit=8)
            return jsonify(facts)
        except Exception as e:
            return jsonify({"error": f"Facts generation failed: {e}"}), 500


@bp.post("/api/longform-analysis")
def api_longform_analysis():
    """
    Payload:
      {
        "title": "...",
        "description": "...",
        "topic": "..."
      }

    Response:
      {
        "what_happened": "...",
        "impact_analysis": "...",
        "historical_context": "...",
        "why_it_matters": "...",
        "future_outlook": "..."
      }
    """
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    description = (data.get("description") or data.get("summary") or data.get("summery") or "").strip()
    topic = (data.get("topic") or "").strip()

    if not (title or description or topic):
        return jsonify({"error": "Provide at least one of: title, description, topic"}), 400

    try:
        analysis = llm_longform_analysis(title=title, description=description, topic=topic)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": f"Longform analysis failed: {e}"}), 500
