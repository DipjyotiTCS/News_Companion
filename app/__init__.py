import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from .routes import bp

from app.services.neo4j_service import Neo4jService
from app.services.vector_service import VectorService
from app.services.pdf_ingest import ingest_pdf_to_articles
from flask_cors import CORS

def _try_ingest():
    """Ingest PDF -> Neo4j + Chroma on startup, but only if graph is empty (unless forced).

    Env flags:
      - INGEST_ON_STARTUP=1 (default) to enable
      - FORCE_REINGEST=1 to ingest even if data exists
      - CLEAR_GRAPH_ON_REINGEST=1 (with FORCE_REINGEST) to wipe the graph before ingesting
    """
    if os.environ.get("INGEST_ON_STARTUP", "1").strip().lower() not in ("1", "true", "yes", "on"):
        return

    force = os.environ.get("FORCE_REINGEST", "0").strip().lower() in ("1", "true", "yes", "on")
    clear = os.environ.get("CLEAR_GRAPH_ON_REINGEST", "0").strip().lower() in ("1", "true", "yes", "on")

    pdf_path = os.environ.get("NEWS_PDF_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "news.pdf"))
    pdf_path = os.path.abspath(pdf_path)

    neo = Neo4jService()
    try:
        neo.ensure_schema()

        existing = neo.run("MATCH (a:Article) RETURN count(a) AS c")[0]["c"]
        if existing and not force:
            print(f"[startup] Detected {existing} existing Article nodes; skipping ingestion.")
            return

        if existing and force and clear:
            neo.run("MATCH (n) DETACH DELETE n")
            neo.ensure_schema()

        articles = ingest_pdf_to_articles(pdf_path)
        if not articles:
            print(f"[startup] No articles found in {pdf_path}; skipping ingestion.")
            return

        neo.upsert_articles(articles)
        neo.build_relationships()

        vector = VectorService()
        vector.ingest_articles(articles)

        print(f"[startup] Ingested {len(articles)} articles from {pdf_path}")
    except Exception as e:
        print(f"[startup] Ingestion skipped/failed: {e}")
    finally:
        neo.close()


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

    CORS(app)
    app.register_blueprint(bp)

    try:
        _try_ingest()
    except Exception as e:
        print(f"[startup] Ingestion skipped/failed: {e}")

    return app
