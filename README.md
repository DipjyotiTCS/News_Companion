# Agentic News Chat (Flask + LangGraph + Neo4j Aura)

This demo app:
- Ingests a **local PDF** (`data/news.pdf`) into **Neo4j Aura** at startup (idempotent).
- Builds a LangGraph agent pipeline:
  1) Intent analysis (LLM)
  2) Domain classification (LLM)
  3) Neo4j retrieval (full-text search)
  4) Summarize retrieved articles (LLM)
  5) Guardrails
  6) Confidence scoring
  7) Response formatting (UI list of titles + right-side summary panel)

## 1) Prereqs
- A running **Neo4j Aura** database
- Your Aura connection details: **URI, username, password**
- OpenAI API key

## 2) Place your PDF
Put your PDF at:
```
data/news.pdf
```

## 3) Setup Python
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4) Configure `.env` for Aura
Create a `.env` file in the project root (you can copy `.env.example`):

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# IMPORTANT: Aura uses neo4j+s:// (TLS)
NEO4J_URI=neo4j+s://<your-aura-host>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

INGEST_ON_STARTUP=1
MAX_RETRIEVAL=6
# NEWS_PDF_PATH=./data/news.pdf
```

## 5) Run
```bash
python run.py
```

Open:
- Home: http://127.0.0.1:5000/
- Chat: http://127.0.0.1:5000/chat

### Notes
- Ingestion is best-effort; if Aura/PDF isn't ready, the server still starts and prints a warning.
- Ingestion creates:
  - `(:Article {id,title,description,body,topic,domain,references})`
  - Relationships: `:SAME_TOPIC`, `:SAME_DOMAIN`, `:SHARES_REFERENCE`
- Retrieval uses a Neo4j 5 full-text index (`articleIndex`).


## Knowledge base PDF
This project includes a sample PDF at `data/news.pdf` (U.S. News Digest – February 10, 2026).

## Neo4j Article schema
Each article becomes a `:Article` node with properties:
- title, details, summery, author, date, isheadline, topic, domain, country, location, reference1, reference2

The app also keeps backward-compatible properties for retrieval:
- description (same as `summery`)
- body (same as `details`)
- references (list of URLs)

Relationships created:
- :SAME_DOMAIN, :SAME_TOPIC, :SAME_AUTHOR, :SAME_LOCATION, :SAME_COUNTRY, :SHARES_REFERENCE


## Rich graph model
This version creates Topic/Domain/Author/Country/Location/Reference nodes and links articles to them.


## Startup ingestion is idempotent
By default, the app ingests the PDF only if no `:Article` nodes exist.
To force re-ingestion:

```env
FORCE_REINGEST=1
# optionally wipe graph before ingest
CLEAR_GRAPH_ON_REINGEST=1
```
