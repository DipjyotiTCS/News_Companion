import os
import re
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase

class Neo4jService:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687").strip()
        self.user = os.environ.get("NEO4J_USER", "neo4j").strip()
        self.password = os.environ.get("NEO4J_PASSWORD", "password").strip()

        if self.uri.startswith("bolt://localhost") or self.uri.startswith("neo4j://localhost"):
            print("[neo4j] Using local Neo4j instance at localhost.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session() as session:
            res = session.run(cypher, params)
            return [r.data() for r in res]

    # -------------------------
    # Schema extraction helpers
    # -------------------------
    def get_schema(self) -> Dict[str, Any]:
        """Return a compact, LLM-friendly schema description.

        Uses Neo4j built-in schema procedures (no APOC dependency).
        """
        schema: Dict[str, Any] = {
            "node_labels": [],
            "relationship_types": [],
            "node_properties": {},
            "relationship_properties": {},
        }

        # Node labels + their properties
        try:
            rows = self.run("CALL db.schema.nodeTypeProperties()")
            # Neo4j returns keys: nodeType, nodeLabels, propertyName, propertyTypes, mandatory
            for r in rows:
                labels = r.get("nodeLabels") or []
                prop = r.get("propertyName")
                ptypes = r.get("propertyTypes") or []
                mandatory = bool(r.get("mandatory"))
                for lab in labels:
                    schema["node_properties"].setdefault(lab, [])
                    if prop:
                        schema["node_properties"][lab].append({
                            "name": prop,
                            "types": ptypes,
                            "mandatory": mandatory,
                        })
        except Exception:
            # If the procedure is unavailable for any reason, fall back to simple introspection.
            pass

        # Relationship types + their properties
        try:
            rows = self.run("CALL db.schema.relTypeProperties()")
            # keys: relType, propertyName, propertyTypes, mandatory
            for r in rows:
                rel = r.get("relType")
                prop = r.get("propertyName")
                ptypes = r.get("propertyTypes") or []
                mandatory = bool(r.get("mandatory"))
                if rel:
                    schema["relationship_properties"].setdefault(rel, [])
                    if prop:
                        schema["relationship_properties"][rel].append({
                            "name": prop,
                            "types": ptypes,
                            "mandatory": mandatory,
                        })
        except Exception:
            pass

        # Relationship topology (start/end labels)
        try:
            rows = self.run("CALL db.schema.visualization()")
            # returns nodes + relationships arrays
            if rows:
                r0 = rows[0]
                rels = r0.get("relationships") or []
                node_map = {}
                for n in (r0.get("nodes") or []):
                    # Each node is a Node object; .labels may not be directly accessible via dict.
                    # The driver returns it as a neo4j.graph.Node, which becomes a dict-like in r.data().
                    # Best-effort extract:
                    try:
                        node_id = n.id  # type: ignore[attr-defined]
                        labels = list(getattr(n, "labels", []) or [])
                    except Exception:
                        node_id = None
                        labels = []
                    if node_id is not None:
                        node_map[node_id] = labels

                edges = []
                rel_types = set()
                for rel in rels:
                    try:
                        rtype = getattr(rel, "type", None)
                        start = getattr(rel, "start_node", None)
                        end = getattr(rel, "end_node", None)
                        s_labels = node_map.get(getattr(start, "id", None), [])
                        e_labels = node_map.get(getattr(end, "id", None), [])
                    except Exception:
                        rtype, s_labels, e_labels = None, [], []
                    if rtype:
                        rel_types.add(rtype)
                        edges.append({
                            "type": rtype,
                            "from": s_labels,
                            "to": e_labels,
                        })
                schema["relationship_types"] = sorted(list(rel_types))
                schema["relationships"] = edges
        except Exception:
            schema["relationships"] = []

        schema["node_labels"] = sorted(list(schema.get("node_properties", {}).keys()))
        return schema

    def get_schema_text(self, max_props_per_label: int = 25) -> str:
        """Human-readable schema string for prompting."""
        s = self.get_schema()
        out: List[str] = []
        out.append("Node labels and properties:")
        for lab in s.get("node_labels", []):
            props = s.get("node_properties", {}).get(lab, [])
            # de-dupe property names (the procedure can return multiple lines per label)
            seen = set()
            uniq = []
            for p in props:
                name = p.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)
                uniq.append(p)
            uniq = uniq[: max_props_per_label]
            plist = ", ".join([
                f"{p.get('name')}:{'|'.join(p.get('types') or [])}{'!' if p.get('mandatory') else ''}"
                for p in uniq
            ])
            out.append(f"- {lab}({plist})")

        out.append("\nRelationship types:")
        for rt in s.get("relationship_types", []):
            rprops = s.get("relationship_properties", {}).get(rt, [])
            seen = set()
            uniq = []
            for p in rprops:
                name = p.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)
                uniq.append(p)
            plist = ", ".join([
                f"{p.get('name')}:{'|'.join(p.get('types') or [])}{'!' if p.get('mandatory') else ''}"
                for p in uniq
            ])
            if plist:
                out.append(f"- {rt}({plist})")
            else:
                out.append(f"- {rt}")

        out.append("\nCommon relationship patterns:")
        for e in (s.get("relationships") or [])[:40]:
            out.append(f"- ({'|'.join(e.get('from') or ['?'])})-[:{e.get('type','?')}]->({'|'.join(e.get('to') or ['?'])})")

        return "\n".join(out)

    # -------------------------
    # Safer read-only execution
    # -------------------------
    def run_readonly(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query but block obvious write operations."""
        c = (cypher or "").strip()
        # Strip leading comments
        c_no_comments = re.sub(r"(?m)^//.*$", "", c).strip()
        upper = c_no_comments.upper()
        forbidden = [
            "CREATE ", "MERGE ", "DELETE ", "DETACH DELETE", "SET ", "DROP ", "CALL DB.CREATE",
            "CALL DBMS", "APOC", "LOAD CSV", "PERIODIC COMMIT",
        ]
        if any(tok in upper for tok in forbidden):
            raise ValueError("Write/administrative Cypher is not allowed in read-only mode.")
        return self.run(cypher, params)

    def ensure_schema(self):
        # Article node
        self.run("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")

        # Dimension nodes (richer graph)
        self.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
        self.run("CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE")
        self.run("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (au:Author) REQUIRE au.name IS UNIQUE")
        self.run("CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE")
        self.run("CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE")
        self.run("CREATE CONSTRAINT ref_url IF NOT EXISTS FOR (r:Reference) REQUIRE r.url IS UNIQUE")

        # Helpful indexes
        self.run("CREATE INDEX article_domain IF NOT EXISTS FOR (a:Article) ON (a.domain)")
        self.run("CREATE INDEX article_topic IF NOT EXISTS FOR (a:Article) ON (a.topic)")
        self.run("CREATE INDEX article_author IF NOT EXISTS FOR (a:Article) ON (a.author)")
        self.run("CREATE INDEX article_country IF NOT EXISTS FOR (a:Article) ON (a.country)")
        self.run("CREATE INDEX article_location IF NOT EXISTS FOR (a:Article) ON (a.location)")

        # Fulltext index for retrieval (chat uses this)
        self.run("""CREATE FULLTEXT INDEX articleIndex IF NOT EXISTS
                 FOR (a:Article) ON EACH [
                   a.title, a.details, a.summery, a.description, a.body,
                   a.topic, a.domain, a.author, a.country, a.location
                 ]""")

    def upsert_articles(self, articles: List[Dict[str, Any]]):
        # 12-article PDF -> simple per-article merges are fine (clear + debuggable).
        for a in articles:
            self.run(
                """MERGE (n:Article {id: $id})
                   SET n.title=$title,
                       n.details=$details,
                       n.summery=$summery,
                       n.description=$description,
                       n.body=$body,
                       n.author=$author,
                       n.date=$date,
                       n.isheadline=$isheadline,
                       n.topic=$topic,
                       n.domain=$domain,
                       n.country=$country,
                       n.location=$location,
                       n.reference1=$reference1,
                       n.reference2=$reference2,
                       n.references=$references""",
                a,
            )

            # Link to dimension nodes
            if (a.get("topic") or "").strip():
                self.run(
                    """MATCH (n:Article {id:$id})
                       MERGE (t:Topic {name:$topic})
                       MERGE (n)-[:HAS_TOPIC]->(t)""",
                    {"id": a["id"], "topic": a["topic"].strip()},
                )

            if (a.get("domain") or "").strip():
                self.run(
                    """MATCH (n:Article {id:$id})
                       MERGE (d:Domain {name:$domain})
                       MERGE (n)-[:IN_DOMAIN]->(d)""",
                    {"id": a["id"], "domain": a["domain"].strip()},
                )

            if (a.get("author") or "").strip():
                self.run(
                    """MATCH (n:Article {id:$id})
                       MERGE (au:Author {name:$author})
                       MERGE (n)-[:WRITTEN_BY]->(au)""",
                    {"id": a["id"], "author": a["author"].strip()},
                )

            if (a.get("country") or "").strip():
                self.run(
                    """MATCH (n:Article {id:$id})
                       MERGE (c:Country {name:$country})
                       MERGE (n)-[:IN_COUNTRY]->(c)""",
                    {"id": a["id"], "country": a["country"].strip()},
                )

            if (a.get("location") or "").strip():
                self.run(
                    """MATCH (n:Article {id:$id})
                       MERGE (l:Location {name:$location})
                       MERGE (n)-[:IN_LOCATION]->(l)""",
                    {"id": a["id"], "location": a["location"].strip()},
                )

            refs = a.get("references") or []
            for url in refs:
                u = (url or "").strip()
                if not u:
                    continue
                self.run(
                    """MATCH (n:Article {id:$id})
                       MERGE (r:Reference {url:$url})
                       MERGE (n)-[:HAS_REFERENCE]->(r)""",
                    {"id": a["id"], "url": u},
                )

    def build_relationships(self):
        # Optional direct article-to-article edges for convenience
        self.run("""MATCH (a:Article), (b:Article)
                   WHERE a.id < b.id AND a.topic <> '' AND a.topic = b.topic
                   MERGE (a)-[:SAME_TOPIC]->(b)""")
        self.run("""MATCH (a:Article), (b:Article)
                   WHERE a.id < b.id AND a.domain <> '' AND a.domain = b.domain
                   MERGE (a)-[:SAME_DOMAIN]->(b)""")
        self.run("""MATCH (a:Article), (b:Article)
                   WHERE a.id < b.id AND a.author <> '' AND a.author = b.author
                   MERGE (a)-[:SAME_AUTHOR]->(b)""")
        self.run("""MATCH (a:Article), (b:Article)
                   WHERE a.id < b.id AND a.country <> '' AND a.country = b.country
                   MERGE (a)-[:SAME_COUNTRY]->(b)""")
        self.run("""MATCH (a:Article), (b:Article)
                   WHERE a.id < b.id AND a.location <> '' AND a.location = b.location
                   MERGE (a)-[:SAME_LOCATION]->(b)""")
        self.run("""MATCH (a:Article), (b:Article)
                   WHERE a.id < b.id AND size(a.references) > 0 AND size(b.references) > 0
                   WITH a,b, [r IN a.references WHERE r IN b.references] AS shared
                   WHERE size(shared) > 0
                   MERGE (a)-[:SHARES_REFERENCE {count:size(shared)}]->(b)""")

    def search(self, query: str, limit: int = 6) -> List[Dict[str, Any]]:
        return self.run(
            """CALL db.index.fulltext.queryNodes('articleIndex', $q) YIELD node, score
               RETURN node.id AS id,
                      node.title AS title,
                      node.description AS description,
                      node.topic AS topic,
                      node.domain AS domain,
                      node.references AS references,
                      score
               ORDER BY score DESC
               LIMIT $limit""",
            {"q": query, "limit": int(limit)},
        )

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single article record by id."""
        rows = self.run(
            """MATCH (a:Article {id:$id})
               RETURN a.id AS id,
                      a.title AS title,
                      a.details AS details,
                      a.summery AS summery,
                      a.description AS description,
                      a.body AS body,
                      a.author AS author,
                      a.date AS date,
                      a.topic AS topic,
                      a.domain AS domain,
                      a.country AS country,
                      a.location AS location,
                      a.references AS references,
                      a.reference1 AS reference1,
                      a.reference2 AS reference2
               LIMIT 1""",
            {"id": article_id},
        )
        return rows[0] if rows else None
