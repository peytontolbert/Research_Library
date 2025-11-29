"""
Health check for Neo4j connectivity using Neo4jGraphClient.
Loads NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD (dotenv supported).

Usage:
  python -m models.mirrormind.scripts.check_neo4j
"""

from models.mirrormind.graph_neo4j import Neo4jGraphClient


def main():
    try:
        client = Neo4jGraphClient()
        res = client.search("test", top_k=1)
        print("Neo4j connection OK; search result length:", len(res))
        client.close()
    except Exception as exc:
        print(f"Neo4j connection failed: {exc}")
        print("Check NEO4J_URI/USER/PASSWORD (dotenv supported).")


if __name__ == "__main__":
    main()
