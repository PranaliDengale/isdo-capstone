"""ISDO Lab C2 - Mock Jira REST API (port 5002).
Run from project root:  python mcp_server/jira_shim.py
"""
import csv
from pathlib import Path

from flask import Flask, jsonify, request

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "requests.csv"
FILTERS = ["request_type", "priority", "assignee", "status"]

app = Flask(__name__)


def load_requests():
    """Load requests.csv into an in-memory dict keyed by request key."""
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found - starting empty.")
        return {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return {row["key"]: dict(row) for row in csv.DictReader(f)}


REQUESTS = load_requests()


def to_jira(req):
    """Convert a flat CSV row into Jira's nested 'fields' shape."""
    return {
        "key": req["key"],
        "fields": {
            "summary": req.get("summary"),
            "issuetype": {"name": req.get("request_type")},
            "priority": {"name": req.get("priority")},
            "status": {"name": req.get("status")},
            "assignee": {"displayName": req.get("assignee")},
            "customfield_sla": req.get("sla"),
        },
    }


@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    """All requests, filtered by ?request_type=, ?priority=, ?assignee=, ?status=."""
    results = list(REQUESTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.get("/rest/api/2/issue/<key>")
def get_request(key):
    req = REQUESTS.get(key)
    if not req:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(req))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock",
                    "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print("Jira Mock API starting on http://localhost:5002")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    app.run(port=5002, debug=True)