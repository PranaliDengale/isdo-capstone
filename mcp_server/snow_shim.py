"""ISDO Lab C2 - Mock ServiceNow Table API (port 5001).
Run from project root:  python mcp_server/snow_shim.py
"""
import csv
from pathlib import Path

from flask import Flask, jsonify, request

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "incidents.csv"
FILTERS = ["category", "priority", "state", "assignment_group"]

app = Flask(__name__)


def load_incidents():
    """Load incidents.csv into an in-memory dict keyed by incident number."""
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found - starting empty.")
        return {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return {row["number"]: dict(row) for row in csv.DictReader(f)}


INCIDENTS = load_incidents()


@app.get("/api/now/table/incident")
def list_incidents():
    """All incidents, filtered by ?category=, ?priority=, ?state=, ?assignment_group=."""
    results = list(INCIDENTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"result": results, "total": len(results)})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    """Update fields in memory, e.g. {"state": "Escalated"}."""
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not updates:
        return jsonify({"error": "JSON body required"}), 400
    updates.pop("number", None)  # the key field cannot change
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock",
                    "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print("ServiceNow Mock API starting on http://localhost:5001")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    app.run(port=5001, debug=True)