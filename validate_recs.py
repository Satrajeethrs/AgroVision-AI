"""CLI to validate recommendations and print a neat, human-readable summary."""
from __future__ import annotations
import json
import sys
from typing import List, Dict, Any

from llm_validator import validate_recommendations


def print_table(results: List[Dict[str, Any]]) -> None:
    # Simple table-like output
    col_widths = {"id": 12, "verdict": 10, "confidence": 10, "notes": 50}
    header = f"{ 'ID'.ljust(col_widths['id'])}  { 'VERDICT'.ljust(col_widths['verdict'])}  { 'CONF'.ljust(col_widths['confidence'])}  NOTES"
    print(header)
    print("-" * (sum(col_widths.values()) + 10))
    for r in results:
        nid = str(r.get("id", "")).ljust(col_widths["id"])[:col_widths["id"]]
        verdict = str(r.get("verdict", "")).ljust(col_widths["verdict"])[:col_widths["verdict"]]
        conf = (f"{r.get('confidence', 0):.2f}").ljust(col_widths["confidence"])[:col_widths["confidence"]]
        notes = str(r.get("notes", "")).replace("\n", " ")[:col_widths["notes"]]
        print(f"{nid}  {verdict}  {conf}  {notes}")


def main():
    # For demo purposes accept a JSON file with recommendations or use a builtin sample
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "r") as f:
            payload = json.load(f)
            recs = payload.get("recommendations")
            data = payload.get("data_summary")
    else:
        recs = [{"id": "r1", "text": "Apply 50 kg/ha of nitrogen at planting."},
                {"id": "r2", "text": "Irrigate twice per week during fruiting."}]
        data = {"soil_n": 12, "rainfall_mm": 200, "crop": "maize"}

    out = validate_recommendations(recs, data)
    print(f"Provider: {out['provider']}")
    print_table(out["results"])


if __name__ == "__main__":
    main()
