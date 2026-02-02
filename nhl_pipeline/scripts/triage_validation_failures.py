import json
import sys
from pathlib import Path
from typing import Dict, List

def triage_failures(report_path: str) -> Dict[str, List[str]]:
    """
    Read validation report, categorize by severity.
    """
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"Report file {report_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {report_path}.")
        return {}

    failures = report.get("failures", [])
    results = report.get("results", {})

    triage = {
        'CRITICAL': [],
        'HIGH': [],
        'MEDIUM': [],
        'LOW': []
    }

    # Helper to categorize based on check name
    def categorize(check_name: str, details: str) -> str:
        check_lower = check_name.lower()
        if "schema" in check_lower or "integrity" in check_lower:
            return "CRITICAL"
        if "completeness" in check_lower and "game_count" in check_lower:
             return "CRITICAL"
        
        if "range_check" in check_lower:
             return "HIGH"
        if "correlation" in check_lower:
             return "HIGH"
        
        if "shift" in check_lower or "coverage" in check_lower:
             return "MEDIUM"
        if "stint" in check_lower:
             return "MEDIUM"
        
        if "outlier" in check_lower or "warning" in details.lower():
             return "LOW"
        
        return "MEDIUM" # Default

    # Process explicit failures list
    for failure in failures:
        # Format: "Source: check_name"
        parts = failure.split(": ", 1)
        if len(parts) == 2:
            category = parts[0]
            check = parts[1]
            severity = categorize(check, "")
            triage[severity].append(failure)
        else:
            triage["MEDIUM"].append(failure)

    # Also check detailed results for things that might be warnings but not in top-level failures
    # (The runner puts failed checks in failures list, but let's double check)
    
    return triage

def main():
    if len(sys.argv) < 2:
        print("Usage: python triage_validation_failures.py <report_json>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    triage = triage_failures(report_path)
    
    print("# Validation Failure Triage\n")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        items = triage.get(severity, [])
        if items:
            print(f"## {severity} ({len(items)})")
            for item in items:
                print(f"- {item}")
            print("")

if __name__ == "__main__":
    main()
