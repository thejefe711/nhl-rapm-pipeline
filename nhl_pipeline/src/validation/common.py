from dataclasses import dataclass, field
from typing import List, Optional, Any
from datetime import datetime

@dataclass
class ValidationResult:
    check: str
    passed: bool
    details: str
    severity: str = "INFO"  # INFO, WARNING, ERROR
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationReport:
    results: List[ValidationResult]
    
    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "ERROR")

    @property
    def failures(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "check": r.check,
                    "passed": r.passed,
                    "details": r.details,
                    "severity": r.severity,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.results
            ],
            "failures": [
                {
                    "check": r.check,
                    "details": r.details,
                    "severity": r.severity
                }
                for r in self.failures
            ]
        }
